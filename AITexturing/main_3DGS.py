import os
import pickle
import random
import cv2
import time
import tqdm
import numpy as np
from PIL import Image
import dearpygui.dearpygui as dpg

import torch
import torch.nn.functional as F
from pytorch_msssim import SSIM, MS_SSIM

import rembg

from common_utils import get_persistent_directory
from camera_utils import orbit_camera, OrbitCamera, MiniCam, calculate_fovX, get_projection_matrix, get_look_at_camera_pose
from mesh_based_GS_renderer import Renderer, K_nearest_neighbors, find_points_within_radius

from grid_put import mipmap_linear_grid_put_2d
from mesh import Mesh, safe_normalize

class GUI:
    def __init__(self, opt):
        self.opt = opt  # shared with the trainer's opt to support in-place modification of rendering parameters.
        self.gui = opt.gui # enable gui
        self.W = opt.W
        self.H = opt.H
        self.cam = OrbitCamera(opt.W, opt.H, r=opt.radius, fovy=opt.fovy)
        
        # default projection matrix
        self.projection_matrix = get_projection_matrix(self.cam.near, self.cam.far, self.cam.fovx, self.cam.fovy).transpose(0, 1).cuda()

        self.mode = "image"
        self.seed = "random"

        self.buffer_image = np.ones((self.W, self.H, 3), dtype=np.float32)
        self.need_update = True  # update buffer_image

        # models
        self.device = torch.device("cuda")
        self.bg_remover = None

        # renderer
        self.renderer = Renderer(sh_degree=self.opt.sh_degree)
        self.gaussain_scale_factor = 1
        
        # reference image
        self.ref_imgs = []  # [H, W, 3] in [0, 1]
        self.ref_imgs_torch = None
        self.ref_masks = []
        self.ref_masks_torch = None

        # training stuff
        self.training = False
        self.optimizer = None
        self.step = 0
        self.train_steps = 1  # steps per rendering loop
        
        # load input data
        self.load_ref_imgs()

        # override if provide a checkpoint
        self.uv_grids_data = self.load_3d_uv_data()
        #self.renderer.initialize(self.uv_grids_data, num_pts=self.opt.num_pts)
        self.renderer.initialize(self.load_3d_mesh(), num_pts=self.opt.num_pts)

        if self.gui:
            dpg.create_context()
            self.register_dpg()
            self.test_step()

    def __del__(self):
        if self.gui:
            dpg.destroy_context()
            
    def load_3d_uv_data(self):
        path = os.path.join(
            get_persistent_directory(
                os.path.join(
                    self.opt.persistent_folder, 
                    self.opt.persistent_data_folder
                )
            ), 
            self.opt.uv_data_name
        )
        
        uv_grids_data = None

        if os.path.exists(path):
            with open(path, 'rb') as f:
                uv_grids_data = pickle.load(f)
            
            print(f"[INFO] loaded 3D UV data from {path}")
        else:
            print(f"[ERROR] 3D UV data {path} does not exist!")
            
        return uv_grids_data
    
    def load_3d_mesh(self):
        path = os.path.join(
            get_persistent_directory(
                os.path.join(
                    self.opt.persistent_folder, 
                    self.opt.persistent_meshes_folder
                )
            ), 
            self.opt.mesh_name
        )
        
        mesh = None

        if os.path.exists(path):
            mesh = Mesh.load(path, resize=False)
            
            print(f"[INFO] loaded 3D Mesh data from {path}")
        else:
            print(f"[ERROR] 3D Mesh data {path} does not exist!")
            
        return mesh
            
    def load_ref_imgs(self):
        # load reference images with ideal style from the folder
        folder_path = os.path.join(
            get_persistent_directory(
                os.path.join(
                    self.opt.persistent_folder, 
                    self.opt.persistent_inputs_folder
                )
            ), 
            self.opt.reference
        )
        print(f'[INFO] load reference images with ideal style from the {folder_path}...')
        
        list_files = os.listdir(folder_path)
        list_files.sort()
        num_ref_imgs = 0
        for file_name in list_files:
            num_ref_imgs += self.load_ref_img(os.path.join(folder_path, file_name))
        
        print(f'[INFO] loaded {num_ref_imgs} reference images with ideal style from the {folder_path}...')
    
    def load_ref_img(self, file):
        # load reference image with ideal style
        #print(f'[INFO] load reference image with ideal style from {file}...')
        img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        if img is not None:
            
            if img.shape[-1] == 3:
                if self.bg_remover is None:
                    self.bg_remover = rembg.new_session()
                img = rembg.remove(img, session=self.bg_remover)
            
            img = cv2.resize(
                img, (self.opt.ref_size_W, self.opt.ref_size_H), interpolation=cv2.INTER_AREA
            )
            img = img.astype(np.float32) / 255.0

            mask = img[..., 3:]
            # white bg
            img = img[..., :3] * mask + (1 - mask)

            # bgr to rgb
            img = img[..., ::-1].copy()
            
            self.ref_imgs.append(img)
            self.ref_masks.append(mask)
            
            return 1
        else:
            return 0

    def prepare_train(self):

        self.step = 0

        # setup training
        self.renderer.gaussians.training_setup(self.opt)
        # do not do progressive sh-level
        self.renderer.gaussians.active_sh_degree = self.renderer.gaussians.max_sh_degree
        self.optimizer = self.renderer.gaussians.optimizer
        
        self.ssim_loss = SSIM(data_range=1, size_average=True, channel=3)
        
        # reference images
        ref_imgs_torch_list = []
        ref_masks_torch_list = []
        for i in range(len(self.ref_imgs)):
            ref_imgs_torch_list.append(self.img_to_torch(self.ref_imgs[i]))
            ref_masks_torch_list.append(self.img_to_torch(self.ref_masks[i]))
            
        self.ref_imgs_torch = torch.cat(ref_imgs_torch_list)
        self.ref_masks_torch = torch.cat(ref_masks_torch_list)
        
        # calculate all the reference camera azimuth angles
        self.ref_imgs_num = len(self.ref_imgs)
        self.all_ref_angles = []
        angle_interval = 360 / self.ref_imgs_num
        now_angle = self.opt.init_angle
        for i in range(self.ref_imgs_num):
            self.all_ref_angles.append(now_angle)
            now_angle = (now_angle + angle_interval) % 360
            print(f"Reference angle: {now_angle}")

    def img_to_torch(self, img):
        img_torch = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.device)
        img_torch = F.interpolate(img_torch, (self.opt.ref_size_H, self.opt.ref_size_W), mode="bilinear", align_corners=False).contiguous()
        return img_torch

    def train_step(self):
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()

        for _ in range(self.train_steps):

            self.step += 1
            step_ratio = min(1, self.step / self.opt.iters)

            # update lr
            self.renderer.gaussians.update_learning_rate(self.step)

            loss = 0

            ### calculate loss between reference and rendered image from known view
            #for i in range(self.ref_imgs_num):
                
            i = random.randint(0, self.ref_imgs_num-1)
                
            ref_pose = orbit_camera(self.opt.elevation, self.all_ref_angles[i], self.opt.radius)
            ref_cam = MiniCam(ref_pose, self.opt.ref_size_W, self.opt.ref_size_H, self.cam.fovy, self.cam.fovx, self.cam.near, self.cam.far, self.projection_matrix)
            
            bg_color = torch.tensor([1, 1, 1] if np.random.rand() > self.opt.invert_bg_prob else [0, 0, 0], dtype=torch.float32, device=self.device)
            out = self.renderer.render(ref_cam, bg_color=bg_color)
            
            # rgb loss
            image = out["image"] # [3, H, W] in [0, 1]
            ref_mask = self.ref_masks_torch[i]
            image_masked = image * ref_mask
            ref_image_masked = self.ref_imgs_torch[i] * ref_mask
            loss += (1 - self.opt.lambda_ssim) * 10000 * step_ratio * F.mse_loss(image_masked, ref_image_masked)

            # alpha loss
            mask = out["alpha"] # [1, H, W] in [0, 1]
            loss += self.opt.lambda_alpha * 10000 * step_ratio * F.mse_loss(mask, ref_mask)
            
            # D-SSIM loss
            # [1, 3, H, W] in [0, 1]
            X = ref_image_masked.unsqueeze(0)
            Y = image_masked.unsqueeze(0)
            loss += self.opt.lambda_ssim * 10000 * step_ratio * (1 - self.ssim_loss(X, Y))
            
            # Reference offset loss
            offset_norm = self.renderer.gaussians.get_xyz_offset.norm(dim=-1, keepdim=True)
            loss += self.opt.lambda_offset * 10000 * step_ratio * torch.mean(offset_norm)
            
            # Alpha penalty loss
            loss += self.opt.lambda_offset_opacity * 10000 * step_ratio * torch.mean(offset_norm.detach() * self.renderer.gaussians.get_opacity)

            # optimize step
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            # densify and prune
            if self.step >= self.opt.density_start_iter and self.step <= self.opt.density_end_iter:
                viewspace_point_tensor, visibility_filter, radii = out["viewspace_points"], out["visibility_filter"], out["radii"]
                self.renderer.gaussians.max_radii2D[visibility_filter] = torch.max(self.renderer.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                self.renderer.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if self.step % self.opt.densification_interval == 0:
                    #self.renderer.gaussians.densify_and_prune(self.opt.densify_grad_threshold, min_opacity=0.01, extent=4, max_screen_size=1)
                    #self.renderer.gaussians.densify_and_prune_by_compatness(self.opt.K, min_opacity=0.01, extent=4, max_screen_size=1)
                    
                    #self.renderer.gaussians.densify_by_clone_and_split(self.opt.densify_grad_threshold, extent=4)
                    self.renderer.gaussians.densify_by_compatness(self.opt.K)
                    self.renderer.gaussians.prune(min_opacity=0.01, extent=4, max_screen_size=1, max_offset=0.1)
                
                if self.step % self.opt.opacity_reset_interval == 0:
                    self.renderer.gaussians.reset_opacity()

        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)

        self.need_update = True

        if self.gui:
            dpg.set_value("_log_train_time", f"{t:.4f}ms")
            dpg.set_value(
                "_log_train_log",
                f"step = {self.step: 5d} (+{self.train_steps: 2d}) loss = {loss.item():.4f}",
            )

        # dynamic train steps (no need for now)
        # max allowed train time per-frame is 500 ms
        # full_t = t / self.train_steps * 16
        # train_steps = min(16, max(4, int(16 * 500 / full_t)))
        # if train_steps > self.train_steps * 1.2 or train_steps < self.train_steps * 0.8:
        #     self.train_steps = train_steps

    @torch.no_grad()
    def test_step(self):
        # ignore if no need to update
        if not self.need_update:
            return

        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()

        # should update image
        if self.need_update:
            # render image

            cur_cam = MiniCam(
                self.cam.pose,
                self.W,
                self.H,
                self.cam.fovy,
                self.cam.fovx,
                self.cam.near,
                self.cam.far,
                self.projection_matrix
            )

            out = self.renderer.render(cur_cam, self.gaussain_scale_factor)

            buffer_image = out[self.mode]  # [3, H, W]

            if self.mode in ['depth', 'alpha']:
                buffer_image = buffer_image.repeat(3, 1, 1)
                if self.mode == 'depth':
                    buffer_image = (buffer_image - buffer_image.min()) / (buffer_image.max() - buffer_image.min() + 1e-20)

            buffer_image = F.interpolate(
                buffer_image.unsqueeze(0),
                size=(self.H, self.W),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

            self.buffer_image = (
                buffer_image.permute(1, 2, 0)
                .contiguous()
                .clamp(0, 1)
                .contiguous()
                .detach()
                .cpu()
                .numpy()
            )

            self.need_update = False

        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)

        if self.gui:
            dpg.set_value("_log_gaussians_number", self.renderer.gaussians.get_gaussians_num)
            dpg.set_value("_log_infer_time", f"{t:.4f}ms ({int(1000/t)} FPS)")
            dpg.set_value(
                "_texture", self.buffer_image
            )  # buffer must be contiguous, else seg fault!
            
    @torch.no_grad()
    def bake_3DGS_texture(self):
        """
            
        """
        start = time.time()

        if self.uv_grids_data is not None:
            uv_coords, uv_coords_3d, uv_normals_3d = self.uv_grids_data

            uv_coords_np = np.asarray(uv_coords)
            uv_coords_3d_np = np.asarray(uv_coords_3d, dtype=np.float32)
            uv_normals_3d_np = np.asarray(uv_normals_3d, dtype=np.float32)
            uv_coords_3d_tensor = torch.tensor(uv_coords_3d_np).float().cuda()
            
            # Make sure there is no zero value inside uv_normals_3d_np otherwise w2c = np.linalg.inv(c2w) in MiniCam() will throw numpy.linalg.LinAlgError: Singular matrix error
            eps = np.finfo(np.float32).eps
            uv_normals_3d_np[abs(uv_normals_3d_np) < eps] = eps
            
            res = 101
            znear, zfar = 0.0001, 100.
            fovY = np.deg2rad(self.opt.fovy_pixel_cam)
            fovX = calculate_fovX(res, res, fovY)
            
            pixel_projection_matrix = get_projection_matrix(znear, zfar, fovX, fovY).transpose(0, 1).cuda()
            
            uv_pixels_color = []
            
            pixels_num = uv_coords_3d_tensor.shape[0]
            coords_group_num = 1000
            groups = [i_g for i_g in range(coords_group_num, pixels_num, coords_group_num)]
            if pixels_num % coords_group_num > 0:
                groups.append(pixels_num)
                
            i_g_last = 0
            for i_g in groups:
                print(f"[INFO] calculate the {self.opt.K_bake} nearest gaussians for each uv sample points")
                _, idx = K_nearest_neighbors(self.renderer.gaussians.get_xyz, self.opt.K_bake+1, uv_coords_3d_tensor[i_g_last:i_g])
                #idx = find_points_within_radius(uv_coords_3d_tensor, self.renderer.gaussians.get_xyz, 0.1)
                
                print(f"[INFO] render the color for {i_g - i_g_last} number of pixels")
                
                i_g_last = i_g

                for i in range(idx.shape[0]):

                    pose = get_look_at_camera_pose(uv_coords_3d_np[i], uv_normals_3d_np[i], look_distance=self.opt.pixel_look_distance)
                    
                    pixel_cam = MiniCam(pose, res, res, fovY, fovX, znear, zfar, pixel_projection_matrix)

                    out = self.renderer.render(pixel_cam, self.gaussain_scale_factor, idx[i])
                    #out = self.renderer.render(pixel_cam, self.gaussain_scale_factor, torch.tensor(idx[i], dtype=torch.int).cuda())
                    
                    #out = self.renderer.render(pixel_cam, self.gaussain_scale_factor)

                    img = out["image"]
                    pixel_color = img[:, img.shape[1] // 2 + 1, img.shape[2] // 2 + 1].tolist()
                    pixel_color.append(
                        1. #torch.squeeze(out["alpha"], dim=(1, 2)).tolist()[0]
                    )
                    uv_pixels_color.append(pixel_color)
                    
                    #print(f"Sample i: {i}; uv_coords_3d[i]: {uv_coords_3d[i]}, uv_normals_3d[i]: {uv_normals_3d[i]}, cam_pos: {pixel_cam.camera_center}")
                    #self.save_rgba_image(out["image"].permute(1, 2, 0).contiguous().detach().cpu().numpy(), "C:\\Users\\reall\\AppData\\Local\\BlenderAI3D\\Outputs", f"rendered_texture_{i}")
                    #break
                
            # bake and save the texture
            uv_pixels_color_np = np.asarray(uv_pixels_color, dtype=np.float32)
            self.bake_uv_pixels_to_texture(uv_coords_np, uv_pixels_color_np)
            
        end = time.time()
        print(f"Time taken to bake 3DGS texture was {end-start} seconds")
        
    def bake_uv_pixels_to_texture(self, uv_coords_np, uv_pixels_color_np):
        # bake the pixels color into texture
        texture = np.zeros([self.opt.ref_size_H, self.opt.ref_size_W, 4], dtype=np.float32)
        u_indices = uv_coords_np[:, 0]
        v_indices = uv_coords_np[:, 1]
        texture[u_indices, v_indices] = uv_pixels_color_np
        
        # save baked texture to a image
        folder_path = get_persistent_directory(
            os.path.join(
                self.opt.persistent_folder, 
                self.opt.persistent_outputs_folder
            )
        )
        self.save_rgba_image(texture, folder_path, self.opt.bake_texture_name)
        
    def save_rgba_image(self, image, folder_path, prefix="img"):
        """
        Args:
            image (NDArray[float32]), shape: (H, W, 4 or 3), value range: [0., 1.]
        """
        path = os.path.join(folder_path, prefix)
        
        if image.shape[2] == 4:
            img_format = 'RGBA'
        elif image.shape[2] == 3:
            img_format = 'RGB'
        else:
            print(f"[ERROR] Cannot save image to {path}, the shape of the image has to be ({image.shape[0]}, {image.shape[1]}, 4 or 3) instead of {image.shape}")
            return
        
        img = Image.fromarray(np.uint8(image * 255), img_format)
        img.save(f"{path}.png")
        
        print(f"[INFO] Image saved to {path}")
        
    def save_3DGS_model(self):        
        path = os.path.join(
            get_persistent_directory(
                os.path.join(
                    self.opt.persistent_folder, 
                    self.opt.persistent_data_folder
                )
            ), 
            self.opt.save_3DGS_model_name
        )

        model_args = self.renderer.gaussians.capture()
        with open(path, 'wb') as f:
            pickle.dump(model_args, f)
            
        print(f"[INFO] saved 3DGS model to {path}")
            
    def load_3DGS_model(self):        
        path = os.path.join(
            get_persistent_directory(
                os.path.join(
                    self.opt.persistent_folder, 
                    self.opt.persistent_data_folder
                )
            ), 
            self.opt.save_3DGS_model_name
        )
        
        if os.path.exists(path):
            with open(path, 'rb') as f:
                model_args = pickle.load(f)
                
            self.renderer.gaussians.restore(model_args, self.opt)
            
            print(f"[INFO] loaded 3DGS model from {path}")
        else:
            print(f"[ERROR] 3DGS model {path} does not exist!")

    @torch.no_grad()
    def save_model(self, mode='geo', texture_size=1024):
        os.makedirs(self.opt.outdir, exist_ok=True)
        if mode == 'geo':
            path = os.path.join(self.opt.outdir, self.opt.save_path + '_mesh.ply')
            mesh = self.renderer.gaussians.extract_mesh(path, self.opt.density_thresh)
            mesh.write_ply(path)

        elif mode == 'geo+tex':
            path = os.path.join(self.opt.outdir, self.opt.save_path + '_mesh.' + self.opt.mesh_format)
            mesh = self.renderer.gaussians.extract_mesh(path, self.opt.density_thresh)

            # perform texture extraction
            print(f"[INFO] unwrap uv...")
            h = w = texture_size
            mesh.auto_uv()
            mesh.auto_normal()

            albedo = torch.zeros((h, w, 3), device=self.device, dtype=torch.float32)
            cnt = torch.zeros((h, w, 1), device=self.device, dtype=torch.float32)

            # self.prepare_train() # tmp fix for not loading 0123
            # vers = [0]
            # hors = [0]
            vers = [0] * 8 + [-45] * 8 + [45] * 8 + [-89.9, 89.9]
            hors = [0, 45, -45, 90, -90, 135, -135, 180] * 3 + [0, 0]

            render_resolution = 512

            import nvdiffrast.torch as dr

            if not self.opt.force_cuda_rast and (not self.opt.gui or os.name == 'nt'):
                glctx = dr.RasterizeGLContext()
            else:
                glctx = dr.RasterizeCudaContext()

            for ver, hor in zip(vers, hors):
                # render image
                pose = orbit_camera(ver, hor, self.cam.radius)

                cur_cam = MiniCam(
                    pose,
                    render_resolution,
                    render_resolution,
                    self.cam.fovy,
                    self.cam.fovx,
                    self.cam.near,
                    self.cam.far,
                )
                
                cur_out = self.renderer.render(cur_cam)

                rgbs = cur_out["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]

                # enhance texture quality with zero123 [not working well]
                # if self.opt.guidance_model == 'zero123':
                #     rgbs = self.guidance.refine(rgbs, [ver], [hor], [0])
                    # import kiui
                    # kiui.vis.plot_image(rgbs)
                    
                # get coordinate in texture image
                pose = torch.from_numpy(pose.astype(np.float32)).to(self.device)
                proj = torch.from_numpy(self.cam.perspective.astype(np.float32)).to(self.device)

                v_cam = torch.matmul(F.pad(mesh.v, pad=(0, 1), mode='constant', value=1.0), torch.inverse(pose).T).float().unsqueeze(0)
                v_clip = v_cam @ proj.T
                rast, rast_db = dr.rasterize(glctx, v_clip, mesh.f, (render_resolution, render_resolution))

                depth, _ = dr.interpolate(-v_cam[..., [2]], rast, mesh.f) # [1, H, W, 1]
                depth = depth.squeeze(0) # [H, W, 1]

                alpha = (rast[0, ..., 3:] > 0).float()

                uvs, _ = dr.interpolate(mesh.vt.unsqueeze(0), rast, mesh.ft)  # [1, 512, 512, 2] in [0, 1]

                # use normal to produce a back-project mask
                normal, _ = dr.interpolate(mesh.vn.unsqueeze(0).contiguous(), rast, mesh.fn)
                normal = safe_normalize(normal[0])

                # rotated normal (where [0, 0, 1] always faces camera)
                rot_normal = normal @ pose[:3, :3]
                viewcos = rot_normal[..., [2]]

                mask = (alpha > 0) & (viewcos > 0.5)  # [H, W, 1]
                mask = mask.view(-1)

                uvs = uvs.view(-1, 2).clamp(0, 1)[mask]
                rgbs = rgbs.view(3, -1).permute(1, 0)[mask].contiguous()
                
                # update texture image
                cur_albedo, cur_cnt = mipmap_linear_grid_put_2d(
                    h, w,
                    uvs[..., [1, 0]] * 2 - 1,
                    rgbs,
                    min_resolution=256,
                    return_count=True,
                )
                
                # albedo += cur_albedo
                # cnt += cur_cnt
                mask = cnt.squeeze(-1) < 0.1
                albedo[mask] += cur_albedo[mask]
                cnt[mask] += cur_cnt[mask]

            mask = cnt.squeeze(-1) > 0
            albedo[mask] = albedo[mask] / cnt[mask].repeat(1, 3)

            mask = mask.view(h, w)

            albedo = albedo.detach().cpu().numpy()
            mask = mask.detach().cpu().numpy()

            # dilate texture
            from sklearn.neighbors import NearestNeighbors
            from scipy.ndimage import binary_dilation, binary_erosion

            inpaint_region = binary_dilation(mask, iterations=32)
            inpaint_region[mask] = 0

            search_region = mask.copy()
            not_search_region = binary_erosion(search_region, iterations=3)
            search_region[not_search_region] = 0

            search_coords = np.stack(np.nonzero(search_region), axis=-1)
            inpaint_coords = np.stack(np.nonzero(inpaint_region), axis=-1)

            knn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(
                search_coords
            )
            _, indices = knn.kneighbors(inpaint_coords)

            albedo[tuple(inpaint_coords.T)] = albedo[tuple(search_coords[indices[:, 0]].T)]

            mesh.albedo = torch.from_numpy(albedo).to(self.device)
            mesh.write(path)

        else:
            path = os.path.join(self.opt.outdir, self.opt.save_path + '_model.ply')
            self.renderer.gaussians.save_ply(path)

        print(f"[INFO] save model to {path}.")

    def register_dpg(self):
        ### register texture

        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                self.W,
                self.H,
                self.buffer_image,
                format=dpg.mvFormat_Float_rgb,
                tag="_texture",
            )

        ### register window

        # the rendered image, as the primary window
        with dpg.window(
            tag="_primary_window",
            width=self.W,
            height=self.H,
            pos=[0, 0],
            no_move=True,
            no_title_bar=True,
            no_scrollbar=True,
        ):
            # add the texture
            dpg.add_image("_texture")

        # dpg.set_primary_window("_primary_window", True)

        # control window
        with dpg.window(
            label="Control",
            tag="_control_window",
            width=600,
            height=self.H,
            pos=[self.W, 0],
            no_move=True,
            no_title_bar=True,
        ):
            # button theme
            with dpg.theme() as theme_button:
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (23, 3, 18))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (51, 3, 47))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (83, 18, 83))
                    dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                    dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)

            # timer stuff
            with dpg.group(horizontal=True):
                dpg.add_text("Infer time: ")
                dpg.add_text("no data", tag="_log_infer_time")

            def callback_setattr(sender, app_data, user_data):
                setattr(self, user_data, app_data)

            # save stuff
            with dpg.collapsing_header(label="Save", default_open=True):

                # save current model
                with dpg.group(horizontal=True):
                    dpg.add_text("Save: ")
                    dpg.add_text("Gaussians Num: ", tag="_log_gaussians_number")

                    def callback_save(sender, app_data, user_data):
                        self.save_model(mode=user_data)
                        
                    def callback_save_3DGS_model(sender, app_data):
                        self.save_3DGS_model()

                    dpg.add_button(
                        label="3DGS_model",
                        tag="_button_save_3DGS_model",
                        callback=callback_save_3DGS_model,
                    )
                    dpg.bind_item_theme("_button_save_3DGS_model", theme_button)

                    dpg.add_button(
                        label="model",
                        tag="_button_save_model",
                        callback=callback_save,
                        user_data='model',
                    )
                    dpg.bind_item_theme("_button_save_model", theme_button)

                    dpg.add_button(
                        label="geo",
                        tag="_button_save_mesh",
                        callback=callback_save,
                        user_data='geo',
                    )
                    dpg.bind_item_theme("_button_save_mesh", theme_button)

                    dpg.add_button(
                        label="geo+tex",
                        tag="_button_save_mesh_with_tex",
                        callback=callback_save,
                        user_data='geo+tex',
                    )
                    dpg.bind_item_theme("_button_save_mesh_with_tex", theme_button)

                    dpg.add_input_text(
                        label="",
                        default_value=self.opt.save_path,
                        callback=callback_setattr,
                        user_data="save_path",
                    )
                    
            # load stuff
            with dpg.collapsing_header(label="Load", default_open=True):

                # load lastest model
                with dpg.group(horizontal=True):
                    dpg.add_text("Load: ")
                        
                    def callback_load_3DGS_model(sender, app_data):
                        self.load_3DGS_model()
                        self.need_update = True

                    dpg.add_button(
                        label="3DGS_model",
                        tag="_button_load_3DGS_model",
                        callback=callback_load_3DGS_model,
                    )
                    dpg.bind_item_theme("_button_load_3DGS_model", theme_button)
                    
            # bake stuff
            with dpg.collapsing_header(label="Bake Texture Data", default_open=True):

                # bake current 3DGS render into texture data
                with dpg.group(horizontal=True):
                    dpg.add_text("Bake: ")
                        
                    def callback_bake_3DGS_texture(sender, app_data):
                        self.bake_3DGS_texture()

                    dpg.add_button(
                        label="bake_3DGS_texture",
                        tag="_button_bake_3DGS_texture",
                        callback=callback_bake_3DGS_texture,
                    )
                    dpg.bind_item_theme("_button_bake_3DGS_texture", theme_button)

            # training stuff
            with dpg.collapsing_header(label="Train", default_open=True):
                # lr and train button
                with dpg.group(horizontal=True):
                    dpg.add_text("Train: ")

                    def callback_train(sender, app_data):
                        if self.training:
                            self.training = False
                            dpg.configure_item("_button_train", label="start")
                        else:
                            self.prepare_train()
                            self.training = True
                            dpg.configure_item("_button_train", label="stop")

                    # dpg.add_button(
                    #     label="init", tag="_button_init", callback=self.prepare_train
                    # )
                    # dpg.bind_item_theme("_button_init", theme_button)

                    dpg.add_button(
                        label="start", tag="_button_train", callback=callback_train
                    )
                    dpg.bind_item_theme("_button_train", theme_button)

                with dpg.group(horizontal=True):
                    dpg.add_text("", tag="_log_train_time")
                    dpg.add_text("", tag="_log_train_log")

            # rendering options
            with dpg.collapsing_header(label="Rendering", default_open=True):
                # mode combo
                def callback_change_mode(sender, app_data):
                    self.mode = app_data
                    self.need_update = True

                dpg.add_combo(
                    ("image", "depth", "alpha"),
                    label="mode",
                    default_value=self.mode,
                    callback=callback_change_mode,
                )

                # fov slider
                def callback_set_fovy(sender, app_data):
                    self.cam.fovy = np.deg2rad(app_data)
                    self.need_update = True

                dpg.add_slider_int(
                    label="FoV (vertical)",
                    min_value=1,
                    max_value=120,
                    format="%d deg",
                    default_value=np.rad2deg(self.cam.fovy),
                    callback=callback_set_fovy,
                )

                def callback_set_gaussain_scale(sender, app_data):
                    self.gaussain_scale_factor = app_data
                    self.need_update = True

                dpg.add_slider_float(
                    label="gaussain scale",
                    min_value=0,
                    max_value=1,
                    format="%.2f",
                    default_value=self.gaussain_scale_factor,
                    callback=callback_set_gaussain_scale,
                )

        ### register camera handler

        def callback_camera_drag_rotate_or_draw_mask(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.orbit(dx, dy)
            self.need_update = True

        def callback_camera_wheel_scale(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            delta = app_data

            self.cam.scale(delta)
            self.need_update = True

        def callback_camera_drag_pan(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.pan(dx, dy)
            self.need_update = True

        def callback_set_mouse_loc(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            # just the pixel coordinate in image
            self.mouse_loc = np.array(app_data)

        with dpg.handler_registry():
            # for camera moving
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Left,
                callback=callback_camera_drag_rotate_or_draw_mask,
            )
            dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Middle, callback=callback_camera_drag_pan
            )

        dpg.create_viewport(
            title="3DGSTexturing",
            width=self.W + 600,
            height=self.H + (45 if os.name == "nt" else 0),
            resizable=False,
        )

        ### global theme
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                # set all padding to 0 to avoid scroll bar
                dpg.add_theme_style(
                    dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core
                )

        dpg.bind_item_theme("_primary_window", theme_no_padding)

        dpg.setup_dearpygui()

        ### register a larger font
        # get it from: https://github.com/lxgw/LxgwWenKai/releases/download/v1.300/LXGWWenKai-Regular.ttf
        if os.path.exists("LXGWWenKai-Regular.ttf"):
            with dpg.font_registry():
                with dpg.font("LXGWWenKai-Regular.ttf", 18) as default_font:
                    dpg.bind_font(default_font)

        # dpg.show_metrics()

        dpg.show_viewport()

    def render(self):
        assert self.gui
        while dpg.is_dearpygui_running():
            # update texture every frame
            if self.training:
                self.train_step()
            self.test_step()
            dpg.render_dearpygui_frame()
    
    # no gui mode
    def train(self, iters=500):
        if iters > 0:
            self.prepare_train()
            for i in tqdm.trange(iters):
                self.train_step()
            # do a last prune
            self.renderer.gaussians.prune(min_opacity=0.01, extent=1, max_screen_size=1)
        # save
        self.save_model(mode='model')
        self.save_model(mode='geo+tex')

if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to the yaml config file")
    args, extras = parser.parse_known_args()

    # override default config from cli
    opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))

    gui = GUI(opt)

    if opt.gui:
        gui.render()
    else:
        gui.train(opt.iters)
