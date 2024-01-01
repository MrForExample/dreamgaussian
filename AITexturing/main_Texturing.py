import os
import cv2
import random
import time
import tqdm
import numpy as np
import dearpygui.dearpygui as dpg
from PIL import Image

import torch
import torch.nn.functional as F
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from pytorch_msssim import SSIM, MS_SSIM

import trimesh
import rembg

from common_utils import get_persistent_directory
from camera_utils import orbit_camera, OrbitCamera
from mesh_renderer import Renderer

# from kiui.lpips import LPIPS

class GUI:
    def __init__(self, opt):
        self.opt = opt  # shared with the trainer's opt to support in-place modification of rendering parameters.
        self.gui = opt.gui # enable gui
        self.W = opt.W
        self.H = opt.H
        self.cam = OrbitCamera(opt.W, opt.H, r=opt.radius, fovy=opt.fovy)

        self.mode = "image"

        self.buffer_image = np.ones((self.W, self.H, 3), dtype=np.float32)
        self.need_update = True  # update buffer_image

        # models
        self.device = torch.device("cuda")

        # renderer
        self.renderer = Renderer(opt).to(self.device)

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
        
        if self.gui:
            dpg.create_context()
            self.register_dpg()
            self.test_step()

    def __del__(self):
        if self.gui:
            dpg.destroy_context()
            
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
        self.optimizer = torch.optim.Adam(self.renderer.get_params())
        #self.ssim_loss = SSIM(data_range=1, size_average=True, channel=3)
        self.ms_ssim_loss = MS_SSIM(data_range=1, size_average=True, channel=3)
        # Also tried lpips library, in this setup lpips loss can only produce noise artifact
        self.lpips_loss = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to(self.device) if self.opt.lambda_lpips > 0 else None
        
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

            ### calculate loss between reference and rendered image from known view
            loss = 0
            for i_b in range(self.opt.batch_size):
                
                i = random.randint(0, self.ref_imgs_num-1)

                #ssaa = min(2.0, max(0.125, 2 * np.random.random()))
                ssaa = 1
                
                # render output
                ref_pose = orbit_camera(self.opt.elevation, self.all_ref_angles[i], self.opt.radius)
                ref_cam = (ref_pose, self.cam.perspective)
                out = self.renderer.render(*ref_cam, self.opt.ref_size_H, self.opt.ref_size_W, ssaa=ssaa)

                image = out["image"]    # [H, W, 3] in [0, 1]
                image = image.permute(2, 0, 1).contiguous()  # [3, H, W] in [0, 1]
                ref_mask = self.ref_masks_torch[i]
                image_masked = image * ref_mask
                ref_image_masked = self.ref_imgs_torch[i] * ref_mask
                
                # rgb loss
                loss += (1 - self.opt.lambda_ssim) * F.mse_loss(image_masked, ref_image_masked)
                
                # D-SSIM loss
                # [1, 3, H, W] in [0, 1]
                X = ref_image_masked.unsqueeze(0)
                Y = image_masked.unsqueeze(0)
                #loss += self.opt.lambda_ssim * (1 - self.ssim_loss(X, Y))
                loss += self.opt.lambda_ssim * (1 - self.ms_ssim_loss(X, Y))
                
                # lpips loss
                if self.lpips_loss is not None:
                    loss += self.opt.lambda_lpips * (1 - self.lpips_loss(Y, X))
                
                #self.save_test_images(image, "./logs/FrontRender", "test_angle_48")
                
            # import kiui
            # kiui.lo(hor, ver)
            # kiui.vis.plot_image(image)

            # optimize step
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

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

            out = self.renderer.render(self.cam.pose, self.cam.perspective, self.H, self.W)

            buffer_image = out[self.mode]  # [H, W, 3]

            if self.mode in ['depth', 'alpha']:
                buffer_image = buffer_image.repeat(1, 1, 3)
                if self.mode == 'depth':
                    buffer_image = (buffer_image - buffer_image.min()) / (buffer_image.max() - buffer_image.min() + 1e-20)

            self.buffer_image = buffer_image.contiguous().clamp(0, 1).detach().cpu().numpy()

            self.need_update = False

        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)

        if self.gui:
            dpg.set_value("_log_infer_time", f"{t:.4f}ms ({int(1000/t)} FPS)")
            dpg.set_value(
                "_texture", self.buffer_image
            )  # buffer must be contiguous, else seg fault!
            
    def save_test_images(self, image, save_path, prefix=""):
        print(f"[SAVE Image] Shape: {image.shape}")
        img = image.contiguous().clamp(0, 1).detach().cpu().numpy() * 255
        img = Image.fromarray(np.uint8(img), 'RGB')
        img.save(f"{save_path}/img_{prefix}.png")
    
    def save_model(self):
        mesh_path = os.path.join(
            get_persistent_directory(
                os.path.join(
                    self.opt.persistent_folder, 
                    self.opt.persistent_outputs_folder
                )
            ), 
            self.opt.saved_mesh_name
        )
    
        self.renderer.export_mesh(mesh_path)

        print(f"[INFO] save model to {mesh_path}.")

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

            # init stuff
            with dpg.collapsing_header(label="Initialize", default_open=True):

                # save current model
                with dpg.group(horizontal=True):
                    dpg.add_text("Save: ")

                    dpg.add_button(
                        label="model",
                        tag="_button_save_model",
                        callback=self.save_model,
                    )
                    dpg.bind_item_theme("_button_save_model", theme_button)

                    dpg.add_input_text(
                        label="",
                        default_value=self.opt.save_path,
                        callback=callback_setattr,
                        user_data="save_path",
                    )

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
                    ("image", "depth", "alpha", "normal"),
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
            title="AI Texturing",
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
        # save
        self.save_model()
        

if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to the yaml config file")
    args, extras = parser.parse_known_args()

    # override default config from cli
    opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))

    # auto find mesh from stage 1
    mesh_path = os.path.join(
        get_persistent_directory(
            os.path.join(
                opt.persistent_folder, 
                opt.persistent_meshes_folder
            )
        ), 
        opt.mesh_name
    )
    
    if os.path.exists(mesh_path):
        opt.mesh = mesh_path
    else:
        raise ValueError(f"Cannot find mesh from {mesh_path}, must specify --mesh explicitly!")

    gui = GUI(opt)

    if opt.gui:
        gui.render()
    else:
        gui.train(opt.iters_refine)
