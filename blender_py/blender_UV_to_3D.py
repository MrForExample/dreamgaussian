import os
import sys
import pickle
import bpy
import bmesh
import time
import math
from typing import Tuple, List, Union
import numpy as np
import mathutils
from mathutils.geometry import barycentric_transform, intersect_point_tri_2d


ob = bpy.context.object
me = ob.data
bm = bmesh.new()
bm.from_mesh(me)

bmesh.ops.triangulate(bm, faces=bm.faces)

uv_layer = bm.loops.layers.uv.active


IS_ZERO_TOL = 1

__all__ = (
    'Rasterizer',
)

class Rasterizer(object):
    def __init__(
            self,
            resolution: Tuple[int, int],
            min_x: int = 0,
            max_x: int = None,
            min_y: int = 0,
            max_y: int = None,
    ):
        """
        Standard Algorithm for Filling Triangles with grids
        
        Algorithm: http://www.sunshine2k.de/coding/java/TriangleRasterization/TriangleRasterization.html
        Code: https://github.com/mikecokina/trast/blob/master/src/trast/core.py
        
        coordinates origin is at top left corner of the screen
        
        :param resolution: Tuple[int, int];  grids resolution (W, H)
        :param min_x: int; coordinate on X-axis of triangles to consider as 0
        :param max_x: int; coordinate on X-axis of trinagles to consider on WIDTH
        :param min_y:
        :param max_y:
        """
        
        # validate_resolution
        if len(resolution) != 2:
            raise ValueError('Invalid resolution, expected `W x H`')
        if any([s < 1 for s in resolution]):
            raise ValueError('Invalid resolution, expected `W x H`, where any of values has to be higher than 0')

        self._resolution = resolution
        self._w = resolution[0]
        self._h = resolution[1]

        self._min_x = min_x
        self._max_x = max_x or self._w
        self._min_y = min_y
        self._max_y = max_y or self._h

        self._dx = (self._max_x - self._min_x) / self._w
        self._dy = (self._max_y - self._min_y) / self._h
        
        self._tri_grids_coords = []

    @property
    def raster(self):
        return self._tri_grids_coords

    def _x_coo(self, x: Union[float, int]) -> int:
        return int((x - self._min_x) // self._dx)

    def _y_coo(self, y: Union[float, int]) -> int:
        return int((self._max_y - y) // self._dy)

    def _preprocessor(self, face: List, is_bottom_flat: bool) -> Union[Tuple, None]:
        v = [None, None, None]

        # Sort by y (v is sorted by y from top to bottom in perspective of points perceptual visibility).
        face = sorted(face, key=lambda i: i[1], reverse=is_bottom_flat)
        v[0] = face[0]
        # Sort by x remaning coordinates in row (to keep upcomming logic simple).
        v[1], v[2] = tuple(sorted(face[1:], key=lambda i: i[0], reverse=False))

        # Result for bottom flatted.
        # top of screen
        #      v[0]
        #   v[1] - v[2]
        # bottom of screen

        # Result for top flatted.
        # top of screen
        #   v[1] - v[2]
        #      v[0]
        # bottom of screen

        v = np.array(v)
        xs, ys = v.T[0], v.T[1]

        # Compute discrete coordinate within screen (transform origin from standard trinagle XY, to WH within screen).
        # dy1/dy2 - padding in pixels computed from top left corner.
        dy1, dy2 = self._y_coo(ys[0]), self._y_coo(ys[1])

        # Skip too small trinagles.
        if abs(dy2 - dy1) <= 1 and (self._dy / 2.0) > (ys[0] - ys[1]):
            return None

        # Step of row fill range in left (1) and right (2) direction from origin point
        # (vertex 1 on the max top position) in scale of new scale.
        invslope1 = ((xs[1] - xs[0]) / (ys[1] - ys[0])) * self._dy
        invslope2 = ((xs[2] - xs[0]) / (ys[2] - ys[0])) * self._dy

        # Current position to fill pixels within screen in x-direction.
        curx1, curx2 = xs[0], xs[0]

        return dy1, dy2, curx1, curx2, invslope1, invslope2
    
    def _add_x_line_grids(self, y, curdx1, curdx2):
        for x in range(curdx1, curdx2 + 1):
            self._tri_grids_coords.append((x, self._h - y))

    def _fill_top_flat_triangle(self, face: List):
        _ = self._preprocessor(face, is_bottom_flat=False)
        # Skip too small / unprocessable triangle.
        if _ is None:
            return
        dy1, dy2, curx1, curx2, invslope1, invslope2 = _
        curdx1, curdx2 = self._x_coo(curx1), self._x_coo(curx2)

        # Fill rows except of last line.
        for y in range(dy1, dy2, -1):            
            self._add_x_line_grids(y, curdx1, curdx2)
            
            curx1 += invslope1
            curx2 += invslope2
            curdx1, curdx2 = self._x_coo(curx1), self._x_coo(curx2)

        # Process last line.
        self._add_x_line_grids(dy2, curdx1, curdx2)
        
    def _fill_bottom_flat_triangle(self, face: List):
        _ = self._preprocessor(face, is_bottom_flat=True)
        # Skip too small / unprocessable triangle.
        if _ is None:
            return
        dy1, dy2, curx1, curx2, invslope1, invslope2 = _
        curdx1, curdx2 = self._x_coo(curx1), self._x_coo(curx2)
        # Fill rows except of last line.
        for y in range(dy1, dy2, 1):            
            self._add_x_line_grids(y, curdx1, curdx2)

            # Move in y-direction - more to bottom.
            curx1 -= invslope1
            curx2 -= invslope2

            # Recompute max-left and max-right position.
            curdx1, curdx2 = self._x_coo(curx1), self._x_coo(curx2)

        # Process last line.
        self._add_x_line_grids(dy2, curdx1, curdx2)
        
    def rasterize_triangle(self, face: Union[List, np.ndarray]) -> List:
        self._tri_grids_coords = []
        
        if isinstance(face, np.ndarray):
            face = face.tolist()

        # Sort face coordinate by Y.
        face.sort(key=lambda i: i[1], reverse=True)

        # Case when triangle si flat from top
        if abs(face[0][1] - face[1][1]) < IS_ZERO_TOL:
            self._fill_top_flat_triangle(face=face)
        elif abs(face[1][1] - face[2][1]) < IS_ZERO_TOL:
            self._fill_bottom_flat_triangle(face=face)

        # Case when vertex 4 has to be esetimated.
        else:
            """

                       o v1


                v2 o       o v4


                               o v3
            """

            x = face[0][0] + ((face[1][1] - face[0][1]) / (face[2][1] - face[0][1])) * (face[2][0] - face[0][0])
            # Split triangle to 2 separated (top flat and bottom flat)
            face_top_flat, face_bottom_flat = [face[1], face[2], [x, face[1][1]]], [face[0], face[1], [x, face[1][1]]]

            self._fill_bottom_flat_triangle(face=face_bottom_flat)
            self._fill_top_flat_triangle(face=face_top_flat)
            
        return self._tri_grids_coords

def find_3d_coord(uv_coord, active_uv_layer, face):
    u, v, w = [l[active_uv_layer].uv.to_3d() for l in face.loops]
    x, y, z = [v.co for v in face.verts]
    co = barycentric_transform(uv_coord, u, v, w, x, y, z)
    return ob.matrix_world @ co

def all_uv_coords_to_3d_data(all_faces, active_uv_layer, texture_res=1024):
    uv_grids_data_dict = {}
    
    r = Rasterizer(resolution=(texture_res, texture_res))
    
    texture_res_minus_1 = texture_res - 1
    for face in all_faces:
        uv_face = []
        for l in face.loops:
            u, v = l[active_uv_layer].uv
            i_u = min(math.floor(u * texture_res), texture_res_minus_1)
            i_v = min(math.floor((1. - v) * texture_res), texture_res_minus_1)
            uv_face.append((i_v, i_u))
        
        grids_coords = r.rasterize_triangle(face=uv_face)
        
        for coord in grids_coords:
            if coord not in uv_grids_data_dict:
                uv_coord = mathutils.Vector(( (coord[1] + 0.5) / texture_res, 1. - (coord[0] + 0.5) / texture_res )).to_3d()
                uv_coord_3d = tuple(find_3d_coord(uv_coord, active_uv_layer, face))
                uv_normal_3d = tuple(face.normal)
                # Change the coordinate system to the our 3DGS are using
                uv_coord_3d = (uv_coord_3d[0], uv_coord_3d[2], -uv_coord_3d[1])
                uv_normal_3d = (uv_normal_3d[0], uv_normal_3d[2], -uv_normal_3d[1])
                uv_grids_data_dict[coord] = (uv_coord_3d, uv_normal_3d)
                
    print(f"{len(uv_grids_data_dict)} number of uv pixels sampled from {len(all_faces)} number of faces")
                
    uv_coords, uv_coords_3d, uv_normals_3d = [], [], []
    for coord in uv_grids_data_dict:
        uv_coords.append(coord)
        uv_coords_3d.append(uv_grids_data_dict[coord][0])
        uv_normals_3d.append(uv_grids_data_dict[coord][1])
        
    uv_grids_data = (uv_coords, uv_coords_3d, uv_normals_3d)    
    
    return uv_grids_data

def save_3d_uv_data(uv_grids_data):
    path = os.path.join(
        get_persistent_directory(os.path.join("BlenderAI3D", "Data")), 
        '3d_uv_data.pkl'
    )
    with open(path, 'wb') as f:
        pickle.dump(uv_grids_data, f)
        
    print(f"[INFO] saved 3D UV data to {path}")

def get_persistent_directory(folder_name):
    if sys.platform == "win32":
        folder = os.path.join(os.path.expanduser("~"), "AppData", "Local", folder_name)
    else:
        folder = os.path.join(os.path.expanduser("~"), "." + folder_name)
    
    os.makedirs(folder, exist_ok=True)
    return folder

"""
    Draw Debug Shape
"""

def debug_cursor_uv_coord_to_3d(all_faces, active_uv_layer, texture_res=1024):
    for area in bpy.context.screen.areas:
        if area.type == 'IMAGE_EDITOR':
            loc = area.spaces.active.cursor_location.to_3d()
            break
    
    r = Rasterizer(resolution=(texture_res, texture_res))
    
    loc_uv_coord = loc.to_3d()
    texture_res_minus_1 = texture_res - 1
    for face in all_faces:
        u, v, w = [l[uv_layer].uv.to_3d() for l in face.loops]
        if intersect_point_tri_2d(loc_uv_coord, u, v, w):
            print(f"Found intersecting triangle for cursor UV coordinate: {loc_uv_coord}")
            
            # Test Rasterizer by rasterize triangle face into grids and transform it back to UV coordinates
            uv_face = []
            for l in face.loops:
                u, v = l[active_uv_layer].uv
                i_u = min(math.floor(u * texture_res), texture_res_minus_1)
                i_v = min(math.floor((1. - v) * texture_res), texture_res_minus_1)
                uv_face.append((i_v, i_u))
            
            grids_coords = r.rasterize_triangle(face=uv_face)
            coord = grids_coords[0]
            uv_coord = mathutils.Vector(( (coord[1] + 0.5) / texture_res, 1. - (coord[0] + 0.5) / texture_res )).to_3d()
            
            i_u = min(math.floor(loc_uv_coord[0] * texture_res), texture_res_minus_1)
            i_v = min(math.floor((1. - loc_uv_coord[1]) * texture_res), texture_res_minus_1)
            loc_coord = (i_v, i_u)
            
            loc_uv_coord_re = mathutils.Vector(( (loc_coord[1] + 0.5) / texture_res, 1. - (loc_coord[0] + 0.5) / texture_res )).to_3d()
            print(f"grids num: {len(grids_coords)}; coord: {coord}; uv_coord: {uv_coord}; loc_coord: {loc_coord}; loc_uv_coord: {loc_uv_coord}; loc_uv_coord_re: {loc_uv_coord_re}")
            
            bpy.context.scene.cursor.location = find_3d_coord(uv_coord, active_uv_layer, face)
            #bpy.context.scene.cursor.location = find_3d_coord(loc_uv_coord, active_uv_layer, face)
            break
    else:
        print(f"Did not find any 3D point for UV coordinate: {uv_coord}")

def debug_draw_3d_uv_data(uv_grids_data, max_sample_num=10):    
    #num_point = len(uv_grids_data)
    i = 0
    for k in uv_grids_data:
        if i < max_sample_num:
            i += 1
        else:
            break
        
        point_a = uv_grids_data[k][0]
        normal_dir = uv_grids_data[k][1]
        point_b = [point_a[j] + normal_dir[j] * 0.025 for j in range(3)]
        
        bpy.ops.mesh.primitive_uv_sphere_add(
            radius=0.001,
            location=point_a,
            scale=(1, 1, 1)
        )
        bpy.ops.mesh.primitive_uv_sphere_add(
            radius=0.001,
            location=point_b,
            scale=(1, 1, 1)
        )

# Main
start = time.time()
#all_faces = bm.faces
all_faces = [f for f in bm.faces if f.select]
uv_grids_data = all_uv_coords_to_3d_data(all_faces, uv_layer)
save_3d_uv_data(uv_grids_data)
#debug_draw_3d_uv_data(uv_grids_data)
#debug_cursor_uv_coord_to_3d(bm.faces, uv_layer)
end = time.time()
print(f"Time taken to run the code was {end-start} seconds")