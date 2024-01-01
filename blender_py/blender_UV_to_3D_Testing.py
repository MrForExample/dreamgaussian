import bpy
import bmesh
import mathutils
from mathutils.geometry import barycentric_transform, intersect_point_tri_2d, closest_point_on_tri


ob = bpy.context.object
me = ob.data
bm = bmesh.new()
bm.from_mesh(me)

bmesh.ops.triangulate(bm, faces=bm.faces)

uv_layer = bm.loops.layers.uv.active

def tri_rasterization():
    """
        Bresenham Algorithm for Filling Triangles with grids
        
        According to: http://www.sunshine2k.de/coding/java/TriangleRasterization/TriangleRasterization.html
        - Bresenham Algorithm is usually the fastest due to its integer-only calculations.
        - Standard Algorithm is efficient but may involve floating-point operations.
    """
    
    
    return

def find_3d_coord(uv_coord, face):
    u, v, w = [l[uv_layer].uv.to_3d() for l in face.loops]
    x, y, z = [v.co for v in face.verts]
    co = barycentric_transform(uv_coord, u, v, w, x, y, z)
    return ob.matrix_world @ co
    
def uv_coord_to_3d(uv_coord):
    uv_coord = uv_coord.to_3d()
    for face in bm.faces:
        u, v, w = [l[uv_layer].uv.to_3d() for l in face.loops]
        if intersect_point_tri_2d(uv_coord, u, v, w):
            print(closest_point_on_tri(uv_coord, u, v, w))
            print(f"Found intersecting triangle for UV coordinate: {uv_coord}")
            find_3d_coord(uv_coord, face)
            break
    else:
        print(f"Did not find any 3D point for UV coordinate: {uv_coord}")
        
def all_uv_coord_to_3d(texture_res=1024):
    for i_u in range(texture_res):
        for i_v in range(texture_res):
            uv_coord = mathutils.Vector((i_u, i_v)) / texture_res
            uv_coord_3d = uv_coord_to_3d(uv_coord)
        print(uv_coord_3d)
        
#all_uv_coord_to_3d()

# Testing using cursor location in UV editor
#"""
for area in bpy.context.screen.areas:
    if area.type == 'IMAGE_EDITOR':
        loc = area.spaces.active.cursor_location.to_3d()
        break

bpy.context.scene.cursor.location = uv_coord_to_3d(loc)
#"""