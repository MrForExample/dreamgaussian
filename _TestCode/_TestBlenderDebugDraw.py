import gpu
from gpu_extras.batch import batch_for_shader
from math import pi, sin, cos

"""
    Draw Debug Shape
    Modified from: https://devtalk.blender.org/t/graphic-debugging-proof-of-concept/5163
    https://devtalk.blender.org/t/what-is-the-new-replacement-for-bgl-module-code-in-blender-3-5-0/28691
"""

def debug_draw_3d_uv_data(uv_coords_3d, uv_normals_3d):
    debugger = DebugDrawCallback()
    
    num_point = len(uv_coords_3d)
    for i in range(num_point):
        point_a = uv_coords_3d[i]
        point_b = point_a + uv_normals_3d[i]
        
        debugger.add_circle(point_a, 0.05)
        debugger.add_line(point_a, point_b)
        
    debugger()

class DebugDrawCallback:
	def __init__(self, thickness = 2):
		self.vertices = []
		self.colors = []
		self.thickness = thickness
		self.shader = gpu.shader.from_builtin("POLYLINE_SMOOTH_COLOR")
		self.batch_redraw = False
		self.batch = None

	def __call__(self):
		self.draw()

	def add_circle(self, center, radius, resolution=8, color=(0, 0, 0, 1)):
		self.batch_redraw = True
		for i in range(resolution):
			line_point_a = (sin(i / resolution * 2 * pi) * radius + center[0],
							cos(i / resolution * 2 * pi) * radius + center[1])
			line_point_b = (sin((i + 1) / resolution * 2 * pi) * radius + center[0],
							cos((i + 1) / resolution * 2 * pi) * radius + center[1])
			self.add_line(line_point_a, line_point_b, color)

	def add_line(self, point_a, point_b, color_a=(0, 0, 0, 1), color_b=None):
		self.batch_redraw = True
		self.vertices.append(point_a)
		self.vertices.append(point_b)
		self.colors.append(color_a)
		self.colors.append(color_b if color_b else color_a)

	def remove_last_line(self):
		self.vertices.pop(-1)
		self.vertices.pop(-1)

	def clear(self):
		self.vertices.clear()
		self.colors.clear()

	def update_batch(self):
		self.batch_redraw = False
		self.batch = batch_for_shader(self.shader, "LINES", {"pos": self.vertices})

	def draw(self):
		gpu.state.blend_set("ALPHA")
		if self.batch_redraw or not self.batch:
			self.update_batch()
		gpu.state.line_width_set(self.thickness)
		self.shader.bind()
		self.shader.uniform_float("color", (1, 0, 0, 1))
		self.batch.draw(self.shader)
		gpu.state.blend_set("NONE")