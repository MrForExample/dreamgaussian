from typing import Tuple, List, Union
import numpy as np

"""
    Standard Algorithm for Filling Triangles with grids
    
    Algorithm: http://www.sunshine2k.de/coding/java/TriangleRasterization/TriangleRasterization.html
    Code: https://github.com/mikecokina/trast/blob/master/src/trast/core.py
"""

IS_ZERO_TOL = 1

__all__ = (
    'Rasterizer',
)


def validate_resolution(res: Tuple) -> None:
    if len(res) != 2:
        raise ValueError('Invalid resolution, expected `W x H`')
    if any([s < 1 for s in res]):
        raise ValueError('Invalid resolution, expected `W x H`, where any of values has to be higher than 0')


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

        :param resolution: Tuple[int, int];  grids resolution (W, H)
        :param min_x: int; coordinate on X-axis of triangles to consider as 0
        :param max_x: int; coordinate on X-axis of trinagles to consider on WIDTH
        :param min_y:
        :param max_y:
        
        coordinates origin is at top left corner of the screen
        """
        validate_resolution(resolution)

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
            self._tri_grids_coords.append((self._h - y, x))

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

    def quick_plot(self, pixel=255):
        from matplotlib import pyplot as plt
        # noinspection PyUnresolvedReferences
        
        screen = np.zeros(shape=(self._h, self._w))
        
        for y, x in self._tri_grids_coords:
            screen[y, x] = pixel
        
        plt.imshow(screen, cmap=plt.cm.gray_r, origin='upper', interpolation="nearest")
        plt.show()

"""
def test_plot():
    t = [[200, 200],
        [800, 400],
        [400, 800]]

    r = Rasterizer(resolution=(1024, 1024))
    r.rasterize_triangle(face=t)

    r.quick_plot()
"""