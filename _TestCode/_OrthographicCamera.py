import torch

class OrthographicCamera:
    def __init__(self, width, height, znear, zfar):
        """
        Initializes an orthographic camera.

        Args:
            width (float): Width of the camera view.
            height (float): Height of the camera view.
            znear (float): Near clipping plane distance.
            zfar (float): Far clipping plane distance.
        """
        self.image_width = width
        self.image_height = height
        self.znear = znear
        self.zfar = zfar

    def project(self, point_3d):
        """
        Projects a 3D point onto the 2D camera plane.

        Args:
            point_3d (torch.Tensor): 3D point in world coordinates.

        Returns:
            torch.Tensor: 2D projected point on the camera plane.
        """
        # Perform orthographic projection
        x = point_3d[0]
        y = point_3d[1]
        z = point_3d[2]

        x_proj = (x / self.image_width) * 2 - 1
        y_proj = (y / self.image_height) * 2 - 1
        z_proj = (z - self.znear) / (self.zfar - self.znear)

        return torch.tensor([x_proj, y_proj, z_proj])

    def unproject(self, point_2d):
        """
        Unprojects a 2D point back to 3D.

        Args:
            point_2d (torch.Tensor): 2D point on the camera plane.

        Returns:
            torch.Tensor: 3D point in world coordinates.
        """
        # Perform inverse orthographic projection
        x_proj = point_2d[0]
        y_proj = point_2d[1]
        z_proj = point_2d[2]

        x = (x_proj + 1) * 0.5 * self.image_width
        y = (y_proj + 1) * 0.5 * self.image_height
        z = z_proj * (self.zfar - self.znear) + self.znear

        return torch.tensor([x, y, z])

def get_orthographic_projection_matrix_simple(width, height, near, far):
    """
    Calculates the orthographic projection matrix.

    Args:
        width (float): Width of the view frustum.
        height (float): Height of the view frustum.
        near (float): Near clipping plane distance.
        far (float): Far clipping plane distance.

    Returns:
        torch.Tensor: 4x4 orthographic projection matrix.
    """
    P = torch.zeros(4, 4)
    P[0, 0] = 2 / width
    P[1, 1] = 2 / height
    P[2, 2] = 1 / (far - near)
    P[3, 2] = -near / (far - near)
    P[3, 3] = 1

    return P
    
def get_orthographic_projection_matrix(width, height, near, far):
    """
    Computes the orthographic projection matrix based on width and height.

    Args:
        width (float): Width of the view frustum.
        height (float): Height of the view frustum.
        near (float): Near clipping plane distance.
        far (float): Far clipping plane distance.

    Returns:
        torch.Tensor: The 4x4 orthographic projection matrix.
    """
    left = -width / 2
    right = width / 2
    bottom = -height / 2
    top = height / 2

    tx = -(right + left) / (right - left)
    ty = -(top + bottom) / (top - bottom)
    tz = -(far + near) / (far - near)

    P = torch.zeros(4, 4)
    P[0, 0] = 2 / width
    P[1, 1] = 2 / height
    P[2, 2] = -2 / (far - near)
    P[0, 3] = tx
    P[1, 3] = ty
    P[2, 3] = tz
    P[3, 3] = 1.0
    
    return P
    
#cam = OrthographicCamera(512, 512, 0.01, 100)
print(get_orthographic_projection_matrix(1024, 1024, 0.01, 100))
print(get_orthographic_projection_matrix_simple(1024, 1024, 0.01, 100).transpose(0, 1))