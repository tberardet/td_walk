import numpy as np

from scipy.spatial.transform import Rotation as R


def rot_x(alpha):
    """Return the 4x4 homogeneous transform corresponding to a rotation of
    alpha around x
    """
    c = np.cos(alpha)
    s = np.sin(alpha)
    return np.array([[1, 0, 0, 0],
                     [0, c, -s, 0],
                     [0, s, c, 0],
                     [0, 0, 0, 1]], dtype=np.double)


def rot_y(alpha):
    """Return the 4x4 homogeneous transform corresponding to a rotation of
    alpha around y
    """
    c = np.cos(alpha)
    s = np.sin(alpha)
    return np.array([[c, 0, s, 0],
                     [0, 1, 0, 0],
                     [-s, 0, c, 0],
                     [0, 0, 0, 1]], dtype=np.double)


def rot_z(alpha):
    """Return the 4x4 homogeneous transform corresponding to a rotation of
    alpha around z
    """
    c = np.cos(alpha)
    s = np.sin(alpha)
    return np.array([[c, -s, 0, 0],
                     [s, c, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]], dtype=np.double)


def translation(vec):
    """Return the 4x4 homogeneous transform corresponding to a translation of
    vec
    """
    return np.array([[1, 0, 0, vec[0]],
                     [0, 1, 0, vec[1]],
                     [0, 0, 1, vec[2]],
                     [0, 0, 0, 1]], dtype=np.double)

def d_rot_x(alpha):
    """Return the 4x4 homogeneous transform corresponding to the derivative of a rotation of
    alpha around x
    """
    c = np.cos(alpha)
    s = np.sin(alpha)
    return np.array([[0, 0, 0, 0],
                     [0, -s, -c, 0],
                     [0, c, -s, 0],
                     [0, 0, 0, 0]], dtype=np.double)

def d_rot_y(alpha):
    """Return the 4x4 homogeneous transform corresponding to the derivative of a rotation of
    alpha around y
    """
    c = np.cos(alpha)
    s = np.sin(alpha)
    return np.array([[-s, 0, c, 0],
                     [0, 0, 0, 0],
                     [-c, 0, -s, 0],
                     [0, 0, 0, 0]], dtype=np.double)


def d_rot_z(alpha):
    """Return the 4x4 homogeneous transform corresponding to the derivative of a rotation of
    alpha around z
    """
    c = np.cos(alpha)
    s = np.sin(alpha)
    return np.array([[-s, -c, 0, 0],
                     [c, -s, 0, 0],
                     [0, 0, 0, 0],
                     [0, 0, 0, 0]], dtype=np.double)


def d_translation(vec):
    """Return the 4x4 homogeneous transform corresponding to the derivative of a translation of
    vec
    """
    v = vec / np.linalg.norm(vec)
    T = np.zeros((4,4), dtype=np.double)
    T[:3,3] = v
    return T

def invert_transform(T):
    I = T.copy()
    RI = T[:3, :3].transpose()
    I[:3, :3] = RI
    I[:3, 3] = -RI @ T[:3, 3]
    return I

def get_quat(T):
    """
    Parameters
    ----------
    T : np.ndarray shape(4,4)
        A 3d homogeneous transformation matrix

    Returns
    -------
    quat : np.ndarray shape(4,)
        a quaternion representing the rotation part of the homogeneous
        transformation matrix
    """
    return R.from_dcm(T[:3,:3]).as_quat()



if __name__ == "__main__":
    unittest.main()
