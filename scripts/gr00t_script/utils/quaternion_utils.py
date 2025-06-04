import numpy as np

def quat_wxyz_to_xyzw(q_wxyz: np.ndarray) -> np.ndarray:
    """
    Converts quaternion from (w, x, y, z) to (x, y, z, w).
    Handles both 1D (single quaternion) and 2D (batch of quaternions) arrays.
    """
    q_wxyz = np.asarray(q_wxyz)
    if q_wxyz.shape[-1] != 4:
        raise ValueError("Last dimension of quaternion array must be 4.")
    return q_wxyz[..., [1, 2, 3, 0]]

def quat_xyzw_to_wxyz(q_xyzw: np.ndarray) -> np.ndarray:
    """
    Converts quaternion from (x, y, z, w) back to (w, x, y, z).
    Handles both 1D (single quaternion) and 2D (batch of quaternions) arrays.
    """
    q_xyzw = np.asarray(q_xyzw)
    if q_xyzw.shape[-1] != 4:
        raise ValueError("Last dimension of quaternion array must be 4.")
    return q_xyzw[..., [3, 0, 1, 2]]