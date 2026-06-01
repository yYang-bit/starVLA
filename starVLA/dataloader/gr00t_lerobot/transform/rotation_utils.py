"""Rotation representation conversion utilities for SO(3) operations.

Provides conversions between rotation_6d and rotation matrices for geometrically
correct delta/relative rotation computation in self-mode action transforms.
"""

import numpy as np


def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """Convert rotation_6d to rotation matrix via Gram-Schmidt orthonormalization.

    Args:
        rot6d: (..., 6) array representing first two columns of rotation matrix

    Returns:
        R: (..., 3, 3) rotation matrices

    Reference:
        Zhou et al. "On the Continuity of Rotation Representations in Neural Networks"
        https://arxiv.org/abs/1812.07035
    """
    rot6d = np.asarray(rot6d, dtype=np.float64)
    batch_shape = rot6d.shape[:-1]

    # Reshape to (..., 2, 3) for easier manipulation
    x_raw = rot6d[..., :3]  # first column
    y_raw = rot6d[..., 3:]  # second column

    # Gram-Schmidt orthonormalization
    x = x_raw / (np.linalg.norm(x_raw, axis=-1, keepdims=True) + 1e-8)

    # y_orth = y_raw - (y_raw · x) * x
    y_orth = y_raw - np.sum(y_raw * x, axis=-1, keepdims=True) * x
    y = y_orth / (np.linalg.norm(y_orth, axis=-1, keepdims=True) + 1e-8)

    # z = x × y
    z = np.cross(x, y, axis=-1)

    # Stack into rotation matrix
    R = np.stack([x, y, z], axis=-1)  # (..., 3, 3)
    return R.astype(np.float32)


def matrix_to_rot6d(R: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to rotation_6d (first two columns flattened).

    Args:
        R: (..., 3, 3) rotation matrices

    Returns:
        rot6d: (..., 6) array [R[:, 0], R[:, 1]] flattened
    """
    R = np.asarray(R, dtype=np.float32)
    # Extract first two columns and flatten
    rot6d = np.concatenate([R[..., :, 0], R[..., :, 1]], axis=-1)
    return rot6d.astype(np.float32)


def compute_relative_rotation_rot6d(
    rot6d_t: np.ndarray,
    rot6d_base: np.ndarray,
) -> np.ndarray:
    """Compute relative rotation in SO(3) and return as rotation_6d.

    R_relative = R_t @ R_base^T

    Args:
        rot6d_t: (..., 6) current rotation
        rot6d_base: (..., 6) base rotation

    Returns:
        rot6d_relative: (..., 6) relative rotation encoded as rotation_6d
    """
    R_t = rot6d_to_matrix(rot6d_t)
    R_base = rot6d_to_matrix(rot6d_base)
    # Transpose last two axes: (..., 3, 3) → (..., 3, 3)
    axes = list(range(R_base.ndim - 2)) + [R_base.ndim - 1, R_base.ndim - 2]
    R_relative = R_t @ R_base.transpose(axes)  # R_t @ R_base^T
    return matrix_to_rot6d(R_relative)


def identity_rotation_rot6d(shape: tuple = ()) -> np.ndarray:
    """Return identity rotation as rotation_6d.

    I = [[1, 0, 0],
         [0, 1, 0],
         [0, 0, 1]]
    rot6d = [1, 0, 0, 0, 1, 0]  # first two columns flattened

    Args:
        shape: batch shape, e.g. (10,) for 10 identity rotations

    Returns:
        rot6d_identity: (*shape, 6)
    """
    identity = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    if shape:
        identity = np.broadcast_to(identity, shape + (6,)).copy()
    return identity
