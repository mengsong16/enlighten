import collections
import os

import numpy as np

# The function to retrieve the rotation matrix changed from as_dcm to as_matrix in version 1.4
# We will use the version number for backcompatibility
import scipy
import yaml
from packaging import version
from PIL import Image

from habitat_sim.utils.common import quat_from_angle_axis
import math

import textwrap
from typing import Dict, List, Optional, Tuple

import imageio
import tqdm
 
from enlighten.utils.path import *

from enlighten.utils.image_utils import try_cv2_import
cv2 = try_cv2_import()

# euler (degree) to quaternion
# The default order for Euler angle rotations is "ZYX"
def get_rotation_quat(rotation):
    x,y,z = rotation
    # z * y * x
    quat = quat_from_angle_axis(math.radians(z), np.array([0, 0, 1.0])
        ) * quat_from_angle_axis(math.radians(y), np.array([0, 1.0, 0])
        ) * quat_from_angle_axis(math.radians(x), np.array([1.0, 0, 0]))

    return quat

def quaternion_rotate_vector(quat: np.quaternion, v: np.array) -> np.array:
    r"""Rotates a vector by a quaternion
    Args:
        quaternion: The quaternion to rotate by [w,x,y,z]
        v: The vector to rotate
    Returns:
        np.array: The rotated vector
    """
    vq = np.quaternion(0, 0, 0, 0)
    vq.imag = v
    return (quat * vq * quat.inverse()).imag

def euclidean_distance(position_a, position_b):
    return np.linalg.norm(position_b - position_a, ord=2)


def cartesian_to_polar(x, y):
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return rho, phi


def quaternion_from_coeff(coeffs: np.ndarray) -> np.quaternion:
    r"""Creates a quaternions [w,x,y,z] from coeffs in [x, y, z, w] format"""
    quat = np.quaternion(0, 0, 0, 0)
    quat.real = coeffs[3]
    quat.imag = coeffs[0:3]
    return quat    

