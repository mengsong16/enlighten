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

# File I/O related


def parse_config(config):

    """
    Parse iGibson config file / object
    """
    try:
        collectionsAbc = collections.abc
    except AttributeError:
        collectionsAbc = collections

    if isinstance(config, collectionsAbc.Mapping):
        return config
    else:
        assert isinstance(config, str)

    if not os.path.exists(config):
        raise IOError(
            "config path {} does not exist. Please either pass in a dict or a string that represents the file path to the config yaml.".format(
                config
            )
        )
    with open(config, "r") as f:
        config_data = yaml.load(f, Loader=yaml.FullLoader)
    return config_data


def parse_str_config(config):
    """
    Parse string config
    """
    return yaml.safe_load(config)


def dump_config(config):
    """
    Converts YML config into a string
    """
    return yaml.dump(config)


def get_rotation_quat(rotation):
    x,y,z = rotation
    # z * y * x
    quat = quat_from_angle_axis(math.radians(z), np.array([0, 0, 1.0])
        ) * quat_from_angle_axis(math.radians(y), np.array([0, 1.0, 0])
        ) * quat_from_angle_axis(math.radians(x), np.array([1.0, 0, 0]))

    return quat