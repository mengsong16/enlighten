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

import cv2

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

def images_to_video(
    images: List[np.ndarray],
    output_dir: str,
    video_name: str,
    fps: int = 10,
    quality: Optional[float] = 5,
    **kwargs,
):
    r"""Calls imageio to run FFMPEG on a list of images. For more info on
    parameters, see https://imageio.readthedocs.io/en/stable/format_ffmpeg.html
    Args:
        images: The list of images. Images should be HxWx3 in RGB order.
        output_dir: The folder to put the video in.
        video_name: The name for the video.
        fps: Frames per second for the video. Not all values work with FFMPEG,
            use at your own risk.
        quality: Default is 5. Uses variable bit rate. Highest quality is 10,
            lowest is 0.  Set to None to prevent variable bitrate flags to
            FFMPEG so you can manually specify them using output_params
            instead. Specifying a fixed bitrate using ‘bitrate’ disables
            this parameter.
    """
    assert 0 <= quality <= 10
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    video_name = video_name.replace(" ", "_").replace("\n", "_") + ".mp4"
    writer = imageio.get_writer(
        os.path.join(output_dir, video_name),
        fps=fps,
        quality=quality,
        **kwargs,
    )
    print("Video created: {os.path.join(output_dir, video_name)}")
    for im in tqdm.tqdm(images):
        writer.append_data(im)
    writer.close()  

def create_video():
    images = []
    for file in os.listdir(output_path):
        if file.endswith(".jpg"):
            #print(os.path.join(output_path, file))
            img = cv2.imread(os.path.join(output_path, file))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)

    if len(images) > 0:
        images_to_video(images, output_dir=output_path, video_name="video")
    else:
        print("Error: no images exist!")    

if __name__ == "__main__":
    create_video()
