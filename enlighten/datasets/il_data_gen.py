from enlighten.utils.path import *
from enlighten.utils.config_utils import parse_config
from enlighten.datasets.pointnav_dataset import PointNavDatasetV1

import math
import os
import numpy as np

def load_pointgoal_dataset(yaml_name):
    config_file=os.path.join(config_path, yaml_name)
    config = parse_config(config_file)

    dataset = PointNavDatasetV1(config)
    print(dataset.episodes[0])

    
def test_get_scene_names(yaml_name):
    config_file=os.path.join(config_path, yaml_name)
    config = parse_config(config_file)

    dataset = PointNavDatasetV1()
    scenes = dataset.get_scene_names_to_load(config)
    
    print("Loaded")
    print(scenes)
    print(len(scenes))

if __name__ == "__main__":
    load_pointgoal_dataset("imitation_learning.yaml")  
    #test_get_scene_names("imitation_learning.yaml")