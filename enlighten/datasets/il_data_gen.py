from enlighten.utils.path import *
from enlighten.utils.config_utils import parse_config
from enlighten.datasets.pointnav_dataset import PointNavDatasetV1
from enlighten.datasets.pointnav_dataset import NavigationEpisode, NavigationGoal, ShortestPathPoint
from enlighten.datasets.dataset import EpisodeIterator

import math
import os
import numpy as np


import habitat
from habitat.core.utils import try_cv2_import
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import images_to_video

cv2 = try_cv2_import()

def load_pointgoal_dataset(yaml_name):
    config_file=os.path.join(config_path, yaml_name)
    config = parse_config(config_file)

    dataset = PointNavDatasetV1(config)
    #print(dataset.episodes[0])

    
def test_get_scene_names(yaml_name):
    config_file=os.path.join(config_path, yaml_name)
    config = parse_config(config_file)

    dataset = PointNavDatasetV1()
    scenes = dataset.get_scene_names_to_load(config)
    
    print("Loaded scene names.")
    print(scenes)
    print(len(scenes))

def shortest_path_example():
    config = habitat.get_config(config_paths="configs/tasks/pointnav.yaml")
    config.defrost()
    config.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
    config.freeze()
    with SimpleRLEnv(config=config) as env:
        goal_radius = env.episodes[0].goals[0].radius
        if goal_radius is None:
            goal_radius = config.SIMULATOR.FORWARD_STEP_SIZE
        follower = ShortestPathFollower(
            env.habitat_env.sim, goal_radius, False
        )

        print("Environment creation successful")
        for episode in range(3):
            env.reset()
            dirname = os.path.join(
                IMAGE_DIR, "shortest_path_example", "%02d" % episode
            )
            if os.path.exists(dirname):
                shutil.rmtree(dirname)
            os.makedirs(dirname)
            print("Agent stepping around inside environment.")
            images = []
            while not env.habitat_env.episode_over:
                best_action = follower.get_next_action(
                    env.habitat_env.current_episode.goals[0].position
                )
                if best_action is None:
                    break

                observations, reward, done, info = env.step(best_action)
                im = observations["rgb"]
                top_down_map = draw_top_down_map(info, im.shape[0])
                output_im = np.concatenate((im, top_down_map), axis=1)
                images.append(output_im)
            images_to_video(images, dirname, "trajectory")
            print("Episode finished")


if __name__ == "__main__":
    #load_pointgoal_dataset("imitation_learning.yaml")  
    test_get_scene_names("imitation_learning.yaml")