from enlighten.utils.path import *
from enlighten.utils.config_utils import parse_config
from enlighten.datasets.pointnav_dataset import PointNavDatasetV1
from enlighten.datasets.pointnav_dataset import NavigationEpisode, NavigationGoal, ShortestPathPoint
from enlighten.datasets.dataset import EpisodeIterator
from enlighten.envs.multi_nav_env import MultiNavEnv
from enlighten.utils.geometry_utils import euclidean_distance

import math
import os
import numpy as np

import pickle
from tqdm import tqdm
import random


import habitat
from habitat.core.utils import try_cv2_import
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import images_to_video

cv2 = try_cv2_import()

def load_pointgoal_dataset(yaml_name, split=None):
    config_file=os.path.join(config_path, yaml_name)
    config = parse_config(config_file)

    if split is None:
        split = config.get("split")

    print("Split: %s"%(split))    

    dataset = PointNavDatasetV1(split=split, config=config)
    
    #print("Loaded %d episodes"%len(dataset.episodes))

    # for episode in dataset.episodes:
    #     print(episode.scene_id)

    return dataset

    
def test_get_scene_names(yaml_name):
    config_file=os.path.join(config_path, yaml_name)
    config = parse_config(config_file)

    dataset = PointNavDatasetV1() # dummy
    scenes = dataset.get_scene_names_to_load(config, config.get("split"))
    
    print("Loaded scene names.")
    print(scenes)
    print("Number of scenes: %d"%len(scenes))

def shortest_path_follower(yaml_name):
    env = MultiNavEnv(config_file=yaml_name)
    dataset = load_pointgoal_dataset(yaml_name)
    
    for i, episode in enumerate(dataset.episodes):
        obs = env.reset(episode=episode, plan_shortest_path=True)
        print('Episode: {}'.format(i+1))
        print("Goal position: %s"%(env.goal_position))
        #print(env.goal_position)
        #env.print_agent_state()
        print("Start position: %s"%(env.start_position))
        #print(env.agent.get_state().position)
        print("Optimal action sequence: %s"%env.optimal_action_seq)


        for action in env.optimal_action_seq:
            #action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            #env.render()
            #print(obs["pointgoal"])
            #print(env.goal_position)
            #print(obs["state_sensor"])
            #print(env.agent.get_state().position)
            print(env.get_current_distance())
            print("---------------------")

        # not empty
        if env.optimal_action_seq:
            print("Distance to goal at the end of the trajectory: %f"%(env.get_current_distance()))
            assert done == True, "done should be true after following the shortest path"
            assert env.is_success() == True, "success should be true after following the shortest path"
        print("===============================")


        # dirname = os.path.join(
        #     IMAGE_DIR, "shortest_path_example", "%02d" % episode
        # )
        # if os.path.exists(dirname):
        #     shutil.rmtree(dirname)
        # os.makedirs(dirname)
        # print("Agent stepping around inside environment.")
        # images = []
        # while not env.habitat_env.episode_over:
        #     best_action = follower.get_next_action(
        #         env.habitat_env.current_episode.goals[0].position
        #     )
        #     if best_action is None:
        #         break

        #     observations, reward, done, info = env.step(best_action)
        #     im = observations["rgb"]
        #     top_down_map = draw_top_down_map(info, im.shape[0])
        #     output_im = np.concatenate((im, top_down_map), axis=1)
        #     images.append(output_im)
        # images_to_video(images, dirname, "trajectory")
        #print("Episode finished")

# pointgoal dataset split: {'train', 'val', 'val_mini'}
def generate_pointgoal_dataset_meta(yaml_name, split):
    episode_dataset = load_pointgoal_dataset(yaml_name, split)
    episodes = {}
    # divide episodes into scenes
    for episode in tqdm(episode_dataset.episodes):
        #print(episode.shortest_paths)
        #print(episode.scene_id)
        data = {"episode": episode, 
                "start_goal_distance": euclidean_distance(np.array(episode.start_position, dtype=np.float32), np.array(episode.goals[0].position, dtype=np.float32))}

        if episode.scene_id not in episodes:
            episodes[episode.scene_id] = []
            
        episodes[episode.scene_id].append(data)
    
    # save meta data
    config_file=os.path.join(config_path, yaml_name)
    config = parse_config(config_file)
    pointgoal_dataset_path = config.get("pointgoal_dataset_path")
    pointgoal_dataset_meta_data_path = os.path.join(pointgoal_dataset_path, "meta_data")

    if not os.path.exists(pointgoal_dataset_meta_data_path):
        os.makedirs(pointgoal_dataset_meta_data_path)

    with open(os.path.join(pointgoal_dataset_meta_data_path, '%s.pickle'%(split)), 'wb') as handle:
        pickle.dump(episodes, handle)

    print("Split %s: Done."%(split))

# pointgoal dataset split: {'train', 'val', 'val_mini'}
def load_pointgoal_dataset_meta(config, split):
    pointgoal_dataset_path = config.get("pointgoal_dataset_path")
    pointgoal_dataset_meta_data_path = os.path.join(pointgoal_dataset_path, "meta_data")
    # load meta data from pointgoal dataset
    pointgoal_meta = pickle.load(open(os.path.join(pointgoal_dataset_meta_data_path, '%s.pickle'%(split)), "rb" ))
    print("Pointgoal dataset meta data Loaded")
    scene_num = len(pointgoal_meta.keys())
    episode_num = 0
    for scene_id, val in tqdm(pointgoal_meta.items()):
        cur_scene_episode_num = len(val)
        print("Scene: %s: %d"%(scene_id, cur_scene_episode_num))
        episode_num += cur_scene_episode_num
    print("===========================")
    print("Scenes: %d"%scene_num)
    print("Episodes: %d"%episode_num)
    print("===========================")

    return pointgoal_meta, scene_num, episode_num

# pointgoal dataset split: {'train', 'val', 'val_mini'}
def generate_behavior_dataset_meta(yaml_name, pointgoal_dataset_split, behavior_dataset_split,
    sample_scene_num, sample_epidode_num):
    config_file=os.path.join(config_path, yaml_name)
    config = parse_config(config_file)

    pointgoal_meta, scene_num, episode_num = load_pointgoal_dataset_meta(config, pointgoal_dataset_split)

    behavior_dataset_path = config.get("behavior_dataset_path")
    behavior_dataset_meta_data_path = os.path.join(behavior_dataset_path, "meta_data")

    # sample scenes
    if sample_scene_num <= scene_num:
        scene_list = pointgoal_meta.keys()
        selected_scenes = random.sample(scene_list, sample_scene_num)
        print("Selected scenes: %s"%(selected_scenes))
    else:
        print("Error: want to sample %d from %d scenes"%(sample_scene_num, scene_num))  
        exit()
    
    if sample_epidode_num > episode_num:
        print("Error: want to sample %d from %d episodes"%(sample_epidode_num, episode_num))  
        exit()

    sampled_episode_num = 0
    sampled_episodes = []
    for i, scene_id in enumerate(selected_scenes):
        # not last batch
        if i < sample_scene_num - 1:
            batch_size = sample_epidode_num // sample_scene_num
        # last batch
        else:
            batch_size = sample_epidode_num - sampled_episode_num
        
        #print(batch_size)
        sampled_episode_num += batch_size

        # sample episode in each scene
        cur_scene_data = pointgoal_meta[scene_id]
        #print(cur_scene_episodes)
        exit()
    
    assert sampled_episode_num == sample_epidode_num, "Sampled episdoe num %d, desired episdoe num %d"%(sampled_episode_num, sample_epidode_num)
        #pointgoal_meta[scene]  

    # save meta data
    if not os.path.exists(behavior_dataset_meta_data_path):
        os.makedirs(behavior_dataset_meta_data_path)

    with open(os.path.join(behavior_dataset_meta_data_path, '%s.pickle'%(behavior_dataset_split)), 'wb') as handle:
        pickle.dump(sampled_episodes, handle)

    print("Split %s: Done."%(behavior_dataset_split))

if __name__ == "__main__":
    #load_pointgoal_dataset("imitation_learning.yaml")  
    #test_get_scene_names("imitation_learning.yaml")
    #shortest_path_follower("imitation_learning.yaml")
    #generate_pointgoal_dataset_meta(yaml_name="imitation_learning.yaml", split="train")
    generate_behavior_dataset_meta(yaml_name="imitation_learning.yaml", 
        pointgoal_dataset_split="val_mini", 
        behavior_dataset_split="across_scene_test_mini", 
        sample_scene_num=2, sample_epidode_num=20)
