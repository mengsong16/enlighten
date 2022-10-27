from enlighten.utils.path import *
from enlighten.utils.config_utils import parse_config
from enlighten.datasets.pointnav_dataset import PointNavDatasetV1
from enlighten.datasets.pointnav_dataset import NavigationEpisode, NavigationGoal
from enlighten.envs.multi_nav_env import MultiNavEnv, NavEnv
from enlighten.utils.geometry_utils import euclidean_distance
from enlighten.agents.common.seed import set_seed_except_env_seed
from enlighten.utils.image_utils import try_cv2_import
from enlighten.datasets.common import load_behavior_dataset_meta
from enlighten.envs.vec_env import chunks
from enlighten.envs.vec_env import construct_envs_based_on_dataset
from enlighten.utils.ddp_utils import is_slurm_batch_job
from enlighten.datasets.common import PolarActionSpace
from enlighten.datasets.common import update_episode_data, extract_observation

cv2 = try_cv2_import()

import math
import os
import numpy as np

import pickle
from tqdm import tqdm
import random
import copy
import time
import multiprocessing as mp


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
        #print(obs["pointgoal"])
        print('Episode: {}'.format(i+1))
        print("Goal position: %s"%(env.goal_position))
        #print(env.goal_position)
        #env.print_agent_state()
        print("Start position: %s"%(env.start_position))
        # print(env.agent.get_state().position)
        # print(env.get_current_distance())
        # print("------------------")
        #print("Optimal action sequence: %s"%env.optimal_action_seq)


        for action in env.optimal_action_seq:
            #action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            env.render()
            # print(action)
            # print(obs["pointgoal"])
            # print(env.goal_position)
            # #print(obs["state_sensor"])
            # print(env.agent.get_state().position)
            # print(env.get_current_distance())
            # print("---------------------")


        # not empty
        if env.optimal_action_seq:
            print("Distance to goal at the end of the trajectory: %f"%(env.get_current_distance()))
            assert done == True, "done should be true after following the shortest path"
            assert env.is_success() == True, "success should be true after following the shortest path"
        print("===============================")
    
    env.close()

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
        pickle.dump(episodes, handle, protocol=pickle.HIGHEST_PROTOCOL)

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

# make sure num_episode is divisible by num_scene
def check_episode_per_scene(train_scene_num, train_episode_num, 
    across_scene_val_scene_num, across_scene_val_episode_num,
    same_scene_val_episode_num,
    same_start_goal_val_episode_num,
    across_scene_test_scene_num, across_scene_test_episode_num,
    same_scene_test_episode_num,
    same_start_goal_test_episode_num):

    if train_episode_num is not None:
        assert train_episode_num % train_scene_num == 0, "Error: train: episode num is not divisible by scene num"
    assert across_scene_val_episode_num % across_scene_val_scene_num == 0, "Error: Across scene val: episode num is not divisible by scene num"
    assert across_scene_test_episode_num % across_scene_test_scene_num == 0, "Error: Across scene test: episode num is not divisible by scene num"
    assert same_scene_val_episode_num % train_scene_num == 0, "Error: Same scene val: episode num is not divisible by train scene num"
    assert same_scene_test_episode_num % train_scene_num == 0, "Error: Same scene test: episode num is not divisible by train scene num"
    if same_start_goal_val_episode_num is not None:
        assert same_start_goal_val_episode_num % train_scene_num == 0, "Error: same start goal val: episode num is not divisible by train scene num"
    if same_start_goal_test_episode_num is not None:
        assert same_start_goal_test_episode_num % train_scene_num == 0, "Error: same start goal test: episode num is not divisible by train scene num"


# save a list of episodes to pickle file
def save_behavior_dataset_meta(sampled_episodes, behavior_dataset_path, split_name):
    behavior_dataset_meta_data_path = os.path.join(behavior_dataset_path, "meta_data")

    if not os.path.exists(behavior_dataset_meta_data_path):
        os.makedirs(behavior_dataset_meta_data_path)

    with open(os.path.join(behavior_dataset_meta_data_path, '%s.pickle'%(split_name)), 'wb') as handle:
        pickle.dump(sampled_episodes, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Behavior dataset episode generation Done: %s, %d episodes"%(split_name, len(sampled_episodes)))

# save scene information
def save_scene_info(scene_list, behavior_dataset_path, split_name):
    scene_folder = os.path.join(behavior_dataset_path, "scenes")
    if not os.path.exists(scene_folder):
        os.makedirs(scene_folder)
    # dump data
    with open(os.path.join(scene_folder, '%s.pickle'%(split_name)), 'wb') as handle:
        pickle.dump(scene_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Behavior dataset scene generation Done: %s, %d scenes"%(split_name, len(scene_list)))

# sample episodes from training scenes
# train/val/test are exclusive
def sample_train_episodes_v1(train_scenes, train_episode_num, 
    same_scene_val_episode_num, same_scene_test_episode_num,
    same_start_goal_val_episode_num, same_start_goal_test_episode_num,
    behavior_dataset_path, pointgoal_meta):

    train_scene_num = len(train_scenes)
    
    train_episode_per_scene = train_episode_num // train_scene_num
    val_episode_per_scene = same_scene_val_episode_num // train_scene_num
    test_episode_per_scene = same_scene_test_episode_num // train_scene_num
    batch_size = train_episode_per_scene + val_episode_per_scene + test_episode_per_scene

    same_start_goal_val_per_scene = same_start_goal_val_episode_num // train_scene_num
    same_start_goal_test_per_scene = same_start_goal_test_episode_num // train_scene_num
    sub_batch_size = same_start_goal_val_per_scene + same_start_goal_test_per_scene

    sampled_episode_num = 0
    sampled_train_episodes = []
    sampled_val_episodes = []
    sampled_test_episodes = []
    sampled_same_start_goal_val_episodes = []
    sampled_same_start_goal_test_episodes = []

    for scene_id in train_scenes:
        # collect all episodes from current scene
        episodes = []
        for data in pointgoal_meta[scene_id]:
            episodes.append(data["episode"])
        # sample episodes from current scene without replacement
        selected_episodes = random.sample(episodes, batch_size)
        # split into train, val, test scenes
        train_episodes = selected_episodes[0:train_episode_per_scene]
        val_episodes = selected_episodes[train_episode_per_scene:train_episode_per_scene+val_episode_per_scene]
        test_episodes = selected_episodes[-test_episode_per_scene:]
        
        sampled_episode_num += (len(train_episodes)+len(val_episodes)+len(test_episodes))
        sampled_train_episodes.extend(train_episodes)
        sampled_val_episodes.extend(val_episodes)
        sampled_test_episodes.extend(test_episodes)

        # sample same start and goal val and test episodes from training set of current scene
        sampled_sub_episodes = random.sample(train_episodes, sub_batch_size)
        # split into val and test
        sampled_same_start_goal_val_episodes.extend(sampled_sub_episodes[0:same_start_goal_val_per_scene])
        sampled_same_start_goal_test_episodes.extend(sampled_sub_episodes[-same_start_goal_test_per_scene:])

    # check sampled episode number
    desired_episode_num = train_episode_num + same_scene_val_episode_num + same_scene_test_episode_num
    assert sampled_episode_num == desired_episode_num, "Sampled episode num %d, desired episode num %d"%(sampled_episode_num, desired_episode_num)

    
    # save episode metadata
    save_behavior_dataset_meta(sampled_train_episodes, 
        behavior_dataset_path, "train")
    save_behavior_dataset_meta(sampled_val_episodes, 
        behavior_dataset_path, "same_scene_val")
    save_behavior_dataset_meta(sampled_test_episodes, 
        behavior_dataset_path, "same_scene_test")
    save_behavior_dataset_meta(sampled_same_start_goal_val_episodes, 
        behavior_dataset_path, "same_start_goal_val")
    save_behavior_dataset_meta(sampled_same_start_goal_test_episodes, 
        behavior_dataset_path, "same_start_goal_test")
    
    # save scene metadata
    save_scene_info(train_scenes, behavior_dataset_path, "train")

# sample episodes from training scenes
# same_scene: train/val are exclusive, val_mini is a subset of val
# same_start_goal: val is a subset of train, val_mini is a subset of val 
def sample_train_episodes(train_scenes, train_episode_num, 
    same_scene_val_episode_num, same_scene_val_mini_episode_num,
    same_start_goal_val_episode_num, same_start_goal_val_mini_episode_num,
    behavior_dataset_path, pointgoal_train_meta):

    train_scene_num = len(train_scenes)
    
    train_episode_per_scene = train_episode_num // train_scene_num
    val_episode_per_scene = same_scene_val_episode_num // train_scene_num
    val_mini_episode_per_scene = same_scene_val_mini_episode_num // train_scene_num
    batch_size = train_episode_per_scene + val_episode_per_scene

    same_start_goal_val_per_scene = same_start_goal_val_episode_num // train_scene_num
    same_start_goal_val_mini_per_scene = same_start_goal_val_mini_episode_num // train_scene_num


    sampled_train_episodes = []
    sampled_val_episodes = []
    sampled_val_mini_episodes = []
    sampled_same_start_goal_val_episodes = []
    sampled_same_start_goal_val_mini_episodes = []

    for scene_id in train_scenes:
        # collect all episodes from current scene
        episodes = []
        for data in pointgoal_train_meta[scene_id]:
            episodes.append(data["episode"])
        # sample episodes from current scene
        # without replacement
        selected_episodes = random.sample(episodes, batch_size)
        # split into train, val
        train_episodes = selected_episodes[0:train_episode_per_scene]
        val_episodes = selected_episodes[train_episode_per_scene:]
        # sample val_mini episodes from val episodes
        # without replacement
        val_mini_episodes = random.sample(val_episodes, val_mini_episode_per_scene)
        
        sampled_train_episodes.extend(train_episodes)
        sampled_val_episodes.extend(val_episodes)
        sampled_val_mini_episodes.extend(val_mini_episodes)

        # sample same start and goal val episodes from training set of current scene
        # without replacement
        sampled_sub_episodes = random.sample(train_episodes, same_start_goal_val_per_scene)
        sampled_same_start_goal_val_episodes.extend(sampled_sub_episodes)
        # sample same start and goal val_mini episodes from val episodes
        # without replacement
        sampled_sub_sub_episodes = random.sample(sampled_sub_episodes, same_start_goal_val_mini_per_scene)
        sampled_same_start_goal_val_mini_episodes.extend(sampled_sub_sub_episodes)

    # check sampled episode number
    assert train_episode_num == len(sampled_train_episodes), "Sampled episode num is not desired episode num"
    assert same_scene_val_episode_num == len(sampled_val_episodes), "Sampled episode num is not desired episode num"
    assert same_scene_val_mini_episode_num == len(sampled_val_mini_episodes), "Sampled episode num is not desired episode num"
    assert same_start_goal_val_episode_num == len(sampled_same_start_goal_val_episodes), "Sampled episode num is not desired episode num"
    assert same_start_goal_val_mini_episode_num == len(sampled_same_start_goal_val_mini_episodes), "Sampled episode num is not desired episode num"

    # save episode metadata
    save_behavior_dataset_meta(sampled_train_episodes, 
        behavior_dataset_path, "train")
    save_behavior_dataset_meta(sampled_val_episodes, 
        behavior_dataset_path, "same_scene_val")
    save_behavior_dataset_meta(sampled_val_mini_episodes, 
        behavior_dataset_path, "same_scene_val_mini")
    save_behavior_dataset_meta(sampled_same_start_goal_val_episodes, 
        behavior_dataset_path, "same_start_goal_val")
    save_behavior_dataset_meta(sampled_same_start_goal_val_mini_episodes, 
        behavior_dataset_path, "same_start_goal_val_mini")
    
    # save scene metadata
    save_scene_info(train_scenes, behavior_dataset_path, "train")

# use all episodes from training scenes
# same_scene: val is a subset of train episodes, val_mini is a subset of val episodes
def sample_train_episodes_whole_scene(train_scenes,  
    same_scene_val_episode_num, same_scene_val_mini_episode_num,
    behavior_dataset_path, pointgoal_train_meta):

    train_scene_num = len(train_scenes)
    
    val_episode_per_scene = same_scene_val_episode_num // train_scene_num
    val_mini_episode_per_scene = same_scene_val_mini_episode_num // train_scene_num

    train_episodes = []
    sampled_val_episodes = []
    sampled_val_mini_episodes = []
    

    for scene_id in train_scenes:
        # collect all episodes from current scene
        episodes = []
        for data in pointgoal_train_meta[scene_id]:
            episodes.append(data["episode"])
       
        # sample episodes from current scene for validation
        # without replacement
        val_episodes = random.sample(episodes, val_episode_per_scene)
        
        # sample val_mini episodes from val episodes
        # without replacement
        val_mini_episodes = random.sample(val_episodes, val_mini_episode_per_scene)
        
        # use all episodes in the scene for training
        train_episodes.extend(episodes)
        sampled_val_episodes.extend(val_episodes)
        sampled_val_mini_episodes.extend(val_mini_episodes)

    # check sampled episode number
    assert same_scene_val_episode_num == len(sampled_val_episodes), "Sampled episode num is not desired episode num"
    assert same_scene_val_mini_episode_num == len(sampled_val_mini_episodes), "Sampled episode num is not desired episode num"
    
    # save episode metadata
    save_behavior_dataset_meta(train_episodes, 
        behavior_dataset_path, "train")
    save_behavior_dataset_meta(sampled_val_episodes, 
        behavior_dataset_path, "same_scene_val")
    save_behavior_dataset_meta(sampled_val_mini_episodes, 
        behavior_dataset_path, "same_scene_val_mini")
    
    # save scene metadata
    save_scene_info(train_scenes, behavior_dataset_path, "train")


# sample episodes from val or test scenes
def sample_across_scene_episodes(scenes, episode_num, 
    behavior_dataset_path, pointgoal_meta, split_name):

    scene_num = len(scenes)
    
    batch_size = episode_num // scene_num

    sampled_episode_num = 0
    sampled_episodes = []

    for scene_id in scenes:
        # collect all episodes from current scene
        episodes = []
        for data in pointgoal_meta[scene_id]:
            episodes.append(data["episode"])
        # sample episodes from current scene
        # without replacement
        cur_scene_selected_episodes = random.sample(episodes, batch_size)
        
        sampled_episode_num += len(cur_scene_selected_episodes)
        sampled_episodes.extend(cur_scene_selected_episodes)

    # check sampled episode number
    assert sampled_episode_num == episode_num, "Sampled episode num %d, desired episode num %d"%(sampled_episode_num, episode_num)
    
    # save episode meta data
    save_behavior_dataset_meta(sampled_episodes, 
        behavior_dataset_path, split_name)

    # save scene metadata
    save_scene_info(scenes, behavior_dataset_path, split_name)

# sample episodes from val and val_mini scenes
def sample_across_scene_val_val_mini_episodes(val_scenes, 
    across_scene_val_mini_episode_num, 
    behavior_dataset_path, pointgoal_val_meta):

    val_scene_num = len(val_scenes)
    
    sub_batch_size = across_scene_val_mini_episode_num // val_scene_num

    
    val_episodes = []
    sampled_val_mini_episodes = []

    for scene_id in val_scenes:
        # collect all episodes from current scene and add them to val set
        cur_scene_val_episodes = []
        for data in pointgoal_val_meta[scene_id]:
            cur_scene_val_episodes.append(data["episode"])
        
        val_episodes.extend(cur_scene_val_episodes)
        # sample val_mini episodes from val episodes
        # without replacement
        cur_scene_val_mini_episodes = random.sample(cur_scene_val_episodes, sub_batch_size)
        sampled_val_mini_episodes.extend(cur_scene_val_mini_episodes)

    # check sampled episode number
    assert len(sampled_val_mini_episodes) == across_scene_val_mini_episode_num, "Sampled episode num is not desired episode num"
    
    # save episode meta data
    save_behavior_dataset_meta(val_episodes, 
        behavior_dataset_path, "across_scene_val")
    save_behavior_dataset_meta(sampled_val_mini_episodes, 
        behavior_dataset_path, "across_scene_val_mini")

    # save scene metadata
    save_scene_info(val_scenes, behavior_dataset_path, "across_scene_val")
    save_scene_info(val_scenes, behavior_dataset_path, "across_scene_val_mini")

# pointgoal gibson dataset: train --> subset scenes --> behavior: train
# pointgoal gibson dataset: val --> full --> behavior: val
# behavior: val --> all scenes, subset episodes --> behavior: val_mini
def generate_behavior_dataset_meta_whole_scene(yaml_name, 
    behavior_dataset_path,
    train_scene_num,  
    same_scene_val_episode_num,
    across_scene_val_mini_episode_num,
    same_scene_val_mini_episode_num):

    config_file = os.path.join(config_path, yaml_name)
    config = parse_config(config_file)

    pointgoal_train_meta, total_train_scene_num, total_train_episode_num = load_pointgoal_dataset_meta(config, "train")
    pointgoal_val_meta, total_val_scene_num, total_val_episode_num = load_pointgoal_dataset_meta(config, "val")
   
    # use all val scenes
    across_scene_val_scene_num = total_val_scene_num
    across_scene_val_mini_scene_num = total_val_scene_num
    val_scenes = list(pointgoal_val_meta.keys())
    across_scene_val_episode_num = total_val_episode_num

    # check divisible
    check_episode_per_scene(train_scene_num, None, 
    across_scene_val_scene_num, across_scene_val_episode_num,
    same_scene_val_episode_num,
    None,
    across_scene_val_mini_scene_num, across_scene_val_mini_episode_num,
    same_scene_val_mini_episode_num,
    None)

    # sample train scenes
    if train_scene_num <= total_train_scene_num:
        train_scene_list = pointgoal_train_meta.keys()
        # sample without replacement
        #train_scenes = random.sample(train_scene_list, train_scene_num)
        train_scenes = ['/dataset/gibson/Rancocas.glb']
        # verbose
        print("Sampled scenes:")
        print("train scenes: %s"%(train_scenes))
    else:
        print("Error: want to sample %d from %d scenes"%(train_scene_num, total_train_scene_num))  
        exit()
    
    # sample train episodes
    sample_train_episodes_whole_scene(train_scenes, 
    same_scene_val_episode_num, same_scene_val_mini_episode_num,
    behavior_dataset_path, pointgoal_train_meta) 

    # use all val episodes
    # and sample a subset of val episodes as val_mini
    sample_across_scene_val_val_mini_episodes(val_scenes, 
    across_scene_val_mini_episode_num,
    behavior_dataset_path, pointgoal_val_meta)

# pointgoal gibson dataset: train --> subset scenes --> behavior: train
# pointgoal gibson dataset: val --> full --> behavior: val
# behavior: val --> all scenes, subset episodes --> behavior: val_mini
def generate_behavior_dataset_meta(yaml_name, 
    behavior_dataset_path,
    train_scene_num, train_episode_num, 
    same_scene_val_episode_num,
    same_start_goal_val_episode_num,
    across_scene_val_mini_episode_num,
    same_scene_val_mini_episode_num,
    same_start_goal_val_mini_episode_num):

    config_file = os.path.join(config_path, yaml_name)
    config = parse_config(config_file)

    pointgoal_train_meta, total_train_scene_num, total_train_episode_num = load_pointgoal_dataset_meta(config, "train")
    pointgoal_val_meta, total_val_scene_num, total_val_episode_num = load_pointgoal_dataset_meta(config, "val")
   
    
    # use all val scenes
    across_scene_val_scene_num = total_val_scene_num
    across_scene_val_mini_scene_num = total_val_scene_num
    val_scenes = list(pointgoal_val_meta.keys())
    across_scene_val_episode_num = total_val_episode_num

    # check divisible
    check_episode_per_scene(train_scene_num, train_episode_num, 
    across_scene_val_scene_num, across_scene_val_episode_num,
    same_scene_val_episode_num,
    same_start_goal_val_episode_num,
    across_scene_val_mini_scene_num, across_scene_val_mini_episode_num,
    same_scene_val_mini_episode_num,
    same_start_goal_val_mini_episode_num)

    # sample train scenes
    if train_scene_num <= total_train_scene_num:
        train_scene_list = pointgoal_train_meta.keys()
        # sample without replacement
        train_scenes = random.sample(train_scene_list, train_scene_num)
        #train_scenes = ['/dataset/gibson/Rancocas.glb']
        # verbose
        print("Sampled scenes:")
        print("train scenes: %s"%(train_scenes))
    else:
        print("Error: want to sample %d from %d scenes"%(train_scene_num, total_train_scene_num))  
        exit()
    
    # sample train episodes
    sample_train_episodes(train_scenes, train_episode_num, 
    same_scene_val_episode_num, same_scene_val_mini_episode_num,
    same_start_goal_val_episode_num, same_start_goal_val_mini_episode_num,
    behavior_dataset_path, pointgoal_train_meta) 

    # use all val episodes
    # and sample a subset of val episodes as val_mini
    sample_across_scene_val_val_mini_episodes(val_scenes, 
    across_scene_val_mini_episode_num,
    behavior_dataset_path, pointgoal_val_meta)

# split a specific pointgoal dataset split: {'train', 'val', 'val_mini'} 
# (train by default) into train, validate, test
# behavior_dataset_path: "/dataset/behavior_dataset_gibson"
def generate_behavior_dataset_meta_v1(yaml_name, 
    pointgoal_dataset_split, 
    behavior_dataset_path,
    train_scene_num, train_episode_num, 
    across_scene_val_scene_num, across_scene_val_episode_num,
    same_scene_val_episode_num,
    same_start_goal_val_episode_num,
    across_scene_test_scene_num, across_scene_test_episode_num,
    same_scene_test_episode_num,
    same_start_goal_test_episode_num):

    config_file = os.path.join(config_path, yaml_name)
    config = parse_config(config_file)

    pointgoal_meta, total_scene_num, total_episode_num = load_pointgoal_dataset_meta(config, pointgoal_dataset_split)


    behavior_dataset_path = config.get("behavior_dataset_path")
    
    # check divisible
    check_episode_per_scene(train_scene_num, train_episode_num, 
    across_scene_val_scene_num, across_scene_val_episode_num,
    same_scene_val_episode_num,
    same_start_goal_val_episode_num,
    across_scene_test_scene_num, across_scene_test_episode_num,
    same_scene_test_episode_num,
    same_start_goal_test_episode_num)

    # sample scenes
    sample_scene_num = train_scene_num + across_scene_val_scene_num + across_scene_test_scene_num
    if sample_scene_num <= total_scene_num:
        scene_list = pointgoal_meta.keys()
        # sample without replacement
        selected_scenes = random.sample(scene_list, sample_scene_num)
        # split into train, val, test scenes
        train_scenes = selected_scenes[0:train_scene_num]
        val_scenes = selected_scenes[train_scene_num:train_scene_num+across_scene_val_scene_num]
        test_scenes = selected_scenes[-across_scene_test_scene_num:]
        print("Sampled scenes:")
        print("train scenes: %s"%(train_scenes))
        print("val scenes: %s"%(val_scenes))
        print("test scenes: %s"%(test_scenes))
    else:
        print("Error: want to sample %d from %d scenes"%(sample_scene_num, total_scene_num))  
        exit()
    
    # sample train episodes
    sample_train_episodes_v1(train_scenes, train_episode_num, 
    same_scene_val_episode_num, same_scene_test_episode_num,
    same_start_goal_val_episode_num, same_start_goal_test_episode_num,
    behavior_dataset_path, pointgoal_meta) 

    # sample across scene val episodes
    sample_across_scene_episodes(val_scenes, across_scene_val_episode_num, 
    behavior_dataset_path, pointgoal_meta, "across_scene_val")

    # sample across scene test episodes
    sample_across_scene_episodes(test_scenes, across_scene_test_episode_num, 
    behavior_dataset_path, pointgoal_meta, "across_scene_test")
    

def load_behavior_dataset_scenes(behavior_dataset_path, split_name):

    behavior_dataset_scene_path = os.path.join(behavior_dataset_path, "scenes")
    behavior_dataset_scene_file = os.path.join(behavior_dataset_scene_path, '%s.pickle'%(split_name))

    if not os.path.exists(behavior_dataset_scene_file):
        print("Error: file does not exist: %s"%(behavior_dataset_scene_file))
        exit()
    
    scene_list = pickle.load(open(behavior_dataset_scene_file, "rb" ))

    print("Behavior data split: %s"%split_name)
    print("Loaded %d scenes"%(len(scene_list)))
    
    return scene_list


def generate_one_episode(env, episode, goal_dimension, goal_coord_system):
    observations = []
    actions = []
    rel_goals = []
    distance_to_goals = []
    goal_positions = []
    state_positions = []
    abs_goals = []
    dones = []
    rewards = []

    traj = {}

    # reset the env
    obs = env.reset(episode=episode, plan_shortest_path=True)
    # add (s0, g0, d0, r0)
    # d0=False, r0=0
    update_episode_data(env=env,
        obs=obs, 
        reward=0.0, 
        done=False, 
        goal_dimension=goal_dimension, 
        goal_coord_system=goal_coord_system,
        observations=observations,
        actions=actions,
        rel_goals=rel_goals,
        distance_to_goals=distance_to_goals,
        goal_positions=goal_positions,
        state_positions=state_positions,
        abs_goals=abs_goals,
        dones=dones,
        rewards=rewards,
        action=None,
        qs=None,
        q=None)
            
    for action in env.optimal_action_seq:
        obs, reward, done, info = env.step(action)
        # add (s_i, a_{i-1}, g_i, d_i, r_i)
        update_episode_data(env=env,
            obs=obs, 
            reward=reward, 
            done=done, 
            goal_dimension=goal_dimension, 
            goal_coord_system=goal_coord_system,
            observations=observations,
            actions=actions,
            rel_goals=rel_goals,
            distance_to_goals=distance_to_goals,
            goal_positions=goal_positions,
            state_positions=state_positions,
            abs_goals=abs_goals,
            dones=dones,
            rewards=rewards,
            action=action,
            qs=None,
            q=None)
    
    # append an additional action STOP (besides the one at the end of the optimal action sequence)
    actions.append(0)

    # print(len(observations)) # n+1
    # print(len(actions)) # n+1
    # print(len(rel_goals)) # n+1
    # print(len(distance_to_goals)) # n+1
    # print(len(goal_positions)) # n+1
    # print(len(state_positions)) # n+1
    # print(len(abs_goals)) # n+1
    # print(len(dones)) # n+1
    # print(len(rewards)) # n+1
    # print(len(env.optimal_action_seq)) # n

    # print(actions)
    # print(env.optimal_action_seq)

    
    # print(actions[-1])
    # print(actions[-2])
    # print(actions[-3])
    # print(env.optimal_action_seq[-1])
    # print(env.optimal_action_seq[-2])
    # print(abs_goals)
    # exit()

    # check validity at the end of the trajectory
    if env.optimal_action_seq:
        if done == False:  # optimal policy did not done
            print("Error: done should be true after following the shortest path")
            print("Distance to goal at the end of the trajectory: %f"%(env.get_current_distance()))
            valid_episode = False
        else:
            if env.is_success() == False: # optimal policy did not succeed
                print("Error: success should be true after following the shortest path")
                print("Distance to goal at the end of the trajectory: %f"%(env.get_current_distance()))
                valid_episode = False
            else:
                # if env.optimal_action_seq[-1] != 0: # the last action of optimal action sequence is not STOP
                #     print("Error: the last action of optimal action sequence is not STOP")
                #     valid_episode = False
                # else:    
                valid_episode = True
                traj["observations"] = observations
                traj["actions"] = actions
                traj["rel_goals"] = rel_goals
                traj["distance_to_goals"] = distance_to_goals
                traj["goal_positions"] = goal_positions
                traj["state_positions"] = state_positions
                traj["abs_goals"] = abs_goals
                traj["dones"] = dones
                traj["rewards"] = rewards
                
    else:
        print("Error: shortest path not found")
        valid_episode = False
    
    return valid_episode, traj, env.optimal_action_seq


def generate_train_behavior_data_with_q(yaml_name, behavior_dataset_path, 
    split_name):

    env = MultiNavEnv(config_file=yaml_name)

    config = parse_config(os.path.join(config_path, yaml_name))
    
    train_episodes = load_behavior_dataset_meta(behavior_dataset_path, split_name)

    traj_lens = []
    trajectories = [] # real observation sequence
    action_sequences = [] # optimal action sequences generated by path planner
    
    for i, episode in tqdm(enumerate(train_episodes)):
        start_time = time.time()

        # generate one episode with q
        traj, act_seq = env.generate_one_episode_with_q(episode)
        
        print("Time per episode: %s"%(time.time()-start_time))
        
        trajectories.append(traj)
        # act_seq include the first STOP, not the second STOP
        action_sequences.append(act_seq)
        traj_lens.append(len(act_seq)+1)

        if i >= 0:
            break
    
    print("==============================================")
    print("Generated %d training trajectories"%(len(trajectories)))
    traj_lens = np.array(traj_lens, dtype=np.float32)  
    print("Total steps: %d"%(np.sum(traj_lens, axis=0)))
    print("Min length: %d"%(np.min(traj_lens, axis=0)))
    print("Mean length: %d"%(np.mean(traj_lens, axis=0)))
    print("Max length: %d"%(np.max(traj_lens, axis=0)))
    print("Std of length: %f"%(np.std(traj_lens, axis=0)))
    print("Length is measured in the number of polar actions")
    print("Number of actions in the polar action space: %s"%(str(env.polar_action_space.polar_action_number)))
    print("==============================================")

    # close env
    env.close()
    
    # save
    # dump training trajectories
    # each part include 1000 episodes
    total_trajectory_num = len(trajectories)
    
    if total_trajectory_num <= 1000:
        with open(os.path.join(behavior_dataset_path, '%s_data.pickle'%(split_name)), 'wb') as handle:
            pickle.dump(trajectories, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        part_num = total_trajectory_num // 1000
        for i in range(part_num):
            with open(os.path.join(behavior_dataset_path, '%s_data_part%d.pickle'%(split_name, i+1)), 'wb') as handle:
                pickle.dump(trajectories[(1000*(i)):(1000*(i+1))], handle, protocol=pickle.HIGHEST_PROTOCOL)
        rest_num = total_trajectory_num % 1000
        if rest_num > 0:
            with open(os.path.join(behavior_dataset_path, '%s_data_part%d.pickle'%(split_name, part_num+1)), 'wb') as handle:
                pickle.dump(trajectories[(1000*part_num):], handle, protocol=pickle.HIGHEST_PROTOCOL)
                
    # dump action sequence
    with open(os.path.join(behavior_dataset_path, '%s_action_sequences.pickle'%(split_name)), 'wb') as handle:
        pickle.dump(action_sequences, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Behavior training dataset %s generation Done: %s"%(split_name, behavior_dataset_path))

def load_trajectories(behavior_dataset_path):
    # load all trajectories from the training dataset
    dataset_path = behavior_dataset_path
    trajectories = []
    # collect all file parts
    for file in os.listdir(dataset_path):
        if file.endswith(".pickle") and file.startswith("train_data"):
            current_train_dataset_path = os.path.join(dataset_path, file)
            print("Loading trajectories from %s"%(current_train_dataset_path))
            with open(current_train_dataset_path, 'rb') as f:
                trajectories_current_file = pickle.load(f)
                trajectories.extend(trajectories_current_file)
    
    print("Loaded %d training trajectories"%(len(trajectories)))

    return trajectories

# each parallel task creates one environment
# episodes is a list of episode
def parallel_generation_task(episodes):
    # fixed config file name
    config_file_name = "imitation_learning_sqn.yaml"
    env = MultiNavEnv(config_file=config_file_name)
    
    trajectories = []
    for episode in episodes:
        # generate one episode with q
        traj, act_seq = env.generate_one_episode_with_q(episode)
        
        trajectories.append(traj)

    env.close()

    return trajectories

def generate_train_behavior_data_with_q_parallel(behavior_dataset_path, 
    split_name):
    # process number
    n_process = mp.cpu_count()  # 20
    # assign training episode to each process
    total_train_episodes = load_behavior_dataset_meta(behavior_dataset_path, split_name)
    #total_train_episodes = total_train_episodes[:20]
    
    # episode_groups must be a list
    episode_groups = chunks(lst=total_train_episodes, n=n_process)

    # start n processes (# cpu cores)
    with mp.Pool(n_process) as pool:
        print("Created a pool of %d processes"%n_process)
        # call the same function with different data in parallel asynchronously
        # can only accept a list of single argument

        for i, trajectories in enumerate(pool.imap(parallel_generation_task, episode_groups)):
            print("---------------------------------------------------")
            print("Process %d generated %d trajectories"%(i+1, len(trajectories)))
            print("---------------------------------------------------")

            # save trajectories
            with open(os.path.join(behavior_dataset_path, '%s_data_part%d.pickle'%(split_name, i+1)), 'wb') as handle:
                pickle.dump(trajectories, handle, protocol=pickle.HIGHEST_PROTOCOL)
       
    # verify by loading the saved trajectories
    total_trajectories = load_trajectories(behavior_dataset_path)
    
    # get trajectory lengths
    traj_lens = []
    for traj in total_trajectories:
        traj_lens.append(len(traj["actions"]))

    print("==============================================")
    print("Generated %d training trajectories"%(len(traj_lens)))
    traj_lens = np.array(traj_lens, dtype=np.float32) 
    print("Total steps: %d"%(np.sum(traj_lens, axis=0)))
    print("Min length: %d"%(np.min(traj_lens, axis=0)))
    print("Mean length: %d"%(np.mean(traj_lens, axis=0)))
    print("Max length: %d"%(np.max(traj_lens, axis=0)))
    print("Std of length: %f"%(np.std(traj_lens, axis=0)))
    print("==============================================")

    print("Behavior training dataset %s generation Done: %s"%(split_name, behavior_dataset_path))

    #print(traj_lens)

# behavior_dataset_path: "/dataset/behavior_dataset_gibson"
def generate_train_behavior_data(yaml_name, behavior_dataset_path, 
    split_name):
    env = MultiNavEnv(config_file=yaml_name)
    
    train_episodes = load_behavior_dataset_meta(behavior_dataset_path, split_name)

    goal_dimension = int(env.config.get("goal_dimension"))
    goal_coord_system = env.config.get("goal_coord_system")

    #train_episodes = train_episodes[:10]
    
    traj_lens = []
    trajectories = [] # real observation sequence
    action_sequences = [] # optimal action sequences generated by path planner 
   
    for episode in tqdm(train_episodes):
        # generate one episode
        valid_episode, traj, act_seq = generate_one_episode(env, episode, goal_dimension, goal_coord_system)
        
        if valid_episode == False:
            print("Error: invalid episode, need to resample train episodes!")
            exit()
        else:
            trajectories.append(traj)
            # act_seq include the first STOP, not the second STOP
            action_sequences.append(act_seq)
            traj_lens.append(len(act_seq)+1)
    
    print("==============================================")
    print("Generated %d training trajectories"%(len(trajectories)))
    traj_lens = np.array(traj_lens, dtype=np.float32)  
    print("Total steps: %d"%(np.sum(traj_lens, axis=0)))
    print("Min length: %d"%(np.min(traj_lens, axis=0)))
    print("Mean length: %d"%(np.mean(traj_lens, axis=0)))
    print("Max length: %d"%(np.max(traj_lens, axis=0)))
    print("Std of length: %f"%(np.std(traj_lens, axis=0)))
    print("Length is measured in the number of cartesian actions")
    print("Number of actions in the cartesian action space: %d"%(env.action_space.n))
    print("==============================================")
    
    # save
    #config_file=os.path.join(config_path, yaml_name)
    #config = parse_config(config_file)

    # with open(os.path.join(behavior_dataset_path, '%s_data.pickle'%(split_name)), 'wb') as handle:
    #     pickle.dump(trajectories, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # dump training trajectories
    # each part include 1000 episodes
    total_trajectory_num = len(trajectories)
    
    if total_trajectory_num <= 1000:
        with open(os.path.join(behavior_dataset_path, '%s_data.pickle'%(split_name)), 'wb') as handle:
            pickle.dump(trajectories, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        part_num = total_trajectory_num // 1000
        for i in range(part_num):
            with open(os.path.join(behavior_dataset_path, '%s_data_part%d.pickle'%(split_name, i+1)), 'wb') as handle:
                pickle.dump(trajectories[(1000*(i)):(1000*(i+1))], handle, protocol=pickle.HIGHEST_PROTOCOL)
        rest_num = total_trajectory_num % 1000
        if rest_num > 0:
            with open(os.path.join(behavior_dataset_path, '%s_data_part%d.pickle'%(split_name, part_num+1)), 'wb') as handle:
                pickle.dump(trajectories[(1000*part_num):], handle, protocol=pickle.HIGHEST_PROTOCOL)
                
    # dump action sequence
    with open(os.path.join(behavior_dataset_path, '%s_action_sequences.pickle'%(split_name)), 'wb') as handle:
        pickle.dump(action_sequences, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Behavior training dataset %s generation Done: %s"%(split_name, behavior_dataset_path))

    env.close()



# used when generate augmented episodes
def assign_episode_to_scene_behavior_dataset(episodes):
    scene_episode_list = {}
    for episode in tqdm(episodes):
        
        if episode.scene_id in scene_episode_list:
            scene_episode_list[episode.scene_id].append(episode)
        else:
            scene_episode_list[episode.scene_id] = [episode]

    for key, value in scene_episode_list.items():
        print("%s: %d"%(key, len(value)))

    return scene_episode_list

# extract (s,g) from episode in the original form
def extract_sg_from_episode(episode):
    sg_dict = {}

    # (3,) cartesian
    sg_dict["start_position"] = episode.start_position
    # (4,) quarternion
    sg_dict["start_rotation"] = episode.start_rotation
    # (3,) cartesian
    sg_dict["goal_position"] = episode.goals[0].position

    return sg_dict

def get_sg_pairs_one_scene(scene_episodes):
    sg_pairs = []
    for episode in tqdm(scene_episodes):
        sg_dict = extract_sg_from_episode(episode)
        sg_pairs.append(sg_dict)

    return sg_pairs

def sample_new_navigable_goal_position(env, orignal_goal_position):
    while True:
        # sample a non-obstacle point
        new_goal_position = env.sim.pathfinder.get_random_navigable_point()
        navigable = env.sim.pathfinder.is_navigable(new_goal_position)
        
        overlap = np.array_equal(new_goal_position, orignal_goal_position)
        if (not overlap) and navigable:
            return new_goal_position

# generate one augment episode
def generate_one_episode_for_one_sg_pair(sg_pairs, scene_name, env):
    if not env.sim.pathfinder.is_loaded:
        print("Error: env.sim.pathfinder NOT loaded")
        exit()
    
    while True:
        # sample a new (s,g) pair
        sg_pair = random.sample(sg_pairs, 1)[0]
        
        orignal_goal_position = sg_pair["goal_position"]
        orignal_goal_position = np.array(orignal_goal_position, dtype=np.float32)
        # sample a valid new goal position (not overlap with original goal position)
        # done and succeed will be checked when executing the trajectory in the environment later
        # (3,) array
        new_goal_position = sample_new_navigable_goal_position(env, orignal_goal_position)
        new_nav_goal = NavigationGoal(position=new_goal_position)
        new_episode = NavigationEpisode(episode_id="", 
            scene_id=scene_name,
            start_position=sg_pair["start_position"],
            start_rotation=sg_pair["start_rotation"],
            goals=[new_nav_goal]
            )
        # reset env and get the shortest path
        env.reset(episode=new_episode, plan_shortest_path=True)
        # the shortest path exists return the episode, otherwise resample
        if env.optimal_action_seq:
            return new_episode
    

def generate_augment_episode_one_scene(scene_name, scene_episodes, env, aug_episode_num_per_scene):
    sg_pairs = get_sg_pairs_one_scene(scene_episodes)
    
    augment_episodes_one_scene = []
    for i in tqdm(range(aug_episode_num_per_scene)):
        episode = generate_one_episode_for_one_sg_pair(sg_pairs, scene_name, env)
        augment_episodes_one_scene.append(episode)
    
    return augment_episodes_one_scene

def episode_in_list(episode, episode_list):
    for ep in episode_list:
        if ep.episode_id == episode.episode_id:
            return True

    return False

def exclude_episodes(all_scene_episodes, exclude_scene_episodes):
    rest_scene_episodes = []
    for episode in all_scene_episodes:
        if not episode_in_list(episode=episode, episode_list=exclude_scene_episodes):
            rest_scene_episodes.append(episode)
    
    assert len(all_scene_episodes) - len(exclude_scene_episodes) == len(rest_scene_episodes), "Error: scene episode exclusion is wrong"
    return rest_scene_episodes

def add_more_episode_one_scene(scene_name, pointgoal_meta, exclude_scene_episodes, aug_episode_num_per_scene):
    # collect all episodes from current scene
    all_scene_episodes = []
    for data in pointgoal_meta[scene_name]:
        all_scene_episodes.append(data["episode"])
    
    # exclude used episodes
    rest_scene_episodes = exclude_episodes(all_scene_episodes, exclude_scene_episodes)
    
    # sample episodes from rest episodes without replacement
    augment_scene_episodes = random.sample(rest_scene_episodes, aug_episode_num_per_scene)
    
    return augment_scene_episodes

# augment training data with same s different g
def generate_behavior_dataset_train_aug_meta(yaml_name, behavior_dataset_path, total_aug_episode_num):
    env = MultiNavEnv(config_file=yaml_name)
    train_episodes = load_behavior_dataset_meta(behavior_dataset_path, 'train')
    scene_episode_list = assign_episode_to_scene_behavior_dataset(train_episodes)
    scene_list = list(scene_episode_list.keys())
    scene_num = len(scene_list)
    assert total_aug_episode_num % scene_num == 0, "Error: train: total augment episode number is not divisible by scene number"
    aug_episode_num_per_scene = int(total_aug_episode_num / scene_num)
    print("Augmenta each scene with %d episodes"%aug_episode_num_per_scene)

    augment_episodes = []
    for scene_name, scene_episodes in scene_episode_list.items():
        augment_episodes_one_scene = generate_augment_episode_one_scene(scene_name, scene_episodes, env, aug_episode_num_per_scene)
        augment_episodes.extend(augment_episodes_one_scene)
    
    # save train_aug episode meta data
    save_behavior_dataset_meta(augment_episodes, 
        behavior_dataset_path, "train_aug")

    env.close()

# add more training episodes, keep the original training, same scene validation set and across scene validation set
def add_more_training_episodes(yaml_name, 
    source_behavior_dataset_path,
    total_aug_episode_num):

    config_file=os.path.join(config_path, yaml_name)
    config = parse_config(config_file)

    train_episodes = load_behavior_dataset_meta(source_behavior_dataset_path, 'train')
    same_scene_val_episodes = load_behavior_dataset_meta(source_behavior_dataset_path, 'same_scene_val')
    exclude_episodes = train_episodes + same_scene_val_episodes
    exclude_scene_episode_list = assign_episode_to_scene_behavior_dataset(exclude_episodes)
    print("Exisiting training and validation episodes: %d"%(len(exclude_episodes)))
    scene_list = list(exclude_scene_episode_list.keys())
    scene_num = len(scene_list)
    print("Exisiting scenes: %d"%(scene_num))
    assert total_aug_episode_num % scene_num == 0, "Error: train: total augment episode number is not divisible by scene number"
    aug_episode_num_per_scene = int(total_aug_episode_num / scene_num)
    print("Augment each scene with %d episodes"%aug_episode_num_per_scene)

    # load all episodes from pointgoal dataset
    pointgoal_train_meta, total_train_scene_num, total_train_episode_num = load_pointgoal_dataset_meta(config=config, split="train")
    
    # generate augment train episodes
    augment_train_episodes = []
    for scene_name, exclude_scene_episodes in exclude_scene_episode_list.items():
        augment_episodes_one_scene = add_more_episode_one_scene(scene_name, pointgoal_train_meta, exclude_scene_episodes, aug_episode_num_per_scene)
        augment_train_episodes.extend(augment_episodes_one_scene)
    
    # save train_aug episode meta data
    save_behavior_dataset_meta(augment_train_episodes, 
        source_behavior_dataset_path, "train_aug")

    print("Done")

def get_scenes_not_in_behavior_dataset(yaml_name, behavior_dataset_path):
    config_file=os.path.join(config_path, yaml_name)
    config = parse_config(config_file)
    
    # get train scenes of the original pointgoal dataset
    pointgoal_train_meta, total_scene_num, total_episode_num = load_pointgoal_dataset_meta(config, "train")
    pointgoal_train_scene_list = pointgoal_train_meta.keys()
    print("===============================================")
    print("Pointgoal dataset train split: %d scenes"%(len(pointgoal_train_scene_list)))
    print(pointgoal_train_scene_list)
    print("===============================================")
    

    # get train scenes in behavior dataset
    train_scene_list = load_behavior_dataset_scenes(behavior_dataset_path, 'train')
    print("===============================================")
    print("Behavior dataset train split: %d scenes"%(len(train_scene_list)))
    print(train_scene_list)
    print("===============================================")

    
    # get rest train scenes
    rest_scene_list = list(set(pointgoal_train_scene_list) - set(train_scene_list))
    
    print("===============================================")
    print("Remaining scenes: %d"%(len(rest_scene_list)))
    print("===============================================")

    return rest_scene_list

def generate_image_dataset_scenes(yaml_name, behavior_dataset_path, image_dataset_path, scene_number):
    # get rest training scenes
    total_scenes = get_scenes_not_in_behavior_dataset(yaml_name, behavior_dataset_path)

    
    # sample without replacement
    assert len(total_scenes) >= scene_number, "Error: Sample scenes %d is larger than total available scenes %d"%(scene_number, len(total_scenes))
    selected_scenes = random.sample(total_scenes, scene_number)

    print("===============================================")
    print(selected_scenes)
    print("Total scenes: %d"%(len(total_scenes)))
    print("Sampled scenes: %d"%(len(selected_scenes)))
    print("===============================================")

    # dump data
    image_meta_folder = os.path.join(image_dataset_path, "meta_data")
    with open(os.path.join(image_meta_folder, 'train_scenes.pickle'), 'wb') as handle:
        pickle.dump(selected_scenes, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print("Training scenes saved.")

    return selected_scenes

def generate_images_in_one_scene(scene_id, image_number_per_scene, yaml_name):
    env =  NavEnv(config_file=os.path.join(config_path, yaml_name),
        scene_id=scene_id)

    assert env.random_start, "Error: s0 should be randomized to generate a random image whenever reset is called."

    images = []
    for i in range(image_number_per_scene):
        obs = env.reset()
        # (C,H,W)
        obs_array = extract_observation(obs, env.observation_space.spaces)
        images.append(obs_array)

    env.close()

    return images

def generate_image_dataset_data(yaml_name, image_dataset_path, image_number_per_scene):
    image_meta_folder = os.path.join(image_dataset_path, "meta_data")
    selected_scenes_path = os.path.join(image_meta_folder, 'train_scenes.pickle')
    selected_scenes = pickle.load(open(selected_scenes_path, "rb" ))
    print("Train scenes: %d"%(len(selected_scenes)))

    total_images = {}
    total_number = 0
    for scene_id in tqdm(selected_scenes):
        images = generate_images_in_one_scene(scene_id, 
            image_number_per_scene, yaml_name)
        total_images[scene_id] = images
        print("---------------------------------------")
        print(scene_id)
        print("Generated %d images"%(len(images)))
        total_number += len(images)
    
    print("---------------------------------------")
    print("Generated total train images: %d"%(total_number))

    # dump data
    with open(os.path.join(image_dataset_path, 'train_data.pickle'), 'wb') as handle:
        pickle.dump(total_images, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Images saved.")

def visualize_image_dataset(image_dataset_path):
    train_image_path = os.path.join(image_dataset_path, 'train_data.pickle')
    train_images = pickle.load(open(train_image_path, "rb" ))
    for scene_id, scene_images in train_images.items():
        for i in range(2):
            img = np.asarray(scene_images[i]).astype(np.uint8)
            # (C,H,W) --> (H,W,C)
            img = np.transpose(img, (1, 2, 0))
            
            cv2.imshow('RobotView', img)

            # wait for 0.5s
            key = cv2.waitKey(500)

            #if ESC is pressed, exit loop
            if key == 27:
                exit()

# from s0, end with STOP
def get_geodesic_q_along_optimal_path_from_s0(env, episode, config, episode_index=0):
    # reset and plan the optimal action sequence
    obs = env.reset(episode=episode, plan_shortest_path=True)
   
    action_num = env.action_space.n

    print("="*20)
    print('Episode: {}'.format(episode_index+1))
    print("Goal position: %s"%(env.goal_position))
    print("Start position: %s"%(env.start_position))
    print("Optimal action sequence: %s"%env.optimal_action_seq)
    print("Optimal action sequence length: %s"%len(env.optimal_action_seq))
    print("="*20)

    done = False
    optimal_actions = []
    i = 0
    while not done:
        current_state = env.get_agent_state()
        # print("-------------------")
        # print(current_state.position)
        # print(current_state.rotation)
        # print("-------------------")

        current_q_values = []
        
        for action in list(range(action_num)):
            # take one step along each direction
            obs, reward, done, info = env.step(action)

            # compute q value
            q = env.get_geodesic_distance_based_q_current_state()

            current_q_values.append(q)

            # get back to the original state
            env.set_agent_state(
                new_position=current_state.position,
                new_rotation=current_state.rotation,
                is_initial=False,
                quaternion=True
            )
        
        # max q should include the optimal action
        current_q_values = np.array(current_q_values, dtype="float32")
        
        # list of indices where max q happen
        optimal_action_list = list(np.argwhere(current_q_values == np.amax(current_q_values)).squeeze(axis=1))
        # multiple equal paths, always pick the one move forward
        optimal_action = optimal_action_list[0]
        optimal_actions.append(optimal_action)
        
        # print q values at current state
        print("="*20)
        print("Step: %d"%(i+1))
        print(current_q_values)
        print(optimal_action)

        #exit()

        # take one action along the optimal path
        obs, reward, done, info = env.step(optimal_action)
        i += 1
    
    if info["success"] != 1:
        print("Error: planned path did not succeed!")
    
    print("="*20)
    print(optimal_actions)
    print("Q based optimal path length: %d"%(len(optimal_actions)))
    print("Path planner optimal path length: %d"%(len(env.optimal_action_seq)))
    print("="*20)

def test_q(config_file="imitation_learning_dqn.yaml"):
    env = MultiNavEnv(config_file=config_file)

    config = parse_config(os.path.join(config_path, config_file))

    episodes = load_behavior_dataset_meta(
                behavior_dataset_path=config.get("behavior_dataset_path"), 
                split_name="same_start_goal_val_mini")
    
    # print(env.action_space.n)
    # exit()
    for i, episode in enumerate(episodes):
        get_geodesic_q_along_optimal_path_from_s0(env, episode, config, episode_index=i)

        if i >= 3:
            break

def test_path_follower(config_file="imitation_learning_dqn.yaml"):
    env = MultiNavEnv(config_file=config_file)

    config = parse_config(os.path.join(config_path, config_file))

    episodes = load_behavior_dataset_meta(
                behavior_dataset_path=config.get("behavior_dataset_path"), 
                split_name="same_start_goal_val_mini")
    
    # print(env.action_space.n)
    # exit()
    for i, episode in enumerate(episodes):
        # reset and plan the optimal action sequence
        obs = env.reset(episode=episode, plan_shortest_path=True)

        print("="*20)
        print('Episode: {}'.format(i+1))
        print("Goal position: %s"%(env.goal_position))
        print("Start position: %s"%(env.start_position))
        print("Optimal action sequence: %s"%env.optimal_action_seq)
        print("Optimal action sequence length: %s"%len(env.optimal_action_seq))
        

        
        actions = []
        done = False
        while not done:
            a = env.follower.next_action_along(goal_pos=env.goal_position)
            actions.append(a)
            obs, reward, done, info = env.step(a)
            
        print("Step by step planning path: %s"%actions)
        print("Step by step planning path length: %s"%len(actions))
        print("="*20)

        if i >= 3:
            break

def test_dataset(config_file="imitation_learning_dqn.yaml"):
    env = MultiNavEnv(config_file=config_file)

    config = parse_config(os.path.join(config_path, config_file))

    episodes = load_behavior_dataset_meta(
                behavior_dataset_path=config.get("behavior_dataset_path"), 
                split_name="same_start_goal_val_mini")
    
    # print(env.action_space.n)
    # exit()
    for i, episode in enumerate(episodes):
        obs = env.reset(episode=episode, plan_shortest_path=True)
        print("="*20)
        print('Episode: {}'.format(i+1))
        print("Goal position: %s"%(env.goal_position))
        #env.print_agent_state()
        print("Start position: %s"%(env.start_position))
        #print(env.get_optimal_trajectory())
        print("Optimal action sequence: %s"%env.optimal_action_seq)
        print("Optimal action sequence length: %s"%len(env.optimal_action_seq))
        print("="*20)

        
        for action in env.optimal_action_seq:
            obs, reward, done, info = env.step(action)
            #print(action)
            #print(obs)
            #exit()
            #print(obs["color_sensor"].shape)
            #print(obs["pointgoal"].shape)
            env.render()   

if __name__ == "__main__":
    # ====== first set seed =======
    set_seed_except_env_seed(seed=0)

    # ====== test =======
    #load_pointgoal_dataset("imitation_learning_rnn_bc.yaml")  
    #test_get_scene_names("imitation_learning_rnn_bc.yaml")
    #shortest_path_follower("imitation_learning_rnn_bc.yaml")
    
    # ====== generate pointgoal meta data =======
    # generate_pointgoal_dataset_meta(yaml_name="imitation_learning_dt.yaml", split="train")
    
    # ====== generate train / val /test split =======
    # generate_behavior_dataset_meta_v1(yaml_name="imitation_learning_rnn_bc.yaml", 
    #     behavior_dataset_path="/dataset/behavior_dataset_gibson_large",
    #     train_scene_num=4, train_episode_num=3000, 
    #     across_scene_val_scene_num=2, across_scene_val_episode_num=10,
    #     same_scene_val_episode_num=20,
    #     same_start_goal_val_episode_num=20,
    #     across_scene_test_scene_num=2, across_scene_test_episode_num=50,
    #     same_scene_test_episode_num=100,
    #     same_start_goal_test_episode_num=100)
    
    # generate_behavior_dataset_meta(yaml_name="imitation_learning_rnn_bc.yaml", 
    #     behavior_dataset_path="/dataset/behavior_dataset_gibson_4_scene",
    #     train_scene_num=4, train_episode_num=2000, 
    #     same_scene_val_episode_num=400,
    #     same_start_goal_val_episode_num=400,
    #     across_scene_val_mini_episode_num=28,
    #     same_scene_val_mini_episode_num=28,
    #     same_start_goal_val_mini_episode_num=28)

    # generate_behavior_dataset_meta(yaml_name="imitation_learning_rnn_bc.yaml", 
    #     behavior_dataset_path="/dataset/behavior_dataset_gibson_72_scene",
    #     train_scene_num=72, train_episode_num=2160, #2880 2160
    #     same_scene_val_episode_num=720,
    #     same_start_goal_val_episode_num=720,
    #     across_scene_val_mini_episode_num=28,
    #     same_scene_val_mini_episode_num=72,
    #     same_start_goal_val_mini_episode_num=72)
    
    # generate_behavior_dataset_meta(yaml_name="imitation_learning_rnn_bc.yaml", 
    #     behavior_dataset_path="/dataset/behavior_dataset_gibson_1_scene",
    #     train_scene_num=1, train_episode_num=2000, #2880 2160
    #     same_scene_val_episode_num=200,
    #     same_start_goal_val_episode_num=200,
    #     across_scene_val_mini_episode_num=28,
    #     same_scene_val_mini_episode_num=20,
    #     same_start_goal_val_mini_episode_num=20)

    # generate_behavior_dataset_meta(yaml_name="imitation_learning_rnn_bc.yaml", 
    #     behavior_dataset_path="/dataset/behavior_dataset_gibson_1_scene",
    #     train_scene_num=1, train_episode_num=4000, #2880 2160
    #     same_scene_val_episode_num=200,
    #     same_start_goal_val_episode_num=200,
    #     across_scene_val_mini_episode_num=28,
    #     same_scene_val_mini_episode_num=20,
    #     same_start_goal_val_mini_episode_num=20)

    # generate_behavior_dataset_meta_whole_scene(yaml_name="imitation_learning_mlp_bc.yaml", 
    #     behavior_dataset_path="/dataset/behavior_dataset_gibson_1_scene_Rancocas_2000",
    #     train_scene_num=1,
    #     same_scene_val_episode_num=1000,
    #     across_scene_val_mini_episode_num=28,
    #     same_scene_val_mini_episode_num=30)

    # ====== generate train episodes =======
    # generate_train_behavior_data(yaml_name="imitation_learning_rnn_bc.yaml", 
    #     behavior_dataset_path="/dataset/behavior_dataset_gibson_1_scene_Rancocas_2000",
    #     split_name="train")

    # ====== add more train episodes to an existing dataset =======
    # add_more_training_episodes(yaml_name="imitation_learning_rnn_bc.yaml", 
    #     source_behavior_dataset_path="/dataset/behavior_dataset_gibson_1_scene_Rancocas_2000",
    #     total_aug_episode_num=2000)
    # add_more_training_episodes(yaml_name="imitation_learning_rnn_bc.yaml", 
    #     source_behavior_dataset_path="/dataset/behavior_dataset_gibson_4_scene_2000",
    #     total_aug_episode_num=2000)

    # ====== generate train augment episodes =======
    # generate_train_behavior_data(yaml_name="imitation_learning_rnn_bc.yaml", 
    #      behavior_dataset_path="/dataset/behavior_dataset_gibson_1_scene_Rancocas_2000",
    #      split_name="train_aug")
    
    # generate_train_behavior_data(yaml_name="imitation_learning_rnn_bc.yaml", 
    #      behavior_dataset_path="/dataset/behavior_dataset_gibson_4_scene_2000",
    #      split_name="train_aug")
    
    # ====== regenerate train episodes, others kept same =======
    # generate_train_behavior_data_with_q(yaml_name="imitation_learning_sqn.yaml", 
    #      behavior_dataset_path="/dataset/behavior_dataset_gibson_1_scene_Rancocas_2000_polar_q",
    #      split_name="train")
    
    generate_train_behavior_data_with_q_parallel(
         behavior_dataset_path="/dataset/behavior_dataset_gibson_1_scene_Rancocas_2000_polar_q",
         split_name="train")
    
    # ====== generate train augment meta data =======
    # generate_behavior_dataset_train_aug_meta(
    #     yaml_name="imitation_learning_rnn_bc.yaml",
    #     behavior_dataset_path="/dataset/behavior_dataset_gibson", 
    #     total_aug_episode_num=1000)
    
    # ====== generate train augment episodes =======
    # generate_train_behavior_data(yaml_name="imitation_learning_rnn_bc.yaml", 
    #      behavior_dataset_path="/dataset/behavior_dataset_gibson",
    #      split_name="train_aug")

    # ====== generate DA target domain scenes =======
    # generate_image_dataset_scenes(yaml_name="imitation_learning_rnn_bc.yaml", 
    #     behavior_dataset_path="/dataset/behavior_dataset_gibson", 
    #     image_dataset_path="/dataset/image_dataset_gibson", 
    #     scene_number=50)
    
    # ====== generate DA target domain images =======
    # generate_image_dataset_data(yaml_name="pointgoal_ppo_baseline.yaml", 
    #     image_dataset_path="/dataset/image_dataset_gibson", 
    #     image_number_per_scene=200)

    # ====== visualize DA target domain images =======
    #visualize_image_dataset(image_dataset_path="/dataset/image_dataset_gibson")

    #test_dataset()
    #test_q()
    #test_path_follower()
