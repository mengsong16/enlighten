from enlighten.utils.path import *
from enlighten.utils.config_utils import parse_config
from enlighten.datasets.pointnav_dataset import PointNavDatasetV1
from enlighten.datasets.pointnav_dataset import NavigationEpisode, NavigationGoal, ShortestPathPoint
from enlighten.datasets.dataset import EpisodeIterator
from enlighten.envs.multi_nav_env import MultiNavEnv, NavEnv
from enlighten.utils.geometry_utils import euclidean_distance
from enlighten.agents.common.seed import set_seed_except_env_seed
from enlighten.utils.geometry_utils import quaternion_rotate_vector, cartesian_to_polar
from enlighten.datasets.dataset import Episode
from enlighten.utils.image_utils import try_cv2_import
cv2 = try_cv2_import()

import math
import os
import numpy as np

import pickle
from tqdm import tqdm
import random
import copy


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

    assert train_episode_num % train_scene_num == 0, "Error: train: episode num is not divisible by scene num"
    assert across_scene_val_episode_num % across_scene_val_scene_num == 0, "Error: Across scene val: episode num is not divisible by scene num"
    assert across_scene_test_episode_num % across_scene_test_scene_num == 0, "Error: Across scene test: episode num is not divisible by scene num"
    assert same_scene_val_episode_num % train_scene_num == 0, "Error: Same scene val: episode num is not divisible by train scene num"
    assert same_scene_test_episode_num % train_scene_num == 0, "Error: Same scene test: episode num is not divisible by train scene num"
    assert same_start_goal_val_episode_num % train_scene_num == 0, "Error: same start goal val: episode num is not divisible by train scene num"
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
        # sample episodes from current scene
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
    
# behavior_dataset_path: "/dataset/behavior_dataset_gibson"
def load_behavior_dataset_meta(behavior_dataset_path, split_name):

    behavior_dataset_meta_data_path = os.path.join(behavior_dataset_path, "meta_data")
    behavior_dataset_meta_split_path = os.path.join(behavior_dataset_meta_data_path, '%s.pickle'%(split_name))

    if not os.path.exists(behavior_dataset_meta_split_path):
        print("Error: path does not exist: %s"%(behavior_dataset_meta_split_path))
        exit()
    
    episode_list = pickle.load(open(behavior_dataset_meta_split_path, "rb" ))

    print("Behavior data split: %s"%split_name)
    print("Loaded %d episodes"%(len(episode_list)))
    
    return episode_list

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

# [CHANNEL x HEIGHT X WIDTH]
# CHANNEL = {1,3,4}
# output numpy array
def extract_observation(obs, observation_spaces):
    n_channel = 0
    obs_array = None
    if "color_sensor" in observation_spaces:
        rgb_obs = obs["color_sensor"]
        # permute tensor from [HEIGHT X WIDTH x CHANNEL] to [CHANNEL x HEIGHT X WIDTH]
        rgb_obs = np.transpose(rgb_obs, (2, 0, 1))

        obs_array = rgb_obs
        n_channel += 3

    if "depth_sensor" in observation_spaces:
        depth_obs = obs["depth_sensor"]
        # permute tensor from [HEIGHT X WIDTH x CHANNEL] to [CHANNEL x HEIGHT X WIDTH]
        depth_obs = np.transpose(depth_obs, (2, 0, 1))
        
        if obs_array is not None:
            obs_array = np.concatenate((obs_array, depth_obs), axis=0)
        else:
            obs_array = depth_obs
        n_channel += 1

    # check if observation is valid
    if n_channel == 0:
        print("Error: channel of observation input is 0")
        exit()
    
    return obs_array

def generate_one_episode(env, episode, goal_dimension, goal_coord_system):
    observations = []
    actions = []
    rel_goals = []
    distance_to_goals = []
    goal_positions = []
    state_positions = []
    abs_goals = []

    traj = {}

    # add (s0, g0)
    obs = env.reset(episode=episode, plan_shortest_path=True)
    obs_array = extract_observation(obs, env.observation_space.spaces)
    observations.append(obs_array) # (channel, height, width)
    rel_goal = np.array(obs["pointgoal"], dtype="float32")
    rel_goals.append(rel_goal) # (goal_dim,)
    distance_to_goals.append(env.get_current_distance())  # float
    goal_position = np.array(env.goal_position, dtype="float32")
    goal_positions.append(goal_position) # (3,)
    abs_goal = goal_position_to_abs_goal(goal_position,
            goal_dimension, goal_coord_system) # (2,) or (3,)
    abs_goals.append(abs_goal)
    state_position = np.array(env.agent.get_state().position, dtype="float32")
    state_positions.append(state_position)  # (3,)
            
    for action in env.optimal_action_seq:
        obs, reward, done, info = env.step(action)
        # add (s_i, a_{i-1}, g_i)
        obs_array = extract_observation(obs, env.observation_space.spaces)
        observations.append(obs_array) # (channel, height, width)
        actions.append(action) # integer
        rel_goal = np.array(obs["pointgoal"], dtype="float32")
        rel_goals.append(rel_goal) # (goal_dim,)
        distance_to_goals.append(env.get_current_distance()) # float
        goal_position = np.array(env.goal_position, dtype="float32")
        goal_positions.append(goal_position) # (3,)
        abs_goal = goal_position_to_abs_goal(goal_position,
            goal_dimension, goal_coord_system) # (2,) or (3,)
        abs_goals.append(abs_goal)
        state_position = np.array(env.agent.get_state().position, dtype="float32")
        state_positions.append(state_position) # (3,)

        # print(rel_goal.shape)
        # print(goal_position.shape)
        # print(state_position.shape)
    
    # append an additional action STOP
    actions.append(0)

    # print(len(observations)) # n+1
    # print(len(actions)) # n+1
    # print(len(rel_goals)) # n+1
    # print(len(distance_to_goals)) # n+1
    # print(len(goal_positions)) # n+1
    # print(len(state_positions)) # n+1
    # print(len(abs_goals)) # n+1
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
                
    else:
        print("Error: shortest path not found")
        valid_episode = False
    
    return valid_episode, traj, env.optimal_action_seq
        
# behavior_dataset_path: "/dataset/behavior_dataset_gibson"
def generate_train_behavior_data(yaml_name, behavior_dataset_path, split_name):
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
    print("==============================================")
    
    # save
    config_file=os.path.join(config_path, yaml_name)
    config = parse_config(config_file)

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

# borrowed from class PointGoal
# source_rotation: quartenion representing a 3D rotation
# goal_coord_system: ["polar", "cartesian"]
# goal_dimension: 2, 3
def compute_pointgoal(source_position, source_rotation, goal_position,
    goal_dimension, goal_coord_system):
    # use source local coordinate system as global coordinate system
    # step 1: align origin 
    direction_vector = goal_position - source_position
    # step 2: align axis 
    direction_vector_agent = quaternion_rotate_vector(
        source_rotation.inverse(), direction_vector
    )

    if goal_coord_system == "polar":
        # 2D movement: r, -phi
        # -phi: angle relative to positive z axis (i.e reverse to robot forward direction)
        # -phi: azimuth, around y axis
        if goal_dimension == 2:
            # -z, x --> x, y --> (r, -\phi)
            rho, phi = cartesian_to_polar(
                -direction_vector_agent[2], direction_vector_agent[0]
            )
            return np.array([rho, -phi], dtype=np.float32)
        #  3D movement: r, -phi, theta 
        #  -phi: azimuth, around y axis
        #  theta: around z axis   
        else:
            # -z, x --> x, y --> -\phi
            _, phi = cartesian_to_polar(
                -direction_vector_agent[2], direction_vector_agent[0]
            )
            theta = np.arccos(
                direction_vector_agent[1]
                / np.linalg.norm(direction_vector_agent)
            )
            # r = l2 norm
            rho = np.linalg.norm(direction_vector_agent)

            return np.array([rho, -phi, theta], dtype=np.float32)
    else:
        # 2D movement: [-z,x]
        # reverse the direction of z axis towards robot forward direction
        if goal_dimension == 2:
            return np.array(
                [-direction_vector_agent[2], direction_vector_agent[0]],
                dtype=np.float32,
            )
        # 3D movement: [x,y,z]    
        # do not reverse the direction of z axis
        else:
            return np.array(direction_vector_agent, dtype=np.float32)

# return abs_goal numpy: (2,), (3,)
def goal_position_to_abs_goal(goal_position, goal_dimension, goal_coord_system):
    goal_world_position = np.array(goal_position, dtype=np.float32)
    return compute_pointgoal(source_position=np.array([0,0,0], dtype="float32"), 
        source_rotation=np.quaternion(1,0,0,0), 
        goal_position=goal_world_position,
        goal_dimension=goal_dimension, 
        goal_coord_system=goal_coord_system)

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


if __name__ == "__main__":
    # ====== first set seed =======
    set_seed_except_env_seed(seed=0)

    # ====== test =======
    #load_pointgoal_dataset("imitation_learning_rnn.yaml")  
    #test_get_scene_names("imitation_learning_rnn.yaml")
    #shortest_path_follower("imitation_learning_rnn.yaml")
    
    # ====== generate pointgoal meta data =======
    # generate_pointgoal_dataset_meta(yaml_name="imitation_learning_dt.yaml", split="train")
    
    # ====== generate train / val /test split =======
    # generate_behavior_dataset_meta_v1(yaml_name="imitation_learning_rnn.yaml", 
    #     behavior_dataset_path="/dataset/behavior_dataset_gibson_large",
    #     train_scene_num=4, train_episode_num=3000, 
    #     across_scene_val_scene_num=2, across_scene_val_episode_num=10,
    #     same_scene_val_episode_num=20,
    #     same_start_goal_val_episode_num=20,
    #     across_scene_test_scene_num=2, across_scene_test_episode_num=50,
    #     same_scene_test_episode_num=100,
    #     same_start_goal_test_episode_num=100)
    
    # generate_behavior_dataset_meta(yaml_name="imitation_learning_rnn.yaml", 
    #     behavior_dataset_path="/dataset/behavior_dataset_gibson",
    #     train_scene_num=4, train_episode_num=2000, 
    #     same_scene_val_episode_num=400,
    #     same_start_goal_val_episode_num=400,
    #     across_scene_val_mini_episode_num=28,
    #     same_scene_val_mini_episode_num=28,
    #     same_start_goal_val_mini_episode_num=28)

    # generate_behavior_dataset_meta(yaml_name="imitation_learning_rnn.yaml", 
    #     behavior_dataset_path="/dataset/behavior_dataset_gibson_72_scene",
    #     train_scene_num=72, train_episode_num=2160, #2880 2160
    #     same_scene_val_episode_num=720,
    #     same_start_goal_val_episode_num=720,
    #     across_scene_val_mini_episode_num=28,
    #     same_scene_val_mini_episode_num=72,
    #     same_start_goal_val_mini_episode_num=72)
    
    # ====== generate train episodes =======
    generate_train_behavior_data(yaml_name="imitation_learning_rnn.yaml", 
        behavior_dataset_path="/dataset/behavior_dataset_gibson_72_scene",
        split_name="train")
    
    # ====== generate train augment meta data =======
    # generate_behavior_dataset_train_aug_meta(
    #     yaml_name="imitation_learning_rnn.yaml",
    #     behavior_dataset_path="/dataset/behavior_dataset_gibson", 
    #     total_aug_episode_num=1000)
    
    # ====== generate train augment episodes =======
    # generate_train_behavior_data(yaml_name="imitation_learning_rnn.yaml", 
    #      behavior_dataset_path="/dataset/behavior_dataset_gibson",
    #      split_name="train_aug")

    # ====== generate DA target domain scenes =======
    # generate_image_dataset_scenes(yaml_name="imitation_learning_rnn.yaml", 
    #     behavior_dataset_path="/dataset/behavior_dataset_gibson", 
    #     image_dataset_path="/dataset/image_dataset_gibson", 
    #     scene_number=50)
    
    # ====== generate DA target domain images =======
    # generate_image_dataset_data(yaml_name="pointgoal_baseline.yaml", 
    #     image_dataset_path="/dataset/image_dataset_gibson", 
    #     image_number_per_scene=200)

    # ====== visualize DA target domain images =======
    #visualize_image_dataset(image_dataset_path="/dataset/image_dataset_gibson")
