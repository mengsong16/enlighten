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
from enlighten.datasets.common import load_behavior_dataset_meta
from enlighten.datasets.common import PolarActionSpace, get_first_effective_action_sequence
from enlighten.utils.geometry_utils import euclidean_distance

import habitat_sim
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






def check_optimal_path_polar_q(env, episode):
    # reset the env
    obs = env.reset(episode=episode, plan_shortest_path=True)
    shortest_path_length = 0.0
    previous_position = env.get_agent_position()


    print("="*20)
    print("Goal position: %s"%(env.goal_position))
    print("Start position: %s"%(env.start_position))
    print("[Shortest path] Optimal cartesian action sequence: %s"%env.optimal_action_seq)
    print("[Shortest path] Optimal cartesian action sequence length: %d"%len(env.optimal_action_seq))

    shortest_path_planner_polar_seq = env.polar_action_space.map_cartesian_action_seq_to_polar_seq(env.optimal_action_seq)
    print("[Shortest path] Optimal polar action sequence: %s"%shortest_path_planner_polar_seq)
    print("[Shortest path] Optimal polar action sequence length: %d"%len(shortest_path_planner_polar_seq))

    
    # act according to the shortest path, and compute its polar q at each state
    for i, action in enumerate(env.optimal_action_seq):
        q, polar_optimal_action, cartesian_optimal_action_seq = env.compute_polar_q_current_state()
        
        # print("-----------------------------")
        # print("Step: %d"%(i+1))
        # print("[Polar Q] Optimal cartesian action sequence: %s"%(cartesian_optimal_action_seq))
        # print("[Shortest path] Optimal cartesian action: %d"%(action))
        # print("-----------------------------")

        # take one step according to the shortest path
        obs, reward, done, info = env.step(action)
        # accumulate path length
        current_position = env.get_agent_position()
        shortest_path_length += euclidean_distance(current_position, previous_position)
        previous_position = current_position
    
    print("[Shortest path] Optimal path length: %f"%shortest_path_length)

def test_polar_q_planner(config_file="imitation_learning_mlp_sqn.yaml"):
    env = MultiNavEnv(config_file=config_file)

    config = parse_config(os.path.join(config_path, config_file))

    episodes = load_behavior_dataset_meta(
                behavior_dataset_path=config.get("behavior_dataset_path"), 
                split_name="same_start_goal_val_mini")
    
    for i, episode in enumerate(episodes):
        polar_optimal_actions, cartesian_optimal_actions, polar_path_length = env.polar_q_planner(episode)
        
        #if i >= 3:
        break

def test_check_optimal_path_polar_q(config_file="imitation_learning_mlp_sqn.yaml"):
    env = MultiNavEnv(config_file=config_file)

    config = parse_config(os.path.join(config_path, config_file))

    episodes = load_behavior_dataset_meta(
                behavior_dataset_path=config.get("behavior_dataset_path"), 
                split_name="same_start_goal_val_mini")
    
    for i, episode in enumerate(episodes):
        check_optimal_path_polar_q(env, episode)
        
        #if i >= 3:
        break

def compare_greedy_planner_polar_q_planner(config_file="imitation_learning_mlp_sqn.yaml"):
    env = MultiNavEnv(config_file=config_file)

    config = parse_config(os.path.join(config_path, config_file))

    episodes = load_behavior_dataset_meta(
                behavior_dataset_path=config.get("behavior_dataset_path"), 
                split_name="same_start_goal_val_mini")
    
    for i, episode in enumerate(episodes):
        check_optimal_path_polar_q(env, episode)
        env.polar_q_planner(episode)
        
        #if i >= 3:
        break

def test_polar_action_space(config_file="imitation_learning_mlp_sqn.yaml"):
    env = MultiNavEnv(config_file=config_file)

    config = parse_config(os.path.join(config_path, config_file))

    rotate_resolution = int(config.get("rotate_resolution"))

    pa = PolarActionSpace(env, rotate_resolution)


# from s0, end with STOP
def check_optimal_path_geodesic_q(env, episode):
    # reset the env
    obs = env.reset(episode=episode, plan_shortest_path=True)
    
    print("="*20)
    print("Goal position: %s"%(env.goal_position))
    print("Start position: %s"%(env.start_position))
    print("[Shortest path] Optimal cartesian action sequence: %s"%env.optimal_action_seq)
    print("[Shortest path] Optimal cartesian action sequence length: %d"%len(env.optimal_action_seq))

    for i, optimal_action in enumerate(env.optimal_action_seq):
        current_q_values, max_q_action, cartesian_optimal_action_list = env.compute_cartesian_q_current_state()
        
        print("Q values: %s"%current_q_values)
        print("Max Q actions: %s"%max_q_action)
        print("Planned action: %s"%optimal_action)
        if optimal_action not in cartesian_optimal_action_list:
            print("Planned optimal action does not have max Q!")
        print("-----------------------------")

        # take one optimal action along the optimal path
        obs, reward, done, info = env.step(optimal_action)

    print("="*20)

def test_check_optimal_path_geodesic_q(config_file="imitation_learning_mlp_sqn.yaml"):
    env = MultiNavEnv(config_file=config_file)

    config = parse_config(os.path.join(config_path, config_file))

    episodes = load_behavior_dataset_meta(
                behavior_dataset_path=config.get("behavior_dataset_path"), 
                split_name="same_start_goal_val_mini")
    
    for i, episode in enumerate(episodes):
        check_optimal_path_geodesic_q(env, episode)
        
        #if i >= 3:
        break

if __name__ == "__main__":
    # ====== first set seed =======
    set_seed_except_env_seed(seed=0)  
    # ========= test ==========  
    #test_polar_action_space()
    #test_polar_q_planner()
    #test_check_optimal_path_polar_q()
    #compare_greedy_planner_polar_q_planner()
    test_check_optimal_path_geodesic_q()



