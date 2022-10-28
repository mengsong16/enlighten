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
from enlighten.datasets.common import load_behavior_dataset_meta, get_optimal_path
from enlighten.datasets.common import PolarActionSpace
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

def get_first_effective_action_sequence(cartesian_action_seq,
    cartesian_stop_action_index,
    cartesian_forward_action_index):
    i = 0 
    sub_seq = []
    while i < len(cartesian_action_seq):
        if cartesian_action_seq[i] == cartesian_stop_action_index or cartesian_action_seq[i] == cartesian_forward_action_index:
            sub_seq.append(cartesian_action_seq[i])
            break
        else:
            sub_seq.append(cartesian_action_seq[i])
            
        i += 1
    
    return sub_seq




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


    # act according to the shortest path, and compute its polar q at each state
    cartesian_action_number = len(env.action_mapping)
    cartesian_stop_action_index = env.action_name_to_index("stop")
    cartesian_forward_action_index = env.action_name_to_index("move_forward")
    cartesian_turn_left_action_index = env.action_name_to_index("turn_left")
    cartesian_turn_right_action_index = env.action_name_to_index("turn_right")
        

    for i, optimal_action in enumerate(env.optimal_action_seq):
        current_state = env.get_agent_state()
        
        current_q_values = []
    
        print("-----------------------------")
        print("Step: %d"%(i+1))
        # "stop" q
        print("Executed actions: None")
        current_q_values.append(env.get_geodesic_distance_based_q_current_state())

        # "move_forward" q
        # take one step forward
        obs, reward, done, info = env.step(cartesian_forward_action_index)
        print("Executed actions: %s"%([cartesian_forward_action_index]))
        # get current geodesic distance to goal as q
        current_q_values.append(env.get_geodesic_distance_based_q_current_state())
            
        # get back to the original state
        env.set_agent_state(
            new_position=current_state.position,
            new_rotation=current_state.rotation,
            is_initial=False,
            quaternion=True
        )

        # "turn_left" or "turn_right" q
        for action in [cartesian_turn_left_action_index, cartesian_turn_right_action_index]:
            # take one step along current direction
            obs, reward, done, info = env.step(action)
            
            # plan the shortest path from the current state to see where move_forward or stop happen
            current_optimal_action_seq = get_optimal_path(env)
            # step the environment until move_forward or stop happen
            execute_action_seq = get_first_effective_action_sequence(current_optimal_action_seq,
                cartesian_stop_action_index,
                cartesian_forward_action_index)
            env.step_cartesian_action_seq(execute_action_seq)

            print("Executed actions: %s"%([action]+execute_action_seq))

            # get current geodesic distance to goal as q
            q = env.get_geodesic_distance_based_q_current_state()
            current_q_values.append(q)
            
            # get back to the original state
            env.set_agent_state(
                new_position=current_state.position,
                new_rotation=current_state.rotation,
                is_initial=False,
                quaternion=True
            )
        
        current_q_values = np.array(current_q_values, dtype="float32")
        # actions where max q happen
        max_q_action_list = list(np.argwhere(current_q_values == np.amax(current_q_values)).squeeze(axis=1))
        
        
        print("Q values: %s"%current_q_values)
        print("Max Q actions: %s"%max_q_action_list)
        print("Planned action: %s"%optimal_action)
        if optimal_action not in max_q_action_list:
            print("Optimal action does not have max Q!")
        print("-----------------------------")

        # take one action along the optimal path
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



