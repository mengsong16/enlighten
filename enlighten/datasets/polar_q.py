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
from enlighten.agents.common.other import get_optimal_q, get_geodesic_distance_based_q_current_state
from enlighten.datasets.il_data_gen import load_behavior_dataset_meta

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

# rotate_resolution: in degree
def compute_polar_q_current_state(env, rotate_resolution:int):
    
    rotate_resolution = int(rotate_resolution)
    assert int(360) % rotate_resolution == 0, "360 should be divisible by the rotation resolution %d"%(rotate_resolution)
    
    action_number = len(env.action_mapping)
    assert  action_number== 4, "Can only compute polar q when we have 4 actions: stop, move_forward, turn_left, turn_right" 
    
    q = np.zeros(action_number, dtype="float")
    current_state = env.get_agent_state()

    # get action indices
    stop_action_index = env.action_name_to_index("stop")
    forward_action_index = env.action_name_to_index("move_forward")
    turn_left_action_index = env.action_name_to_index("turn_left")
    turn_right_action_index = env.action_name_to_index("turn_right")
    
    # q["stop"], always be the smallest q comparing to qs on the circle
    q[stop_action_index] = get_geodesic_distance_based_q_current_state(env)

    
    # q["move_forward"]
    # take one step forward
    obs, reward, done, info = env.step(forward_action_index)
    q[forward_action_index] = get_geodesic_distance_based_q_current_state(env)
    # get back to the original state (i.e. circle center)
    env.set_agent_state(
        new_position=current_state.position,
        new_rotation=current_state.rotation,
        is_initial=False,
        quaternion=True
    )

    # check q at all angles rotate over 360 degrees
    circular_qs = []
    circular_qs.append(q[forward_action_index])

    rotate_degrees = list(range(rotate_resolution, 360, rotate_resolution))
    rotate_steps = len(rotate_degrees)

    circle_states = []
    for n in list(range(1,rotate_steps+1)):
        # rotate counterclockwise (i.e. turn left) one more time, i.e. n times in total
        obs, reward, done, info = env.step(turn_left_action_index)
        circle_states.append(env.get_agent_state())
        # take one step forward
        obs, reward, done, info = env.step(forward_action_index)
        cur_q = get_geodesic_distance_based_q_current_state(env)
        circular_qs.append(cur_q)

        # get back to the last circle state
        env.set_agent_state(
            new_position=circle_states[-1].position,
            new_rotation=circle_states[-1].rotation,
            is_initial=False,
            quaternion=True
        )
    
    # get back to the original state (i.e. circle center)
    env.set_agent_state(
        new_position=current_state.position,
        new_rotation=current_state.rotation,
        is_initial=False,
        quaternion=True
    )

    # q["turn_left"]=q["turn_right"]=max q over the circular points requiring rotation
    rotate_circular_qs = np.array(circular_qs[1:], dtype="float")
    # can exist only one max value
    rotate_q_max = np.amax(rotate_circular_qs)
    q[turn_left_action_index] = rotate_q_max
    q[turn_right_action_index] = rotate_q_max

    # move forward is optimal
    if q[forward_action_index] >= rotate_q_max:
        optimal_actions = [forward_action_index]
    # rotate is optimal
    else:
        # find minimum number of rotations
        rotate_q_max_index = np.argmax(rotate_circular_qs)
        turn_left_num = rotate_q_max_index + 1
        turn_right_num = int(360) // rotate_resolution - turn_left_num

        # print(rotate_q_max_index)
        # print(turn_left_num)
        # print(turn_right_num)
        # exit()

        # turn left until move forward
        if turn_left_num <= turn_right_num:
            optimal_actions = [turn_left_action_index] * turn_left_num
        else:
        # turn right until move forward
            optimal_actions = [turn_right_action_index] * turn_right_num
        # move forward
        optimal_actions.append(forward_action_index)

    return q, optimal_actions

def reached_goal(env, config):
    distance_to_goal = env.get_current_distance()

    if distance_to_goal < config.get("success_distance"):
        return True
    else:
        return False

def euclidean_distance(position_a, position_b):
    return np.linalg.norm(position_b - position_a, ord=2)

def polar_q_planner(env, episode, config):
    rotate_resolution = int(config.get("rotate_resolution"))

    # reset the env
    obs = env.reset(episode=episode, plan_shortest_path=True)
    path_length = 0.0
    previous_position = env.get_agent_position()

    print("="*20)
    print('Episode: ')
    print("Goal position: %s"%(env.goal_position))
    print("Start position: %s"%(env.start_position))
    print("[Shortest path] Optimal action sequence: %s"%env.optimal_action_seq)
    print("[Shortest path] Optimal action sequence length: %s"%len(env.optimal_action_seq))
    

    reach_q_flag = reached_goal(env, config)
    optimal_actions = []
    
    while not reach_q_flag:
        cur_q, cur_optimal_actions = compute_polar_q_current_state(env, rotate_resolution)
        
        print(cur_optimal_actions)
        print("----------------------")

        optimal_actions.extend(cur_optimal_actions)
        # execute current optimal action sequence
        for action in cur_optimal_actions:
            # take one step
            obs, reward, done, info = env.step(action)
            # accumulate path length
            current_position = env.get_agent_position()
            path_length += euclidean_distance(current_position, previous_position)
            previous_position = current_position

        reach_q_flag = reached_goal(env, config)
    
    # append STOP as the final step
    stop_action_index = env.action_name_to_index("stop")
    optimal_actions.append(stop_action_index)

    print("[Polar Q] Optimal action sequence: %s"%optimal_actions)
    print("[Polar Q] Optimal action sequence length: %s"%len(optimal_actions))
    print("[Polar Q] Optimal action path length: %f"%path_length)
    
    print("="*20)

    return optimal_actions, path_length

def polar_q_planner_new(env, episode, config):
    rotate_resolution = int(config.get("rotate_resolution"))

    # reset the env
    obs = env.reset(episode=episode, plan_shortest_path=True)
    path_length = 0.0
    previous_position = env.get_agent_position()

    print("="*20)
    print('Episode: ')
    print("Goal position: %s"%(env.goal_position))
    print("Start position: %s"%(env.start_position))
    print("[Shortest path] Optimal action sequence: %s"%env.optimal_action_seq)
    print("[Shortest path] Optimal action sequence length: %s"%len(env.optimal_action_seq))
    
    reach_q_flag = reached_goal(env, config)
    optimal_actions = []
    
    while not reach_q_flag:
        cur_q, cur_optimal_actions = compute_polar_q_current_state(env, rotate_resolution)
        optimal_action_list = list(np.argwhere(cur_q == np.amax(cur_q)).squeeze(axis=1))

        # take one step
        action = optimal_action_list[0]
        obs, reward, done, info = env.step(action)
        optimal_actions.append(action)

        print(action)
        print("----------------------")

        # accumulate path length
        current_position = env.get_agent_position()
        path_length += euclidean_distance(current_position, previous_position)
        previous_position = current_position
            
        reach_q_flag = reached_goal(env, config)
    
    # append STOP as the final step
    stop_action_index = env.action_name_to_index("stop")
    optimal_actions.append(stop_action_index)

    print("[Polar Q] Optimal action sequence: %s"%optimal_actions)
    print("[Polar Q] Optimal action sequence length: %s"%len(optimal_actions))
    print("[Polar Q] Optimal action path length: %f"%path_length)
    
    print("="*20)

    return optimal_actions, path_length

def check_optimal_path_polar_q(env, episode, config):
    rotate_resolution = int(config.get("rotate_resolution"))

    # reset the env
    obs = env.reset(episode=episode, plan_shortest_path=True)
    shortest_path_length = 0.0
    previous_position = env.get_agent_position()


    print("="*20)
    print('Episode: ')
    print("Goal position: %s"%(env.goal_position))
    print("Start position: %s"%(env.start_position))
    print("[Shortest path] Optimal action sequence: %s"%env.optimal_action_seq)
    print("[Shortest path] Optimal action sequence length: %s"%len(env.optimal_action_seq))
    
    # act according to the shortest path, and compute its polar q at each state
    for i, action in enumerate(env.optimal_action_seq):
        cur_q, cur_optimal_actions = compute_polar_q_current_state(env, rotate_resolution)
        
        # max q predicted by polar q
        optimal_action_list = list(np.argwhere(cur_q == np.amax(cur_q)).squeeze(axis=1))
        print("-----------------------------")
        print("Step: %d"%(i+1))
        print("[Polar Q] Optimal action: %s"%(optimal_action_list))
        print("[Polar Q] Optimal action sequence: %s"%(cur_optimal_actions))
        print("[Shortest path] Optimal action: %d"%(action))
        print("-----------------------------")

        # take one step according to the shortest path
        obs, reward, done, info = env.step(action)
        # accumulate path length
        current_position = env.get_agent_position()
        shortest_path_length += euclidean_distance(current_position, previous_position)
        previous_position = current_position
    
    print("[Shortest path] Optimal action path length: %f"%shortest_path_length)

def test_polar_q_planner(config_file="imitation_learning_dqn.yaml"):
    env = MultiNavEnv(config_file=config_file)

    config = parse_config(os.path.join(config_path, config_file))

    episodes = load_behavior_dataset_meta(
                behavior_dataset_path=config.get("behavior_dataset_path"), 
                split_name="same_start_goal_val_mini")
    
    for i, episode in enumerate(episodes):
        _, path_length = polar_q_planner_new(env, episode, config)
        
        #if i >= 3:
        break

def test_check_optimal_path_polar_q(config_file="imitation_learning_dqn.yaml"):
    env = MultiNavEnv(config_file=config_file)

    config = parse_config(os.path.join(config_path, config_file))

    episodes = load_behavior_dataset_meta(
                behavior_dataset_path=config.get("behavior_dataset_path"), 
                split_name="same_start_goal_val_mini")
    
    for i, episode in enumerate(episodes):
        check_optimal_path_polar_q(env, episode, config)
        
        #if i >= 3:
        break

def compare_greedy_planner_polar_q(config_file="imitation_learning_dqn.yaml"):
    env = MultiNavEnv(config_file=config_file)

    config = parse_config(os.path.join(config_path, config_file))

    episodes = load_behavior_dataset_meta(
                behavior_dataset_path=config.get("behavior_dataset_path"), 
                split_name="same_start_goal_val_mini")
    
    for i, episode in enumerate(episodes):
        check_optimal_path_polar_q(env, episode, config)
        polar_q_planner(env, episode, config)
        
        if i >= 3:
            break

    
if __name__ == "__main__":
    # ====== first set seed =======
    set_seed_except_env_seed(seed=0)  
    # ========= test ==========  
    test_polar_q_planner()
    #test_check_optimal_path_polar_q()
    #compare_greedy_planner_polar_q()



