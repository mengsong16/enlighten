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
from enlighten.datasets.il_data_gen import load_behavior_dataset_meta, get_optimal_path

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

class PolarActionSpace:
    def __init__(self, env, rotate_resolution):
        self.rotate_resolution = int(rotate_resolution)
        assert int(360) % self.rotate_resolution == 0, "360 should be divisible by the rotation resolution %d"%(self.rotate_resolution)
    
        # get cartesian action indices
        self.cartesian_action_number = len(env.action_mapping)
        assert  self.cartesian_action_number == 4, "Can only convert to Polar action space when we have 4 actions in Cartesian action space: stop, move_forward, turn_left, turn_right" 
        self.cartesian_stop_action_index = env.action_name_to_index("stop")
        self.cartesian_forward_action_index = env.action_name_to_index("move_forward")
        self.cartesian_turn_left_action_index = env.action_name_to_index("turn_left")
        self.cartesian_turn_right_action_index = env.action_name_to_index("turn_right")
        
        # counter-clockwise: [0,360)
        self.rotate_degrees = [None] + list(range(0, 360, self.rotate_resolution))

        self.polar_action_number = len(self.rotate_degrees)
        
        # stop is the first action
        # move forward is the second action
        self.cartesian_action_mapping = [[self.cartesian_stop_action_index], [self.cartesian_forward_action_index]]

        # then rotate counter-clockwise step by step
        # to minimize the number of rotations:
        # first half: turn left, second half: turn right
        left_rotate_steps = int(180) // self.rotate_resolution
        for i in list(range(1, left_rotate_steps+1)):
            current_action_sequence = [self.cartesian_turn_left_action_index]*i + [self.cartesian_forward_action_index]
            self.cartesian_action_mapping.append(current_action_sequence)

        right_rotate_steps = int(360) // self.rotate_resolution - 1 - left_rotate_steps
        for i in list(range(right_rotate_steps, 0, -1)):
            current_action_sequence = [self.cartesian_turn_right_action_index]*i + [self.cartesian_forward_action_index]
            self.cartesian_action_mapping.append(current_action_sequence)

        # print("---------------------------")
        # for e in self.cartesian_action_mapping:
        #     print(e)
        #     print("---------------------------")
        # print(len(self.cartesian_action_mapping))

        # print(self.cartesian_actions_to_polar_action([0]))
        # print(self.cartesian_actions_to_polar_action([1]))
        # print(self.cartesian_actions_to_polar_action([2,2,1]))
        # print(self.cartesian_actions_to_polar_action([2,3]))

    # map any degree to [0,360)
    def map_degree_to_2pi(self, degrees):
        return int(degrees + 360) % int(360)
    
    def polar_action_to_cartesian_actions(self, polar_action_index):
        return self.cartesian_action_mapping[polar_action_index]
    
    def cartesian_actions_to_polar_action(self, cartesian_action_seq):
        return self.cartesian_action_mapping.index(cartesian_action_seq)
    
    def convert_angles_in_seq_to_2pi(self, cartesian_action_seq):
        # cartesian action sequence must end with stop or move_forward
        assert cartesian_action_seq[-1] == self.cartesian_stop_action_index or cartesian_action_seq[-1] == self.cartesian_forward_action_index

        # sequence length == 1
        if len(cartesian_action_seq) == 1:
            return cartesian_action_seq

        # sequence length >= 2    
        rotation_seq = cartesian_action_seq[:-1]

        # counter-clockwise
        rotation_degree = 0
        for rotate_action in rotation_seq:
            if rotate_action == self.cartesian_turn_left_action_index:
                rotation_degree += self.rotate_resolution
            elif rotate_action == self.cartesian_turn_right_action_index:
                rotation_degree -= self.rotate_resolution    
            else:
                print("Error: unknown rotation action index: %d"%(rotate_action))
                exit()
        
        rotation_degree = self.map_degree_to_2pi(rotation_degree)

        polar_index = self.rotate_degrees.index(rotation_degree)

        return self.polar_action_to_cartesian_actions(polar_index)

    def map_cartesian_action_seq_to_polar_seq(self, cartesian_action_seq):
        i = 0
        polar_seq = []
        sub_seq = []
        while i < len(cartesian_action_seq):
            if cartesian_action_seq[i] == self.cartesian_stop_action_index or cartesian_action_seq[i] == self.cartesian_forward_action_index:
                sub_seq.append(cartesian_action_seq[i])
                # make sure the rotation happened in this sub sequence is in the range of [0,360)
                new_sub_seq = self.convert_angles_in_seq_to_2pi(sub_seq)
                polar_action_index = self.cartesian_actions_to_polar_action(new_sub_seq)
                polar_seq.append(polar_action_index)
                sub_seq = []
            else:
                sub_seq.append(cartesian_action_seq[i])
            
            i += 1

        return polar_seq


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

# rotate_resolution: in degree
def compute_polar_q_current_state(env, polar_action_space):
    q = []
    current_state = env.get_agent_state()

    # q["stop"], always not the max q
    q.append(get_geodesic_distance_based_q_current_state(env))

    
    # q["move_forward"]
    # take one step forward
    obs, reward, done, info = env.step(polar_action_space.cartesian_forward_action_index)
    q.append(get_geodesic_distance_based_q_current_state(env))
    # get back to the original state (i.e. circle center)
    env.set_agent_state(
        new_position=current_state.position,
        new_rotation=current_state.rotation,
        is_initial=False,
        quaternion=True
    )

    # compute q at all angles rotate from 10 to 350 degrees
    rotate_num = polar_action_space.polar_action_number - 2
    circle_states = []
    for n in list(range(1, rotate_num+1)):
        # rotate counterclockwise (i.e. turn left) one more time, i.e. n times in total
        obs, reward, done, info = env.step(polar_action_space.cartesian_turn_left_action_index)
        circle_states.append(env.get_agent_state())
        # take one step forward
        obs, reward, done, info = env.step(polar_action_space.cartesian_forward_action_index)
        cur_q = get_geodesic_distance_based_q_current_state(env)
        q.append(cur_q)

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

    assert len(q) == polar_action_space.polar_action_number
    q = np.array(q, dtype="float")

    polar_optimal_action_list = list(np.argwhere(q == np.amax(q)).squeeze(axis=1))

    if len(polar_optimal_action_list) > 1:
        print("More than one polar optimal action has been found")
    
    polar_optimal_action = polar_optimal_action_list[0]
    cartesian_optimal_action_seq = polar_action_space.polar_action_to_cartesian_actions(polar_optimal_action)

    return q, polar_optimal_action, cartesian_optimal_action_seq

def reached_goal(env, config):
    distance_to_goal = env.get_current_distance()

    if distance_to_goal < config.get("success_distance"):
        return True
    else:
        return False

def euclidean_distance(position_a, position_b):
    return np.linalg.norm(position_b - position_a, ord=2)

def step_cartesian_action_seq(env, cartesian_action_seq):
    for action in cartesian_action_seq:
        obs, reward, done, info = env.step(action)

def step_one_polar_action(env, polar_action, polar_action_space):
    cartesian_action_seq = polar_action_space.polar_action_to_cartesian_actions(polar_action)
    step_cartesian_action_seq(env, cartesian_action_seq)

def polar_q_planner(env, polar_action_space, episode, config):
    # reset the env
    obs = env.reset(episode=episode, plan_shortest_path=True)
    polar_path_length = 0.0
    previous_position = env.get_agent_position()

    print("="*20)
    print("Goal position: %s"%(env.goal_position))
    print("Start position: %s"%(env.start_position))
    print("[Shortest path] Optimal cartesian action sequence: %s"%env.optimal_action_seq)
    print("[Shortest path] Optimal cartesian action sequence length: %d"%len(env.optimal_action_seq))

    shortest_path_planner_polar_seq = polar_action_space.map_cartesian_action_seq_to_polar_seq(env.optimal_action_seq)
    print("[Shortest path] Optimal polar action sequence: %s"%shortest_path_planner_polar_seq)
    print("[Shortest path] Optimal polar action sequence length: %d"%len(shortest_path_planner_polar_seq))

    
    reach_q_flag = reached_goal(env, config)
    polar_optimal_actions = []
    cartesian_optimal_actions = []
    
    while not reach_q_flag:
        q, polar_optimal_action, cartesian_optimal_action_seq = compute_polar_q_current_state(env, polar_action_space)
        
        # take one polar action step
        step_cartesian_action_seq(env, cartesian_optimal_action_seq)

        polar_optimal_actions.append(polar_optimal_action)
        cartesian_optimal_actions.extend(cartesian_optimal_action_seq)

        # print("-----------------------------")
        # print("[Polar Q] Optimal polar action: %d"%polar_optimal_action)
        # print("[Polar Q] Optimal cartesian action sequence: %s"%(cartesian_optimal_action_seq))
        # print("-----------------------------")

        # accumulate path length
        current_position = env.get_agent_position()
        polar_path_length += euclidean_distance(current_position, previous_position)
        previous_position = current_position
            
        reach_q_flag = reached_goal(env, config)
    
    # append STOP as the final step
    polar_optimal_actions.append(0)
    cartesian_optimal_actions.append(polar_action_space.cartesian_stop_action_index)

    print("[Polar Q] Optimal cartesian action sequence: %s"%cartesian_optimal_actions)
    print("[Polar Q] Optimal cartesian action sequence length: %d"%len(cartesian_optimal_actions))
    print("[Polar Q] Optimal polar action sequence: %s"%polar_optimal_actions)
    print("[Polar Q] Optimal polar action sequence length: %d"%len(polar_optimal_actions))
    print("[Polar Q] Optimal path length: %f"%polar_path_length)
    
    print("="*20)

    return polar_optimal_actions, cartesian_optimal_actions, polar_path_length

def check_optimal_path_polar_q(env, polar_action_space, episode, config):
    # reset the env
    obs = env.reset(episode=episode, plan_shortest_path=True)
    shortest_path_length = 0.0
    previous_position = env.get_agent_position()


    print("="*20)
    print("Goal position: %s"%(env.goal_position))
    print("Start position: %s"%(env.start_position))
    print("[Shortest path] Optimal cartesian action sequence: %s"%env.optimal_action_seq)
    print("[Shortest path] Optimal cartesian action sequence length: %d"%len(env.optimal_action_seq))

    shortest_path_planner_polar_seq = polar_action_space.map_cartesian_action_seq_to_polar_seq(env.optimal_action_seq)
    print("[Shortest path] Optimal polar action sequence: %s"%shortest_path_planner_polar_seq)
    print("[Shortest path] Optimal polar action sequence length: %d"%len(shortest_path_planner_polar_seq))

    
    # act according to the shortest path, and compute its polar q at each state
    for i, action in enumerate(env.optimal_action_seq):
        q, polar_optimal_action, cartesian_optimal_action_seq = compute_polar_q_current_state(env, polar_action_space)
        
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

def test_polar_q_planner(config_file="imitation_learning_sqn.yaml"):
    env = MultiNavEnv(config_file=config_file)

    config = parse_config(os.path.join(config_path, config_file))

    episodes = load_behavior_dataset_meta(
                behavior_dataset_path=config.get("behavior_dataset_path"), 
                split_name="same_start_goal_val_mini")
    
    rotate_resolution = int(config.get("rotate_resolution"))

    polar_action_space = PolarActionSpace(env, rotate_resolution)
    
    for i, episode in enumerate(episodes):
        polar_optimal_actions, cartesian_optimal_actions, polar_path_length = polar_q_planner(env, polar_action_space, episode, config)
        
        #if i >= 3:
        break

def test_check_optimal_path_polar_q(config_file="imitation_learning_sqn.yaml"):
    env = MultiNavEnv(config_file=config_file)

    config = parse_config(os.path.join(config_path, config_file))

    episodes = load_behavior_dataset_meta(
                behavior_dataset_path=config.get("behavior_dataset_path"), 
                split_name="same_start_goal_val_mini")
    
    rotate_resolution = int(config.get("rotate_resolution"))

    polar_action_space = PolarActionSpace(env, rotate_resolution)
    
    for i, episode in enumerate(episodes):
        check_optimal_path_polar_q(env, polar_action_space, episode, config)
        
        #if i >= 3:
        break

def compare_greedy_planner_polar_q_planner(config_file="imitation_learning_sqn.yaml"):
    env = MultiNavEnv(config_file=config_file)

    config = parse_config(os.path.join(config_path, config_file))

    episodes = load_behavior_dataset_meta(
                behavior_dataset_path=config.get("behavior_dataset_path"), 
                split_name="same_start_goal_val_mini")
    
    rotate_resolution = int(config.get("rotate_resolution"))

    polar_action_space = PolarActionSpace(env, rotate_resolution)
    
    for i, episode in enumerate(episodes):
        check_optimal_path_polar_q(env, polar_action_space, episode, config)
        polar_q_planner(env, polar_action_space, episode, config)
        
        #if i >= 3:
        break

def test_polar_action_space(config_file="imitation_learning_sqn.yaml"):
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
        current_q_values.append(get_geodesic_distance_based_q_current_state(env))

        # "move_forward" q
        # take one step forward
        obs, reward, done, info = env.step(cartesian_forward_action_index)
        print("Executed actions: %s"%([cartesian_forward_action_index]))
        # get current geodesic distance to goal as q
        current_q_values.append(get_geodesic_distance_based_q_current_state(env))
            
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
            step_cartesian_action_seq(env, execute_action_seq)

            print("Executed actions: %s"%([action]+execute_action_seq))

            # get current geodesic distance to goal as q
            q = get_geodesic_distance_based_q_current_state(env)
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

def test_check_optimal_path_geodesic_q(config_file="imitation_learning_sqn.yaml"):
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
    #test_polar_q_planner()
    #test_check_optimal_path_polar_q()
    #compare_greedy_planner_polar_q_planner()
    #test_polar_action_space()
    test_check_optimal_path_geodesic_q()



