# behavior_dataset_path: "/dataset/behavior_dataset_gibson"
import habitat_sim
import os
import numpy as np

import pickle
from enlighten.utils.geometry_utils import quaternion_rotate_vector, cartesian_to_polar

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

def update_episode_data(env, 
    obs, 
    reward, 
    done, 
    goal_dimension, 
    goal_coord_system,
    observations,
    actions,
    rel_goals,
    distance_to_goals,
    goal_positions,
    state_positions,
    abs_goals,
    dones,
    rewards,
    action=None,
    qs=None,
    q=None):

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

    #state_position = np.array(env.agent.get_state().position, dtype="float32")
    state_position = np.array(obs["state_sensor"], dtype="float32")
    state_positions.append(state_position)  # (3,)

    if action is not None:
        actions.append(action)

    dones.append(done)  # bool

    rewards.append(reward) # float

    if q is not None and qs is not None:
        qs.append(q)

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

def get_first_forward_action_sequence(cartesian_action_seq,
    cartesian_forward_action_index):
    i = 0 
    sub_seq = []
    while i < len(cartesian_action_seq):
        if cartesian_action_seq[i] == cartesian_forward_action_index:
            sub_seq.append(cartesian_action_seq[i])
            break
        else:
            sub_seq.append(cartesian_action_seq[i])
            
        i += 1
    
    return sub_seq