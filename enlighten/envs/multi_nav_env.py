from enlighten.envs.nav_env import NavEnv
from enlighten.utils.config_utils import parse_config

import os
import numpy as np
import gym
from gym import spaces
import copy

import habitat_sim
from habitat_sim.gfx import LightInfo, LightPositionModel, DEFAULT_LIGHTING_KEY, NO_LIGHT_KEY
from habitat_sim.utils.common import quat_from_angle_axis

from enlighten.utils.image_utils import try_cv2_import
cv2 = try_cv2_import()

from enlighten.envs import HabitatSensor, Dictionary_Observations, HabitatSimRGBSensor, HabitatSimDepthSensor, HabitatSimSemanticSensor, ImageGoal, PointGoal
from enlighten.envs.sensor import StateSensor
from enlighten.utils.path import *
from enlighten.tasks.measures import Measurements
from enlighten.datasets.pointnav_dataset import PointNavDatasetV1
from enlighten.utils.geometry_utils import get_rotation_quat, euclidean_distance, quaternion_rotate_vector, cartesian_to_polar, quaternion_from_coeff 
from enlighten.datasets.pointnav_dataset import NavigationEpisode, NavigationGoal, ShortestPathPoint
from enlighten.datasets.dataset import EpisodeIterator
from enlighten.datasets.common import PolarActionSpace
from enlighten.datasets.common import update_episode_data
from enlighten.datasets.common import load_behavior_dataset_meta, get_first_effective_action_sequence, get_first_forward_action_sequence

# across scene environments
class MultiNavEnv(NavEnv):
    
    # config_file could be a string or a parsed config dict
    def __init__(self, config_file="imitation_learning_dt.yaml"):
        # get config
        if isinstance(config_file, str):
            config_file = os.path.join(config_path, config_file)
        self.config = parse_config(config_file)

        self.NONE_ACTION = -2
    
        # create simulator configuration and sensors
        self.create_sim_config()

        # create simulator
        self.sim = habitat_sim.Simulator(self.sim_config) 

           
        # create gym observation space
        self.observation_space = self.create_gym_observation_space(self.sensors)
        
        # create goal sensors (must be created after the main observation space is created)
        self.create_goal_sensor()

        # create agent and set agent's initial state to a navigable random position and z rotation
        self.agent = self.sim.initialize_agent(agent_id=self.sim._default_agent_id)
       
        # create gym action space: Cartesian
        self.action_space = self.create_gym_action_space()

        # initialize viewer
        self.viewer = None

        # initialize start and goal positions
        self.init_start_goal()

        # set measurements
        measure_ids = list(self.config.get("measurements"))
        self.measurements = Measurements(measure_ids=measure_ids, env=self, config=self.config)
        
        # load goal radius
        self.goal_radius = float(self.config.get("success_distance"))

        # seed action and observation space
        self.seed_action_obs_space()

        # automatically replay episode if not None
        self.episode_iterator = None 
        self.current_episode = None

        print("======> Number of cartesian actions in the environment: %s"%(str(self.action_space.n)))

        # polar action space
        rotate_resolution = int(self.config.get("rotate_resolution"))
        self.polar_action_space = PolarActionSpace(self, rotate_resolution)
        print("======> Number of polar actions in the environment: %d"%(self.polar_action_space.polar_action_number))
        
    def set_episode_dataset(self, episodes):
        self.episode_iterator = EpisodeIterator(episodes=episodes, seed=int(self.config.get('seed')))
        print("===> Set episode iterator over training set, shuffling the episodes")
        self.number_of_episodes = len(episodes)

    def create_sim_cfg(self, scene_id):
        # simulator configuration
        sim_config = habitat_sim.SimulatorConfiguration()
        
        # set scene
        sim_config.scene_id = scene_id
        self.current_scene = sim_config.scene_id

        # set dataset path
        # if not set, the value is "default"
        #sim_config.scene_dataset_config_file = self.config.get('dataset_path')

        # set random seed for the Simulator and Pathfinder
        sim_config.random_seed = int(self.config.get('seed'))

        # sim_config.gpu_device_id = 0 by default use gpu 0 to render

        # enable physics
        sim_config.enable_physics = True

        # create a new lighting for dark mode
        self.dark_mode = self.config.get('dark_mode')
        if self.dark_mode:
            # enable scene lighting change
            sim_config.override_scene_light_defaults = True
            # change global lighting
            sim_config.scene_light_setup = "current_scene_lighting"
        else:
            sim_config.override_scene_light_defaults = False
            # no lights
            print("Daylight mode is on: global scene light setup is: %s"%(sim_config.scene_light_setup))
            #print(sim_config.scene_light_setup)
            #print("****************************")
        
        return sim_config

    def create_sim_config(self):
        # create sim cfg
        sim_config = self.create_sim_cfg(scene_id=self.config.get('scene_id'))  

        # sensors and sensor specifications
        sensor_specs, self.sensors = self.create_sensors_and_sensor_specs()
        
        # agent configuration
        self.agent_cfg = habitat_sim.agent.AgentConfiguration()
        self.agent_cfg.action_space = self.create_sim_action_space()
        self.agent_cfg.sensor_specifications = sensor_specs

        self.sim_config = habitat_sim.Configuration(sim_config, [self.agent_cfg])


    def seed_action_obs_space(self):
        """Set the random seed of the environment."""
        self.action_space.seed(int(self.config.get('seed')))
        self.observation_space.seed(int(self.config.get('seed')))

    def get_episode_start_position(self, episode):
        return np.array(episode.start_position, dtype=np.float32)

    def get_episode_start_rotation(self, episode):
        return quaternion_from_coeff(episode.start_rotation)
    
    def get_episode_goal_position(self, episode):
        return np.array(episode.goals[0].position, dtype=np.float32)

    # reconfig scene, start, goal according to the given episode
    def reconfigure(self, episode):
        # reset scene id
        episode_scene = episode.scene_id
        
        # need to change scene
        if self.current_scene != episode_scene:
            # need to close sim before reconfigure so that memory can be released
            self.sim.close()
            self.sim_config = habitat_sim.Configuration(self.create_sim_cfg(episode_scene), [self.agent_cfg])
            self.sim.reconfigure(self.sim_config)
            # must recreate agent if simulator is reconfiged, otherwise agent will not move when env.step
            self.agent = self.sim.initialize_agent(agent_id=self.sim._default_agent_id)
        
        # reset agent state and goal according to episode
        # is_initial must be true, otherwise will be set to the previous start position and rotation
        self.set_start_goal(new_goal_position=self.get_episode_goal_position(episode), 
            new_start_position=self.get_episode_start_position(episode), 
            new_start_rotation=self.get_episode_start_rotation(episode), 
            is_initial=True)

    def init_start_goal(self):
        goal_position = np.array(self.config.get('goal_position'), dtype="float32")
        start_position = np.array(self.config.get('agent_initial_position'), dtype="float32")
        start_rotation = np.array(self.config.get('agent_initial_rotation'), dtype="float32")
        start_rotation = get_rotation_quat(start_rotation)

        self.set_start_goal(goal_position, start_position, start_rotation, is_initial=True)

    # new_start_rotation: quaternion
    # new_start_position: 3D coord
    # new_goal_position: 3D coord
    def set_start_goal(self, new_goal_position, new_start_position, new_start_rotation, is_initial):
        self.goal_position = new_goal_position
        self.start_position = new_start_position
        self.start_rotation = new_start_rotation
       
        self.set_agent_state(new_position=self.start_position, 
            new_rotation=self.start_rotation, is_initial=is_initial, quaternion=True)

    def set_agent_to_initial_state(self):
        self.set_agent_state(new_position=self.start_position, 
            new_rotation=self.start_rotation, is_initial=True, quaternion=True)
    
    # check validity of the generated optimal action sequence
    def check_optimal_action_sequence(self):
        assert len(self.optimal_action_seq) > 0, "Error: optimal action sequence must have at least one element"
        if self.optimal_action_seq[-1] != self.action_name_to_index("stop"):
            print("Error: the last action in the optimal action sequence must be STOP, but %d now, appending STOP."%(self.optimal_action_seq[-1]))
            self.optimal_action_seq.append(self.action_name_to_index("stop"))
       
    # plan shortest path
    # must be called after agent has been set to the start location, and goal has been reset
    # invalid optimal_action_sequence will be []
    def plan_shortest_path(self):
       
        try:
            # find optimal action sequence from current state to goal state
            self.optimal_action_seq = self.follower.find_path(goal_pos=self.goal_position)
            # append STOP if not appended
            self.check_optimal_action_sequence()
        except habitat_sim.errors.GreedyFollowerError as e:
            print("Error: optimal path NOT found! set optimal action sequence to []")
            self.optimal_action_seq = []
        

        # update optimal action sequence iterator
        self.optimal_action_iter = iter(self.optimal_action_seq)


    def reset(self, episode=None, plan_shortest_path=False):
        # reset scene, agent start and goal
        if episode is None:
            if self.episode_iterator is None:
                self.set_agent_to_initial_state()
            else:
                # iterate to next episode
                self.current_episode = next(self.episode_iterator)
                self.reconfigure(self.current_episode)
        else:    
            self.reconfigure(episode)

        # reset simulator and get the initial observation
        sim_obs = self.sim.reset()
        obs = self.sensor_suite.get_observations(sim_obs=sim_obs, env=self)

        # concat obs with goal
        if self.config.get("goal_conditioned"):  
            obs = self.add_goal_obs(obs)

        # has the agent issued STOP action
        self.is_stop_called = False

        # reset measurements
        self.measurements.reset_measures(measurements=self.measurements, is_stop_called=self.is_stop_called)

        # if navmesh loaded, then pathfinder is loaded, then seed it
        if self.sim.pathfinder.is_loaded:
            self.sim.pathfinder.seed(int(self.config.get("seed")))
            #print("Path finder loaded and seeded.")

        # create optimal action sequence and its iterator
        self.optimal_action_seq = []    
        self.optimal_action_iter = iter(self.optimal_action_seq)

        # create shortest path planner
        self.create_shortest_path_follower()

        # plan the shortest path using path follower at s0
        if plan_shortest_path:
            self.plan_shortest_path()

        return obs
    
    def get_next_optimal_action(self):
        return next(self.optimal_action_iter, self.NONE_ACTION)

    def get_optimal_action_sequence_length(self):
        return len(self.optimal_action_seq)

    # default observation is zero numpy array
    def get_default_observation(self):
        default_obs = {}
        spaces = self.get_combined_goal_obs_space().spaces
        for key, space in spaces.items():
            default_obs[key] = np.zeros(shape=space.shape, dtype=space.dtype)
        
        return default_obs


    # action is an integer in cartesian action space
    def step(self, action):
        if action is self.NONE_ACTION:
            return self.get_default_observation(), None, True, {}

        # action index to action name
        action_name = self.action_mapping[action]
        
        if action_name == "stop":
            self.is_stop_called = True
        else:
            self.is_stop_called = False    
            
        # get current observation
        sim_obs = self.sim.step(action_name)
        obs = self.sensor_suite.get_observations(sim_obs=sim_obs, env=self)

        # goal conditioned
        if self.config.get("goal_conditioned"):
            obs = self.add_goal_obs(obs)

        # update all measurements
        self.measurements.update_measures(
            measurements=self.measurements,
            sim_obs=sim_obs,
            is_stop_called=self.is_stop_called
        )

        reward = self.get_reward()
        
        done = self.is_done()

        # add metrics (e.g, success) to info for tensorboard and evaluation stats
        info = {"success": int(self.is_success()), "spl": self.get_spl()}

        return obs, reward, done, info 
    
    def get_reward(self):
        reward = self.measurements.measures["point_goal_reward"].get_metric()

        return reward   

    # agent location must already been reset
    def create_shortest_path_follower(self):
        # self.follower = self.sim.make_greedy_follower(
        #         agent_id=0,
        #         goal_radius=self.goal_radius,
        #         stop_key=self.action_name_to_index("stop"),  # 0
        #         forward_key=self.action_name_to_index("move_forward"), # 1
        #         left_key=self.action_name_to_index("turn_left"), # 2 
        #         right_key=self.action_name_to_index("turn_right"),  #3
        #     ) 

        # could map to action index or name string, no mapping action will be None
        assert self.sim.pathfinder.is_loaded, "Error: try to create path follower before creating path finder"
        self.follower = habitat_sim.GreedyGeodesicFollower(
            pathfinder=self.sim.pathfinder,
            agent=self.agent,
            goal_radius=self.goal_radius,
            stop_key=self.action_name_to_index("stop"),
            forward_key=self.action_name_to_index("move_forward"),
            left_key=self.action_name_to_index("turn_left"),
            right_key=self.action_name_to_index("turn_right"))
        
        self.follower.reset()

        #print("Path follower created and reset.")      
    
    def set_scene_id_in_config(self, new_scene):
        self.sim_config.sim_cfg.scene_id = new_scene
    
    # ============== The following are for polar q =====================
    def reached_goal(self):
        distance_to_goal = self.get_current_distance()

        if distance_to_goal < self.config.get("success_distance"):
            return True
        else:
            return False
    
    def get_geodesic_distance_based_q_current_state(self):
        # geodesic distance from current state to goal state
        d = self.get_current_distance()
        q = -d
        return q


    # plan the optimal action sequence path from the current state
    def get_optimal_path(self):
        try:
            # No need to reset or recreate the path follower before path planning
            # once create the path follower attaching to an agent 
            # it will always update itself to the current state when find_path is called
            #env.create_shortest_path_follower()
            #env.follower.reset()
            optimal_action_seq = self.follower.find_path(goal_pos=self.goal_position)
            
            assert len(optimal_action_seq) > 0, "Error: optimal action sequence must have at least one element"
            # append STOP if not appended
            if optimal_action_seq[-1] != self.action_name_to_index("stop"):
                print("Error: the last action in the optimal action sequence must be STOP, but %d now, appending STOP."%(optimal_action_seq[-1]))
                optimal_action_seq.append(self.action_name_to_index("stop"))
        
        except habitat_sim.errors.GreedyFollowerError as e:
            print("Error: optimal path NOT found! set optimal action sequence to []")
            optimal_action_seq = []

        return optimal_action_seq

    # wrong
    def compute_cartesian_q_current_state(self):
        # act according to the shortest path, and compute its cartesian q at each state
        cartesian_action_number = len(self.action_mapping)
        cartesian_stop_action_index = self.action_name_to_index("stop")
        cartesian_forward_action_index = self.action_name_to_index("move_forward")
        cartesian_turn_left_action_index = self.action_name_to_index("turn_left")
        cartesian_turn_right_action_index = self.action_name_to_index("turn_right")
            
        q_values = []
        current_state = self.get_agent_state()

        # q["stop"]
        q_values.append(self.get_geodesic_distance_based_q_current_state())
        # print("Executed actions: None")
        
        # q["move_forward"]
        # take one step forward
        obs, reward, done, info = self.step(cartesian_forward_action_index)
        #print("Executed actions: %s"%([self.polar_action_space.cartesian_forward_action_index]))
        
        q_values.append(self.get_geodesic_distance_based_q_current_state())
        # get back to the original state (i.e. circle center)
        self.set_agent_state(
            new_position=current_state.position,
            new_rotation=current_state.rotation,
            is_initial=False,
            quaternion=True
        )

        # "turn_left" or "turn_right" q
        for action in [cartesian_turn_left_action_index, cartesian_turn_right_action_index]:
            # take one cartesian step along current direction
            obs, reward, done, info = self.step(action)
            
            # plan the shortest path from the current state to see where move_forward or stop happen
            current_optimal_action_seq = self.get_optimal_path()
            # step the environment until move_forward or stop happen
            execute_action_seq = get_first_effective_action_sequence(current_optimal_action_seq,
                cartesian_stop_action_index,
                cartesian_forward_action_index)
            
            self.step_cartesian_action_seq(execute_action_seq)

            #print("Executed actions: %s"%([action]+execute_action_seq))

            # get current geodesic distance to goal as q
            q_values.append(self.get_geodesic_distance_based_q_current_state())
            
            # get back to the original state (i.e. circle center)
            self.set_agent_state(
                new_position=current_state.position,
                new_rotation=current_state.rotation,
                is_initial=False,
                quaternion=True
            )

        assert len(q_values) == cartesian_action_number
        q_values = np.array(q_values, dtype="float")

        # plan max q action
        cartesian_optimal_action_list = list(np.argwhere(q_values == np.amax(q_values)).squeeze(axis=1))

        if len(cartesian_optimal_action_list) > 1:
            print("-------------------------------------------------")
            print("More than one cartesian optimal actions have been found: %s"%(str(cartesian_optimal_action_list)))
            for aa in cartesian_optimal_action_list:
                if aa != 0: # find the first action which is not "stop"
                    # We now have the number of actions needs, do we pick the one with the shortest action sequence?
                    # To keep it consistent with polar action q planner, we pick the first action not STOP 
                    cartesian_optimal_action = aa
                    print("Choose action %d"%(cartesian_optimal_action))
                    break
            print("-------------------------------------------------")
        else:
            cartesian_optimal_action = cartesian_optimal_action_list[0]
        
        return q_values, cartesian_optimal_action, cartesian_optimal_action_list

    # rotate_resolution: in degree
    def compute_polar_q_current_state(self):
        q = []
        current_state = self.get_agent_state()

        # q["stop"]
        q.append(self.get_geodesic_distance_based_q_current_state())

        
        # q["move_forward"]
        # take one step forward
        obs, reward, done, info = self.step(self.polar_action_space.cartesian_forward_action_index)
        q.append(self.get_geodesic_distance_based_q_current_state())
        # get back to the original state (i.e. circle center)
        self.set_agent_state(
            new_position=current_state.position,
            new_rotation=current_state.rotation,
            is_initial=False,
            quaternion=True
        )

        # compute q at all angles rotate from 10 to 350 degrees
        rotate_num = self.polar_action_space.polar_action_number - 2
        circle_states = []
        for n in list(range(1, rotate_num+1)):
            # rotate counterclockwise (i.e. turn left) one more time, i.e. n times in total
            obs, reward, done, info = self.step(self.polar_action_space.cartesian_turn_left_action_index)
            circle_states.append(self.get_agent_state())
            # take one step forward
            obs, reward, done, info = self.step(self.polar_action_space.cartesian_forward_action_index)
            cur_q = self.get_geodesic_distance_based_q_current_state()
            q.append(cur_q)

            # get back to the last circle state
            self.set_agent_state(
                new_position=circle_states[-1].position,
                new_rotation=circle_states[-1].rotation,
                is_initial=False,
                quaternion=True
            )
        
        # get back to the original state (i.e. circle center)
        self.set_agent_state(
            new_position=current_state.position,
            new_rotation=current_state.rotation,
            is_initial=False,
            quaternion=True
        )

        assert len(q) == self.polar_action_space.polar_action_number
        q = np.array(q, dtype="float")

        # plan max q action
        polar_optimal_action_list = list(np.argwhere(q == np.amax(q)).squeeze(axis=1))

        # this happened when the optimal direction is exactly opposite to the agent's current orientation
        if len(polar_optimal_action_list) > 1:
            print("-------------------------------------------------")
            print("More than one polar optimal actions have been found: %s"%(str(polar_optimal_action_list)))
            for aa in polar_optimal_action_list:
                if aa != 0: # find the first action which is not "stop"
                    polar_optimal_action = aa
                    print("Choose action %d"%(polar_optimal_action))
                    break
            print("-------------------------------------------------")
        else:
            assert polar_optimal_action_list[0] != 0, "Optimal action should not be STOP"
            polar_optimal_action = polar_optimal_action_list[0]
        
        cartesian_optimal_action_seq = self.polar_action_space.polar_action_to_cartesian_actions(polar_optimal_action)

        return q, polar_optimal_action, cartesian_optimal_action_seq

    def step_cartesian_action_seq(self, cartesian_action_seq):
        for action in cartesian_action_seq:
            obs, reward, done, info = self.step(action)
        
        return obs, reward, done, info
    
    def step_one_polar_action(self, polar_action):
        cartesian_action_seq = self.polar_action_space.polar_action_to_cartesian_actions(polar_action)
        return self.step_cartesian_action_seq(cartesian_action_seq)
    
    def polar_q_planner(self, episode):
        # reset the env
        obs = self.reset(episode=episode, plan_shortest_path=True)
        polar_path_length = 0.0
        previous_position = self.get_agent_position()

        print("="*20)
        print("Goal position: %s"%(self.goal_position))
        print("Start position: %s"%(self.start_position))
        print("[Shortest path] Optimal cartesian action sequence: %s"%self.optimal_action_seq)
        print("[Shortest path] Optimal cartesian action sequence length: %d"%len(self.optimal_action_seq))

        shortest_path_planner_polar_seq = self.polar_action_space.map_cartesian_action_seq_to_polar_seq(self.optimal_action_seq)
        print("[Shortest path] Optimal polar action sequence: %s"%shortest_path_planner_polar_seq)
        print("[Shortest path] Optimal polar action sequence length: %d"%len(shortest_path_planner_polar_seq))

        
        reach_q_flag = self.reached_goal()
        polar_optimal_actions = []
        cartesian_optimal_actions = []
        
        while not reach_q_flag:
            q, polar_optimal_action, cartesian_optimal_action_seq = self.compute_polar_q_current_state()
            
            # take one polar action step
            self.step_cartesian_action_seq(cartesian_optimal_action_seq)

            polar_optimal_actions.append(polar_optimal_action)
            cartesian_optimal_actions.extend(cartesian_optimal_action_seq)

            # print("-----------------------------")
            # print("[Polar Q] Optimal polar action: %d"%polar_optimal_action)
            # print("[Polar Q] Optimal cartesian action sequence: %s"%(cartesian_optimal_action_seq))
            # print("-----------------------------")

            # accumulate path length
            current_position = self.get_agent_position()
            polar_path_length += euclidean_distance(current_position, previous_position)
            previous_position = current_position
                
            reach_q_flag = self.reached_goal()
        
        # append STOP as the final step
        polar_optimal_actions.append(0)
        cartesian_optimal_actions.append(self.polar_action_space.cartesian_stop_action_index)

        print("[Polar Q] Optimal cartesian action sequence: %s"%cartesian_optimal_actions)
        print("[Polar Q] Optimal cartesian action sequence length: %d"%len(cartesian_optimal_actions))
        print("[Polar Q] Optimal polar action sequence: %s"%polar_optimal_actions)
        print("[Polar Q] Optimal polar action sequence length: %d"%len(polar_optimal_actions))
        print("[Polar Q] Optimal path length: %f"%polar_path_length)
        
        print("="*20)

        return polar_optimal_actions, cartesian_optimal_actions, polar_path_length
    
    def generate_one_episode_with_polar_q(self, episode):
        goal_dimension = int(self.config.get("goal_dimension"))
        goal_coord_system = self.config.get("goal_coord_system")

        observations = []
        actions = [] # polar action sequence, not cartesian action sequence
        rel_goals = []
        distance_to_goals = []
        goal_positions = []
        state_positions = []
        abs_goals = []
        dones = []
        rewards = []
        qs = []
        traj = {}
        
        # reset the env
        obs = self.reset(episode=episode, plan_shortest_path=False)
        # add (s0, g0, d0, r0)
        # d0=False, r0=0, q=None
        update_episode_data(env=self,
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
            qs=qs,
            q=None)

        
        reach_q_flag = self.reached_goal()

        while not reach_q_flag:
            q, polar_optimal_action, cartesian_optimal_action_seq = self.compute_polar_q_current_state()

            # take one polar action step
            obs, reward, done, info = self.step_cartesian_action_seq(cartesian_optimal_action_seq)

            # add (s_i, a_{i-1}, g_i, d_i, r_i, q_{i-1})
            update_episode_data(env=self,
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
                action=polar_optimal_action,
                qs=qs,
                q=q)

            reach_q_flag = self.reached_goal()
        
        assert actions[-1] != 0, "The original planned optimal polar action sequence should not end with STOP."
        # print("========================")
        # print(actions)
        
        # compute q when we have already reached the goal
        # align current agent position to goal position
        #q, polar_optimal_action, cartesian_optimal_action_seq = self.compute_polar_q_current_state()
        q = -1.0 * float(self.config.get("forward_resolution")) * np.ones(self.polar_action_space.polar_action_number, dtype="float")
        q[0] = 0.0
        final_polar_optimal_action_list = list(np.argwhere(q == np.amax(q)).squeeze(axis=1))
        assert len(final_polar_optimal_action_list)==1 and final_polar_optimal_action_list[0]==0, "STOP should be the optimal action when we have already reached the goal."
        
        # take one normal env step = STOP
        obs, reward, done, info = self.step(0)
        assert done==True and self.is_success(), "Generated episode did not succeed"
        
        # append the first polar action STOP as the final step
        # add (s_i, a_{i-1}=0, g_i, d_i, r_i, q_{i-1})
        update_episode_data(env=self,
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
            action=0,
            qs=qs,
            q=q)

        # append the second polar action STOP (besides the one at the end of the optimal action sequence)
        actions.append(0)
        # append the second q (same as the previous one) to qs
        qs.append(copy.deepcopy(q))

        # print("========================")
        # print(len(observations)) # n+1
        # print(len(rel_goals)) # n+1
        # print(len(distance_to_goals)) # n+1
        # print(len(goal_positions)) # n+1
        # print(len(state_positions)) # n+1
        # print(len(abs_goals)) # n+1
        # print(len(dones)) # n+1
        # print(len(rewards)) # n+1
        # print(len(qs)) # n+1
        # print(len(actions)) # n+1
        # print("========================")
        # print(actions)
        # print("========================")
        # print(qs)
        # print("========================")
        

        traj["observations"] = observations
        traj["actions"] = actions
        traj["rel_goals"] = rel_goals
        traj["distance_to_goals"] = distance_to_goals
        traj["goal_positions"] = goal_positions
        traj["state_positions"] = state_positions
        traj["abs_goals"] = abs_goals
        traj["dones"] = dones
        traj["rewards"] = rewards
        traj["qs"] = qs
                    
        return traj, actions[:-1]
    
    # action_sequence: [turn_left, turn_left, ..., move_forward]
    def extract_qs_on_circles(self, qs_on_circle, action_sequence, 
        cartesian_turn_left_action_index, cartesian_turn_right_action_index):
        
        rotation_action_sequence = action_sequence[:-1]
        # only one action: move_forward
        if not rotation_action_sequence: # no rotation
            q_extract = np.array([copy.copy(qs_on_circle[1])], dtype="float")
            
        else:
            rotation_action_sequence_len = len(rotation_action_sequence)
            #print(rotation_action_sequence_len)
            # print(len(action_sequence))
            # print("-------------------------")
            # turn left
            if rotation_action_sequence[0] == cartesian_turn_left_action_index:
                #print("turn_left")
                q_extract = np.array(copy.copy(qs_on_circle[2:2+rotation_action_sequence_len]), dtype="float")
            # turn right
            elif rotation_action_sequence[0] == cartesian_turn_right_action_index:
                #print("turn_right")
                q_extract = np.array(copy.copy(qs_on_circle[-1:-1-rotation_action_sequence_len:-1]), dtype="float")
            
            # add move_forward to q
            q_extract = np.insert(q_extract, 0, qs_on_circle[1])
            
        #print(q_extract)
        #print("-------------------------")
        #print(q_extract.shape)
        assert q_extract.shape[0] == len(action_sequence)
        assert q_extract[-1] == np.amax(qs_on_circle), "max q on the circle %f is not the final extracted q %f"%(np.amax(qs_on_circle), q_extract[-1]) 

        #print(q_extract)
        #exit()

        return q_extract

    def generate_one_episode_cartesian_qs(self, episode):
        # action space mapping
        cartesian_stop_action_index = self.action_name_to_index("stop")
        cartesian_forward_action_index = self.action_name_to_index("move_forward")
        cartesian_turn_left_action_index = self.action_name_to_index("turn_left")
        cartesian_turn_right_action_index = self.action_name_to_index("turn_right")

        # how many shares on the circle
        share_num = int(360) // self.polar_action_space.rotate_resolution

        # penalty for each rotation action
        q_penality_per_action = -1.0e-4
        assert q_penality_per_action < 0, "q penality should be less than 0"
       
        # reset the env
        obs = self.reset(episode=episode, plan_shortest_path=False)
        
        real_actions = []
        reach_q_flag = self.reached_goal()


        while not reach_q_flag:
            # now at the circle center
            # get current q
            cur_q = self.get_geodesic_distance_based_q_current_state()

            # plan one polar step
            qs_on_circle, _, next_action_seq = self.compute_polar_q_current_state()
            
            # verify the sequence's optimality
            next_action_seq[-1] == cartesian_forward_action_index
            rotation_seq = next_action_seq[:-1]
            if rotation_seq: # not empty
                assert rotation_seq.count(rotation_seq[0]) == len(rotation_seq), "The sequence should rotate in the same direction"
            
            
            # extract corresponding qs from the circle
            q_extract = self.extract_qs_on_circles(qs_on_circle, next_action_seq, 
                cartesian_turn_left_action_index, cartesian_turn_right_action_index)
            
            # after rotate and move_forward
            final_q = q_extract[-1]

            # print("------------------------")
            # print(next_action_seq)
            # print("------------------------")
            # print(qs_on_circle)
            # print("------------------------")
            # print(q_extract)
            # print("------------------------")

            assert final_q > cur_q, "The final q %f should be greater than the current q %f"%(final_q, cur_q)

            # compute these qs
            q_sequence = []
            for j, action in enumerate(next_action_seq):
                # compute one step q
                qs = np.zeros(4, dtype="float")

                # stop is always stay at current q
                qs[cartesian_stop_action_index] = cur_q

                steps_to_forward = len(next_action_seq) - 1 - j
                
                # forward is optimal
                if action == cartesian_forward_action_index:
                    qs[action] = final_q
                    qs[cartesian_turn_left_action_index] = cur_q
                    qs[cartesian_turn_right_action_index] = cur_q
                # turn left is optimal
                elif action == cartesian_turn_left_action_index:
                    qs[cartesian_forward_action_index] = q_extract[j]
                    qs[action] = final_q + q_penality_per_action * steps_to_forward
                    qs[cartesian_turn_right_action_index] = final_q + q_penality_per_action * (share_num-steps_to_forward)
                # turn right is optimal
                elif action == cartesian_turn_right_action_index:
                    qs[cartesian_forward_action_index] = q_extract[j]
                    qs[cartesian_turn_left_action_index] = final_q + q_penality_per_action * (share_num-steps_to_forward)
                    qs[action] = final_q + q_penality_per_action * steps_to_forward
                
                print("---------------------------------")
                print(action)
                print(final_q)
                print(qs)
                # print(cur_q)
                
                # print(steps_to_forward)
                # print(share_num-steps_to_forward)
                # print(qs)
                print("---------------------------------")

                # check validity of qs
                cur_step_optimal_action_list = list(np.argwhere(qs == np.amax(qs)).squeeze(axis=1))
                assert action in  cur_step_optimal_action_list, "Optimal action %d does not have max q %s"%(action, str( cur_step_optimal_action_list))

                # add q
                q_sequence.append(qs)

            
            # take one polar action step
            self.step_cartesian_action_seq(next_action_seq)
            # add optimal actions
            real_actions.extend(next_action_seq)
                
            reach_q_flag = self.reached_goal()


        print(real_actions)
        # print("==================")
        # print(share_num)
        # print("==================")
        exit()
            

    def generate_one_episode_with_cartesian_q(self, episode):
        goal_dimension = int(self.config.get("goal_dimension"))
        goal_coord_system = self.config.get("goal_coord_system")
        cartesian_action_number = len(self.action_mapping)

        observations = []
        actions = [] # cartesian action sequence
        rel_goals = []
        distance_to_goals = []
        goal_positions = []
        state_positions = []
        abs_goals = []
        dones = []
        rewards = []
        qs = []
        traj = {}
        
        # reset the env
        obs = self.reset(episode=episode, plan_shortest_path=True)
        # add (s0, g0, d0, r0)
        # d0=False, r0=0, q=None
        update_episode_data(env=self,
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
            qs=qs,
            q=None)

        
        reach_q_flag = self.reached_goal()

        while not reach_q_flag:
            q, cartesian_optimal_action, cartesian_optimal_action_list = self.compute_cartesian_q_current_state()

            # take one cartesian action step
            obs, reward, done, info = self.step(cartesian_optimal_action)

            # add (s_i, a_{i-1}, g_i, d_i, r_i, q_{i-1})
            update_episode_data(env=self,
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
                action=cartesian_optimal_action,
                qs=qs,
                q=q)

            reach_q_flag = self.reached_goal()
        
        assert actions[-1] != 0, "The original planned optimal cartesian action sequence should not end with STOP."
        # print("========================")
        # print(actions)
        
        # compute q when we have already reached the goal
        # align current agent position to goal position
        q = -1.0 * float(self.config.get("forward_resolution")) * np.ones(cartesian_action_number, dtype="float")
        q[0] = 0.0
        final_cartesian_optimal_action_list = list(np.argwhere(q == np.amax(q)).squeeze(axis=1))
        assert len(final_cartesian_optimal_action_list)==1 and final_cartesian_optimal_action_list[0]==0, "STOP should be the optimal action when we have already reached the goal."
        
        # take one cartesian env step = STOP
        obs, reward, done, info = self.step(0)
        assert done==True and self.is_success(), "Generated episode did not succeed"
        
        # append the first polar action STOP as the final step
        # add (s_i, a_{i-1}=0, g_i, d_i, r_i, q_{i-1})
        update_episode_data(env=self,
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
            action=0,
            qs=qs,
            q=q)

        # append the second polar action STOP (besides the one at the end of the optimal action sequence)
        actions.append(0)
        # append the second q (same as the previous one) to qs
        qs.append(copy.deepcopy(q))

        # print("========================")
        # print(len(observations)) # n+1
        # print(len(rel_goals)) # n+1
        # print(len(distance_to_goals)) # n+1
        # print(len(goal_positions)) # n+1
        # print(len(state_positions)) # n+1
        # print(len(abs_goals)) # n+1
        # print(len(dones)) # n+1
        # print(len(rewards)) # n+1
        # print(len(qs)) # n+1
        # print(len(actions)) # n+1
        # print("========================")
        # print(actions)
        # print("========================")
        # print(qs)
        # print("========================")
        

        traj["observations"] = observations
        traj["actions"] = actions
        traj["rel_goals"] = rel_goals
        traj["distance_to_goals"] = distance_to_goals
        traj["goal_positions"] = goal_positions
        traj["state_positions"] = state_positions
        traj["abs_goals"] = abs_goals
        traj["dones"] = dones
        traj["rewards"] = rewards
        traj["qs"] = qs
                    
        return traj, actions[:-1]

def test_env(config_file="imitation_learning_rnn_bc.yaml"):
    env = MultiNavEnv(config_file=config_file)
    
    
    for i in range(10):
        obs = env.reset(plan_shortest_path=True)
        print('Episode: {}'.format(i+1))
        print("Goal position: %s"%(env.goal_position))
        #env.print_agent_state()
        print("Start position: %s"%(env.start_position))
        #print(env.get_optimal_trajectory())
        print("Optimal action sequence: %s"%env.optimal_action_seq)
        print("Optimal action sequence length: %s"%len(env.optimal_action_seq))

        done = False
        #for j in range(100):
        j = 0
        while not done:
            #action = env.action_space.sample()
            action = env.get_next_optimal_action()
            #print(action)
            obs, reward, done, info = env.step(action)
            #print(obs)
            #print(env.get_combined_goal_obs_space())
            #print(env.get_default_observation())
            #exit()
            #print(obs["color_sensor"].shape)
            #print(obs["pointgoal"].shape)
            env.render()
            j += 1

        print("Total steps: %d"%j)
        print("===============================")

def test_polar_episode_generation(config_file):
    config = parse_config(os.path.join(config_path, config_file))

    episodes = load_behavior_dataset_meta(
                behavior_dataset_path="/dataset/behavior_dataset_gibson_1_scene_Rancocas_2000_polar_q_new", 
                split_name="same_start_goal_val_mini") 

    env = MultiNavEnv(config_file=config_file)
    for episode in episodes[:1]:
        traj, act_seq = env.generate_one_episode_with_polar_q(episode=episode)
        print(traj["actions"])
        print(len(traj["actions"]))
        print(len(traj["qs"]))
        print("Generated trajectory length: %d"%(len(act_seq)+1))
        

    env.close()

if __name__ == "__main__":    
    #test_env(config_file="imitation_learning_dqn.yaml")
    test_polar_episode_generation(config_file="imitation_learning_mlp_sqn.yaml")