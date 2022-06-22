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

# across scene environments
class MultiNavEnv(NavEnv):
    # config_file could be a string or a parsed config dict
    def __init__(self, config_file="imitation_learning.yaml"):
        # get config
        if isinstance(config_file, str):
            config_file = os.path.join(config_path, config_file)
        self.config = parse_config(config_file)
    
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
       
        # create gym action space
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
        
        if self.current_scene != episode_scene:
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
    
    def check_optimal_action_sequence(self):
        assert len(self.optimal_action_seq) > 0, "Error: optimal action sequence must have at least one element"
        if self.optimal_action_seq[-1] != self.action_name_to_index("stop"):
            print("Error: the last action in the optimal action sequence must be STOP, but %d now, appending STOP."%(self.optimal_action_seq[-1]))
            self.optimal_action_seq.append(self.action_name_to_index("stop"))
       
    def reset(self, episode=None, plan_shortest_path=False):
        # reset scene, agent start and goal
        if episode is None:
            if self.episode_iterator is None:
                self.set_agent_to_initial_state()
            else:
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

        # plan shortest path
        # must be called after agent has been set to the start location, and goal has been reset
        if plan_shortest_path:
            # create shortest path planner, must create everytime reset is called
            self.create_shortest_path_follower()
            try:
                self.optimal_action_seq = self.follower.find_path(goal_pos=self.goal_position)
                self.check_optimal_action_sequence()
            except habitat_sim.errors.GreedyFollowerError as e:
                print("Error: optimal path NOT found!")
                self.optimal_action_seq = []
        else:
            self.optimal_action_seq = []    

        return obs
    
    # action is an integer
    def step(self, action):
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

        print("Path follower created and reset.")      
    
    def set_scene_id_in_config(self, new_scene):
        self.sim_config.sim_cfg.scene_id = new_scene

def test_env():
    env = MultiNavEnv(config_file="imitation_learning.yaml")
    for i in range(10):
        obs = env.reset(plan_shortest_path=True)
        print('Episode: {}'.format(i+1))
        print("Goal position: %s"%(env.goal_position))
        #env.print_agent_state()
        print("Start position: %s"%(env.start_position))
        #print(env.get_optimal_trajectory())
        print("Optimal action sequence: %s"%env.optimal_action_seq)

        for j in range(100):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            print(obs["color_sensor"].shape)
            print(obs["pointgoal"].shape)
            #env.render()

        print("===============================")

    


if __name__ == "__main__":    
    test_env()