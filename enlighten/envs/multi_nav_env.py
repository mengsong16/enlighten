from enlighten.envs.nav_env import NavEnv
from enlighten.utils.config_utils import parse_config

import os
import numpy as np
import gym
from gym import spaces

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
        config_file = os.path.join(config_path, config_file)
        self.config = parse_config(config_file)
    
        # create simulator configuration
        self.sim_config, sensors = self.create_sim_config()
        # create simulator
        self.sim = habitat_sim.Simulator(self.sim_config) 

        # register dynamic lighting in simulator for dark mode
        if self.dark_mode:
            # flashlight: point light x m in front of the robot 
            self.flashlight_z = float(self.config.get('flashlight_z'))

            self.sim.set_light_setup([
            LightInfo(vector=[0.0, 0.0, -self.flashlight_z, 1.0], model=LightPositionModel.Camera)
            ], "current_scene_lighting")
       
           
        # create gym observation space
        self.observation_space = self.create_gym_observation_space(sensors)
        
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
    
    def create_sim_config(self):
        # simulator configuration
        sim_config = habitat_sim.SimulatorConfiguration()
        
        # set dummy scene path
        sim_config.scene_id = self.config.get('scene_id')
        self.current_scene = sim_config.scene_id

        # set dataset path
        # if not set, the value is "default"
        #sim_config.scene_dataset_config_file = self.config.get('dataset_path')

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
            

        # sensors and sensor specifications
        sensor_specs, sensors = self.create_sensors_and_sensor_specs()
        
        # agent configuration
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.action_space = self.create_sim_action_space()
        agent_cfg.sensor_specifications = sensor_specs

        return habitat_sim.Configuration(sim_config, [agent_cfg]), sensors

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
            self.current_scene = episode_scene
            self.set_scene_id_in_config(episode_scene)
            self.sim.reconfigure(self.sim_config)
        
        # reset agent state and goal according to episode
        self.set_start_goal(new_goal_position=self.get_episode_goal_position(episode), 
            new_start_position=self.get_episode_start_position(episode), 
            new_start_rotation=self.get_episode_start_rotation(episode), 
            is_initial=False)

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
    
    def reset(self, episode=None):
        # reset scene, agent start and goal
        if episode is None:
            self.set_agent_to_initial_state()
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

        return obs
    
    def step(self, action):
        
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

        # add metrics (e.g, success) to info for tensorboard stats
        info = {"success": int(self.is_success()), "spl": self.get_spl()}

        return obs, reward, done, info 
    
    def get_reward(self):
        reward = self.measurements.measures["point_goal_reward"].get_metric()

        return reward   

    def create_shortest_path_follower(self):
        self.follower = self.sim.make_greedy_follower(
                agent_id=0,
                goal_radius=self.goal_radius,
                stop_key=self.action_name_to_index("stop"),  # 0
                forward_key=self.action_name_to_index("move_forward"), # 1
                left_key=self.action_name_to_index("turn_left"), # 2 
                right_key=self.action_name_to_index("turn_right"),  #3
            )   
    
    def set_scene_id_in_config(self, new_scene):
        self.sim_config.sim_cfg.scene_id = new_scene

def test_env():
    env = MultiNavEnv()
    for i in range(10):
        obs = env.reset()
        print('Episode: {}'.format(i))
        print("Goal position: %s"%(env.goal_position))
        env.print_agent_state()
        #print(env.get_optimal_trajectory())

        for j in range(100):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            env.render()

        print("===============================")

    


if __name__ == "__main__":    
    test_env()