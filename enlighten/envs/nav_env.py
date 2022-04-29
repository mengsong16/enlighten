import math
import os
import numpy as np
from numpy import euler_gamma, ndarray
import sys
import random
from matplotlib import pyplot as plt
import gym
from gym import spaces
#from gym.envs.classic_control.rendering import SimpleImageViewer

import habitat_sim
from habitat_sim.gfx import LightInfo, LightPositionModel, DEFAULT_LIGHTING_KEY, NO_LIGHT_KEY
from habitat_sim.utils.common import quat_from_angle_axis

import attr

from enlighten.utils.config_utils import parse_config
from enlighten.utils.geometry_utils import get_rotation_quat, euclidean_distance 
from enlighten.utils.path import *
from enlighten.tasks.measures import Measurements
from enlighten.envs.explore import State_Visitation

import abc
import copy

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Tuple,
    Optional,
    Sequence,
    Set,
    Union,
    cast,
    Iterable
)

from collections import OrderedDict
from enum import Enum

#from garage import Environment, EnvSpec, EnvStep, StepType
#from garage.envs import GymEnv
from PIL import Image
from enlighten.utils.viewer import MyViewer

from enlighten.utils.image_utils import try_cv2_import
cv2 = try_cv2_import()

from skimage.color import label2rgb 

from habitat.utils.visualizations import maps
import magnum as mn
import quaternion as qt
from habitat_sim.utils.common import quat_from_angle_axis

from enlighten.envs import HabitatSensor, Dictionary_Observations, HabitatSimRGBSensor, HabitatSimDepthSensor, HabitatSimSemanticSensor, ImageGoal, PointGoal
from enlighten.envs.sensor import StateSensor

import torch
import urllib.request

import matplotlib.pyplot as plt

class SensorSuite:
    r"""Represents a set of sensors, with each sensor being identified
    through a unique id.
    """

    sensors: Dict[str, HabitatSensor]
    observation_spaces: spaces.Dict

    def __init__(self, sensors: Iterable[HabitatSensor], dictionary_observation_space) -> None:
        """Constructor

        :param sensors: list containing sensors for the environment, uuid of
            each sensor must be unique.
        """

        # get a dictionary of sensor spaces 
        self.sensors = OrderedDict()
        ordered_spaces: OrderedDict[str, gym.Space] = OrderedDict()
        for sensor in sensors:
            assert (
                sensor.uuid not in self.sensors
            ), "'{}' is duplicated sensor uuid".format(sensor.uuid)
            self.sensors[sensor.uuid] = sensor
            ordered_spaces[sensor.uuid] = sensor.observation_space

        # get a dictionary observation space for gym  
        self.dictionary_observation_space = dictionary_observation_space
    

        # dictionary
        if self.dictionary_observation_space:
            self.observation_spaces = gym.spaces.Dict(spaces=ordered_spaces)
        # numpy array    
        else:
            #if self.goal_conditioned == True and self.config.get("goal_format") == "pointgoal":
            #    raise ValueError("Can only concatenate images not goal vectors when observation space is numpy array")
            if "state_sensor" in self.sensors:
                if "color_sensor" in self.sensors or "depth_sensor" in self.sensors or "semantic_sensor" in self.sensors:
                    print("Error: cannot concatenate state space with image observation space")
                    exit()

            self.observation_spaces = self.concatenate_observation_space(ordered_spaces)   

    def get_specific_sensor(self, uuid: str) -> HabitatSensor:
        return self.sensors[uuid]

    def get_observations(self, *args: Any, **kwargs: Any):
        r"""Collects data from all sensors and returns it packaged inside
        :ref:`Observations`.
        """
        # dictionary
        if self.dictionary_observation_space:
            return Dictionary_Observations(self.sensors, *args, **kwargs)
        # numpy array    
        else:
            #if self.goal_conditioned == True and self.config.get("goal_format") == "pointgoal":
            #    raise ValueError("Can only concatenate images not goal vectors when observation space is numpy array")
            
            return self.get_array_observations(*args, **kwargs)    

    def get_specific_observation(self, uuid: str, *args, **kwargs):
        assert uuid in self.sensors, "mode {} sensor is not active".format(uuid)

        # sensor.get_observation need parameter sim_obs
        return self.sensors[uuid].get_observation(*args, **kwargs)


    # Return a numpy array concatenating all observation modes
    # state observation or concatenated image observations
    def get_array_observations(self, *args: Any, **kwargs: Any):
        r"""Numpy array containing sensor observations"""

        # no need to handle the first item
        data = [
            # sensor.get_observation need parameter sim_obs
            sensor.get_observation(*args, **kwargs)
            for _, sensor in self.sensors.items()
        ]
    
        concat_data = np.dstack(data)

        return concat_data

    def concatenate_observation_space(self, ordered_spaces):
        self.channel_num = 0
        self.image_height = None
        self.image_width = None
        low = np.inf
        high = -np.inf

        for _, space in ordered_spaces.items():
            if self.image_height is None:
                self.image_height = space.shape[0]
            if self.image_width is None:    
                self.image_width = space.shape[1]
                
            self.channel_num += space.shape[2]
            
            low = min(low, np.amin(space.low))
            high = max(high, np.amax(space.high))

        # [H,W,C]
        return gym.spaces.Box(
            low=low,
            high=high,
            shape=(self.image_height, self.image_width, self.channel_num),
            dtype=np.float32
        )    

# define new action stop in habitat simulator
@attr.s(auto_attribs=True, slots=True)
class StopSpec:
    pass
# gym.Env definition: https://github.com/openai/gym/blob/267916b9d268c37cc948bafe35606c665aac53ac/gym/core.py

@habitat_sim.registry.register_move_fn(body_action=True)
class Stop(habitat_sim.SceneNodeControl):
    def __call__(
        self, scene_node: habitat_sim.SceneNode, actuation_spec: StopSpec
    ):
        return

# register the function with a custom name
# move robot body according to this action definition
habitat_sim.registry.register_move_fn(
    Stop, name="stop", body_action=True
)


class NavEnv(gym.Env):
    r"""Base gym navigation environment
    """
    # config_file could be a string or a parsed config
    def __init__(self, config_file=os.path.join(config_path, "navigate_with_flashlight.yaml"), dataset=None):
        self.config = parse_config(config_file)
        self.dataset = dataset
    
        # create simulator configuration
        self.sim_config, sensors = self.create_sim_config()
        # create simulator
        self.sim = habitat_sim.Simulator(self.sim_config) 

        
        # register dynamic lighting in simulator for dark mode
        if self.dark_mode:
            # flashlight: point light x m in front of the robot 
            self.flashlight_z = float(self.config.get('flashlight_z'))
            #self.sim.set_light_setup([], "current_scene_lighting")
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
         
        # set start and goal positions
        self.set_start_goal()

        
        # set measurements
        measure_ids = list(self.config.get("measurements"))
        self.measurements = Measurements(measure_ids=measure_ids, env=self, config=self.config)
        # set current episode
        self.current_episode = 0

        # set visitation dictionaries
        self.train_state_count_dict = State_Visitation(position_resolution=float(self.config.get("forward_resolution")), rotation_resolution=float(self.config.get("rotate_resolution")))
        self.episode_state_count_dict = State_Visitation(position_resolution=float(self.config.get("forward_resolution")), rotation_resolution=float(self.config.get("rotate_resolution")))
        
        # set depth estimation model
        self.depth_reward = self.config.get("depth_reward")
        
        if self.depth_reward:
            assert self.config.get("color_sensor"), "Need RGB sensor to estimate depth"
            assert self.config.get("goal_reward") == False, "Depth reward cannot be used with distance to goal reward together"
            # load model
            model_type = "MiDaS_small"
            self.midas = torch.hub.load("intel-isl/MiDaS", model_type)

            self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            self.midas.to(self.device)
            self.midas.eval()
            # load model transforms
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

            if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
                self.transform = midas_transforms.dpt_transform
            else:
                self.transform = midas_transforms.small_transform


    def create_sim_config(self):
        # simulator configuration
        sim_config = habitat_sim.SimulatorConfiguration()
        
        # set scene path
        sim_config.scene_id = self.config.get('scene_id')

        # set dataset path
        # if not set, the value is "default"
        sim_config.scene_dataset_config_file = self.config.get('dataset_path')

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

    def create_sim_action_space(self):
        self.action_mapping = ["stop", "move_forward", "turn_left", "turn_right", "look_up", "look_down"]
        action_space = {
            "stop": habitat_sim.agent.ActionSpec("stop", habitat_sim.agent.ActuationSpec(amount=0.0)),
            "move_forward": habitat_sim.agent.ActionSpec(
                "move_forward", habitat_sim.agent.ActuationSpec(amount=float(self.config.get("forward_resolution")))  # move -a meter along z axis (translate along local frame)
            ),
            "turn_left": habitat_sim.agent.ActionSpec(
                "turn_left", habitat_sim.agent.ActuationSpec(amount=float(self.config.get("rotate_resolution"))) # rotate a degree along y axis (rotate along local frame) 
            ),
            "turn_right": habitat_sim.agent.ActionSpec(
                "turn_right", habitat_sim.agent.ActuationSpec(amount=float(self.config.get("rotate_resolution"))) # rotate -a degree along y axis (rotate along local frame)
            ),
            
        }

        if int(self.config.get("action_number")) == 6:
            # rotate a degree along x axis (rotate along local frame)
            action_space["look_up"] = habitat_sim.agent.ActionSpec("look_up", habitat_sim.agent.ActuationSpec(amount=float(self.config.get("rotate_resolution"))))
            # rotate -a degree along x axis (rotate along local frame)
            action_space["look_down"] = habitat_sim.agent.ActionSpec("look_down", habitat_sim.agent.ActuationSpec(amount=float(self.config.get("rotate_resolution"))))

        return action_space

    def get_sim_action_space(self):
        return self.sim_config.agents[self.sim._default_agent_id].action_space

    def create_gym_action_space(self):
        return spaces.Discrete(len(self.get_sim_action_space())) 

    def get_gym_action_space(self):
        return self.action_space  

    def get_sim_config(self):
        return self.sim_config.sim_cfg    

    # ref: class ImageExtractor
    def create_sensors_and_sensor_specs(self):
           
        image_height = int(self.config.get("image_height"))
        image_width = int(self.config.get("image_width"))
        sensor_height = 1.5
        sensor_specs = []
        sensors = []
        
        
        # Note: all sensors must have the same resolution
        if self.config.get("color_sensor"):
            color_sensor_spec = habitat_sim.CameraSensorSpec()
            color_sensor_spec.uuid = "color_sensor"
            color_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
            color_sensor_spec.resolution = [image_height, image_width]
            color_sensor_spec.postition = [0.0, sensor_height, 0.0]
            color_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
            sensor_specs.append(color_sensor_spec)
            color_sensor = HabitatSimRGBSensor(config=self.config)
            self.BGR_mode = color_sensor.RGB2BGR
            sensors.append(color_sensor)

        if self.config.get("depth_sensor"):
            depth_sensor_spec = habitat_sim.CameraSensorSpec()
            depth_sensor_spec.uuid = "depth_sensor"
            depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
            depth_sensor_spec.resolution = [image_height, image_width]
            depth_sensor_spec.postition = [0.0, sensor_height, 0.0]
            depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
            sensor_specs.append(depth_sensor_spec)
            sensors.append(HabitatSimDepthSensor(config=self.config))

        if self.config.get("semantic_sensor"):
            semantic_sensor_spec = habitat_sim.CameraSensorSpec()
            semantic_sensor_spec.uuid = "semantic_sensor"
            semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
            semantic_sensor_spec.resolution = [image_height, image_width]
            semantic_sensor_spec.postition = [0.0, sensor_height, 0.0]
            semantic_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
            sensor_specs.append(semantic_sensor_spec) 
            sensors.append(HabitatSimSemanticSensor(config=self.config))
        
        if self.config.get("state_sensor"):
            sensors.append(StateSensor(config=self.config))
        
        # set render mode according to all available sensors
        self.set_render_mode()

        return sensor_specs, sensors 

    def create_goal_sensor(self):
        # add goal to observations
        if self.config.get("goal_conditioned"):
        #if self.config.get("random_goal"):
            assert self.config.get("goal_format") in ["pointgoal", "imagegoal"], "Goal format is not supported!"
            if self.config.get("goal_format") == "pointgoal":
                self.goal_sensor = PointGoal(config=self.config, env=self)
            else:
                self.goal_sensor = ImageGoal(config=self.config, env=self)
        else:
            self.goal_sensor = None
    

    def add_goal_obs(self, obs):
        if self.goal_sensor is not None:
            obs[self.goal_sensor._get_uuid()] = self.get_goal_observation()

        return obs    

    def create_gym_observation_space(self, sensors):
        self.sensor_suite = SensorSuite(sensors, 
            dictionary_observation_space=self.config.get('dictionary_observation_space')) 

        return self.sensor_suite.observation_spaces    

    def get_gym_observation_space(self):
        return self.observation_space

    def get_goal_observation_space(self):
        if self.goal_sensor is not None:
            return self.goal_sensor._get_observation_space()  
        else:
            print("Goal space does not exist since the env is not goal conditioned")
            return None 

    def get_goal_observation(self):
        if self.goal_sensor is not None:
            # for image goal, use the default goal_azimuth
            return self.goal_sensor.get_observation(goal_position=self.goal_position, start_position=self.start_position, start_rotation=self.start_rotation) 
        else:
            print("Goal observation does not exist since the env is not goal conditioned")
            return None 

    def get_combined_goal_obs_space(self):
        combined_observation_space = copy.deepcopy(self.observation_space)
        goal_space = self.get_goal_observation_space()
        if goal_space == None:
            return combined_observation_space
        else:
            combined_observation_space.spaces[self.goal_sensor.uuid] = copy.deepcopy(goal_space) 
            return combined_observation_space                     
    
    def get_sim_sensors(self):
        return self.sim._sensors

    def get_agent_state(self):
        agent_state = self.agent.get_state()
        # position: [x,y,z]
        # rotation: quarternion [x,y,z,q]
        
        return agent_state

    def get_agent_position(self):    
        return self.get_agent_state().position

    def get_agent_rotation_euler(self):
        agent_state = self.agent.get_state()
        
        return qt.as_euler_angles(agent_state.rotation)

    # [x, y, z, alpha, beta, gamma]
    def get_agent_state_xyz_euler(self):
        agent_state = self.agent.get_state()
        # position: [x,y,z]
        # rotation: quarternion [x,y,z,q]

        euler = qt.as_euler_angles(agent_state.rotation)
        
        return np.concatenate((agent_state.position, euler), axis=0)       


    def print_agent_state(self):  
        agent_state = self.agent.get_state() 

        print("agent state: position: ", agent_state.position, ", rotation: ", agent_state.rotation) 

    def create_agent_state(self, new_position, new_rotation=None, quaternion=False):
        new_agent_state = habitat_sim.AgentState()
        # global system, must be casted to float32
        new_agent_state.position = np.array(new_position, dtype="float32")  

        if new_rotation is not None:
            # quaternion
            if quaternion:
                # new_rotation type: quaternion
                new_agent_state.rotation = new_rotation  
            # euler     
            else:    
                new_agent_state.rotation = get_rotation_quat(np.array(new_rotation, dtype="float32"))
        
        return new_agent_state

    def set_agent_state(self, new_position, new_rotation=None, is_initial=False, quaternion=False):
        self.agent.set_state(self.create_agent_state(new_position=new_position, new_rotation=new_rotation, quaternion=quaternion), is_initial=is_initial)

    # get goal observation
    # used in getting image goal
    def get_observations_at(self, position, rotation, keep_agent_at_new_pose=False):
        # reset agent state to the goal location
        current_state = self.get_agent_state()

        self.set_agent_state(new_position=position, new_rotation=rotation, is_initial=False, quaternion=True)

        sim_obs = self.sim.get_sensor_observations(agent_ids=self.sim._default_agent_id)

        obs = self.sensor_suite.get_observations(sim_obs=sim_obs, env=self)

        # get agent back to current pose
        if not keep_agent_at_new_pose:
            self.set_agent_state(
                new_position=current_state.position,
                new_rotation=current_state.rotation,
                is_initial=False,
                quaternion=True
            )
        return obs
        
    
    def action_index_to_name(self, index):
        return self.action_mapping[index]

    def action_name_to_index(self, name):
        index = self.action_mapping.index(name) 
        return index   

    def set_start_goal(self):
        if self.sim.pathfinder.is_loaded:
            trajectory_not_exist = True
            while trajectory_not_exist:
                self.set_start_goal_once()
                found_path, _, _ = self.shortest_path(self.get_agent_position(), self.goal_position)
                if found_path:
                    trajectory_not_exist = False
                else:
                    if (not self.random_goal) and (not self.random_start):
                        print("There is no optimal trajectory exists between the provided start and goal position")
                        exit()    

            print("Optimal trajectory exists between start position %s and goal position %s"%(self.get_agent_position(), self.goal_position))        
        else:
           self.goal_position = np.array(self.config.get('goal_position'), dtype="float32")
           self.start_position = np.array(self.config.get('agent_initial_position'), dtype="float32")
           self.start_rotation = np.array(self.config.get('agent_initial_rotation'), dtype="float32")
           # quarternion
           self.start_rotation = get_rotation_quat(np.array(self.start_rotation, dtype="float32"))

           self.set_agent_state(new_position=self.start_position, 
                        new_rotation=self.start_rotation, is_initial=True, quaternion=True)

    # may need to fix y coordinate of start and goal position
    def set_start_goal_once(self):
        self.random_goal = self.config.get('random_goal') 

        # fixed goal
        if not self.random_goal: 
            end_point = np.array(self.config.get('goal_position'), dtype="float32")
            
            if self.sim.pathfinder.is_loaded:
                # check navigability of goal position
                if self.sim.pathfinder.is_navigable(end_point):
                    self.goal_position = end_point
                else:
                    print("Error: provided goal position is not navigable!")
                    exit()
            else:
                self.goal_position = end_point        
            
            #self.goal_position = end_point  
        # random goal       
        else:
            if self.sim.pathfinder.is_loaded:
                self.goal_position = self.sim.pathfinder.get_random_navigable_point()
            else:
                print("Error: navmesh is not available so is not able to set random goal point")      

        self.random_start = self.config.get('random_start')

        # fixed start
        if not self.random_start:
            self.start_position = np.array(self.config.get('agent_initial_position'), dtype="float32")
            self.start_rotation = np.array(self.config.get('agent_initial_rotation'), dtype="float32")
            # quarternion
            self.start_rotation = get_rotation_quat(np.array(self.start_rotation, dtype="float32"))

            if self.sim.pathfinder.is_loaded:
                if self.sim.pathfinder.is_navigable(self.start_position):
                    self.set_agent_state(new_position=self.start_position, 
                        new_rotation=self.start_rotation, is_initial=True, quaternion=True)
                else:
                    print("Error: provided start position is not navigable!")
                    exit() 
            else:
                self.set_agent_state(new_position=self.start_position, 
                        new_rotation=self.start_rotation, is_initial=True, quaternion=True)
            #print(start_point)
            #self.set_agent_state(new_position=start_point, \
            #    new_rotation=self.config.get('agent_initial_rotation'), is_initial=True)             
        # random start
        else:
            if self.sim.pathfinder.is_loaded:
                self.start_position = self.sim.pathfinder.get_random_navigable_point()
                # quarternion
                self.start_rotation = quat_from_angle_axis(self.sim.random.uniform_float(0, 2.0 * np.pi), np.array([0, 1, 0]))
                self.set_agent_state(new_position=self.start_position, \
                    new_rotation=self.start_rotation, is_initial=True, quaternion=True) 
            else:
                print("Error: navmesh is not available so is not able to set random start point")                

    def estimate_depth(self, obs):
        # input image: [224,224,3], in range of [0,255]
        img = obs["color_sensor"]
        #print(self.BGR_mode)
        # BGR to RGB
        if self.BGR_mode:   
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        input_batch = self.transform(img).to(self.device)

        with torch.no_grad():
            prediction = self.midas(input_batch)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        # output depth: [224,224]
        predicted_depth_map = prediction.cpu().numpy()
        average_predicted_depth = np.mean(predicted_depth_map)
        return average_predicted_depth

    def step(self, action):
        # step simulator
        # transit to s'
        action_name = self.action_mapping[action]
        sim_obs = self.sim.step(action_name)
        # self.did_collide = self.extract_collisions(sim_obs)
        # if self.did_collide:
        #     self.collision_count_per_episode += 1
        
        # sim_obs includes all modes
        obs = self.sensor_suite.get_observations(sim_obs=sim_obs, env=self)

        # goal conditioned (either random goal or fixed goal)
        if self.config.get("goal_conditioned"):
        #if self.config.get("random_goal"):
            obs = self.add_goal_obs(obs)

        #self.step_count_per_episode += 1

        # update all measurements
        self.measurements.update_measures(
            measurements=self.measurements,
            sim_obs=sim_obs,
        )

        # estimate depth from RGB observation
        if self.depth_reward:
            
            current_average_predicted_depth = self.estimate_depth(obs)
            #print(output.shape)
            self.average_predicted_depth_change = self.prev_average_predicted_depth - current_average_predicted_depth
            # normailze
            self.average_predicted_depth_change /= 1000.0
            #print(self.average_predicted_depth_change)
            self.prev_average_predicted_depth = current_average_predicted_depth

        #self.measurements.print_measures()

        # update visitation dictionaries
        # must done before calculating the reward and after env transited to s'
        current_state = self.get_agent_state_xyz_euler()
        # must update visitation count before get reward
        self.train_state_count_dict.add(current_state)
        self.episode_state_count_dict.add(current_state)

        current_train_state_count = self.train_state_count_dict.get(current_state)
        current_episode_state_count = self.episode_state_count_dict.get(current_state)
        
        reward = self.get_reward(current_train_state_count, current_episode_state_count)
        # add visual odometry reward
        if self.depth_reward:
            reward += self.average_predicted_depth_change

        # update prev state count
        self.prev_train_state_count = current_train_state_count
        self.prev_episode_state_count = current_episode_state_count

        done = self.is_done()

        # add metrics (e.g, success) to info for tensorboard stats
        info = {"success": int(self.is_success()), "spl": self.get_spl()}


        return obs, reward, done, info              

    def reset(self):
        # reset agent.initial_state to its initial pose
        sim_obs = self.sim.reset()

        # reset start and goal
        self.set_start_goal()

        # get the initial observation
        obs = self.sensor_suite.get_observations(sim_obs=sim_obs, env=self)

        if self.config.get("goal_conditioned"):
        #if self.config.get("random_goal"):    
            obs = self.add_goal_obs(obs)

        # self.did_collide = self.extract_collisions(sim_obs)
        # self.collision_count_per_episode = 0
        #self.step_count_per_episode = 0

        # reset measurements
        self.measurements.reset_measures(measurements=self.measurements)

        # reset episode visitation count
        self.episode_state_count_dict.reset()

        # count s0 to visitation count
        current_state = self.get_agent_state_xyz_euler()
        self.train_state_count_dict.add(current_state)
        self.episode_state_count_dict.add(current_state)

        # record prev_state_count
        self.prev_train_state_count = self.train_state_count_dict.get(current_state)
        self.prev_episode_state_count = self.episode_state_count_dict.get(current_state)
        #self.previous_measure = self.get_current_distance()

        # initialize prev predicted depth
        if self.depth_reward:
            self.prev_average_predicted_depth = self.estimate_depth(obs)

        return obs

    def seed(self, seed):    
        self.sim.seed(seed)
        if self.sim.pathfinder.is_loaded:
            self.sim.pathfinder.seed(seed)

    def get_render_mode(self):
        return self.metadata['render.modes']

    def set_render_mode(self):
        self.metadata['render.modes'] = ['color_sensor', 'depth_sensor', 'semantic_sensor']  

    def get_current_step(self):
        return self.measurements.measures["steps"].get_metric()

    def get_current_collision_counts(self):
        return self.measurements.measures["collisions"].get_metric()["count"]   

    def is_collision(self):
        return self.measurements.measures["collisions"].get_metric()["is_collision"]         

    def render(self, mode: str = "color_sensor", attention_image=None) -> Any:
        r"""
        Args:
            mode: sensor whose observation is used for returning the frame,
                eg: "rgb", "depth", "semantic"

        Returns:
            rendered frame according to the mode
        """

        assert self.config.get(mode), "render mode should be active in the config file"


        if self.config.get("attention"):
            save_attention_image = True
        else:    
            save_attention_image = False
        # create viewer
        if self.viewer is None:
            #self.viewer = SimpleImageViewer()
            self.viewer = MyViewer(sim=self.sim, show_attention=self.config.get("attention"))

        sim_obs = self.sim.get_sensor_observations(agent_ids=self.sim._default_agent_id)
        obs = self.sensor_suite.get_specific_observation(uuid=mode, sim_obs=sim_obs)

        # show state observation
        if mode == "state_sensor":
            print(obs)
            return

        # show image observation
        if not isinstance(obs, np.ndarray):
            # If it is not a numpy array, it is a torch tensor
            # The function expects the result to be a numpy array
            obs = obs.to("cpu").numpy()

        # get map
        if self.sim.pathfinder.is_loaded:
            path_points = self.get_optimal_trajectory()
            topdown_map = self.get_map(path_points)
        else:
            topdown_map = None    

        #print(obs.shape)
        #print('------------------')
        # show image and map in viewer
        # color RGB
        if obs.shape[2] == 3:
            
            img = np.asarray(obs).astype(np.uint8)
            #img = img[:,:,[2,1,0]]
            #img = Image.fromarray(np.uint8(obs))
            alpha_blend_image = self.viewer.imshow(img, topdown_map, attention_image)
            
        elif obs.shape[2] == 1:
            # depth
            if mode == "depth_sensor":
                img = np.asarray(obs * 255).astype(np.uint8)
                # not the same with dstack the single channel
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                alpha_blend_image = self.viewer.imshow(img, topdown_map, attention_image)
            # label image    
            else:
                img = np.asarray(np.squeeze(obs, axis=2)).astype(np.uint8)
                img = label2rgb(label=img)
                # float to int
                img = np.asarray(img).astype(np.uint8)
                #print("*****************")
                #print(np.ptp(img))
                alpha_blend_image = self.viewer.imshow(img, topdown_map, attention_image)
        else:
            print("Error: image channel is neither 1 nor 3!")

        # save observation images in video dir
        if "disk" in list(self.config.get("eval_video_option")):
            filename = str(self.get_current_episode())+"_"+str(self.get_current_step()) + ".jpg"
            video_path = os.path.join(root_path, self.config.get("video_dir"), self.config.get("experiment_name"))
            if not os.path.exists(video_path):
                os.mkdir(video_path)
            
            if save_attention_image == False:
                cv2.imwrite(os.path.join(video_path, filename), img) 
            else:
                if alpha_blend_image is None:
                    print("Error: attention image is None when trying to save it during rendering")
                    exit()
                
                cv2.imwrite(os.path.join(video_path, filename), alpha_blend_image)       
        
    
        return img

    # whether agent collided with the scene
    def extract_collisions(self, sim_obs):
        # obs contains collided only when sim.step() is called
        if "collided" in sim_obs:
            colls = sim_obs["collided"]
        else:
            colls = None    
        return colls
 

    def print_collide_info(self):
        print('collide: %s'%(str(self.is_collision()))) 

    def get_current_scene_light_vector(self):
        print("******************************************************************")
        if not self.sim.get_current_light_setup():
            print("Current scene light setup: No Light")
        else:    
            print("Current scene light setup: vector=%s"%self.sim.get_current_light_setup()[0].vector)
            print("Current scene light setup: model=%s"%self.sim.get_current_light_setup()[0].model)
        print("******************************************************************")

    def get_specific_scene_light_vector(self, key):
        print("******************************************************************")
        if not self.sim.get_light_setup(key):
            print("Scene light setup: No Light, key=%s"%(key))
        else:    
            print("Scene light setup: key=%s, vector=%s"%(key, self.sim.get_light_setup(key)[0].vector))
            print("Scene light setup: key=%s, model=%s"%(key, self.sim.get_light_setup(key)[0].model))
        print("******************************************************************")                
    
    def shortest_path(self, start_point, end_point):
        if not self.sim.pathfinder.is_loaded:
            print("Pathfinder not initialized, aborting.")
            return

        path = habitat_sim.ShortestPath()
        path.requested_start = start_point
        path.requested_end = end_point

        # compute path
        found_path = self.sim.pathfinder.find_path(path)
        geodesic_distance = path.geodesic_distance
        path_points = path.points
        
        #print("start point: "+str(start_point))
        #print("end point: "+str(end_point))
        #print("found_path : " + str(found_path))
        #print("geodesic_distance : " + str(geodesic_distance))
        #print("path_points : " + str(path_points))

        return found_path, geodesic_distance, path_points

    def get_env_bounds(self):
        # 2*3 array
        return self.sim.pathfinder.get_bounds()

    # from current position to goal
    def get_optimal_trajectory(self):
        _, _, path_points = self.shortest_path(start_point=self.get_agent_position(), end_point=self.goal_position)
        return path_points

    # get geodesic distance from current position to goal
    def get_geodesic_distance_single_goal(self):
        found_path, geodesic_distance, _ = self.shortest_path(start_point=self.get_agent_position(), end_point=self.goal_position)
        return found_path, geodesic_distance    

    def get_geodesic_distance_multi_goals(self, goal_positions):
        found_path, geodesic_distance, _ = self.shortest_path(start_point=self.get_agent_position(), 
        end_point=goal_positions)
        return found_path, geodesic_distance
    # get euclidean distance from current position to goal [x,y,z] distance
    def get_euclidean_distance(self):
        return euclidean_distance(position_a=self.get_agent_position(), position_b=self.goal_position)

    # Display the map with agent and path overlay        
    def get_map(self, path_points):
        #sim_topdown_map = self.sim.pathfinder.get_topdown_view(self.meters_per_pixel, self.height)
        #sim_topdown_map = sim_topdown_map.astype(np.uint8)
        
        #print("path_points : " + str(path_points))

        self.meters_per_pixel = 0.1
        # The height (min y coordinate) in the environment to make the topdown map 
        self.height = self.sim.pathfinder.get_bounds()[0][1]

        top_down_map = maps.get_topdown_map(self.sim.pathfinder, self.height, meters_per_pixel=self.meters_per_pixel)
        
        # background, walkable area, obstacle and boundary
        # MAP_INVALID_POINT = 0, [255, 255, 255] white
        # MAP_VALID_POINT = 1, [128, 128, 128] grey
        # MAP_BORDER_INDICATOR = 2, [0, 0, 0] black
        recolor_map = np.array([[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8)
        top_down_map = recolor_map[top_down_map]

        grid_dimensions = (top_down_map.shape[0], top_down_map.shape[1])

        # no trajectory
        if not path_points:
            start_point = self.get_agent_position()
            end_point = self.goal_position 
            path_points = [start_point, end_point]  

        # convert world trajectory points to maps module grid points
        trajectory = [
                maps.to_grid(
                    path_point[2],  # realworld-z --> grid row
                    path_point[0],  # realworld-x --> grid column
                    grid_dimensions,
                    pathfinder=self.sim.pathfinder,
                )
                for path_point in path_points
            ]     

        '''
        # tangent =  z / x
        # robot always facing towards next state
        grid_tangent = mn.Vector2(
            trajectory[1][1] - trajectory[0][1], trajectory[1][0] - trajectory[0][0]
        )

        # trajectory point 0 and point 1 overlap
        if grid_tangent.is_zero():
            initial_angle = self.get_agent_rotation_euler()[1]  # angle around y 
        else:
            path_initial_tangent = grid_tangent / grid_tangent.length()
            # [-pi, pi]
            initial_angle = math.atan2(path_initial_tangent[0], path_initial_tangent[1])
        '''

        # robot rotation as robot angle on the map
        initial_angle = self.get_agent_rotation_euler()[1]  # angle around y

        # draw the trajectory on the map
        # color: (B,G,R)
        maps.draw_path(top_down_map=top_down_map, path_points=trajectory, color=(0, 0, 255), thickness=1)

        #print('grid_tangent: '+str(grid_tangent))
        #print('path_initial_tangent: '+str(path_initial_tangent))
        #print('agent rotation: '+str(initial_angle))

        # draw the agent
        maps.draw_agent(
            image=top_down_map, agent_center_coord=trajectory[0], agent_rotation=initial_angle, agent_radius_px=4
        )
        
        return top_down_map

    
    def get_current_distance(self):
        if self.sim.pathfinder.is_loaded:
            found_path, geodesic_distance = self.get_geodesic_distance_single_goal()
            if found_path:
                current_measure = geodesic_distance
            else:    
                current_measure = self.get_euclidean_distance()

            return current_measure
        else:
            return self.get_euclidean_distance()    

    def get_reward(self, current_train_state_count, current_episode_state_count):
        extrinsic_reward = self.measurements.measures["point_goal_reward"].get_metric()

        intrinsic_reward_coef = float(self.config.get("intrinsic_reward_coef"))
        if intrinsic_reward_coef != 0 :
            novelty_change = 1.0 / float(current_train_state_count) - float(self.config.get("prev_state_novelty_coef")) / float(self.prev_train_state_count)
            intrinsic_reward = max(novelty_change, 0.0)
            intrinsic_reward *= float(current_episode_state_count == 1)
            intrinsic_reward *= intrinsic_reward_coef
        else:
            #print("no intrinsic reward")
            intrinsic_reward = 0.0

        #print("~~~~~~~~~~~~~~~~~~~~~~~~~")
        #print("Extrinsic reward: %f"%(extrinsic_reward))
        #print("Intrinsic reward: %f"%(intrinsic_reward))
        #print("~~~~~~~~~~~~~~~~~~~~~~~~~")
        #assert intrinsic_reward == 0.0

        reward = extrinsic_reward + intrinsic_reward
        return reward   

    def is_done(self):
        return self.measurements.measures["done"].get_metric()  

    def is_success(self):
        return bool(self.measurements.measures["success"].get_metric()) 

    def get_spl(self):
        return self.measurements.measures["spl"].get_metric()    

    #def number_of_episodes(self):

    def get_current_episode(self):
        return self.current_episode

    def set_current_episode(self, episode_index):
        self.current_episode = episode_index   
    
    def close(self):
        self.sim.close()
        if self.viewer is not None:
            self.viewer.close()



################################ test ##########################################
def move_forward(env):
    env.reset()
    print("***********************************")
    env.print_agent_state()
    print('agent rotation [euler]: '+str(env.get_agent_rotation_euler()))

    # move forward (0.25m)
    action_index = env.action_name_to_index("move_forward")
    for i in range(4):
        obs, reward, done, info = env.step(action_index)
        print('-------------------------------------------')
        print('action: move_forward')
        env.print_agent_state()
        print('agent rotation [euler]: '+str(env.get_agent_rotation_euler()))
        #env.print_collide_info()
    
    print("***********************************")

def turn_left_move_forward(env):
    env.reset()
    print("***********************************")
    env.print_agent_state()
    print('agent rotation [euler]: '+str(env.get_agent_rotation_euler()))

    # turn left (10*9 degree)
    action_index_1 = env.action_name_to_index("turn_left")
    for i in range(9):
        obs, reward, done, info = env.step(action_index_1)
        print('-------------------------------------------')
        print('action: turn left')
        env.print_agent_state()
        print('agent rotation [euler]: '+str(env.get_agent_rotation_euler()))
        #env.print_collide_info()

    # move forward (0.25m)
    action_index_2 = env.action_name_to_index("move_forward")
    for i in range(2):
        obs, reward, done, info = env.step(action_index_2)
        print('-------------------------------------------')
        print('action: move_forward')
        env.print_agent_state()
        print('agent rotation [euler]: '+str(env.get_agent_rotation_euler()))
        #env.print_collide_info()
    
    print("***********************************")

def check_coordinate_system():
    env =  NavEnv()
    #move_forward(env)
    turn_left_move_forward(env)

# def create_garage_env(config_filename="navigate_with_flashlight.yaml"):
#     config_file = os.path.join(config_path, config_filename)
#     config = parse_config(config_file)
#     dictionary_observation_space = config.get('dictionary_observation_space')
#     assert dictionary_observation_space == False, "Garage env does NOT support dictionary observation space"
    
#     env = GymEnv(env=NavEnv(), is_image=True, max_episode_length=int(config.get("max_steps_per_episode"))) 
#     assert isinstance(env.spec, EnvSpec)

#     return env

def save_goal_image(img, current_episode):
    if not os.path.exists(output_path):
            os.mkdir(output_path)

    filename = str(current_episode)+ ".jpg" 
    cv2.imwrite(os.path.join(output_path, filename), img)         
            

def test_env(yaml_name):
    #if gym_env:
    env =  NavEnv(config_file=os.path.join(config_path, yaml_name))
    #else:
    #    env = create_garage_env()
    
    for episode in range(20):
        print("***********************************")
        print('Episode: {}'.format(episode))
        #step = 0
        env.reset()
        print('-----------------------------')
        print('Reset')
        env.print_agent_state()
        #env.print_collide_info()
        print("Goal position: %s"%(env.goal_position))
        if env.config.get("goal_conditioned"):
        #if env.config.get("random_goal"):
            goal = env.get_goal_observation()
            print("Goal observation: "+str(goal.shape))
            if env.config.get("goal_format") == "imagegoal":
                save_goal_image(goal, episode)
        #print("Goal observation: %s"%(env.get_goal_observation()))
        
        
        print('-----------------------------')

       
        for i in range(100):  # max steps per episode
            action = env.action_space.sample()
            #if gym_env:
            obs, reward, done, info = env.step(action)
            # else:
            #     env_step = env.step(action)
            #     obs = env_step.observation
            #     reward = env_step.reward
            #     info = env_step.env_info
            #     done = env_step.terminal

            print('-----------------------------')
            print('step: %d'%(i+1))
            # print('action: %s'%(env.action_index_to_name(action)))
            #print('observation: %s, %s'%(str(obs.shape), str(type(obs))))
            #print('observation: %s'%(str(type(obs))))
            #print('observation: %s'%(obs.keys()))
            print(obs["color_sensor"].shape)
            print(obs["state_sensor"])
            print(obs["pointgoal"])
            #print(obs["imagegoal"].shape)
            #print(obs["color_sensor"])
            print(obs["depth_sensor"].shape)
            #env.print_agent_state()
            #print('agent angle [euler]: '+str(env.get_agent_rotation_euler()))
            #print(obs["depth_sensor"])
            # #print(obs["semantic_sensor"].shape)
            # #print(obs)
            # print('agent rotation [euler]: '+str(env.get_agent_rotation_euler()))
            print('reward: %f'%(reward))
            # print('done: '+str(done))
            # #env.print_collide_info()
            
            # # Garage env needs set render mode explicitly
            if "color_sensor" in obs.keys():
                render_obs = env.render(mode="color_sensor")
            else:    
                render_obs = env.render(mode="depth_sensor")
            # print('render observation: %s, %s'%(str(render_obs.shape), str(type(render_obs))))

            # env.print_agent_state()
            # if env.config.get("goal_conditioned"):
            # print("Goal observation: "+str(env.get_goal_observation()))
            #      print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            # env.get_current_scene_light_vector()
            # print('state: %s'%(env.get_agent_state_xyz_euler()))
            # print(env.train_state_count_dict.state_to_key(env.get_agent_state_xyz_euler()))
            #print(env.get_agent_state_xyz_euler().shape)
            #print('state: %s'%(env.get_agent_state_xyz_euler()[:3]))
            #print('state: %s'%(env.get_agent_state_xyz_euler()[3:]))
            #print('-------------------------------')
            

            #step += 1
            if done:
                break
               
        # env.episode_state_count_dict.print()
        # print('----------------------------------')
        # env.train_state_count_dict.print()
        # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")       
        print('Episode finished after {} timesteps.'.format(i+1))
        print('Collision count: %d'%(env.get_current_collision_counts()))
    
    print('-----------------------------')
    # if gym_env:
    #     print("Gym env")
    # else:
    #     print("Garage env")    
    print("Action space: %s"%(env.action_space)) 
    print("Observation space: %s"%(env.observation_space)) 
    print("Goal observation space: %s"%(env.get_goal_observation_space()))
    print("Combined goal observation space: %s"%(env.get_combined_goal_obs_space()))
    #print(np.amin(env.observation_space.low))
    #print(np.amax(env.observation_space.high))
    bound = env.get_env_bounds()
    print("Scene range: " + str(bound[1]-bound[0]))
    #if not gym_env:
    #print("Env spec: " + str(env.spec))
    print('-----------------------------') 

    print(env.get_sim_config().scene_dataset_config_file)
    print(env.get_sim_config().scene_id)
    
    env.close() 
    

def test_shortest_path(start_point=None, end_point=None):
    env =  NavEnv()
    env.shortest_path(start_point=start_point, end_point=end_point)
    print(env.get_env_bounds())

def test_rollout_storage():
    env =  NavEnv()
    assert env.sensor_suite.dictionary_observation_space
    print(env.observation_space.spaces)
    print(env.observation_space.spaces['color_sensor'].shape[:2])
    print(env.observation_space.spaces['depth_sensor'].shape[:2])
    print(env.observation_space.spaces['color_sensor'].shape[:2] == env.observation_space.spaces['depth_sensor'].shape[:2])
    print('color_sensor' in env.observation_space.spaces)
    print(env.action_space.__class__.__name__)
    print(env.action_space.shape)
    print(env.action_space)
    print("Goal observation space: %s"%(env.get_goal_observation_space()))
    print("Goal observation space dim: %s"%(len(env.get_goal_observation_space().shape)))

def test_stop_action():
    env =  NavEnv()
    for episode in range(2):
        print("***********************************")
        print('Episode: {}'.format(episode))
        step = 0
        env.reset()
        print('-----------------------------')
        print('Reset')
        env.print_agent_state()
        #env.print_collide_info()
        print("Goal position: %s"%(env.goal_position))
        #print("Goal observation: "+str(env.get_goal_observation().shape))
        #print("Goal observation: %s"%(env.get_goal_observation()))
        
        
        print('-----------------------------')
        for i in range(3):  # max steps per episode
            obs, reward, done, info = env.step(0)
            
            print('-----------------------------')
            print('step: %d'%(i+1))
            print('action: %s'%(env.action_index_to_name(0)))
            env.print_agent_state()
            print('agent angle [euler]: '+str(env.get_agent_rotation_euler()))
            print('reward: %f'%(reward))
            print('done: '+str(done))
            #env.print_collide_info()
            
            print('-------------------------------')

            step += 1
            if done:
                break
               
        print('Episode finished after {} timesteps.'.format(step))
        print('Collision count: %d'%(env.get_current_collision_counts()))

        

if __name__ == "__main__":    
    test_env("replica_nav.yaml")
    #test_env("pointgoal_baseline.yaml")
    #test_shortest_path(start_point=[0,0,0], end_point=[1,0,0])
    #check_coordinate_system()
    #test_rollout_storage()
    #test_stop_action()

    