import math
import os
import numpy as np
from numpy import ndarray
import sys
import random
from matplotlib import pyplot as plt
import gym
from gym import spaces
from gym.envs.classic_control.rendering import SimpleImageViewer

import random

import habitat_sim
from habitat_sim.gfx import LightInfo, LightPositionModel, DEFAULT_LIGHTING_KEY, NO_LIGHT_KEY
from habitat_sim.utils.common import quat_from_angle_axis

from enlighten.utils.utils import parse_config, get_rotation_quat 
from enlighten.utils.path import *

import abc

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

from garage import Environment, EnvSpec, EnvStep, StepType
from garage.envs import GymEnv
from PIL import Image
from enlighten.utils.viewer import MyViewer
import cv2
from skimage.color import label2rgb 

VisualObservation = Union[np.ndarray]


class HabitatSensor(metaclass=abc.ABCMeta):
    r"""Represents a sensor that provides data from the environment to agent.

    :data uuid: universally unique id.
    :data observation_space: ``gym.Space`` object corresponding to observation
        of sensor.

    The user of this class needs to implement the get_observation method and
    the user is also required to set the below attributes:
    """

    uuid: str
    observation_space: gym.Space

    def __init__(self, uuid, config, *args: Any, **kwargs: Any) -> None:
        self.config = config
        self.uuid = uuid
        
        self.observation_space = self._get_observation_space(*args, **kwargs)

    def _get_observation_space(self, *args: Any, **kwargs: Any) -> gym.Space:
        raise NotImplementedError

    @abc.abstractmethod
    def get_observation(self, *args: Any, **kwargs: Any) -> Any:
        r"""
        Returns:
            current observation for Sensor.
        """
        raise NotImplementedError


class HabitatSimRGBSensor(HabitatSensor):
    _get_default_spec = habitat_sim.CameraSensorSpec
    sim_sensor_type = habitat_sim.SensorType.COLOR

    RGBSENSOR_DIMENSION = 3

    def __init__(self, uuid, config) -> None:
        super().__init__(uuid=uuid, config=config)

    def _get_observation_space(self, *args: Any, **kwargs: Any) -> gym.spaces.Box:
        return gym.spaces.Box(
            low=0,
            high=255,
            shape=(
                int(self.config.get("image_height")), 
                int(self.config.get("image_width")),
                self.RGBSENSOR_DIMENSION,
            ),
            dtype=np.uint8,
        )

    def get_observation(
        self, sim_obs: Dict[str, Union[ndarray, bool, "Tensor"]]
    ) -> VisualObservation:
        obs = cast(Optional[VisualObservation], sim_obs.get(self.uuid, None))
        
        # remove alpha channel
        obs = obs[:, :, : self.RGBSENSOR_DIMENSION]  # type: ignore[index]
        return obs


class HabitatSimDepthSensor(HabitatSensor):
    _get_default_spec = habitat_sim.CameraSensorSpec
    sim_sensor_type = habitat_sim.SensorType.DEPTH

    min_depth_value: float
    max_depth_value: float

    def __init__(self, uuid, config) -> None:
        self.normalize_depth = config.get("normalize_depth")

        if self.normalize_depth:
            self.min_depth_value = 0.0
            self.max_depth_value = 1.0
        else:
            self.min_depth_value = float(config.get("min_depth"))
            self.max_depth_value = float(config.get("max_depth"))

        #print("========================")
        #print(self.normalize_depth)
        #print(self.min_depth_value)    
        #print(self.max_depth_value) 
        #print("========================")

        # must be after initialize min and max depth value
        super().__init__(uuid=uuid, config=config)

    def _get_observation_space(self, *args: Any, **kwargs: Any) -> gym.spaces.Box:
        return gym.spaces.Box(
            low=self.min_depth_value,
            high=self.max_depth_value,
            shape=(int(self.config.get("image_height")), int(self.config.get("image_width")), 1),
            dtype=np.float32,
        )

    def get_observation(
        self, sim_obs: Dict[str, Union[ndarray, bool, "Tensor"]]
    ) -> VisualObservation:

        obs = cast(Optional[VisualObservation], sim_obs.get(self.uuid, None))
        
        if isinstance(obs, np.ndarray):
            obs = np.clip(obs, self.min_depth_value, self.max_depth_value)

            obs = np.expand_dims(
                obs, axis=2
            )  # make depth observation a 3D array
        else:
            obs = obs.clamp(self.min_depth_value, self.max_depth_value)  # type: ignore[attr-defined]

            obs = obs.unsqueeze(-1)  # type: ignore[attr-defined]

        if self.normalize_depth:
            # normalize depth observation to [0, 1]
            obs = (obs - self.min_depth_value) / (self.max_depth_value - self.min_depth_value)

        return obs


class HabitatSimSemanticSensor(HabitatSensor):
    _get_default_spec = habitat_sim.CameraSensorSpec
    sim_sensor_type = habitat_sim.SensorType.SEMANTIC

    def __init__(self, uuid, config) -> None:
        super().__init__(uuid=uuid, config=config)

    def _get_observation_space(self, *args: Any, **kwargs: Any) -> gym.spaces.Box:
        return gym.spaces.Box(
            low=np.iinfo(np.uint32).min,
            high=np.iinfo(np.uint32).max,
            shape=(int(self.config.get("image_height")), int(self.config.get("image_width")), 1),
            dtype=np.uint32,
        )

    def get_observation(
        self, sim_obs: Dict[str, Union[ndarray, bool, "Tensor"]]
    ) -> VisualObservation:
        obs = cast(Optional[VisualObservation], sim_obs.get(self.uuid, None))
        obs = np.expand_dims(obs, axis=2)
        
        return obs

# A dictionary data structure
class Dictionary_Observations(Dict[str, Any]):
    r"""Dictionary containing sensor observations"""

    def __init__(
        self, sensors: Dict[str, HabitatSensor], *args: Any, **kwargs: Any
    ) -> None:
        """Constructor

        :param sensors: list of sensors whose observations are fetched and
            packaged.
        """

        data = [
            # sensor.get_observation need parameter sim_obs
            (uuid, sensor.get_observation(*args, **kwargs))
            for uuid, sensor in sensors.items()
        ]
        super().__init__(data)



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
            return self.get_array_observations(*args, **kwargs)    

    def get_specific_observation(self, uuid: str, *args, **kwargs):
        assert uuid in self.sensors, "mode {} sensor is not active".format(uuid)

        # sensor.get_observation need parameter sim_obs
        return self.sensors[uuid].get_observation(*args, **kwargs)


    # Return a numpy array concatenating all observation modes
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

# gym.Env definition: https://github.com/openai/gym/blob/267916b9d268c37cc948bafe35606c665aac53ac/gym/core.py

class NavEnv(gym.Env):
    r"""Base gym navigation environment
    """

    def __init__(self, config_file=os.path.join(config_path, "navigate_with_flashlight.yaml")):
        self.config = parse_config(config_file)
    
        # create simulator configuration
        self.sim_config, sensors = self.create_sim_config()
        # create simulator
        self.sim = habitat_sim.Simulator(self.sim_config) 
        # create gym observation space
        self.observation_space = self.create_gym_observation_space(sensors)
        # create agent and set agent's initial state
        self.agent = self.sim.initialize_agent(agent_id=self.sim._default_agent_id, initial_state=self.create_agent_state(new_position=self.config.get('agent_initial_position'), new_rotation=self.config.get('agent_initial_rotation')))
        # create gym action space
        self.action_space = self.create_gym_action_space()
        # viewer
        self.viewer = None
        
    def create_sim_config(self):
        # simulator configuration
        sim_config = habitat_sim.SimulatorConfiguration()
        
        # set scene path
        sim_config.scene_id = self.config.get('scene_id')

        # enable physics
        sim_config.enable_physics = True

        # enable scene lighting change
        sim_config.override_scene_light_defaults = True

        # sensors and sensor specifications
        sensor_specs, sensors = self.create_sensors_and_sensor_specs()
        
        # agent configuration
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.action_space = self.create_sim_action_space()
        agent_cfg.sensor_specifications = sensor_specs

        return habitat_sim.Configuration(sim_config, [agent_cfg]), sensors

    def create_sim_action_space(self):
        self.action_mapping = ["move_forward", "turn_left", "turn_right", "look_up", "look_down"]
        action_space = {
            "move_forward": habitat_sim.agent.ActionSpec(
                "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)  # move -a meter along z axis (global)
            ),
            "turn_left": habitat_sim.agent.ActionSpec(
                "turn_left", habitat_sim.agent.ActuationSpec(amount=10.0) # rotate a degree along y axis (rotate local) 
            ),
            "turn_right": habitat_sim.agent.ActionSpec(
                "turn_right", habitat_sim.agent.ActuationSpec(amount=10.0) # rotate a degree along y axis (rotate local)
            ),
            "look_up": habitat_sim.agent.ActionSpec(
                "look_up", habitat_sim.agent.ActuationSpec(amount=10.0) # rotate a degree along x axis (rotate local)
            ),
            "look_down": habitat_sim.agent.ActionSpec(
                "look_down", habitat_sim.agent.ActuationSpec(amount=10.0) # rotate -a degree along x axis (rotate local)
            )
        }
        return action_space

    def get_sim_action_space(self):
        return self.sim_config.agents[self.sim._default_agent_id].action_space

    def create_gym_action_space(self):
        return spaces.Discrete(len(self.get_sim_action_space())) 

    def get_gym_action_space(self):
        return self.action_space  

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
            sensors.append(HabitatSimRGBSensor(uuid="color_sensor", config=self.config))

        if self.config.get("depth_sensor"):
            depth_sensor_spec = habitat_sim.CameraSensorSpec()
            depth_sensor_spec.uuid = "depth_sensor"
            depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
            depth_sensor_spec.resolution = [image_height, image_width]
            depth_sensor_spec.postition = [0.0, sensor_height, 0.0]
            depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
            sensor_specs.append(depth_sensor_spec)
            sensors.append(HabitatSimDepthSensor(uuid="depth_sensor", config=self.config))

        if self.config.get("semantic_sensor"):
            semantic_sensor_spec = habitat_sim.CameraSensorSpec()
            semantic_sensor_spec.uuid = "semantic_sensor"
            semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
            semantic_sensor_spec.resolution = [image_height, image_width]
            semantic_sensor_spec.postition = [0.0, sensor_height, 0.0]
            semantic_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
            sensor_specs.append(semantic_sensor_spec) 
            sensors.append(HabitatSimSemanticSensor(uuid="semantic_sensor", config=self.config))

        # set render mode according to sensors
        self.set_render_mode()

        return sensor_specs, sensors 

    def create_gym_observation_space(self, sensors):
        self.sensor_suite = SensorSuite(sensors, self.config.get('dictionary_observation_space')) 
        return self.sensor_suite.observation_spaces    

    def get_gym_observation_space(self):
        return self.observation_space
    
    def get_sim_sensors(self):
        return self.sim._sensors

    def get_agent_state(self):
        agent_state = self.agent.get_state()
        # position: [x,y,z]
        # rotation: quarternion [x,y,z,q]
        
        return agent_state

    def print_agent_state(self):  
        agent_state = self.agent.get_state() 

        print("agent state: position: ", agent_state.position, ", rotation: ", agent_state.rotation) 

    def create_agent_state(self, new_position, new_rotation=None):
        new_agent_state = habitat_sim.AgentState()
        # global system, must be casted to float32
        new_agent_state.position = np.array(new_position, dtype="float32")  

        if new_rotation is not None:
            new_agent_state.rotation = get_rotation_quat(np.array(new_rotation, dtype="float32"))
        
        return new_agent_state

    def set_agent_state(self, new_position, new_rotation=None, is_initial=False):
        self.agent.set_state(self.create_agent_state(new_position=new_position, new_rotation=new_rotation), is_initial=is_initial)

    def action_index_to_name(self, index):
        return self.action_mapping[index]

    def step(self, action):
        action_name = self.action_mapping[action]
        sim_obs = self.sim.step(action_name)
        self.did_collide = self.extract_collisions(sim_obs)
        if self.did_collide:
            self.collision_count_per_episode += 1
        # sim_obs includes all modes
        obs = self.sensor_suite.get_observations(sim_obs)

        reward = 0

        #done = random.choice([True, False])
        done = False

        info = {}
        return obs, reward, done, info              

    def reset(self):
        # will reset agent to its initial pose
        sim_obs = self.sim.reset()
        # sim_obs includes all modes
        obs = self.sensor_suite.get_observations(sim_obs)

        self.did_collide = self.extract_collisions(sim_obs)
        self.collision_count_per_episode = 0

        return obs

    def seed(self, seed):    
        self.sim.seed(seed)
    
    def get_render_mode(self):
        return self.metadata['render.modes']

    def set_render_mode(self):
        self.metadata['render.modes'] = ['color_sensor', 'depth_sensor', 'semantic_sensor']  

    def render(self, mode: str = "color_sensor") -> Any:
        r"""
        Args:
            mode: sensor whose observation is used for returning the frame,
                eg: "rgb", "depth", "semantic"

        Returns:
            rendered frame according to the mode
        """

        assert self.config.get(mode), "render mode should be active in the config file"
        # create viewer
        if self.viewer is None:
            #self.viewer = SimpleImageViewer()
            self.viewer = MyViewer()

        sim_obs = self.sim.get_sensor_observations()
        obs = self.sensor_suite.get_specific_observation(uuid=mode, sim_obs=sim_obs)

        if not isinstance(obs, np.ndarray):
            # If it is not a numpy array, it is a torch tensor
            # The function expects the result to be a numpy array
            obs = obs.to("cpu").numpy()

        # show image in viewer
        if obs.shape[2] == 3:
            img = np.asarray(obs).astype(np.uint8)
            # RGB
            self.viewer.imshow(img)
        elif obs.shape[2] == 1:
            if mode == "depth_sensor":
                img = np.asarray(obs * 255).astype(np.uint8)
                # not the same with dstack the single channel
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                self.viewer.imshow(img)
            # label image    
            else:
                img = np.asarray(np.squeeze(obs, axis=2)).astype(np.uint8)
                img = label2rgb(label=img)
                # float to int
                img = np.asarray(img).astype(np.uint8)
                #print("*****************")
                #print(np.ptp(img))
                self.viewer.imshow(img)
        else:
            print("Error: image channel is neither 1 nor 3!")
        
    
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
        if self.did_collide is None:
            print("collide: Unknown")
        else:
            print('collide: %s'%(str(self.did_collide)))         
    
    def close(self):
        if self.viewer is not None:
            self.viewer.close()

def test_env(gym_env=True):
    if gym_env:
        env =  NavEnv()
    else:
        env = GymEnv(NavEnv())     
        assert isinstance(env.spec, EnvSpec)
    
    
    for episode in range(20):
        print("***********************************")
        print('Episode: {}'.format(episode))
        step = 0
        env.reset()
        print('-----------------------------')
        print('Reset')
        env.print_agent_state()
        env.print_collide_info()
        print('-----------------------------')

        for i in range(50):  # max steps per episode
            action = env.action_space.sample()
            if gym_env:
                obs, reward, done, info = env.step(action)
            else:
                env_step = env.step(action)
                obs = env_step.observation
                reward = env_step.reward
                info = env_step.env_info
                done = env_step.terminal
            print('-----------------------------')
            print('step: %d'%(i+1))
            print('action: %s'%(env.action_index_to_name(action)))
            print('observation: %s, %s'%(str(obs.shape), str(type(obs))))
            #print(obs["color_sensor"].shape)
            #print(obs["depth_sensor"].shape)
            #print(obs["semantic_sensor"].shape)
            env.print_agent_state()
            print('reward: %f'%(reward))
            env.print_collide_info()
            
            # Garage env needs set render mode explicitly
            render_obs = env.render(mode="depth_sensor")
            print('render observation: %s, %s'%(str(render_obs.shape), str(type(render_obs))))
            print('-------------------------------')
            
            step += 1
            if done:
                break
               
        print('Episode finished after {} timesteps.'.format(step))
        print('Collision count: %d'%(env.collision_count_per_episode))
    
    print('-----------------------------')
    if gym_env:
        print("Gym env")
    else:
        print("Garage env")    
    print("Action space: %s"%(env.action_space)) 
    print("Observation space: %s"%(env.observation_space)) 
    #print(np.amin(env.observation_space.low))
    #print(np.amax(env.observation_space.high))
    print('-----------------------------') 
    
    env.close() 

if __name__ == "__main__":    
    test_env(gym_env=False)