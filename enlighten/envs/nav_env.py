import math
import os
import numpy as np
from numpy import ndarray
import sys
import random
from matplotlib import pyplot as plt
import gym
from gym import spaces

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
    Optional,
    Sequence,
    Set,
    Union,
    cast,
    Iterable
)

from collections import OrderedDict
from enum import Enum


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

        print("========================")
        print(self.normalize_depth)
        print(self.min_depth_value)    
        print(self.max_depth_value) 
        print("========================")

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
        
        return obs

class Observations(Dict[str, Any]):
    r"""Dictionary containing sensor observations"""

    def __init__(
        self, sensors: Dict[str, HabitatSensor], *args: Any, **kwargs: Any
    ) -> None:
        """Constructor

        :param sensors: list of sensors whose observations are fetched and
            packaged.
        """

        data = [
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

    def __init__(self, sensors: Iterable[HabitatSensor]) -> None:
        """Constructor

        :param sensors: list containing sensors for the environment, uuid of
            each sensor must be unique.
        """
        self.sensors = OrderedDict()
        ordered_spaces: OrderedDict[str, gym.Space] = OrderedDict()
        for sensor in sensors:
            assert (
                sensor.uuid not in self.sensors
            ), "'{}' is duplicated sensor uuid".format(sensor.uuid)
            self.sensors[sensor.uuid] = sensor
            ordered_spaces[sensor.uuid] = sensor.observation_space

        # get a dictionary observation space   
        self.observation_spaces = gym.spaces.Dict(spaces=ordered_spaces)

    def get(self, uuid: str) -> HabitatSensor:
        return self.sensors[uuid]

    def get_observations(self, *args: Any, **kwargs: Any) -> Observations:
        r"""Collects data from all sensors and returns it packaged inside
        :ref:`Observations`.
        """
        return Observations(self.sensors, *args, **kwargs)


class NavEnv(gym.Env):
    r"""Base gym navigation environment
    """

    def __init__(self, config_file=os.path.join(config_path, "navigate_with_flashlight.yaml")):
        self.config = parse_config(config_file)
        self.sim_config = self.create_sim_config()
        # create simulator
        self.sim = habitat_sim.Simulator(self.sim_config) 
        # create agent
        self.agent = self.sim.initialize_agent(agent_id=self.sim._default_agent_id)
        # initialize agent
        self.set_agent_state(new_position=self.config.get('agent_initial_position'), new_rotation=self.config.get('agent_initial_rotation'), is_initial=True)
        # create gym action space and observation space
        self.action_space = self.create_gym_action_space()
       

    def create_sim_config(self):
        # simulator configuration
        sim_config = habitat_sim.SimulatorConfiguration()
        
        # set scene path
        sim_config.scene_id = self.config.get('scene_id')

        # enable physics
        sim_config.enable_physics = True

        # enable scene lighting change
        sim_config.override_scene_light_defaults = True

        # sensor configuration
        sensor_specs = self.create_sim_observation_space()
        
        # agent configuration
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.action_space = self.create_sim_action_space()
        agent_cfg.sensor_specifications = sensor_specs

        return habitat_sim.Configuration(sim_config, [agent_cfg])

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
    def create_sim_observation_space(self):
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

        self.sensor_suite = SensorSuite(sensors)    

        return sensor_specs 

    def get_gym_observation_space(self):
        return self.sensor_suite.observation_spaces
    
    def get_sim_sensors(self):
        return self.sim._sensors

    def get_agent_state(self):
        agent_state = self.agent.get_state()
        # position: [x,y,z]
        # rotation: quarternion [x,y,z,q]
        print("agent_state: position", agent_state.position, "rotation", agent_state.rotation)
        return agent_state

    def set_agent_state(self, new_position, new_rotation=None, is_initial=False):
        new_agent_state = habitat_sim.AgentState()
        # global system, must be casted to float32
        new_agent_state.position = np.array(new_position, dtype="float32")  

        #print("===================")
        #print(new_agent_state.position)
        #print(type(new_agent_state.position))
        #print("===================")

        if new_rotation is not None:
            #print("===================")
            #print(get_rotation_quat(new_rotation))
            #print("===================")
            new_agent_state.rotation = get_rotation_quat(np.array(new_rotation, dtype="float32"))
        self.agent.set_state(new_agent_state, is_initial=is_initial)


    def step(self, action):
        action_name = self.action_mapping[action]
        sim_obs = self.sim.step(action_name)
        
        observations = self.sensor_suite.get_observations(sim_obs)
        return observations               

    def reset(self):
        self.initialize_agent()

    def seed(self, seed):    
        self.sim.seed(seed)

    #def render(self, mode):    

if __name__ == "__main__":    
    env =  NavEnv()
    #print(env.sim_config)
    #print(env.get_sim_action_space())
    print("===========================")
    print(env.get_gym_action_space())
    env.get_agent_state()
    #print(env.get_sim_sensors()["color_sensor"])
    print(env.get_gym_observation_space())
    env.step(0)
    print("===========================")