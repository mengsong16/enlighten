import math
import os
import numpy as np
from numpy import euler_gamma, ndarray
import sys
import random
from matplotlib import pyplot as plt
import gym
from gym import spaces
import habitat_sim
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

from enlighten.utils.config_utils import parse_config
from enlighten.utils.video_utils import BGR_mode

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
        #self.config = parse_config(config)
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

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.uuid

# A dictionary of Observations
# modified from Observations in habitat lab 
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
   

class HabitatSimRGBSensor(HabitatSensor):
    _get_default_spec = habitat_sim.CameraSensorSpec
    sim_sensor_type = habitat_sim.SensorType.COLOR

    RGBSENSOR_DIMENSION = 3

    def __init__(self, config) -> None:
        super().__init__(uuid="color_sensor", config=config)

        self.RGB2BGR = BGR_mode(config)

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

    # [0, 255] RGB
    def get_observation(
        self, sim_obs: Dict[str, Union[ndarray, bool, "Tensor"]]
    ) -> VisualObservation:
        obs = cast(Optional[VisualObservation], sim_obs.get(self.uuid, None))
        
        # remove alpha channel
        obs = obs[:, :, : self.RGBSENSOR_DIMENSION]  # type: ignore[index]

        # RGB to BGR
        if self.RGB2BGR:
            obs = obs[:,:,[2,1,0]]
            
        return obs


class HabitatSimDepthSensor(HabitatSensor):
    _get_default_spec = habitat_sim.CameraSensorSpec
    sim_sensor_type = habitat_sim.SensorType.DEPTH

    min_depth_value: float
    max_depth_value: float

    def __init__(self, config) -> None:
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
        super().__init__(uuid="depth_sensor", config=config)

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

    def __init__(self, config) -> None:
        super().__init__(uuid="semantic_sensor", config=config)

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