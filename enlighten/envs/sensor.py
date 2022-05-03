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
from enlighten.utils.geometry_utils import quaternion_rotate_vector, cartesian_to_polar, quaternion_from_coeff


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
        self, sim_obs: Dict[str, Union[ndarray, bool, "Tensor"]],
        *args: Any, **kwargs: Any
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
        self, sim_obs: Dict[str, Union[ndarray, bool, "Tensor"]],
        *args: Any, **kwargs: Any
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
        self, sim_obs: Dict[str, Union[ndarray, bool, "Tensor"]],
        *args: Any, **kwargs: Any
    ) -> VisualObservation:
        obs = cast(Optional[VisualObservation], sim_obs.get(self.uuid, None))
        obs = np.expand_dims(obs, axis=2)
        
        return obs

class StateSensor(HabitatSensor):
    r"""Sensor for state observations which are used in PointGoal Navigation.

    For the agent in simulator the forward direction is along negative-z.
    In 2D polar coordinate format the angle returned is azimuth from the start.
    The coordinate system is the local coordinate system when the agent is at the start location of current episode.

    Args:
        _state_coord_system: coordinate system for specifying the goal which can be done
            in cartesian or polar coordinates.
        _state_dimension: number of dimensions used to specify the goal
    """

    def __init__(self, config):
        self._state_coord_system = config.get("state_coord_system")
        assert self._state_coord_system in ["cartesian", "polar"], "state coordinate system should be cartesian or polar"

        self._state_dimension = config.get("state_dimension")
        assert self._state_dimension in [2, 3], "state dimension should be 2 or 3"

        self.state_relative_to_origin = config.get("state_relative_to_origin")

        if self.state_relative_to_origin:
            print("====> state relative to 0")
        else:
            print("====> state relative to start")    

        super().__init__(uuid="state_sensor", config=config)
    

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(self._state_dimension,),
            dtype=np.float32,
        )

    def _compute_coordinate_representation(self, source_position, source_rotation, state_position):
        # use source local coordinate system as global coordinate system
        # step 1: align origin 
        direction_vector = state_position - source_position
        # step 2: align axis 
        direction_vector = quaternion_rotate_vector(
            source_rotation.inverse(), direction_vector
        )

        if self._state_coord_system == "polar":
            # 2D movement: r, -phi
            # -phi: angle relative to positive z axis (i.e reverse to robot forward direction)
            # -phi: azimuth, around y axis
            if self._state_dimension == 2:
                rho, phi = cartesian_to_polar(
                    -direction_vector[2], direction_vector[0]
                )
                return np.array([rho, -phi], dtype=np.float32)
            #  3D movement: r, -phi, theta 
            #  -phi: azimuth, around y axis
            #  theta: around z axis   
            else:
                # -z, x --> x, y --> -\phi
                _, phi = cartesian_to_polar(
                    -direction_vector[2], direction_vector[0]
                )
                theta = np.arccos(
                    direction_vector[1]
                    / np.linalg.norm(direction_vector)
                )
                # r = l2 norm
                rho = np.linalg.norm(direction_vector)

                return np.array([rho, -phi, theta], dtype=np.float32)
        else:
            # 2D movement : [-z,x]
            # reverse the direction of z axis towards robot forward direction
            if self._state_dimension == 2:
                return np.array(
                    [-direction_vector[2], direction_vector[0]],
                    dtype=np.float32,
                )
            # 3D movement: [x,y,z] 
            # do not reverse the direction of z axis      
            else:
                return np.array(direction_vector, dtype=np.float32)

    def get_observation(self, sim_obs, env):
        start_position = env.start_position # [x,y,z] in world coord system
        start_rotation = env.start_rotation # quarternion
        current_position = env.agent.get_state().position # [x,y,z] in world coord system
        
        if self.state_relative_to_origin:
            return self._compute_coordinate_representation(
                    np.array([0,0,0], dtype="float32"), np.quaternion(1,0,0,0), np.array(current_position, dtype=np.float32)
                )
        else:
            return self._compute_coordinate_representation(
                    start_position, start_rotation, np.array(current_position, dtype=np.float32)
                )        
