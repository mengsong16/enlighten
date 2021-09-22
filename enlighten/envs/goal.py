from typing import Any, List, Optional, Tuple

import attr
import numpy as np
from gym import spaces

from enlighten.envs import HabitatSensor, Dictionary_Observations
from enlighten.utils.utils import parse_config
from enlighten.utils.utils import quaternion_rotate_vector, cartesian_to_polar, quaternion_from_coeff

class PointGoal(HabitatSensor):
    r"""Sensor for PointGoal observations which are used in PointGoal Navigation.

    For the agent in simulator the forward direction is along negative-z.
    In polar coordinate format the angle returned is azimuth to the goal.

    Args:
        _goal_coord_system: coordinate system for specifying the goal which can be done
            in cartesian or polar coordinates.
        _goal_dimension: number of dimensions used to specify the goal
    """

    def __init__(self, config, env, *args: Any, **kwargs: Any):
        self._goal_coord_system = config.get("goal_coord_system")
        assert self._goal_coord_system in ["cartesian", "polar"], "goal coordinate system should be cartesian or polar"

        self._goal_dimension = config.get("goal_dimension")
        assert self._goal_dimension in [2, 3], "goal dimension should be 2 or 3"

        self.env = env

        super().__init__(uuid="pointgoal", config=config)
    

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(self._goal_dimension,),
            dtype=np.float32,
        )

    def _compute_pointgoal(self, source_position, source_rotation, goal_position):
        direction_vector = goal_position - source_position
        # use source rotation as x axis
        direction_vector_agent = quaternion_rotate_vector(
            source_rotation.inverse(), direction_vector
        )

        if self._goal_coord_system == "polar":
            # rho, -phi
            if self._goal_dimension == 2:
                rho, phi = cartesian_to_polar(
                    -direction_vector_agent[2], direction_vector_agent[0]
                )
                return np.array([rho, -phi], dtype=np.float32)
            #  rho, -phi, theta   
            else:
                _, phi = cartesian_to_polar(
                    -direction_vector_agent[2], direction_vector_agent[0]
                )
                theta = np.arccos(
                    direction_vector_agent[1]
                    / np.linalg.norm(direction_vector_agent)
                )
                rho = np.linalg.norm(direction_vector_agent)

                return np.array([rho, -phi, theta], dtype=np.float32)
        else:
            if self._goal_dimension == 2:
                return np.array(
                    [-direction_vector_agent[2], direction_vector_agent[0]],
                    dtype=np.float32,
                )
            else:
                return np.array(direction_vector_agent, dtype=np.float32)

    # vector from current position to goal position
    def get_observation(
        self, 
        goal_position, # [x,y,z] in world coord system 
        *args: Any, 
        **kwargs: Any
    ):
        agent_state = self.env.get_agent_state()
        agent_position = agent_state.position
        rotation_world_agent = agent_state.rotation
        goal_world_position = np.array(goal_position, dtype=np.float32)

        return self._compute_pointgoal(
            agent_position, rotation_world_agent, goal_world_position
        )

# image goal can only be RGB image
class ImageGoal(HabitatSensor):
    r"""Sensor for ImageGoal observations which are used in ImageGoal Navigation.

    RGBSensor needs to be one of the Simulator sensors.
    This sensor return the rgb image taken from the goal position to reach with
    random rotation.

    Args:
        env: reference to the environment for calculating task observations.
        config: config for the ImageGoal sensor.
    """

    def __init__(self, config, env, *args: Any, **kwargs: Any):
        self.env = env

        assert 'color_sensor' in self.env.observation_space.spaces, "Image goal requires one RGB sensor, but not detected"

        self._current_image_goal = None

        super().__init__(uuid="imagegoal", config=config)

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return self.env.observation_space.spaces["color_sensor"]

    # goal_position: [x, y, z]
    # goal_azimuth: scalar in [0, 2*pi]
    def _get_image_goal(self, goal_position, goal_azimuth):
        goal_world_position = np.array(goal_position, dtype=np.float32)
        
        # quarternion
        goal_rotation_quart = [0, np.sin(goal_azimuth / 2), 0, np.cos(goal_azimuth / 2)]
        goal_observation = self.env.get_observations_at(position=goal_world_position, rotation=goal_rotation_quart, keep_agent_at_new_pose=False)
        
        return goal_observation["color_sensor"]

    def get_observation(
        self, 
        goal_position, # [x,y,z] in world coord system 
        goal_azimuth=0, # scalar in [0, 2*pi]
        *args: Any, 
        **kwargs: Any
    ):

        self._current_image_goal = self._get_image_goal(goal_position, goal_azimuth)

        return self._current_image_goal