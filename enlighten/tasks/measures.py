from typing import Any, List, Optional, Tuple, Dict, Iterable, Union

import attr
import numpy as np
from gym import spaces

from enlighten.datasets.pointnav_dataset import NavigationEpisode

from habitat import logger

from collections import OrderedDict


class Measure:
    r"""Represents a measure that provides measurement on top of environment
    and task.

    :data uuid: universally unique id.
    :data _metric: metric for the :ref:`Measure`, this has to be updated with
        each :ref:`step() <env.Env.step()>` call on :ref:`env.Env`.

    This can be used for tracking statistics when running experiments. The
    user of this class needs to implement the :ref:`reset_metric()` and
    :ref:`update_metric()` method and the user is also required to set the
    :ref:`uuid <Measure.uuid>` and :ref:`_metric` attributes.

    .. (uuid is a builtin Python module, so just :ref:`uuid` would link there)
    """

    _metric: Any
    cls_uuid: str
    #cls_uuid = "measure"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        #self.cls_uuid = self._get_uuid(*args, **kwargs)
        self._metric = None

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        #raise NotImplementedError
        return self.cls_uuid

    def reset_metric(self, *args: Any, **kwargs: Any) -> None:
        r"""Reset :ref:`_metric`, this method is called from :ref:`env.Env` on
        each reset.
        """
        raise NotImplementedError

    def update_metric(self, *args: Any, **kwargs: Any) -> None:
        r"""Update :ref:`_metric`, this method is called from :ref:`env.Env`
        on each :ref:`step() <env.Env.step()>`
        """
        raise NotImplementedError

    def get_metric(self):
        r"""..

        :return: the current metric for :ref:`Measure`.
        """
        return self._metric

    def print_metric(self):
        string = str(self.cls_uuid) + ":" + str(self._metric)
        print(string)
        return string+"\n"

    def set_metric_to_zero(self):
        self._metric = 0   

    def set_metric(self, new_value):
        self._metric = new_value     

class DistanceToGoal(Measure):
    """The measure calculates a distance towards the goal."""

    # class attribute must have value
    cls_uuid = "distance_to_goal"
    def __init__(self, env, config, *args: Any, **kwargs: Any):
        self._previous_position: Optional[Tuple[float, float, float]] = None
        self._env = env
        self._config = config

        super().__init__()

    
    def reset_metric(self, episode=None, *args: Any, **kwargs: Any):
        self._previous_position = None
        self._metric = None
        
        self.update_metric(episode=episode, *args, **kwargs)  # type: ignore

    def update_metric(self, episode: NavigationEpisode=None, *args: Any, **kwargs: Any):
        current_position = self._env.get_agent_position()

        # compute geodesic distance from current position to goal position
        # if episode just starts or current position is far enough from previous position
        if self._previous_position is None or not np.allclose(
            self._previous_position, current_position, atol=1e-4
        ):
            distance_to_target = self._env.get_current_distance()
            
            self._previous_position = current_position
            self._metric = distance_to_target

    

class Collisions(Measure):
    cls_uuid = "collisions"

    def __init__(self, env, config, *args: Any, **kwargs: Any):
        self._env = env
        self._config = config
        #self._metric = None
        super().__init__()


    def reset_metric(self, episode=None, *args: Any, **kwargs: Any):
        self._metric = {"count": 0, "is_collision": False}

    def update_metric(self, sim_obs, episode=None, *args: Any, **kwargs: Any):
        did_collide = self._env.extract_collisions(sim_obs)
        if did_collide is None:
            self._metric["is_collision"] = False
        else:    
            if did_collide:
                self._metric["count"] += 1
                self._metric["is_collision"] = True
            else:
                self._metric["is_collision"] = False

#  count steps per episode
class Steps(Measure):
    cls_uuid = "steps"

    def __init__(self, env, config, *args: Any, **kwargs: Any):
        self._env = env
        self._config = config
        #self._metric = None
        super().__init__()

    def reset_metric(self, episode=None, *args: Any, **kwargs: Any):
        self._metric = 0

    def update_metric(self, episode=None, *args: Any, **kwargs: Any):
        self._metric += 1

class Success(Measure):
    r"""Whether or not the agent succeeded at its task

    This measure depends on DistanceToGoal measure.
    """
    cls_uuid = "success"

    def __init__(self, env, config, *args: Any, **kwargs: Any):
        self._env = env
        self._config = config

        super().__init__()

    
    def reset_metric(self, measurements, episode=None, *args: Any, **kwargs: Any):
        measurements.check_measure_dependencies(self.cls_uuid, [DistanceToGoal.cls_uuid])
        self.update_metric(measurements=measurements, episode=episode, *args, **kwargs)  # type: ignore

    def update_metric(self, measurements, episode=None, *args: Any, **kwargs: Any):
        distance_to_target = measurements.measures[DistanceToGoal.cls_uuid].get_metric()

        # succeed
        if (distance_to_target < self._config.get("success_distance")):
            self._metric = 1.0
        else:
            self._metric = 0.0

class SPL(Measure):
    r"""SPL (Success weighted by Path Length)

    ref: On Evaluation of Embodied Agents - Anderson et. al
    https://arxiv.org/pdf/1807.06757.pdf
    The measure depends on Distance to Goal measure and Success measure
    to improve computational
    performance for sophisticated goal areas.
    """

    cls_uuid = "spl"

    def __init__(self, env, config, *args: Any, **kwargs: Any):
        self._previous_position = None
        self._start_end_episode_distance = None
        self._agent_episode_distance: Optional[float] = None
        self._env = env
        self._config = config

        super().__init__()


    def reset_metric(self, measurements, episode=None, *args: Any, **kwargs: Any):
        measurements.check_measure_dependencies(
            self.cls_uuid, [DistanceToGoal.cls_uuid, Success.cls_uuid]
        )

        self._previous_position = self._env.get_agent_position()
        self._agent_episode_distance = 0.0
        self._start_end_episode_distance = measurements.measures[
            DistanceToGoal.cls_uuid
        ].get_metric()
        self.update_metric(measurements=measurements, episode=episode, *args, **kwargs)

    def _euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(position_b - position_a, ord=2)

    # not averaged by the number of total episodes
    def update_metric(self, measurements, episode=None, *args: Any, **kwargs: Any):
        ep_success = measurements.measures[Success.cls_uuid].get_metric()

        current_position = self._env.get_agent_position()
        self._agent_episode_distance += self._euclidean_distance(current_position, self._previous_position)

        self._previous_position = current_position

        self._metric = ep_success * (
            self._start_end_episode_distance
            / max(
                self._start_end_episode_distance, self._agent_episode_distance
            )
        )


class SoftSPL(SPL):
    r"""Soft SPL

    Similar to SPL with a relaxed soft-success criteria. Instead of a boolean
    success is now calculated as 1 - (ratio of distance covered to target).
    """

    cls_uuid = "softspl"

    def reset_metric(self, measurements, episode=None, *args: Any, **kwargs: Any):
        measurements.check_measure_dependencies(
            self.cls_uuid, [DistanceToGoal.cls_uuid]
        )

        self._previous_position = self._env.get_agent_position()
        self._agent_episode_distance = 0.0
        self._start_end_episode_distance = measurements.measures[
            DistanceToGoal.cls_uuid
        ].get_metric()
        self.update_metric(measurements=measurements, episode=episode, *args, **kwargs)  # type: ignore

    # not averaged by the number of total episodes
    def update_metric(self, measurements, episode=None, *args: Any, **kwargs: Any):
        current_position = self._env.get_agent_position()

        distance_to_target = measurements.measures[
            DistanceToGoal.cls_uuid
        ].get_metric()

        ep_soft_success = max(
            0, (1 - distance_to_target / self._start_end_episode_distance)
        )

        self._agent_episode_distance += self._euclidean_distance(
            current_position, self._previous_position
        )

        self._previous_position = current_position

        self._metric = ep_soft_success * (
            self._start_end_episode_distance
            / max(
                self._start_end_episode_distance, self._agent_episode_distance
            )
        )

class Done(Measure):
    r"""Whether or not the episode ends
    """
    cls_uuid = "done"

    def __init__(self, env, config, *args: Any, **kwargs: Any):
        self._env = env
        self._config = config

        super().__init__()

    
    def reset_metric(self, measurements, episode=None, *args: Any, **kwargs: Any):
        measurements.check_measure_dependencies(self.cls_uuid, [Success.cls_uuid, Steps.cls_uuid, Collisions.cls_uuid])
        self.update_metric(measurements=measurements, episode=episode, *args, **kwargs)  # type: ignore

    def update_metric(self, measurements, episode=None, *args: Any, **kwargs: Any):
        current_collision_count = measurements.measures[Collisions.cls_uuid].get_metric()["count"] 
        current_step_count = measurements.measures[Steps.cls_uuid].get_metric()
        success = measurements.measures[Success.cls_uuid].get_metric()

        if bool(success):
            self._metric = True
        else:    
            if current_collision_count >= int(self._config.get("max_collisions_per_episode")) \
                or current_step_count >= int(self._config.get("max_steps_per_episode")):
                self._metric = True
            else:
                self._metric = False 

class PointGoalReward(Measure):
    
    cls_uuid = "point_goal_reward"

    def __init__(self, env, config, *args: Any, **kwargs: Any):
        self._env = env
        self._config = config
        self._previous_measure = None

        super().__init__()

    
    def reset_metric(self, measurements, episode=None, *args: Any, **kwargs: Any):
        measurements.check_measure_dependencies(self.cls_uuid, [DistanceToGoal.cls_uuid, Success.cls_uuid])
        self._previous_measure = measurements.measures[DistanceToGoal.cls_uuid].get_metric()
        
        self.update_metric(measurements=measurements, episode=episode, *args, **kwargs)  # type: ignore

    def update_metric(self, measurements, episode=None, *args: Any, **kwargs: Any):
        self._metric = float(self._config.get("slack_reward"))

        current_measure = measurements.measures[DistanceToGoal.cls_uuid].get_metric()

        self._metric += (self._previous_measure - current_measure)
        self._previous_measure = current_measure

        success = measurements.measures[Success.cls_uuid].get_metric()
        if bool(success):
            self._metric += float(self._config.get("success_reward"))

# episode return, undiscounted                  
class Return(Measure):
    
    cls_uuid = "return"

    def __init__(self, env, config, *args: Any, **kwargs: Any):
        self._env = env
        self._config = config
        
        super().__init__()

    
    def reset_metric(self, measurements, episode=None, *args: Any, **kwargs: Any):
        measurements.check_measure_dependencies(self.cls_uuid, [PointGoalReward.cls_uuid])
        self._metric = 0
        
    def update_metric(self, measurements, episode=None, *args: Any, **kwargs: Any):
        
        current_reward = measurements.measures[PointGoalReward.cls_uuid].get_metric()

        self._metric += current_reward
        

def create_one_measurement(measure_id, env, config):
        if measure_id == "distance_to_goal":
            return DistanceToGoal(env=env, config=config)
        elif measure_id == "collisions":
            return Collisions(env=env, config=config)
        elif measure_id == "steps":
            return Steps(env=env, config=config)
        elif measure_id == "success":
            return Success(env=env, config=config)
        elif measure_id == "spl":
            return SPL(env=env, config=config)   
        elif measure_id == "softspl":
            return SoftSPL(env=env, config=config)
        elif measure_id == "done":
            return Done(env=env, config=config)   
        elif measure_id == "point_goal_reward":
            return PointGoalReward(env=env, config=config)
        elif measure_id == "return":
            return Return(env=env, config=config)         
        else:
            print("Error: not defined measure id: "+str(measure_id)) 
            return    

# collection of all measurements
class Measurements:
    r"""Represents a set of Measures, with each :ref:`Measure` being
    identified through a unique id.
    """

    measures: Dict[str, Measure]

    def __init__(self, measure_ids, env, config) -> None:
        """Constructor

        :param measures: list containing :ref:`Measure`, uuid of each
            :ref:`Measure` must be unique.
        """
        self.measures = OrderedDict()

        for mid in measure_ids:
            assert (
                mid not in self.measures
            ), "'{}' is duplicated measure uuid".format(mid)
            
            self.measures[mid] = create_one_measurement(measure_id=mid, env=env, config=config)

    def init_all_to_zero(self):
        for measure in self.measures.values():
            measure.set_metric_to_zero()

    # def print(self):
    #     for k,v in self.measures.items():
    #         print(str(k)+": %f"%(v.get_metric()))
    #         #print(str(k))         

    def reset_measures(self, *args: Any, **kwargs: Any) -> None:
        for measure in self.measures.values():
            measure.reset_metric(*args, **kwargs)

    def update_measures(self, *args: Any, **kwargs: Any) -> None:
        for measure in self.measures.values():
            measure.update_metric(*args, **kwargs)

    def _get_measure_index(self, measure_name):
        return list(self.measures.keys()).index(measure_name)


    def print_measures(self):
        print('-------------- Measures ---------------------')
        string = ""
        for measure in self.measures.values():
            s = measure.print_metric()  
            string += s 

        print('---------------------------------------------') 
        return string   

    # check if measure A depends on measure B (requires computing measure B), B should have smaller id
    # e.g. success measure requires distanc to goal measure
    def check_measure_dependencies(
        self, measure_name: str, dependencies: List[str]
    ):
        r"""Checks if dependencies measures are enabled and calculate that the measure
        :param measure_name: a name of the measure for which has dependencies.
        :param dependencies: a list of a measure names that are required by
        the measure.
        :return:
        """
        measure_index = self._get_measure_index(measure_name)
        for dependency_measure in dependencies:
            assert (
                dependency_measure in self.measures
            ), f"""{measure_name} measure requires {dependency_measure}
                listed in the measures list in the config."""

        for dependency_measure in dependencies:
            assert measure_index > self._get_measure_index(
                dependency_measure
            ), f"""{measure_name} measure requires be listed after {dependency_measure}
                in the measures list in the config."""
