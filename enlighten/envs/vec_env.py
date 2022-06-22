#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from tqdm import tqdm
import signal
import warnings
from multiprocessing.connection import Connection
from multiprocessing.context import BaseContext
from queue import Queue
from threading import Thread
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
    cast,
)
from typing import Optional, Type

import attr
import gym
import numpy as np
from gym import spaces

# import habitat
# from habitat.core.env import Env, RLEnv
from habitat.core.logging import logger

from habitat_sim.utils import profiling_utils

from enlighten.utils.image_utils import tile_images
from enlighten.utils.pickle5_multiprocessing import ConnectionWrapper

import os
from enlighten.envs import NavEnv, MultiNavEnv
from enlighten.utils.path import *
from enlighten.utils.config_utils import parse_config

#from garage.envs import GymEnv
#from garage import EnvSpec

import copy

import random

from enlighten.datasets.pointnav_dataset import make_dataset
from enlighten.datasets.il_data_gen import load_behavior_dataset_meta, extract_observation

try:
    # Use torch.multiprocessing if we can.
    # We have yet to find a reason to not use it and
    # you are required to use it when sending a torch.Tensor
    # between processes
    import torch
    from torch import multiprocessing as mp  # type:ignore
except ImportError:
    torch = None
    import multiprocessing as mp  # type:ignore


STEP_COMMAND = "step"
RESET_COMMAND = "reset"
RENDER_COMMAND = "render"
CLOSE_COMMAND = "close"
CALL_COMMAND = "call"
COUNT_EPISODES_COMMAND = "count_episodes"

EPISODE_OVER_NAME = "episode_over"
GET_METRICS_NAME = "get_metrics"
CURRENT_EPISODE_NAME = "current_episode"
NUMBER_OF_EPISODE_NAME = "number_of_episodes"
ACTION_SPACE_NAME = "action_space"
OBSERVATION_SPACE_NAME = "observation_space"
GET_GOAL_OBS_SPACE_NAME = "get_goal_observation_space"
GET_COMBINED_GOAL_OBS_SPACE_NAME = "get_combined_goal_obs_space"

# def _make_env_fn(
#     config_filename: str="navigate_with_flashlight.yaml", rank: int = 0
# ) -> GymEnv:
#     """Constructor for default enlighten :ref:`garage.GymEnv`.

#     :param config_filename: configuration file name for environment.
#     :param rank: rank for setting seed of environment
#     :return: :ref:`garage.GymEnv` object
#     """
#     # get configuration
#     config_file=os.path.join(config_path, config_filename)
#     config = parse_config(config_file)
#     # create env
#     env = GymEnv(env=NavEnv(), is_image=True, max_episode_length=int(config.get("max_steps_per_episode"))) 
#     assert isinstance(env.spec, EnvSpec)
#     # set seed
#     env.seed(rank)

#     return env


@attr.s(auto_attribs=True, slots=True)
class _ReadWrapper:
    r"""Convenience wrapper to track if a connection to a worker process
    should have something to read.
    """
    read_fn: Callable[[], Any]
    rank: int
    is_waiting: bool = False

    def __call__(self) -> Any:
        if not self.is_waiting:
            raise RuntimeError(
                f"Tried to read from process {self.rank}"
                " but there is nothing waiting to be read"
            )
        res = self.read_fn()
        self.is_waiting = False

        return res


@attr.s(auto_attribs=True, slots=True)
class _WriteWrapper:
    r"""Convenience wrapper to track if a connection to a worker process
    can be written to safely.  In other words, checks to make sure the
    result returned from the last write was read.
    """
    write_fn: Callable[[Any], None]
    read_wrapper: _ReadWrapper

    def __call__(self, data: Any) -> None:
        if self.read_wrapper.is_waiting:
            raise RuntimeError(
                f"Tried to write to process {self.read_wrapper.rank}"
                " but the last write has not been read"
            )
        self.write_fn(data)
        self.read_wrapper.is_waiting = True


class VectorEnv:
    r"""Vectorized environment which creates multiple processes where each
    process runs its own environment. Main class for parallelization of
    training and evaluation.


    All the environments are synchronized on step and reset methods.
    """

    observation_spaces: List[spaces.Dict]
    number_of_episodes: List[Optional[int]]
    action_spaces: List[spaces.Dict]
    _workers: List[Union[mp.Process, Thread]]
    _num_envs: int
    _auto_reset_done: bool
    _mp_ctx: BaseContext
    _connection_read_fns: List[_ReadWrapper]
    _connection_write_fns: List[_WriteWrapper]

    def __init__(
        self,
        make_env_fn: Callable[..., Union[NavEnv, MultiNavEnv]],
        env_fn_args: Sequence[Tuple] = None,
        auto_reset_done: bool = True,
        multiprocessing_start_method: str = "forkserver",
        workers_ignore_signals: bool = False,
    ) -> None:
        """..

        :param make_env_fn: function which creates a single environment. An
            environment can be of type :ref:`garage.GymEnv` or :ref:`enlighten.NavEnv`
        :param env_fn_args: tuple of tuple of args to pass to the
            :ref:`_make_env_fn`.
        :param auto_reset_done: automatically reset the environment when
            done. This functionality is provided for seamless training
            of vectorized environments.
        :param multiprocessing_start_method: the multiprocessing method used to
            spawn worker processes. Valid methods are
            :py:`{'spawn', 'forkserver', 'fork'}`; :py:`'forkserver'` is the
            recommended method as it works well with CUDA. If :py:`'fork'` is
            used, the subproccess  must be started before any other GPU usage.
        :param workers_ignore_signals: Whether or not workers will ignore SIGINT and SIGTERM
            and instead will only exit when :ref:`close` is called
        """
        self._is_closed = True

        assert (
            env_fn_args is not None and len(env_fn_args) > 0
        ), "number of environments to be created should be greater than 0"

        self._num_envs = len(env_fn_args)

        assert multiprocessing_start_method in self._valid_start_methods, (
            "multiprocessing_start_method must be one of {}. Got '{}'"
        ).format(self._valid_start_methods, multiprocessing_start_method)
        self._auto_reset_done = auto_reset_done
        self._mp_ctx = mp.get_context(multiprocessing_start_method)
        self._workers = []
        (
            self._connection_read_fns,
            self._connection_write_fns,
        ) = self._spawn_workers(  # noqa
            env_fn_args,
            make_env_fn,
            workers_ignore_signals=workers_ignore_signals,
        )

        self._is_closed = False

        for write_fn in self._connection_write_fns:
            write_fn((CALL_COMMAND, (OBSERVATION_SPACE_NAME, None)))
        self.observation_spaces = [
            read_fn() for read_fn in self._connection_read_fns
        ]
        for write_fn in self._connection_write_fns:
            write_fn((CALL_COMMAND, (ACTION_SPACE_NAME, None)))
        self.action_spaces = [
            read_fn() for read_fn in self._connection_read_fns
        ]        

        # get number of episodes
        for write_fn in self._connection_write_fns:
            write_fn((CALL_COMMAND, (NUMBER_OF_EPISODE_NAME, None)))
        self.number_of_episodes = [
            read_fn() for read_fn in self._connection_read_fns
        ]
        
        self._paused: List[Tuple] = []

    @property
    def num_envs(self):
        r"""number of individual environments."""
        return self._num_envs - len(self._paused)

    @staticmethod
    @profiling_utils.RangeContext("_worker_env")
    def _worker_env(
        connection_read_fn: Callable,
        connection_write_fn: Callable,
        env_fn: Callable,
        env_fn_args: Tuple[Any],
        auto_reset_done: bool,
        mask_signals: bool = False,
        child_pipe: Optional[Connection] = None,
        parent_pipe: Optional[Connection] = None,
    ) -> None:
        r"""process worker for creating and interacting with the environment."""
        if mask_signals:
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            signal.signal(signal.SIGTERM, signal.SIG_IGN)

            signal.signal(signal.SIGUSR1, signal.SIG_IGN)
            signal.signal(signal.SIGUSR2, signal.SIG_IGN)

        env = env_fn(*env_fn_args)
        if parent_pipe is not None:
            parent_pipe.close()
        try:
            command, data = connection_read_fn()
            while command != CLOSE_COMMAND:
                if command == STEP_COMMAND:
                    # check wrapped env types
                    # different step methods for enlighten.NavEnv and garage.GymEnv
                    if isinstance(env, (NavEnv, MultiNavEnv, gym.Env)):
                        # NavEnv or MultiNavEnv should have the same iterface for reset and step
                        observations, reward, done, info = env.step(**data)
                        if auto_reset_done and done:

                            observations = env.reset()
                        with profiling_utils.RangeContext(
                            "worker write after step"
                        ):
                            connection_write_fn(
                                (observations, reward, done, info)
                            )
                    # elif isinstance(env, GymEnv):  # type: ignore
                    #     # garage.GymEnv
                    #     env_step = env.step(**data)
                    #     observations = env_step.observation
                    #     reward = env_step.reward
                    #     info = env_step.env_info
                    #     done = env_step.terminal
                    #     if auto_reset_done and done:
                    #         observations, _ = env.reset()
                    #     connection_write_fn(
                    #         (observations, reward, done, info)
                    #     )
                    else:
                        raise NotImplementedError

                elif command == RESET_COMMAND:
                    observations = env.reset()
                    connection_write_fn(observations)

                elif command == RENDER_COMMAND:
                    connection_write_fn(env.render(*data[0], **data[1]))

                elif command == CALL_COMMAND:
                    function_name, function_args = data
                    if function_args is None:
                        function_args = {}

                    result_or_fn = getattr(env, function_name)

                    if len(function_args) > 0 or callable(result_or_fn):
                        result = result_or_fn(**function_args)
                    else:
                        result = result_or_fn

                    connection_write_fn(result)

                # elif command == COUNT_EPISODES_COMMAND:
                #     connection_write_fn(len(env.episodes))

                else:
                    raise NotImplementedError(f"Unknown command {command}")

                with profiling_utils.RangeContext("worker wait for command"):
                    command, data = connection_read_fn()

        except KeyboardInterrupt:
            logger.info("Worker KeyboardInterrupt")
        finally:
            if child_pipe is not None:
                child_pipe.close()
            env.close()

    def _spawn_workers(
        self,
        env_fn_args: Sequence[Tuple],
        make_env_fn: Callable[..., Union[NavEnv]],
        workers_ignore_signals: bool = False,
    ) -> Tuple[List[_ReadWrapper], List[_WriteWrapper]]:
        parent_connections, worker_connections = zip(
            *[
                [ConnectionWrapper(c) for c in self._mp_ctx.Pipe(duplex=True)]
                for _ in range(self._num_envs)
            ]
        )
        self._workers = []
        for worker_conn, parent_conn, env_args in zip(
            worker_connections, parent_connections, env_fn_args
        ):
            ps = self._mp_ctx.Process(
                target=self._worker_env,
                args=(
                    worker_conn.recv,
                    worker_conn.send,
                    make_env_fn,
                    env_args,
                    self._auto_reset_done,
                    workers_ignore_signals,
                    worker_conn,
                    parent_conn,
                ),
            )
            self._workers.append(cast(mp.Process, ps))
            ps.daemon = True
            ps.start()
            worker_conn.close()

        read_fns = [
            _ReadWrapper(p.recv, rank)
            for rank, p in enumerate(parent_connections)
        ]
        write_fns = [
            _WriteWrapper(p.send, read_fn)
            for p, read_fn in zip(parent_connections, read_fns)
        ]

        return read_fns, write_fns

    def current_episodes(self):
        for write_fn in self._connection_write_fns:
            write_fn((CALL_COMMAND, (CURRENT_EPISODE_NAME, None)))
        results = []
        for read_fn in self._connection_read_fns:
            results.append(read_fn())
        return results

    def count_episodes(self):
        for write_fn in self._connection_write_fns:
            write_fn((COUNT_EPISODES_COMMAND, None))
        results = []
        for read_fn in self._connection_read_fns:
            results.append(read_fn())
        return results

    def episode_over(self):
        for write_fn in self._connection_write_fns:
            write_fn((CALL_COMMAND, (EPISODE_OVER_NAME, None)))
        results = []
        for read_fn in self._connection_read_fns:
            results.append(read_fn())
        return results

    def get_metrics(self):
        for write_fn in self._connection_write_fns:
            write_fn((CALL_COMMAND, (GET_METRICS_NAME, None)))
        results = []
        for read_fn in self._connection_read_fns:
            results.append(read_fn())
        return results

    def get_goal_observation_space(self):
        for write_fn in self._connection_write_fns:
            write_fn((CALL_COMMAND, (GET_GOAL_OBS_SPACE_NAME, None)))
        results = []
        for read_fn in self._connection_read_fns:
            results.append(read_fn())
        return results[0]

    def get_combined_goal_obs_space(self):
        for write_fn in self._connection_write_fns:
            write_fn((CALL_COMMAND, (GET_COMBINED_GOAL_OBS_SPACE_NAME, None)))
        results = []
        for read_fn in self._connection_read_fns:
            results.append(read_fn())
        return results[0]

    def reset(self):
        r"""Reset all the vectorized environments

        :return: list of outputs from the reset method of envs.
        """
        for write_fn in self._connection_write_fns:
            write_fn((RESET_COMMAND, None))
        results = []
        for read_fn in self._connection_read_fns:
            results.append(read_fn())
        return results

    def reset_at(self, index_env: int):
        r"""Reset in the index_env environment in the vector.

        :param index_env: index of the environment to be reset
        :return: list containing the output of reset method of indexed env.
        """
        self._connection_write_fns[index_env]((RESET_COMMAND, None))
        results = [self._connection_read_fns[index_env]()]
        return results

    def async_step_at(
        self, index_env: int, action: Union[int, str, Dict[str, Any]]
    ) -> None:
        # Backward compatibility
        if isinstance(action, (int, np.integer, str)):
            #print("inside")
            # never use this branch, normal format is {"action": action index}
            action = {"action": {"action": action}}
            

        #print("outside")
        #print(action)
        #exit()
        # if index_env < 0 or index_env > 3:
        #     print(index_env)
        #     print("===> env index out of range")
        #     exit()

        # if action["action"] < 0 or action["action"] > 3:
        #     print(action["action"])
        #     print("===> action index out of range")
        #     exit()

        self._warn_cuda_tensors(action)
        try:
            self._connection_write_fns[index_env]((STEP_COMMAND, action))
        except:
            #print(index_env)
            #print(action["action"])
            print("Error: Env %d was incorrectly paused before step is called"%(index_env))
            

    @profiling_utils.RangeContext("wait_step_at")
    def wait_step_at(self, index_env: int) -> Any:
        return self._connection_read_fns[index_env]()

    def step_at(self, index_env: int, action: Union[int, str, Dict[str, Any]]):
        r"""Step in the index_env environment in the vector.

        :param index_env: index of the environment to be stepped into
        :param action: action to be taken
        :return: list containing the output of step method of indexed env.
        """
        self.async_step_at(index_env, action)
        #print('-------after async step at--------')
        r = self.wait_step_at(index_env)
        #print('-------after wait step at--------')
        return r

    def async_step(self, data: List[Union[int, str, Dict[str, Any]]]) -> None:
        r"""Asynchronously step in the environments.

        :param data: list of size _num_envs containing keyword arguments to
            pass to :ref:`step` method for each Environment. For example,
            :py:`[{"action": "TURN_LEFT", "action_args": {...}}, ...]`.
        """

        for index_env, act in enumerate(data):
            self.async_step_at(index_env, act)

    @profiling_utils.RangeContext("wait_step")
    def wait_step(self) -> List[Any]:
        r"""Wait until all the asynchronized environments have synchronized."""
        return [
            self.wait_step_at(index_env) for index_env in range(self.num_envs)
        ]

    # note that action must be in the format of {"action": int}
    def step(self, data: List[Union[int, str, Dict[str, Any]]]) -> List[Any]:
        r"""Perform actions in the vectorized environments.

        :param data: list of size _num_envs containing keyword arguments to
            pass to :ref:`step` method for each Environment. For example,
            :py:`[{"action": "TURN_LEFT", "action_args": {...}}, ...]`.
        :return: list of outputs from the step method of envs.
        """
        
        self.async_step(data)
        
        return self.wait_step()

    def close(self) -> None:
        if self._is_closed:
            return

        for read_fn in self._connection_read_fns:
            if read_fn.is_waiting:
                read_fn()

        for write_fn in self._connection_write_fns:
            write_fn((CLOSE_COMMAND, None))

        for _, _, write_fn, _ in self._paused:
            write_fn((CLOSE_COMMAND, None))

        for process in self._workers:
            process.join()

        for _, _, _, process in self._paused:
            process.join()

        self._is_closed = True

    def pause_at(self, index: int) -> None:
        r"""Pauses computation on this env without destroying the env.

        :param index: which env to pause. All indexes after this one will be
            shifted down by one.

        This is useful for not needing to call steps on all environments when
        only some are active (for example during the last episodes of running
        eval episodes).
        """
        if self._connection_read_fns[index].is_waiting:
            self._connection_read_fns[index]()
        read_fn = self._connection_read_fns.pop(index)
        write_fn = self._connection_write_fns.pop(index)
        worker = self._workers.pop(index)
        self._paused.append((index, read_fn, write_fn, worker))

    def resume_all(self) -> None:
        r"""Resumes any paused envs."""
        for index, read_fn, write_fn, worker in reversed(self._paused):
            self._connection_read_fns.insert(index, read_fn)
            self._connection_write_fns.insert(index, write_fn)
            self._workers.insert(index, worker)
        self._paused = []

    def call_at(
        self,
        index: int,
        function_name: str,
        function_args: Optional[Dict[str, Any]] = None,
    ) -> Any:
        r"""Calls a function or retrieves a property/member variable (which is passed by name)
        on the selected env and returns the result.

        :param index: which env to call the function on.
        :param function_name: the name of the function to call or property to retrieve on the env.
        :param function_args: optional function args.
        :return: result of calling the function.
        """
        self._connection_write_fns[index](
            (CALL_COMMAND, (function_name, function_args))
        )
        result = self._connection_read_fns[index]()
        return result

    def call(
        self,
        function_names: List[str],
        function_args_list: Optional[List[Any]] = None,
    ) -> List[Any]:
        r"""Calls a list of functions (which are passed by name) on the
        corresponding env (by index).

        :param function_names: the name of the functions to call on the envs.
        :param function_args_list: list of function args for each function. If
            provided, :py:`len(function_args_list)` should be as long as
            :py:`len(function_names)`.
        :return: result of calling the function.
        """
        if function_args_list is None:
            function_args_list = [None] * len(function_names)
        assert len(function_names) == len(function_args_list)
        func_args = zip(function_names, function_args_list)
        for write_fn, func_args_on in zip(
            self._connection_write_fns, func_args
        ):
            write_fn((CALL_COMMAND, func_args_on))
        results = []
        for read_fn in self._connection_read_fns:
            results.append(read_fn())
        return results

    def render(
        self, mode: str = "human", *args, **kwargs
    ) -> Union[np.ndarray, None]:
        r"""Render observations from all environments in a tiled image."""
        for write_fn in self._connection_write_fns:
            write_fn((RENDER_COMMAND, (args, {"mode": "rgb", **kwargs})))
        images = [read_fn() for read_fn in self._connection_read_fns]
        tile = tile_images(images)
        if mode == "human":
            from enlighten.utils.image_utils import try_cv2_import

            cv2 = try_cv2_import()

            cv2.imshow("vecenv", tile[:, :, ::-1])
            cv2.waitKey(1)
            return None
        elif mode == "rgb_array":
            return tile
        else:
            raise NotImplementedError

    @property
    def _valid_start_methods(self) -> Set[str]:
        return {"forkserver", "spawn", "fork"}

    def _warn_cuda_tensors(
        self, action: Dict[str, Any], prefix: Optional[str] = None
    ):
        if torch is None:
            return

        for k, v in action.items():
            if isinstance(v, dict):
                subk = f"{prefix}.{k}" if prefix is not None else k
                self._warn_cuda_tensors(v, prefix=subk)
            elif torch.is_tensor(v) and v.device.type == "cuda":
                subk = f"{prefix}.{k}" if prefix is not None else k
                warnings.warn(
                    "Action with key {} is a CUDA tensor."
                    "  This will result in a CUDA context in the subproccess worker."
                    "  Using CPU tensors instead is recommended.".format(subk)
                )

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class ThreadedVectorEnv(VectorEnv):
    r"""Provides same functionality as :ref:`VectorEnv`, the only difference
    is it runs in a multi-thread setup inside a single process.

    The :ref:`VectorEnv` runs in a multi-proc setup. This makes it much easier
    to debug when using :ref:`VectorEnv` because you can actually put break
    points in the environment methods. It should not be used for best
    performance.
    """

    def _spawn_workers(
        self,
        env_fn_args: Sequence[Tuple],
        make_env_fn: Callable[..., Union[NavEnv]],
        workers_ignore_signals: bool = False,
    ) -> Tuple[List[_ReadWrapper], List[_WriteWrapper]]:
        queues: Iterator[Tuple[Any, ...]] = zip(
            *[(Queue(), Queue()) for _ in range(self._num_envs)]
        )
        parent_read_queues, parent_write_queues = queues
        self._workers = []
        for parent_read_queue, parent_write_queue, env_args in zip(
            parent_read_queues, parent_write_queues, env_fn_args
        ):
            thread = Thread(
                target=self._worker_env,
                args=(
                    parent_write_queue.get,
                    parent_read_queue.put,
                    make_env_fn,
                    env_args,
                    self._auto_reset_done,
                ),
            )
            self._workers.append(thread)
            thread.daemon = True
            thread.start()

        read_fns = [
            _ReadWrapper(q.get, rank)
            for rank, q in enumerate(parent_read_queues)
        ]
        write_fns = [
            _WriteWrapper(q.put, read_wrapper)
            for q, read_wrapper in zip(parent_write_queues, read_fns)
        ]
        return read_fns, write_fns

def load_scenes_episodes(config, split_name):
    episodes = load_behavior_dataset_meta(yaml_name=config, 
        split_name=split_name)

    scene_episodes_dict = {}
    for episode in tqdm(episodes):
        
        if episode.scene_id not in scene_episodes_dict:
            scene_episodes_dict[episode.scene_id] = []
            
        scene_episodes_dict[episode.scene_id].append(episode)
    
    return scene_episodes_dict

# construct vector envs from a data set
def construct_envs_based_on_dataset(
    config,
    split_name,
    workers_ignore_signals: bool = False,
) -> VectorEnv:
    r"""Create VectorEnv object with specified config and env class type.
    To allow better performance, dataset are split into small ones for
    each individual env, grouped by scenes.

    :param config: configs that contain num_environments as well as information
    :param necessary to create individual environments.
    :param workers_ignore_signals: Passed to :ref:`habitat.VectorEnv`'s constructor

    :return: VectorEnv object created according to specification.
    """

    num_environments = int(config.get("num_environments"))
    
    
    # get train/val/test scenes
    scene_episodes_dict = load_scenes_episodes(config=config, split_name=split_name)
    scenes = list(scene_episodes_dict.keys())

    # shuffle scenes
    if num_environments > 1:
        if len(scenes) == 0:
            raise RuntimeError(
                "No scenes to load, multiple process logic relies on being able to split scenes uniquely between processes"
            )

        if len(scenes) < num_environments:
            print(
                "reduce the number of environments as there "
                "aren't enough number of scenes.\n"
                "num_environments: {}\tnum_scenes: {}".format(
                    num_environments, len(scenes)
                )
            )
            num_environments = len(scenes)

        random.shuffle(scenes)

    # assign scenes to each env
    # scene_splits is a list of scene list, len = num of environments
    scene_splits: List[List[str]] = [[] for _ in range(num_environments)]
    for idx, scene in enumerate(scenes):
        scene_splits[idx % len(scene_splits)].append(scene)

    assert sum(map(len, scene_splits)) == len(scenes)

    # create episode list according to scene list
    episode_splits = []
    for scene_set in scene_splits:
        episode_list = []
        for sc in scene_set:
            episode_list.extend(scene_episodes_dict[sc])
        
        episode_splits.append(episode_list)    


    # copy configs and compute seeds
    configs = []
    seeds = []
    for i in range(num_environments):
        seed = int(config.get("seed") + i)
        seeds.append(seed)
        cur_config = copy.deepcopy(config)
        configs.append(cur_config)
        
    env_fn_args = tuple(zip(configs, seeds, episode_splits))

    envs = VectorEnv(
        make_env_fn=_make_multi_nav_fn,
        env_fn_args=env_fn_args,
        workers_ignore_signals=workers_ignore_signals,
    ) 

    return envs

# make one multi scene nav environment 
def _make_multi_nav_fn(config, seed, episode_split):
    r"""Creates an env of type env_class with specified config and rank.
    This is to be passed in as an argument when creating VectorEnv.
    Returns:
        env object created according to specification.
    """
    
    env = MultiNavEnv(config_file=config)
    env.seed(seed)
    env.set_episode_dataset(episodes=episode_split)
    return env

# make one single scene nav environment
def _make_nav_env_fn(config: str="navigate_with_flashlight.yaml", seed: int = 0) -> NavEnv:
    # create env
    env = NavEnv(config_file=config)
    # set seed
    env.seed(seed)

    return env

def construct_envs_based_on_singel_scene(config, workers_ignore_signals: bool = False):
    num_environments = int(config.get("num_environments"))
    configs = []
    seeds = []
    for i in range(num_environments):
        seed = int(config.get("seed") + i)
        seeds.append(seed)
        cur_config = copy.deepcopy(config)
        configs.append(cur_config)

    env_fn_args = tuple(zip(configs, seeds))

    envs = VectorEnv(
        env_fn_args=env_fn_args,
        make_env_fn=_make_nav_env_fn,
        workers_ignore_signals=workers_ignore_signals,
    )
    return envs