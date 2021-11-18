#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

from enlighten.envs import VectorEnv, NavEnv
from enlighten.utils.path import *
from enlighten.utils.config_utils import parse_config

import random
import gym

CFG_TEST = "navigate_with_flashlight.yaml"
NUM_ENVS = 4

def _make_nav_env_fn(
    config_filename: str="navigate_with_flashlight.yaml", rank: int = 0
) -> NavEnv:
    """Constructor for default enlighten :ref:`enlighten.NavEnv`.

    :param config_filename: configuration file name for environment.
    :param rank: rank for setting seed of environment
    :return: :ref:`enlighten.NavEnv` object
    """
    # get configuration
    config_file=os.path.join(config_path, config_filename)
    # create env
    env = NavEnv(config_file=config_file)
    # set seed
    env.seed(rank)

    return env

def sample_action(action_space, num_samples=1):
    samples = []
    for _ in range(num_samples):
        action = action_space.sample()
        samples.append({"action": action})

    if num_samples == 1:
        return samples[0] #["action"]
    else:
        return samples

def vec_env_test_fn():
    config_filenames = [CFG_TEST for _ in range(NUM_ENVS)]
    env_fn_args = tuple(zip(config_filenames, range(NUM_ENVS)))

    config_file=os.path.join(config_path, CFG_TEST)
    config = parse_config(config_file)

    with VectorEnv(
        env_fn_args=env_fn_args,
        make_env_fn=_make_nav_env_fn,
        multiprocessing_start_method="forkserver",
    ) as envs:
        envs.reset()
        for step in range(10): # config.get("max_steps_per_episode")
            observations = envs.step(
                sample_action(envs.action_spaces[0], NUM_ENVS)
            )
            assert len(observations) == NUM_ENVS
        
            print("----------- step %d ----------"%step)
            for idx, obs in enumerate(observations):
                # print(idx, obs)
                print("----------- env %d ----------"%idx)
                print(len(obs))
                #cv2.imwrite(f'{idx}.jpg', obs[0]['color_sensor'])

def vec_env_test_async_fn():
    config_filenames = [CFG_TEST for _ in range(NUM_ENVS)]
    env_fn_args = tuple(zip(config_filenames, range(NUM_ENVS)))

    config_file=os.path.join(config_path, CFG_TEST)
    config = parse_config(config_file)

    with VectorEnv(
        env_fn_args=env_fn_args,
        make_env_fn=_make_nav_env_fn,
        multiprocessing_start_method="forkserver",
    ) as envs:
        envs.reset()
        for step in range(10): # config.get("max_steps_per_episode")

            env_id = random.randint(0, NUM_ENVS-1)
            #action_id = envs.action_spaces[0].sample()
            action  =  sample_action(envs.action_spaces[0], 1)
            print(action)
            print("----------- step %d ----------"%step)
            
            print("----------- env %d ----------"%env_id)
            print("----------- action %s ----------"%action)
            observation, reward, done, info = envs.step_at(index_env=env_id, action=action)
            
            
            print(observation.keys())

def test_breakout():
    env  =  gym.make('Breakout-v0')
    
    for episode_index in range(10): 
        env.reset()
        for step_index in range(100):
            env.render()
            action  =  env.action_space.sample()
            
            print(action)
            observation, reward, done, info  =  env.step(action)
            #print(observation.shape)
            
            if done:
                break

def vec_env_test_breakout():
    with VectorEnv(
        env_fn_args=[0,1,2,3],
        make_env_fn=_make_breakout_env_fn,
        multiprocessing_start_method="fork",
    ) as envs:
        envs.reset()
        for step in range(10): # config.get("max_steps_per_episode")
            observations = envs.step(
                sample_action(envs.action_spaces[0], NUM_ENVS)
            )
            assert len(observations) == NUM_ENVS
        
            print("----------- step %d ----------"%step)
            for idx, obs in enumerate(observations):
                # print(idx, obs)
                print("----------- env %d ----------"%idx)
                print(len(obs))
                #cv2.imwrite(f'{idx}.jpg', obs[0]['color_sensor'])

def _make_breakout_env_fn(seed: int = 0):
    """Constructor for default enlighten :ref:`enlighten.NavEnv`.

    :param config_filename: configuration file name for environment.
    :param rank: rank for setting seed of environment
    :return: :ref:`enlighten.NavEnv` object
    """

    # create env
    env = gym.make('Breakout-v0')
    # set seed
    env.seed(seed)

    return env

if __name__ == '__main__':
    #vec_env_test_fn()
    vec_env_test_async_fn()
    #test_breakout()
    #vec_env_test_breakout()