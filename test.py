#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

from enlighten.envs import VectorEnv, NavEnv
from enlighten.utils.path import *
from enlighten.utils.config_utils import parse_config

import cv2


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
        return samples[0]["action"]
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
        for _ in range(config.get("max_steps_per_episode")):
            observations = envs.step(
                sample_action(envs.action_spaces[0], NUM_ENVS)
            )
            assert len(observations) == NUM_ENVS
        
        for idx, obs in enumerate(observations):
            # print(idx, obs)
            print(obs)
            #cv2.imwrite(f'{idx}.jpg', obs[0]['color_sensor'])

if __name__ == '__main__':
    vec_env_test_fn()