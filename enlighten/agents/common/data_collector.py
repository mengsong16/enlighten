import abc
from collections import deque, OrderedDict
from functools import partial

import numpy as np

import copy

import random

from enlighten.agents.common.other import create_stats_ordered_dict
from enlighten.datasets.common import extract_observation

def extract_obs_goal(env, obs):
    # [C,H,W]
    obs_array = extract_observation(obs, env.observation_space.spaces)
    
    rel_goal = np.array(obs["pointgoal"], dtype="float32")
    
    # np arrays
    return obs_array, rel_goal


def get_random_action(observations, goals, sample):
    return random.randint(0,3)

    
def rollout(
        env,
        get_action_fn,
        sample,
        max_path_length=np.inf,
        render=False,
):
    
    observations = []
    goals = []
    actions = []
    rewards = []
    next_observations = []
    next_goals = []
    dones = []

    # how many times env.step has been called
    step_num = 0

    # reset environment
    # plan_shortest_path=False by default
    raw_obs = env.reset()
    o, g = extract_obs_goal(env, raw_obs)

    if render:
        env.render()

    # assume step_num is in [1, max_path_length-1]
    while step_num < max_path_length:
    
        a_tensor = get_action_fn(observations=np.expand_dims(o, axis=0), goals=np.expand_dims(g, axis=0), sample=sample)
        a = a_tensor.item()


        next_raw_obs, r, done, info = env.step(copy.deepcopy(a))
        next_o, next_g = extract_obs_goal(env, next_raw_obs)

        if render:
            env.render()

        observations.append(o)
        goals.append(g)
        rewards.append(r)
        dones.append(done)
        actions.append(a)
        next_observations.append(next_o)
        next_goals.append(next_g)

        step_num += 1

        if done:
            break

        # next turn
        o = next_o
        g = next_g
    
    # assume the path has [s0, s1, ..., sN], there are N transitions
    actions = np.array(actions)  # [N,1]
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations) #[N,C,H,W]
    next_observations = np.array(next_observations) #[N,C,H,W]
    goals = np.array(goals) # [N, goal_dim]
    next_goals = np.array(next_goals) # [N, goal_dim]
    rewards = np.array(rewards) #[N,1]
    if len(rewards.shape) == 1: 
        rewards = rewards.reshape(-1, 1) 
    dones = np.array(dones).reshape(-1, 1)  #[N,1]

    #assert dones[-1,:] == True, "A trajectory should end with done=True"
    
    return dict(
        observations=observations,
        goals=goals,
        actions=actions,
        rewards=rewards,
        next_observations=next_observations,
        next_goals=next_goals,
        dones=dones
    )

class MdpPathCollector(object, metaclass=abc.ABCMeta):
    def __init__(
            self,
            env,
            get_action_fn,
            max_num_epoch_paths_saved=None,
            render=False,
            rollout_fn=rollout,
            sample=True,
    ):
        self._env = env
        self._get_action_fn = get_action_fn
        self._max_num_epoch_paths_saved = max_num_epoch_paths_saved
        self._sample = sample
        # paths collected in the current epoch
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)
        # return of paths collected in the current epoch
        self._epoch_returns = []
        self._render = render
        self._rollout_fn = rollout_fn

        self._num_steps_total = 0
        self._num_paths_total = 0


    # collect n transitions and return them
    # do not check the repeat of (s,a,r,s')
    def collect_new_paths(
            self,
            max_path_length,
            num_steps
    ):
        paths = []
        num_steps_collected = 0
        while num_steps_collected < num_steps:
            max_path_length_this_loop = min(  # Do not go over num_steps
                max_path_length,
                num_steps - num_steps_collected,
            )

            # collect a set of (s,a,r,s') from one path (<= max steps needed)
            path = self._rollout_fn(
                env=self._env,
                get_action_fn=self._get_action_fn,
                max_path_length=max_path_length_this_loop,
                sample=self._sample,
                render=self._render
            )
            path_len = len(path['actions'])

            num_steps_collected += path_len
            paths.append(path)


            path_return = np.sum(path['rewards'])
            self._epoch_returns.append(path_return)
        
        self._num_paths_total += len(paths)
        self._num_steps_total += num_steps_collected
        self._epoch_paths.extend(paths)
        
        return paths

    def get_epoch_paths(self):
        return self._epoch_paths

    # clear epochs collected in the current epoch 
    # call at the end of each epoch
    def end_epoch(self, epoch):
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)
        self._epoch_returns = []

     # get stats: call at the end of each epoch
    def get_diagnostics(self):
        epoch_path_lens = [len(path['actions']) for path in self._epoch_paths]
        stats = OrderedDict([
            ('Exploration/num_steps_total', self._num_steps_total),
            ('Exploration/num_paths_total', self._num_paths_total),
        ])
        # mean/std/max/min
        # dict_A.update(dict_B): append dict_B to dict_A
        stats.update(create_stats_ordered_dict(
            "Exploration/epoch_path_length",
            epoch_path_lens,
            always_show_all_stats=True,
            exclude_max_min=True,
            exclude_std=True
        ))
        return stats

    def get_snapshot(self):
        return {}

if __name__ == "__main__":
    print("Done")

