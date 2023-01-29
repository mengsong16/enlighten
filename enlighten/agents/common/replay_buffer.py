from gym.spaces import Discrete
from collections import OrderedDict
from enlighten.agents.common.other import get_obs_channel_num
import numpy as np
import warnings

# on cpu
class EnvReplayBuffer():
    def __init__(
            self,
            config
    ):
        """
        :param max_replay_buffer_size:
        :param env:
        """
        self.config = config  # config is a dictionary

        self._goal_dim = int(self.config.get("goal_dimension")) 
        self._obs_channel = get_obs_channel_num(self.config)
        if self._obs_channel == 0:
            print("Error: channel of observation input to the encoder is 0")
            exit()
        self._obs_width = int(self.config.get("image_width")) 
        self._obs_height = int(self.config.get("image_height"))

        self._max_replay_buffer_size = int(self.config.get("max_replay_buffer_size"))
        self._observations = np.zeros((self._max_replay_buffer_size, self._obs_channel, self._obs_height, self._obs_width), dtype=np.float32)
        self._goals = np.zeros((self._max_replay_buffer_size, self._goal_dim), dtype=np.float32)
        # It's a bit memory inefficient to save the observations twice,
        # but it makes the code *much* easier since you no longer have to
        # worry about termination conditions.
        self._next_obs = np.zeros((self._max_replay_buffer_size, self._obs_channel, self._obs_height, self._obs_width), dtype=np.float32)
        self._next_goal = np.zeros((self._max_replay_buffer_size, self._goal_dim), dtype=np.float32)
        self._actions = np.zeros((self._max_replay_buffer_size, 1), dtype=np.uint8)
        # Make everything a 2D np array to make it easier for other code to
        # reason about the shape of the data
        self._rewards = np.zeros((self._max_replay_buffer_size, 1), dtype=np.float32)
        # self._dones[i] = a done was received at time i
        self._dones = np.zeros((self._max_replay_buffer_size, 1), dtype=np.uint8)
        
        # sample batch with replacement or not
        self._replace = self.config.get("sample_with_replace")

        self._top = 0
        self._size = 0

    def add_sample(self, observation, 
                goal, 
                action,     
                reward, 
                next_observation, 
                next_goal,
                done):
        
        self._observations[self._top] = observation
        self._goals[self._top] = goal
        self._actions[self._top] = action
        self._rewards[self._top] = reward
        self._dones[self._top] = done
        self._next_obs[self._top] = next_observation
        self._next_goal[self._top] = next_goal

        self._advance()


    def terminate_episode(self):
        pass

    def clear(self):
        self._top = 0
        self._size = 0
        self._episode_starts = []
        self._cur_episode_start = 0

    def _advance(self):
        self._top = (self._top + 1) % self._max_replay_buffer_size
        if self._size < self._max_replay_buffer_size:
            self._size += 1

    # [o,g,a,r,o',g',d]
    def random_batch(self, batch_size):
        indices = np.random.choice(self._size, size=batch_size, replace=self._replace or self._size < batch_size)
        if not self._replace and self._size < batch_size:
            warnings.warn('Replace was set to false, but is temporarily set to true because batch size is larger than current size of replay.')
        
        # return a dictionary of numpy arrays
        batch = dict(
            observations=self._observations[indices],
            goals=self._goals[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            dones=self._dones[indices],
            next_observations=self._next_obs[indices],
            next_goals=self._next_goal[indices]
        )
        
        return batch

    # number of transitions in the replay buffer
    def num_steps_can_sample(self):
        return self._size

    # get stats: call at the end of each epoch
    def get_diagnostics(self):
        return OrderedDict([
            ('Replay_buffer/buffer_size', self._size)
        ])
    
    def add_path(self, path):
        """
        Add a path to the replay buffer.

        This default implementation naively goes through every step, but you
        may want to optimize this.

        NOTE: You should NOT call "terminate_episode" after calling add_path.
        It's assumed that this function handles the episode termination.

        :param path: Dict like one outputted by rlkit.samplers.util.rollout
        """
        for i, (
                obs,
                goal,
                action,
                reward,
                next_obs,
                next_goal,
                done
        ) in enumerate(zip(
            path["observations"],
            path["goals"],
            path["actions"],
            path["rewards"],
            path["next_observations"],
            path["next_goals"],
            path["dones"]
        )):
            self.add_sample(
                observation=obs,
                goal=goal,
                action=action,
                reward=reward,
                next_observation=next_obs,
                next_goal=next_goal,
                done=done,
            )
        self.terminate_episode()

    def add_paths(self, paths):
        for path in paths:
            self.add_path(path)
    
    def get_snapshot(self):
        return {}

    def end_epoch(self, epoch):
        return