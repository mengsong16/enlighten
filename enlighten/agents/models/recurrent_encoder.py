import gym
from gym import spaces
from typing import Dict

import numpy as np
import torch
from torch import nn as nn

from enlighten.envs import ImageGoal, PointGoal, goal
from enlighten.agents.models import build_attention_rnn_state_encoder

import abc

from torchinfo import summary

class Net(nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        pass

    @property
    @abc.abstractmethod
    def output_size(self):
        pass

    @property
    @abc.abstractmethod
    def num_recurrent_layers(self):
        pass

    @property
    @abc.abstractmethod
    def is_blind(self):
        pass

class RecurrentVisualEncoder(Net):
    r"""Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    def __init__(
        self,
        goal_observation_space,
        action_space,
        visual_encoder,
        hidden_size: int, # output size of visual encoder
        num_recurrent_layers: int = 1,
        rnn_type: str="gru",
        polar_point_goal=False,
        goal_visual_encoder=None,
        attention=False
    ):
        super().__init__()

        self.polar_point_goal = polar_point_goal
        # action embedding: index --> embedding vector 
        # num_embeddings: dictionary size
        # make the index start from 1 instead of 0, preserve 0 as unknown
        self.prev_action_encoder = nn.Embedding(num_embeddings=action_space.n+1, embedding_dim=32)
        self._n_prev_action = 32
        other_input_size = self._n_prev_action

        # goal embedding 
        if goal_observation_space is not None: 
            # pointgoal
            if len(goal_observation_space.shape) < 3:
                if self.polar_point_goal:
                    n_input_goal = goal_observation_space.shape[0] + 1
                else:    
                    n_input_goal = goal_observation_space.shape[0]
                self.goal_encoder = nn.Linear(n_input_goal, 32)
                other_input_size += 32
            # imagegoal  
            else:
                self.goal_encoder = goal_visual_encoder
                other_input_size += hidden_size
        else:
            self.goal_encoder = None        


        # visual observation embedding
        self.visual_encoder = visual_encoder

        # RNN state embedding
        self._hidden_size = hidden_size

        if self.is_blind:
            visual_embedding_size = 0
        else:    
            if attention:
                # visual encoder output a feature map where each pixel has channel self.visual_encoder.dim
                visual_embedding_size = self.visual_encoder.dim
            else:
                # visual encoder output a vector which equals to the hidden size of RNN
                visual_embedding_size =  self._hidden_size   
        
        self.state_encoder = build_attention_rnn_state_encoder(
            attention,
            visual_embedding_size,
            other_input_size,
            self._hidden_size,
            rnn_type=rnn_type,
            num_layers=num_recurrent_layers
        )


    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    # augment angle with sin and cos if using polar point goal
    def augment_goal_observation(self, goal_observations):
        if goal_observations.shape[1] == 2:
                # Polar Dimensionality 2 --> 3
                # 2D polar transform
                goal_observations = torch.stack(
                    [
                        goal_observations[:, 0],
                        torch.cos(-goal_observations[:, 1]),
                        torch.sin(-goal_observations[:, 1]),
                    ],
                    -1,
                )
        else:
            assert (
                goal_observations.shape[1] == 3
            ), "Unsupported dimensionality"
            vertical_angle_sin = torch.sin(goal_observations[:, 2])
            # Polar Dimensionality 3 --> 4
            # 3D Polar transformation
            goal_observations = torch.stack(
                [
                    goal_observations[:, 0],
                    torch.cos(-goal_observations[:, 1])
                    * vertical_angle_sin,
                    torch.sin(-goal_observations[:, 1])
                    * vertical_angle_sin,
                    torch.cos(goal_observations[:, 2]),
                ],
                -1,
            )

        return goal_observations    

    # masks: not done masks: True (1): not done, False (0): done
    # single forward: rnn_hidden_states=h_{t-1}
    # sequence forward: rnn_hidden_states=h0
    # visual_input: [T*N,input_size] or [N, 1, hidden_size]
    # rnn_hidden_states: [N, 1, hidden_size] 
    # prev_actions: [T*N,input_size] or [N, 1, hidden_size]
    # note that prev_actions are not cut off to a_0 as rnn_hidden_states as rnn_hidden_states when doing sequence forward
    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        
        # visual observation embedding
        if not self.is_blind:
            visual_input = self.visual_encoder(observations)

        other_input = []
        # goal embedding
        if self.goal_encoder is not None:
            if "pointgoal" in observations:
                goal_observations = observations["pointgoal"]
                if self.polar_point_goal:
                    goal_observations = self.augment_goal_observation(goal_observations)
                goal_embedding = self.goal_encoder(goal_observations)
            elif "imagegoal" in observations:
                image_goal = observations["imagegoal"]
                # input should be a dictionary when using a visual encoder
                goal_embedding = self.goal_encoder({"color_sensor": image_goal})

            other_input.append(goal_embedding)
        
        # action embedding
        prev_actions = prev_actions.squeeze(-1)
        start_token = torch.zeros_like(prev_actions)
        # not done: action index, done: 0
        
        # input of nn.embedding should be long
        prev_action_embedding = self.prev_action_encoder(
            (torch.where(masks.view(-1), prev_actions+1, start_token)).long()
        )

        other_input.append(prev_action_embedding)
        
        other_input = torch.cat(other_input, dim=1)

        
        # forward RNNStateEncoder
        # out will be used to reduce V(s) and policy
        out, rnn_hidden_states, patch_weights = self.state_encoder(
            visual_input, other_input, rnn_hidden_states, masks
        )

        return out, rnn_hidden_states, patch_weights

