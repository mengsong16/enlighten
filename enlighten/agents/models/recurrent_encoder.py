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

    # @property
    # @abc.abstractmethod
    # def is_blind(self):
    #     pass

class RecurrentVisualEncoder(Net):
    r"""Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    def __init__(
        self,
        goal_observation_space,
        action_space,
        visual_encoder,
        hidden_size, # output size of visual encoder
        goal_input_location,
        num_recurrent_layers=1,
        rnn_type="gru",
        attention_type="caption",
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
        self.goal_input_size = 0

        self.goal_input_location = goal_input_location

        # goal embedding 
        if goal_observation_space is not None: 
            # pointgoal
            if len(goal_observation_space.shape) < 3:
                if self.polar_point_goal:
                    n_input_goal = goal_observation_space.shape[0] + 1
                else:    
                    n_input_goal = goal_observation_space.shape[0]

                self.goal_input_size = 32    
                self.goal_encoder = nn.Linear(n_input_goal, self.goal_input_size)

                if self.goal_input_location == "baseline":
                    other_input_size += self.goal_input_size
            # imagegoal  
            else:
                assert goal_visual_encoder is not None, "goal visual encoder is None but image goal is used"
                self.goal_encoder = goal_visual_encoder
                
                self.goal_input_size = hidden_size

                if self.goal_input_location == "baseline":
                    other_input_size += self.goal_input_size
        else:
            self.goal_encoder = None        


        # visual observation embedding
        self.visual_encoder = visual_encoder

        # RNN state embedding
        self._hidden_size = hidden_size

        # let visual encoder check whether its observation space is none
        if self.visual_encoder is None:
            visual_encoder_output_size = 0
        elif self.visual_encoder.is_blind:
            visual_encoder_output_size = 0
        else: 
            visual_encoder_output_size = self.visual_encoder.output_size   
            
            # if attention:
            #     # visual encoder output a feature map where each pixel has channel self.visual_encoder.dim
            #     visual_embedding_size = self.visual_encoder.dim
                
            # else:
            #     # visual encoder output a vector which equals to the hidden size of RNN
            #     visual_embedding_size =  self._hidden_size   
        
        if self.visual_encoder is not None:
            visual_map_size = self.visual_encoder.visual_feature_map_dim
        else:
            visual_map_size = 0

        # create RNN model
        self.state_encoder = build_attention_rnn_state_encoder(
            attention,
            visual_encoder_output_size,
            other_input_size,
            self._hidden_size,
            visual_map_size=visual_map_size,
            rnn_type=rnn_type,
            attention_type=attention_type,
            num_layers=num_recurrent_layers
        )

    # @property
    # def goal_input_size(self):
    #     return self.goal_input_size
    
    @property
    def output_size(self):
        return self._hidden_size 

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
        if self.visual_encoder is None:
            visual_input = None
        elif self.visual_encoder.is_blind:
            visual_input = None
        else:
            visual_input = self.visual_encoder(observations)
      
        other_input = []
        # goal embedding
        goal_embedding = None
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
                

        if goal_embedding is not None:
            if self.goal_input_location == "baseline":
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

        # if self.goal_input_location == "baseline":
        #     return out, rnn_hidden_states, patch_weights
        # elif self.goal_input_location == "value_function":
        #     return out, rnn_hidden_states, patch_weights, goal_embedding
        # else:
        #     print("undefined goal input location in recurrent_encoder.py")

        return out, rnn_hidden_states, patch_weights, goal_embedding    


