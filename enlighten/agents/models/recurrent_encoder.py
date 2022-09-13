import gym
from gym import spaces
from typing import Dict

import numpy as np
import torch
from torch import nn as nn

from enlighten.envs import ImageGoal, PointGoal, goal
from enlighten.agents.models import build_attention_rnn_state_encoder
from enlighten.agents.models.mlp_encoder import MLPEncoder

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
        observation_space,
        action_space,
        visual_encoder,
        hidden_size, # output size of visual encoder, rnn hidden state size
        goal_input_location,
        num_recurrent_layers=1,
        rnn_type="gru",
        attention_type="caption",
        polar_point_goal=False,
        goal_visual_encoder=None,
        attention=False,
        rnn_policy = True,
        state_only = False,
        polar_state = True,
        cos_augmented_goal = False,
        cos_augmented_state = False
    ):
        super().__init__()

        self.rnn_policy = rnn_policy
        self.state_only = state_only
        self.polar_state = polar_state
        self.cos_augmented_goal = cos_augmented_goal
        self.cos_augmented_state = cos_augmented_state

        if self.polar_state:
            print("======> polor state")
        if self.cos_augmented_state:
            print("======> state is cos augmented")
        else:
            print("======> state is NOT cos augmented")   

             

        other_state_encoder_input_size = 0
        # action embedding only required by rnn
        if rnn_policy:
            # index --> embedding vector 
            # num_embeddings: dictionary size
            # make the index start from 1 instead of 0, preserve 0 as unknown
            self.prev_action_encoder = nn.Embedding(num_embeddings=action_space.n+1, embedding_dim=32)
            self._n_prev_action = 32
            other_state_encoder_input_size += self._n_prev_action
        

        # goal embedding 
        self.polar_point_goal = polar_point_goal
        self.goal_input_location = goal_input_location

        if self.polar_point_goal:
            print("======> polor goal")
        if self.cos_augmented_goal:
            print("======> goal is cos augmented")
        else:
            print("======> goal is NOT cos augmented")
         
        if goal_observation_space is not None: 
            # pointgoal
            if len(goal_observation_space.shape) < 3:
                if self.polar_point_goal:
                    if self.cos_augmented_goal:
                        self.input_point_goal_size = goal_observation_space.shape[0] + 1
                    else:    
                        self.input_point_goal_size = goal_observation_space.shape[0]
                else:    
                    self.input_point_goal_size = goal_observation_space.shape[0]

                self.goal_format = "pointgoal"

                # goal will be fed into the following state encoder
                if state_only == False:
                    self.goal_input_size = 32    
                    self.goal_encoder = nn.Linear(self.input_point_goal_size, self.goal_input_size)

                    if self.goal_input_location == "baseline":
                        other_state_encoder_input_size += self.goal_input_size

                   
            # imagegoal  
            else:
                assert goal_visual_encoder is not None, "goal visual encoder is None but image goal is used"
                self.goal_encoder = goal_visual_encoder

                self.goal_format = "imagegoal"
                self.goal_input_size = hidden_size
                # goal will be fed into the following state encoder
                if state_only == False:
                    if self.goal_input_location == "baseline":
                        other_state_encoder_input_size += self.goal_input_size
        # no goal
        else:
            self.goal_encoder = None 
            self.goal_input_size = 0
            self.goal_format = "nogoal"       


        # visual observation embedding
        self.visual_encoder = visual_encoder

        # state encoder
        if state_only:
            if self.polar_state:
                if self.polar_state:
                    if self.cos_augmented_state:
                        input_state_size = observation_space["state_sensor"].shape[0] + 1
                    else:    
                        input_state_size = observation_space["state_sensor"].shape[0]
            else:    
                input_state_size = observation_space["state_sensor"].shape[0]
 
             
            # concatenate state with point goal
            # point goal does not have goal encoder
            if self.goal_format == "pointgoal":
                input_state_size  += self.input_point_goal_size
            # concatenate state with image goal
            # image goal has its own goal encoder
            elif self.goal_format == "imagegoal":
                input_state_size  += self.goal_input_size
    
                   
            self.state_embed_encoder = MLPEncoder(input_dim=input_state_size, output_dim=hidden_size)
            # state output size is equal to hidden size of RNN
            visual_state_encoder_output_size = hidden_size
        # visual encoder
        else:
            # let visual encoder check whether its observation space is none
            if self.visual_encoder is None:
                visual_state_encoder_output_size = 0
            elif self.visual_encoder.is_blind:
                visual_state_encoder_output_size = 0
            else: 
                visual_state_encoder_output_size = self.visual_encoder.output_size   
            

        # hidden size = final output size
        self._hidden_size = hidden_size

        
        # rnn policy
        if rnn_policy:
            # visual map size that will be used in RNN model
            if self.visual_encoder is not None:
                visual_map_size = self.visual_encoder.visual_feature_map_dim
            else:
                visual_map_size = 0
            # create RNN model
            self.state_encoder = build_attention_rnn_state_encoder(
                attention,
                visual_state_encoder_output_size,
                other_state_encoder_input_size,
                self._hidden_size,
                visual_map_size=visual_map_size,
                rnn_type=rnn_type,
                attention_type=attention_type,
                num_layers=num_recurrent_layers
            )
        # mlp policy    
        else:
            # create MLP model
            if self.state_only:
                self.state_encoder_layer = 0
            else:
                self.state_encoder_layer = 2    
            
            if self.state_encoder_layer > 0:
                self.state_encoder = MLPEncoder(input_dim=visual_state_encoder_output_size+other_state_encoder_input_size, 
                    hidden_layer=self.state_encoder_layer, output_dim=hidden_size)  
            else:
                self.state_encoder = None     

    # hidden size = final output size
    @property
    def output_size(self):
        return self._hidden_size 

    @property
    def num_recurrent_layers(self):
        if self.rnn_policy:
            return self.state_encoder.num_recurrent_layers
        else:
            return 0    

    # augment angle with sin and cos if using polar observation
    def augment_polar_observation(self, goal_observations):
        if goal_observations.shape[1] == 2:
                # Polar Dimensionality 2 --> 3
                # (r, -phi) --> (r, cos (-phi), sin (-phi))
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
            # (r, -phi, theta) --> (r, cos (-phi), sin (-phi), cos theta)
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
      
        other_input = []
       
        # goal embedding
        if "pointgoal" in observations:
            goal_observations = observations["pointgoal"]
            if self.polar_point_goal:
                if self.cos_augmented_goal:
                    goal_observations = self.augment_polar_observation(goal_observations)    
            
            if self.state_only == False:
                goal_embedding = self.goal_encoder(goal_observations)
                if self.goal_input_location == "baseline":
                    other_input.append(goal_embedding)
            else:
                goal_embedding = None        
        elif "imagegoal" in observations:
            image_goal = observations["imagegoal"]
            # input should be a dictionary when using a visual encoder
            goal_embedding = self.goal_encoder({"color_sensor": image_goal})
            if self.state_only == False:
                if self.goal_input_location == "baseline":
                    other_input.append(goal_embedding)
        else:
            goal_embedding = None        

        
        # action embedding only required by rnn
        if self.rnn_policy:
            # action embedding
            prev_actions = prev_actions.squeeze(-1)
            start_token = torch.zeros_like(prev_actions)
            # not done: action index, done: 0
            
            # input of nn.embedding should be long
            # done=True: prev_action = 0, done=False: prev_action+1
            prev_action_embedding = self.prev_action_encoder(
                (torch.where(masks.view(-1), prev_actions+1, start_token)).long()
            )

            other_input.append(prev_action_embedding)
            
        # concate other input
        if other_input:
            other_input = torch.cat(other_input, dim=1)
        else:    
            other_input = None

        # state encoder
        if self.state_only:
            state_observations = observations["state_sensor"]
            if self.polar_state:
                if self.cos_augmented_state:
                    state_observations = self.augment_polar_observation(state_observations)
            
            # concate with point goal
            if "pointgoal" in observations:
                state_observations = torch.cat((state_observations, goal_observations), dim=1)
            # concate with image goal
            elif "imagegoal" in observations: 
                state_observations = torch.cat((state_observations, goal_embedding), dim=1)


            visual_input = self.state_embed_encoder(state_observations)  
    
        # visual encoder
        else:
            # visual observation embedding
            if self.visual_encoder is None:
                visual_input = None
            elif self.visual_encoder.is_blind:
                visual_input = None
            else:
                visual_input = self.visual_encoder(observations)
        # rnn policy
        if self.rnn_policy:
            # forward RNNStateEncoder
            # out will be used to produce V(s) and policy
            out, rnn_hidden_states, patch_weights = self.state_encoder(
                visual_input, other_input, rnn_hidden_states, masks
            )

            return out, rnn_hidden_states, patch_weights, goal_embedding
        # mlp policy
        else:
            
            if other_input is not None:
                input_tensor = torch.cat((visual_input, other_input), dim=1)
            else:
                input_tensor = visual_input    

            if self.state_encoder_layer > 0:    
                out = self.state_encoder(input_tensor)
            else:
                out = input_tensor
                assert out.size(dim=1) == self._hidden_size, "If no state encoder, the output should have size: %d"%(self._hidden_size)
            return out, None, None, None

            


