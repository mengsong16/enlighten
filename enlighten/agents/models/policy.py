#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import abc
from enlighten.agents.models.state_encoder import MLPEncoder

import torch
from gym import spaces
from torch import nn as nn

from typing import Dict, List, Optional, Type, Union, cast
from torch import Size, Tensor

from enlighten.agents.models import CNNEncoder, ResNetEncoder, RecurrentVisualEncoder
from enlighten.envs import NavEnv
from enlighten.agents.models import resnet

from gym.spaces import Dict as SpaceDict
import matplotlib.pyplot as plt
import skimage
import matplotlib.cm as cm
import torchvision
#from torchsummary import summary

import numpy as np
import gym

# one fc layer
class CriticHead(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Linear(input_size, 1)
        nn.init.orthogonal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        return self.fc(x)

# input: logits of Categorial distribution
# output: log prob distribution
# outupt: samples
class CustomFixedCategorical(torch.distributions.Categorical):  # type: ignore
    def sample(
        self, sample_shape: Size = torch.Size()  # noqa: B008
    ) -> Tensor:
        return super().sample(sample_shape).unsqueeze(-1)

    def log_probs(self, actions: Tensor) -> Tensor:
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)

# one fc layer
class CategoricalNet(nn.Module):
    def __init__(self, num_inputs: int, num_outputs: int) -> None:
        super().__init__()

        self.linear = nn.Linear(num_inputs, num_outputs)

        nn.init.orthogonal_(self.linear.weight, gain=0.01)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x: Tensor) -> CustomFixedCategorical:
        x = self.linear(x)
        return CustomFixedCategorical(logits=x)

# policy network, only support discrete action space
# observations, [prev_actions, hidden_states] --> net --> features, hidden_states
# Two heads:
# features --> categoricalNet --> current action distribution
# features --> critic --> V(s)
class Policy(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, net, dim_actions, goal_input_location, attention=False):
        super().__init__()
        # visual recurrent encoder, rnn or mlp
        self.net = net
        self.dim_actions = dim_actions
        self.attention = attention
        self.goal_input_location = goal_input_location


        if self.goal_input_location == "baseline":
            self.action_distribution = CategoricalNet(
                self.net.output_size, self.dim_actions
            )
            self.critic = CriticHead(self.net.output_size)
        elif self.goal_input_location == "value_function":
            self.action_distribution = CategoricalNet(
                self.net.output_size+self.net.goal_input_size, self.dim_actions
            )
            self.critic = CriticHead(self.net.output_size+self.net.goal_input_size)    
        else:
            print("Error: undefined goal input location in polcy.py")    

    def forward(self, *x):
        raise NotImplementedError

    # input: obs, rnn_hidden_states, prev_action
    # output: cur_action
    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
    ):
           
        rnn_hidden_states, distribution, value = self.get_net_output(observations, rnn_hidden_states, prev_actions, masks)

        if deterministic:  # greedy
            action = distribution.mode()
        else:  # sample
            action = distribution.sample()

        # get probs
        action_log_probs = distribution.log_probs(action)

        return value, action, action_log_probs, rnn_hidden_states

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks):
        # visual rnn encoder
        features, _, _, goal_embedding = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        if self.goal_input_location == "value_function":
            features = torch.cat([features, goal_embedding], dim=1)
        
        return self.critic(features)

    
    def get_attention_map(self, observations, rnn_hidden_states, prev_actions, masks):
        assert self.attention==True, "Error: attention should be set to True"
        # visual rnn encoder
        _, _, attention_map, _ = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        return attention_map

    def get_resized_attention_map(self, observations, rnn_hidden_states, prev_actions, masks):
        attention_map =  self.get_attention_map(observations, rnn_hidden_states, prev_actions, masks)
        shape_size = 7
        attention_map = attention_map.reshape(shape_size, shape_size)
        
        attention_map = torch.unsqueeze(attention_map, dim=0)

        size = (observations["color_sensor"].size()[1], observations["color_sensor"].size()[2])
        resize_operator = torchvision.transforms.Resize(size=size)
        attention_map = resize_operator(attention_map)
        # # [C,H,W] --> [H,W,C]
        attention_map = attention_map.permute(1,2,0)
        #alpha_img = skimage.transform.pyramid_expand(, upscale=224/14, sigma=20)
        #plt.imshow(alpha_img, alpha=0.8)
        #plt.set_cmap(cm.Greys_r)
        #plt.axis('off')

        
        return attention_map

    # get actor-critic output 
    def get_net_output(self, observations, rnn_hidden_states, prev_actions, masks):
        # get RecurrentVisualEncoder (visual rnn encoder) output
        # note that features is h sequence (T*N) during training and is h_t (N) during evaluation
        # therefore, value and distribution is also a sequence (T*N) during training
        
        # features are a sequence of hidden states (history) during training
        features, rnn_hidden_states, _, goal_embedding = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )

        # both actor and critic conditioned on goal
        if self.goal_input_location == "value_function":
            features = torch.cat([features, goal_embedding], dim=1)
        #exit()
        # actor 
        distribution = self.action_distribution(features)
        # critic
        value = self.critic(features)

        # print(features.size())
        # print(value.size())
        # print(rnn_hidden_states.size())
        # print("**************************")

        return rnn_hidden_states, distribution, value

    # rnn_hidden_states: h0 when evaluate a sequence, h_t when evaluate one step
    def evaluate_actions(
        self, observations, rnn_hidden_states, prev_actions, masks, action
    ):
        #print("==================================")
        #print(observations.size())
        # hidden_states size: [3,1,512]: no attention --> RNN --> [3,1,512]
        # no attention: the hidden state is independent of input and output of RNN unit, thus its batch size could be different
        # prev_actions size: [384, 1]
        # masks size: [384, 1]
        # observations size: [384, 1]
        #print(rnn_hidden_states.size())
        #print(prev_actions.size())  
        #print(masks.size())
        #print("==================================")
        #exit()
        #assert rnn_hidden_states.size()[0] == prev_actions.size()[0], "Error: size does not match"
        rnn_hidden_states, distribution, value = self.get_net_output(observations, rnn_hidden_states, prev_actions, masks)
        # get action distribution's entropy
        action_log_probs = distribution.log_probs(action)
        distribution_entropy = distribution.entropy()

        return value, action_log_probs, distribution_entropy, rnn_hidden_states


class CNNPolicy(Policy):
    def __init__(
        self,
        observation_space: spaces.Dict,
        goal_observation_space,
        polar_point_goal,
        action_space,
        rnn_type,
        attention_type,
        goal_input_location,
        hidden_size: int = 512,
        attention: bool=False,
        blind_agent = False,
        rnn_policy = True,
        state_only = False,
        polar_state = True,
        cos_augmented_goal = False,
        cos_augmented_state = False,
        **kwargs
    ):
        # CNN output dimension = RNN hidden size
        if blind_agent is False:
            if state_only:
                assert "state_sensor" in observation_space.spaces, "state sensor should be true"
                visual_encoder = None # the state encoder will be created later in the recurrent encoder
                print("===> agent is state only")   
            else:    
                visual_encoder = CNNEncoder(observation_space=observation_space, 
                    output_size=hidden_size)

        else:
            visual_encoder = None
            print("===> agent is blind")

        # point goal or no goal
        if goal_observation_space is None or len(goal_observation_space.shape) < 3:
            goal_visual_encoder = None   
        # image goal      
        else:
            goal_visual_encoder = CNNEncoder(observation_space=SpaceDict({"color_sensor": goal_observation_space}), 
                output_size=hidden_size)       
        
        super().__init__(
            net = RecurrentVisualEncoder(  
                goal_observation_space=goal_observation_space,
                observation_space=observation_space,
                action_space=action_space,
                visual_encoder=visual_encoder,
                goal_visual_encoder=goal_visual_encoder,
                hidden_size=hidden_size,
                goal_input_location=goal_input_location,
                polar_point_goal=polar_point_goal,
                rnn_type=rnn_type,
                attention_type=attention_type,
                rnn_policy = rnn_policy,
                state_only = state_only,
                polar_state = polar_state,
                cos_augmented_goal = cos_augmented_goal,
                cos_augmented_state = cos_augmented_state,
                **kwargs,
            ),
            dim_actions = action_space.n,
            goal_input_location = goal_input_location,
            attention = attention
        )  

class ResNetPolicy(Policy):
    def __init__(
        self,
        observation_space: spaces.Dict,
        goal_observation_space,
        polar_point_goal,
        action_space,
        rnn_type,
        attention_type,
        goal_input_location,
        baseplanes: int = 32,
        backbone: str = "resnet18",
        normalize_visual_inputs: bool = False,
        hidden_size: int = 512,
        attention: bool=False,
        blind_agent = False,
        rnn_policy = True,
        state_only = False,
        polar_state = True,
        cos_augmented_goal = False,
        cos_augmented_state = False,
        **kwargs
    ):
        # ResNet output dimension = RNN hidden size
        if blind_agent is False:
            if state_only:
                assert "state_sensor" in observation_space.spaces, "state sensor should be true"
                visual_encoder = None # the state encoder will be created later in the recurrent encoder
                print("===> agent is state only")   
            else:    
                visual_encoder = ResNetEncoder(observation_space=observation_space, 
                    output_size=hidden_size,
                    baseplanes=baseplanes,
                    make_backbone=getattr(resnet, backbone),
                    normalize_visual_inputs=normalize_visual_inputs,
                    attention=attention)
        else:
            visual_encoder = None        
            print("===> agent is blind")
        
        # point goal or no goal
        if goal_observation_space is None or len(goal_observation_space.shape) < 3:
            goal_visual_encoder = None
        # image goal    
        else:    
            goal_visual_encoder = ResNetEncoder(observation_space=SpaceDict({"color_sensor": goal_observation_space}), 
                output_size=hidden_size,
                baseplanes=baseplanes,
                make_backbone=getattr(resnet, backbone),
                normalize_visual_inputs=normalize_visual_inputs)    

        super().__init__(
            net = RecurrentVisualEncoder( 
                goal_observation_space=goal_observation_space,
                observation_space=observation_space,
                action_space=action_space,
                visual_encoder=visual_encoder,
                goal_visual_encoder=goal_visual_encoder,
                hidden_size=hidden_size,
                goal_input_location=goal_input_location,
                polar_point_goal=polar_point_goal,
                rnn_type=rnn_type,
                attention_type=attention_type,
                attention=attention,
                rnn_policy = rnn_policy,
                state_only = state_only,
                polar_state = polar_state,
                cos_augmented_goal = cos_augmented_goal,
                cos_augmented_state = cos_augmented_state,
                **kwargs,
            ),
            dim_actions = action_space.n,
            goal_input_location = goal_input_location, 
            attention = attention
        ) 

if __name__ == "__main__": 
    print('Done.')