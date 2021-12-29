#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import abc

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
    def __init__(self, net, dim_actions, attention=False):
        super().__init__()
        # visual recurrent encoder
        self.net = net
        self.dim_actions = dim_actions
        self.attention = attention

        self.action_distribution = CategoricalNet(
            self.net.output_size, self.dim_actions
        )
        self.critic = CriticHead(self.net.output_size)

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

        if deterministic:
            action = distribution.mode()
        else:
            action = distribution.sample()

        action_log_probs = distribution.log_probs(action)

        return value, action, action_log_probs, rnn_hidden_states

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks):
        # visual rnn encoder
        features, _, _ = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        return self.critic(features)

    
    def get_attention_map(self, observations, rnn_hidden_states, prev_actions, masks):
        assert self.attention==True, "Error: attention should be set to True"
        # visual rnn encoder
        _, _, attention_map = self.net(
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
        features, rnn_hidden_states, _ = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        # actor
        distribution = self.action_distribution(features)
        # critic
        value = self.critic(features)

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
        hidden_size: int = 512,
        attention: bool=False,
        **kwargs
    ):
        visual_encoder = CNNEncoder(observation_space=observation_space, 
            output_size=hidden_size)

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
                action_space=action_space,
                visual_encoder=visual_encoder,
                goal_visual_encoder=goal_visual_encoder,
                hidden_size=hidden_size,
                polar_point_goal=polar_point_goal,
                rnn_type=rnn_type,
                **kwargs,
            ),
            dim_actions = action_space.n,
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
        baseplanes: int = 32,
        backbone: str = "resnet18",
        normalize_visual_inputs: bool = False,
        hidden_size: int = 512,
        attention: bool=False,
        **kwargs
    ):
        
        visual_encoder = ResNetEncoder(observation_space=observation_space, 
            output_size=hidden_size,
            baseplanes=baseplanes,
            make_backbone=getattr(resnet, backbone),
            normalize_visual_inputs=normalize_visual_inputs,
            attention=attention)

        
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
                action_space=action_space,
                visual_encoder=visual_encoder,
                goal_visual_encoder=goal_visual_encoder,
                hidden_size=hidden_size,
                polar_point_goal=polar_point_goal,
                rnn_type=rnn_type,
                attention=attention,
                **kwargs,
            ),
            dim_actions = action_space.n,
            attention = attention
        ) 

if __name__ == "__main__": 
    print('Done.')