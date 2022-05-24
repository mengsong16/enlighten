from gym import spaces
import numpy as np

import torch
from torch import nn as nn
from torch.nn import functional as F

from enlighten.agents.models import RunningMeanAndVar

from torchinfo import summary

from gym.spaces import Dict
import gym

from enlighten.agents.models import resnet18  # restnet backbone

class ObservationEncoder(nn.Module):
    def __init__(self, 
        observation_space: spaces.Dict,
        output_size: int,  # fc output size
        baseplanes: int = 32,
        ngroups: int = 32):

        super().__init__()

        if "color_sensor" in observation_space.spaces:
            self._n_input_rgb = observation_space.spaces["color_sensor"].shape[2]
        else:
            self._n_input_rgb = 0

        
        if "depth_sensor" in observation_space.spaces:
            self._n_input_depth = observation_space.spaces["depth_sensor"].shape[2]
        else:
            self._n_input_depth = 0

        # check if observation is valid
        if self._n_input_depth + self._n_input_rgb == 0:
            print("Error: channel of observation input to the encoder is 0")
            exit()

        # for group norm
        self.running_mean_and_var: nn.Module = RunningMeanAndVar(
            self._n_input_depth + self._n_input_rgb
        )
        

        self.fc_output_size = output_size
        
        # create model and initialize
        self.create_model(baseplanes, 
            ngroups,
            make_backbone=resnet18)

        self.layer_init()   

        

    def _conv_output_dim(
        self, dimension, padding, dilation, kernel_size, stride):
        r"""Calculates the output height and width based on the input
        height and width to the convolution layer.

        ref: https://pytorch.org/docs/master/nn.html#torch.nn.Conv2d
        """
        assert len(dimension) == 2
        out_dimension = []
        for i in range(len(dimension)):
            out_dimension.append(
                int(
                    np.floor(
                        (
                            (
                                dimension[i]
                                + 2 * padding[i]
                                - dilation[i] * (kernel_size[i] - 1)
                                - 1
                            )
                            / stride[i]
                        )
                        + 1
                    )
                )
            )
        return tuple(out_dimension)

    def create_model(self, 
        baseplanes,
        ngroups,
        make_backbone):  

        input_channels = self._n_input_depth + self._n_input_rgb
        self.backbone = make_backbone(input_channels, baseplanes, ngroups)

        output_shape = (256, 7, 7)
        self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(
                    np.prod(output_shape), self.fc_output_size
                ),
                nn.ReLU(True),
        )

        
        self.visual_encoder = nn.Sequential(
            self.running_mean_and_var,
            self.backbone,
            self.fc
        )
        
       
        
        print(type(self.visual_encoder))
        summary(self.visual_encoder, (3,224,224), device="cpu")
        
    
    # initialize layers in visual_encoder
    def layer_init(self):
        for layer in self.visual_encoder:  # type: ignore
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(
                    layer.weight, nn.init.calculate_gain("relu")
                )
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)

    def get_vision_inputs(self, observations):
        vision_inputs = []
        if self._n_input_rgb > 0:
            rgb_observations = observations["color_sensor"]
            # permute tensor from [BATCH x HEIGHT X WIDTH x CHANNEL] to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            rgb_observations = rgb_observations.permute(0, 3, 1, 2)
            rgb_observations = (
                rgb_observations.float() / 255.0
            )  # normalize RGB to [0,1]
            vision_inputs.append(rgb_observations)

        if self._n_input_depth > 0:
            depth_observations = observations["depth_sensor"]
            # permute tensor from [BATCH x HEIGHT X WIDTH x CHANNEL] to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            depth_observations = depth_observations.permute(0, 3, 1, 2)
            vision_inputs.append(depth_observations)
        
        # concat all channels
        vision_inputs = torch.cat(vision_inputs, dim=1)

        return vision_inputs

    def forward(self, observations) -> torch.Tensor:
        vision_inputs = self.get_vision_inputs(observations)

        return self.visual_encoder(vision_inputs)

class TimestepEncoder(nn.Module):
    def __init__(self, max_len_episode, output_size):
        super().__init__()
        self.model = nn.Embedding(max_len_episode, output_size)

    def forward(self, timesteps):
        super().__init__()
        return self.model(timesteps)

class ReturnToGoEncoder(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.model = torch.nn.Linear(1, output_size)

    def forward(self, RtGs):
        return self.model(RtGs)

class ActionEncoder(nn.Module):
    def __init__(self, action_dim, output_size):
        super().__init__()
        self.model = torch.nn.Linear(action_dim, output_size)

    def forward(self, actions):
        return self.model(actions)

class GoalEncoder(nn.Module):
    def __init__(self, goal_dim, output_size):
        super().__init__()
        self.model = torch.nn.Linear(goal_dim, output_size)

    def forward(self, goals):
        return self.model(goals)

class DiscreteActionDecoder(nn.Module):
    def __init__(self, input_size, action_dim):
        super().__init__()
        # no bias
        self.model = torch.nn.Linear(input_size, action_dim, bias=False)

    def forward(self, hidden_states):
        return self.model(hidden_states)  # logits

if __name__ == "__main__":
    rgb_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(
                224, 
                224,
                3,
            ),
            dtype=np.uint8,
        )
    oe = ObservationEncoder(observation_space=Dict({"color_sensor": rgb_space}),
                                output_size=512)
    print('Done')    