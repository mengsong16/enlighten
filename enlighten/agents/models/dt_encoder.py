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
        channel_num: int,
        output_size: int,  # fc output size
        baseplanes: int = 32,
        ngroups: int = 32):

        super().__init__()

        self.channel_num = channel_num
        # for group norm
        self.running_mean_and_var: nn.Module = RunningMeanAndVar(
            self.channel_num
        )
        

        self.fc_output_size = output_size
        
        # create Resnet model and initialize
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

        input_channels = self.channel_num
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
        
        #summary(self.visual_encoder, (16,3,224,224), device="cpu")
        
    
    # initialize layers in visual_encoder
    def layer_init(self):
        for layer in self.visual_encoder:  # type: ignore
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(
                    layer.weight, nn.init.calculate_gain("relu")
                )
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)

    # (B,C,H,W)
    def normalize_vision_inputs(self, observations):
        channel_n = observations.size()[1]
        # normalize RGB to [0,1]
        if channel_n >= 3:
            observations[:,0:3,:,:] = observations[:,0:3,:,:] / 255.0
        
        return observations

    def get_device(self):
        return next(self.parameters()).device

    def from_numpy_to_tensor(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).float()
        
        x = x.to(self.get_device())
        return x


    def forward(self, observations) -> torch.Tensor:
         # ensure states are tensors on the same device
        observations = self.from_numpy_to_tensor(observations)

        observations = self.normalize_vision_inputs(observations)

        return self.visual_encoder(observations)

# adapt from dt for gym
class TimestepEncoder(nn.Module):
    def __init__(self, max_len_episode, output_size):
        super().__init__()
        # a lookup table for the embeddings by index
        self.model = nn.Embedding(max_len_episode, output_size)

    def forward(self, timesteps):
        return self.model(timesteps)

# adapt from return to go encoder from dt for gym
class DistanceToGoalEncoder(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        
        self.model = torch.nn.Linear(1, output_size)

    def forward(self, RtGs):
        return self.model(RtGs)

# adapt from action encoder in ppo
class DiscreteActionEncoder(nn.Module):
    def __init__(self, num_actions, output_size):
        super().__init__()
        # a lookup table for the embeddings by index
        self.model = nn.Embedding(num_actions, output_size)

    def forward(self, actions):
        return self.model(actions)

# relative goal, adapt from goal encoder in ppo
class GoalEncoder(nn.Module):
    def __init__(self, goal_dim, output_size):
        super().__init__()
        self.model = torch.nn.Linear(goal_dim, output_size)

    def get_device(self):
        return next(self.parameters()).device

    def from_numpy_to_tensor(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).float()
        
        x = x.to(self.get_device())
        return x

    def forward(self, goals):
        # ensure states are tensors on the same device
        goals = self.from_numpy_to_tensor(goals)


        return self.model(goals)

# adapt from action decoder from dt for atari
class DiscreteActionDecoder(nn.Module):
    def __init__(self, input_size, action_number):
        super().__init__()
        # no bias
        self.model = torch.nn.Linear(input_size, action_number, bias=False)

    def forward(self, hidden_states):
        return self.model(hidden_states)  # logits

# value decoder
class ValueDecoder(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        # no bias
        self.model = torch.nn.Linear(input_size, 1, bias=False)

    def forward(self, hidden_states):
        return self.model(hidden_states)  # logits

# domain discriminator
class BinaryDiscriminator(nn.Module):
    def __init__(self, feature_size, hidden_size=512):
        super(BinaryDiscriminator, self).__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        self.ad_layer1 = nn.Linear(feature_size, hidden_size)
        self.ad_layer1.weight.data.normal_(0, 0.01)
        self.ad_layer1.bias.data.fill_(0.0)

        # output: one class logit
        self.ad_layer2 = nn.Linear(hidden_size, 1)
        self.ad_layer2.weight.data.normal_(0, 0.3)
        self.ad_layer2.bias.data.fill_(0.0)
        self.model = nn.Sequential(self.ad_layer1, nn.ReLU(), nn.Dropout(0.5), self.ad_layer2)

    # features: [B, feature_size]
    def forward(self, features):
        return self.model(features)

class AdversarialLayer(torch.autograd.Function):
    iter_num = 0
    max_iter = 200000 #12500
    @staticmethod
    def forward(ctx, input):
        AdversarialLayer.iter_num += 1
        return input * 1.0

    @staticmethod
    def backward(ctx, gradOutput):
        alpha = 10
        low = 0.0
        high = 1.0
        lamb = 2.0
        iter_num, max_iter = AdversarialLayer.iter_num, AdversarialLayer.max_iter 
        # print('iter_num {}'.format(iter_num))
        coeff = np.float(lamb * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)
        return -coeff * gradOutput
        
if __name__ == "__main__":
    oe = ObservationEncoder(channel_num=3, output_size=512)
    print('Done')    
