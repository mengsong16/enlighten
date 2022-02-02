from typing import Dict, List, Optional, Type, Union, cast
from gym import spaces
import numpy as np

import torch
from torch import nn as nn
from torch.nn import functional as F
from torch import Tensor
from torch.nn.modules.container import Sequential
from torch.nn.modules.conv import Conv2d

from enlighten.agents.models import RunningMeanAndVar

#from torchsummary import summary
from torchinfo import summary

from gym.spaces import Dict as SpaceDict
import gym

from enlighten.agents.models import resnet

# Allow rgb and depth
class VisualEncoder(nn.Module):
    r"""A Simple 3-Conv CNN followed by a fully connected layer

    Takes in observations and produces an embedding of the rgb and/or depth components

    Args:
        observation_space: The observation_space of the agent
        output_size: The size of the embedding vector
    """

    def __init__(self, observation_space: spaces.Dict):
        super().__init__()

        if "color_sensor" in observation_space.spaces:
            self._n_input_rgb = observation_space.spaces["color_sensor"].shape[2]
        else:
            self._n_input_rgb = 0

        
        if "depth_sensor" in observation_space.spaces:
            self._n_input_depth = observation_space.spaces["depth_sensor"].shape[2]
        else:
            self._n_input_depth = 0

    def layer_init(self):
        # for layer in self.modules():
        for layer in self.visual_encoder:  # type: ignore
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(
                    layer.weight, nn.init.calculate_gain("relu")
                )
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)
    @property
    def is_blind(self):
        return self._n_input_rgb + self._n_input_depth == 0

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

        return vision_inputs


class CNNEncoder(VisualEncoder):
    def __init__(self, observation_space: spaces.Dict, output_size):
        super().__init__(observation_space)

        # kernel size for different CNN layers
        self._cnn_layers_kernel_size = [(8, 8), (4, 4), (3, 3)]

        # strides for different CNN layers
        self._cnn_layers_stride = [(4, 4), (2, 2), (1, 1)]

        if not self.is_blind:
            self.create_model(observation_space, output_size)  
            self.layer_init()

        #print("is blind: "+str(self.is_blind))
        

    def create_model(self, observation_space, output_size):
        # calculate the output height and width of each layer
        if self._n_input_rgb > 0:
            cnn_dims = np.array(
                observation_space.spaces["color_sensor"].shape[:2], dtype=np.float32
            )
        elif self._n_input_depth > 0:
            cnn_dims = np.array(
                observation_space.spaces["depth_sensor"].shape[:2], dtype=np.float32
            )

        for kernel_size, stride in zip(
            self._cnn_layers_kernel_size, self._cnn_layers_stride
        ):
            cnn_dims = self._conv_output_dim(
                dimension=cnn_dims,
                padding=np.array([0, 0], dtype=np.float32),
                dilation=np.array([1, 1], dtype=np.float32),
                kernel_size=np.array(kernel_size, dtype=np.float32),
                stride=np.array(stride, dtype=np.float32),
            )

        # create cnn layers
        self.visual_encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=self._n_input_rgb + self._n_input_depth,
                out_channels=32,
                kernel_size=self._cnn_layers_kernel_size[0],
                stride=self._cnn_layers_stride[0],
            ),
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=self._cnn_layers_kernel_size[1],
                stride=self._cnn_layers_stride[1],
            ),
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=64,
                out_channels=32,
                kernel_size=self._cnn_layers_kernel_size[2],
                stride=self._cnn_layers_stride[2],
            ),
            #  nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(32 * cnn_dims[0] * cnn_dims[1], output_size),
            nn.ReLU(True),
        )

        #summary(self.visual_encoder, (3,224,224), device="cpu")
        #summary(self.visual_encoder, input_size=(1,3,224,224))
        #exit()
    
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

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.is_blind:
            return None

        vision_inputs = self.get_vision_inputs(observations)
        vision_inputs = torch.cat(vision_inputs, dim=1)

        
        return self.visual_encoder(vision_inputs)
           

class ResNetEncoder(VisualEncoder):
    def __init__(self, 
        observation_space: spaces.Dict,
        output_size: int = 32,
        baseplanes: int = 32,
        ngroups: int = 32,
        spatial_size: int = 128,
        make_backbone=None,
        normalize_visual_inputs: bool = False,
        attention: bool=False):

        super().__init__(observation_space)

        if "color_sensor" in observation_space.spaces:
            spatial_size = observation_space.spaces["color_sensor"].shape[0] // 2

        if "depth_sensor" in observation_space.spaces:
            spatial_size = observation_space.spaces["depth_sensor"].shape[0] // 2

        if normalize_visual_inputs:
            self.running_mean_and_var: nn.Module = RunningMeanAndVar(
                self._n_input_depth + self._n_input_rgb
            )
        else:
            self.running_mean_and_var = nn.Sequential()

        self.output_size = output_size
        self.attention = attention

        #/////////////////
        if self.attention:
            self.output_size = 256
        #/////////////////    

        if not self.is_blind:
            self.create_model(baseplanes, 
                ngroups,
                spatial_size,
                make_backbone,
                attention)

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
        spatial_size,
        make_backbone,
        attention,
        compression=False):  # by default, no compression, to be consistent with attention mode

        input_channels = self._n_input_depth + self._n_input_rgb

        self.backbone = make_backbone(input_channels, baseplanes, ngroups)

        if attention == False:

            # expected spatial size after compression 
            final_spatial = int(
                spatial_size * self.backbone.final_spatial_compress
            )

            after_compression_flat_size = 2048
            num_compression_channels = int(
                round(after_compression_flat_size / (final_spatial ** 2))
            )

            if compression:
                self.compression = nn.Sequential(
                    nn.Conv2d(
                        self.backbone.final_channels,
                        num_compression_channels,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                    nn.GroupNorm(1, num_compression_channels),
                    nn.ReLU(True),
                )

                # after_compression_dims = np.array(
                #         (before_dims, before_dims), dtype=np.float32
                #     )
                # final_dims = self._conv_output_dim(
                #         dimension=after_compression_dims,
                #         padding=np.array([1, 1], dtype=np.float32),
                #         dilation=np.array([1, 1], dtype=np.float32),
                #         kernel_size=np.array([3, 3], dtype=np.float32),
                #         stride=np.array([1, 1], dtype=np.float32),
                #     )
                # after_compression_spatial = 7
                # print(final_spatial)
                # print(final_dims)
                
                output_shape = (
                        num_compression_channels,
                        final_spatial,
                        final_spatial,
                    )
            else:
                output_shape = (
                        256,
                        7,
                        7,
                    )
                #print(self.output_shape)    

            # 2048 print(after_compression_flat_size)
            # 228 print(num_compression_channels)
            # 512 print(self.output_size)
            # 3 print(final_spatial)
            # 0.03125 print(self.backbone.final_spatial_compress)
            # 112 print(spatial_size)

            self.fc = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(
                        np.prod(output_shape), self.output_size
                    ),
                    nn.ReLU(True),
            )

            if compression:
                self.visual_encoder = nn.Sequential(
                    self.running_mean_and_var,
                    self.backbone,
                    self.compression,
                    self.fc
                )
            else:    
                self.visual_encoder = nn.Sequential(
                    self.running_mean_and_var,
                    self.backbone,
                    self.fc
                )
            # dummy
            self.visual_feature_map_dim = 0    
        # add attention  
        # remove the last block 
        # for res18, output shape is (128, 14, 14)
        # don't remove the last block
        # for res18, output shape is (256, 7, 7)
        else:
            self.visual_encoder = nn.Sequential(
                self.running_mean_and_var,
                #*list(self.backbone.modules())#[:-1]
                *list(self.backbone.children())
                #*list(self.backbone.children())[:-1],
                #self.backbone
            ) 
            # visual feature map dimension (only for attention)
            #self.visual_feature_map_dim = 128 # 14*14
            self.visual_feature_map_dim = 256 #256  # 7*7     

        #print(self.visual_encoder.children().children())
        #print(type(self.visual_encoder))
        #summary(self.visual_encoder, (3,224,224), device="cpu")
        #print(attention)
        #print(*list(self.backbone.modules())[:-1])
        #summary(self.visual_encoder, (3,224,224), device="cpu")
        #summary(self.visual_encoder, input_size=(4,3,224,224))
        #exit()
    
    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.is_blind:
            return None

        vision_inputs = self.get_vision_inputs(observations)
        vision_inputs = torch.cat(vision_inputs, dim=1)
        #vision_inputs = F.avg_pool2d(vision_inputs, 2)

        if self.attention == False:
            return self.visual_encoder(vision_inputs)
        else:
            x = self.visual_encoder(vision_inputs)
            # （1，128，14，14） --> (1,14, 14, 128)
            x = x.permute(0, 2, 3, 1)
            # (1, 14, 14, 128) --> (1, 196, 128)
            x = x.view(x.size(0), -1, x.size(-1)) 
            return x    

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
    resnet_policy = ResNetEncoder(observation_space=SpaceDict({"color_sensor": rgb_space}),
                                make_backbone=getattr(resnet, "resnet18"),
                                output_size=512            )
    print('Done')      