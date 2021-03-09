import jax.numpy as jnp
from flax import linen as nn
from typing import Any, Callable
from functools import partial
from jax import random
import numpy as onp
import jax
from IPython import embed

ModuleDef = Any

kaiming_normal = partial(jax.nn.initializers.variance_scaling, 2.0, "fan_out", "truncated_normal")


def dilated_conv3x3(x, features, strides=1, groups=1, dilation=1, name='dilated_conv3x3'):
    """3x3 convolution with padding"""

    d = max(1, dilation)
    return nn.Conv(features, kernel_size=(3, 3), strides=(strides,strides), padding=((dilation,dilation),(dilation,dilation)),
                   kernel_dilation=(d, d), feature_group_count=groups, use_bias=False, name=name)(x)
    # return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
    #                  padding=dilation, groups=groups, bias=False, dilation=dilation)



def conv1x1(features, stride=1):
    """1x1 convolution"""
    return nn.Conv(features=features, kernel_size=(1, 1), strides=(stride,stride), padding='VALID', use_bias=False)
    # return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)



#
# class FeaturePyramidNetwork(nn.Module):
#     def __init__(self, in_channels, out_channels=128,
#                  num_levels=3):
#         # FPN paper uses 256 out channels by default
#         super(FeaturePyramidNetwork, self).__init__()
#
#         assert isinstance(in_channels, list)
#
#         self.in_channels = in_channels
#
#         self.lateral_convs = nn.ModuleList()
#         self.fpn_convs = nn.ModuleList()
#
#         for i in range(num_levels):
#             lateral_conv = nn.Conv2d(in_channels[i], out_channels, 1)
#             fpn_conv = nn.Sequential(
#                 nn.Conv2d(out_channels, out_channels, 3, padding=1),
#                 nn.BatchNorm2d(out_channels),
#                 nn.ReLU(inplace=True))
#
#             self.lateral_convs.append(lateral_conv)
#             self.fpn_convs.append(fpn_conv)
#
#         # Initialize weights
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.xavier_uniform_(m.weight, gain=1)
#                 if hasattr(m, 'bias'):
#                     nn.init.constant_(m.bias, 0)
#
#     def forward(self, inputs):
#         # Inputs: resolution high -> low
#         assert len(self.in_channels) == len(inputs)
#
#         # Build laterals
#         laterals = [lateral_conv(inputs[i])
#                     for i, lateral_conv in enumerate(self.lateral_convs)]
#
#         # Build top-down path
#         used_backbone_levels = len(laterals)
#         for i in range(used_backbone_levels - 1, 0, -1):
#             laterals[i - 1] += F.interpolate(
#                 laterals[i], scale_factor=2, mode='nearest')
#
#         # Build outputs
#         out = [
#             self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
#         ]
#
#         return out

class Bottleneck(nn.Module):
  """Bottleneck ResNet block."""
  expansion = 4
  #__constants__ = ['downsample']
  features: int
  # norm_layer: ModuleDef
  strides: int = 1
  downsample: Any = None
  groups: int = 1
  base_width: int = 64

  dilation: int = 1
  norm_layer: ModuleDef = nn.BatchNorm
  dtype: Any = jnp.float32

  def setup(self):
      self.width = int((self.features * (self.base_width / 64.)) * self.groups)
      self.norm_layer1 = nn.BatchNorm(self.width, scale_init=nn.initializers.ones, bias_init=nn.initializers.zeros)
      self.norm_layer2 = nn.BatchNorm(self.features * self.expansion, scale_init=nn.initializers.ones,
                                      bias_init=nn.initializers.zeros)

  @nn.compact
  def __call__(self, x):
    width = int(self.features * (self.base_width / 64.)) * self.groups
    identity = x

    #1
    out = conv1x1(width)(x)
    out = self.norm_layer1(out)  # width
    out = nn.relu(out)

    #2
    out = dilated_conv3x3(out, width, strides=self.strides, groups=self.groups, dilation=self.dilation, name='conv2')
    out = self.norm_layer1(out)  # width
    out = nn.relu(out)

    #3
    out = conv1x1(self.features * self.expansion)(out)  # ie self.features * 4
    out = self.norm_layer2(out)  # self.features * self.expansion

    if self.downsample:
        identity = self.downsample(x)

    out += identity
    out = nn.relu(out)
    return out


class BasicBlock(nn.Module):
  expansion = 1
  __constants__ = ['downsample']
  features: int
  # norm: Any = nn.BatchNorm
  strides: int = 1
  downsample: Any = None
  #downsample: bool = False
  groups: int = 1
  base_width: int = 64
  dilation: int = 1
  dtype: Any = jnp.float32

  def setup(self):
    self.norm = nn.BatchNorm(self.features, scale_init=nn.initializers.ones, bias_init=nn.initializers.zeros)

  @nn.compact
  def __call__(self, inputs):
        if self.groups != 1 or self.base_width != 64:
          raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if self.dilation > 1:
          raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        identity = inputs
        out = dilated_conv3x3(inputs, self.features, strides=self.strides, name='conv1')
        out = self.norm_layer(out)
        out = nn.relu(out)

        out = dilated_conv3x3(out, self.features, name='conv2')
        out = self.norm(out)

        if self.downsample is not None:
            identity = self.downsample(out)

        out += identity
        out = nn.relu(out)

        return out


class AANetFeature(nn.Module):
    in_channels = int = 32
    groups: int = 1
    width_per_group: int = 64
    feature_mdconv: bool = True
    norm_layer: Callable = nn.BatchNorm

    def setup(self):

        # self.inplanes = 64
        self.inplanes = self.in_channels
        self.dilation = 1

        #self.groups = self.groups
        self.base_width = self.width_per_group


    def apply_layer(self, x, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        # dilation = self.dilation
        if dilate:
            self.dilation *= stride #TODO: local variable: dilation
            stride = 1


        if stride != 1 or self.inplanes != planes * block.expansion:
            def downsample(x):
                out = conv1x1(planes * block.expansion, stride)(x)
                # embed()
                out = self.norm_layer(use_running_average=False)(out)
                return out


        x = block(planes, stride, downsample, self.groups,
                            self.width_per_group, previous_dilation, self.norm_layer)(x)

        #TODO: find work-around for this bc immutable
        #self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            x = block(planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=self.norm_layer)(x)

        return x

    @nn.compact
    def __call__(self, x):  # TODO: call
        stride = 3

        x = nn.Conv(self.inplanes, kernel_size=(7,7), strides=(stride, stride), padding=((3,3),(3,3)), use_bias=False,
                    kernel_init=kaiming_normal(dtype=jnp.float64))(x)
        x = nn.BatchNorm(use_running_average=False, scale_init=nn.initializers.ones, bias_init=nn.initializers.zeros)(x)
        x = nn.relu(x)  # H/3

        layers = [3, 4, 6]  # ResNet-40

        layer1 = self.apply_layer(x, Bottleneck, self.in_channels, layers[0])  # H/3
        layer2 = self.apply_layer(x, Bottleneck, self.in_channels * 2, layers[1], stride=2)  # H/6

        #block = DeformBottleneck if self.feature_mdconv else Bottleneck
        block = Bottleneck  # TODO: change this back to above
        layer3 = self.apply_layer(x, block, self.in_channels * 4, layers[2], stride=2)  # H/12


        return [layer1, layer2, layer3]


class FeaturePyrmaid(nn.Module):
    in_channel: int = 32

    @nn.compact
    def __call__(self, x):
        # x: [B, 32, H, W]

        # out1 = [B, 64, H/2, W/2]
        out1 = nn.Conv(self.in_channel * 2, kernel_size=(3,3), strides=(2,2), padding=((1,1),(1,1)), use_bias=False)(x)
        out1 = nn.BatchNorm(self.in_channel * 2)(out1)
        out1 = nn.leaky_relu(out1, negative_slope=0.2)
        out1 = nn.Conv(self.in_channel * 2, kernel_size=(1,1), strides=(1, 1), padding='VALID', use_bias=False)(out1)
        out1 = nn.BatchNorm(self.in_channel * 2)(out1)
        out1 = nn.leaky_relu(out1, negative_slope=0.2)

        # out2 = [B, 128, H/4, W/4]
        out2 = nn.Conv(self.in_channel * 4, kernel_size=(3,3), strides=(2,2), padding=((1,1), (1,1)), use_bias=False)(out1)
        out2 = nn.BatchNorm(self.in_channel * 4)(out2)
        out2 = nn.leaky_relu(out2, negative_slope=0.2)
        out2 = nn.Conv(self.in_channel * 4, kernel_size=(1,1), strides=(1,1), padding='VALID', use_bias=False)(out2)
        out2 = nn.BatchNorm(self.in_channel * 4)(out2)
        out2 = nn.leaky_relu(out2, negative_slope=0.2)

        return [x, out1, out2]



key1, key2 = random.split(random.PRNGKey(0), 2)
# x = random.uniform(key1, (15, 32, 32, 3))  # for AANet
# init_variables = model.init(key2, x)
#
# feature_extractor = AANetFeature(feature_mdconv=(not False))
# x = random.uniform(key1, (15, 3, 32, 32))  # for AANet
# init_variables = feature_extractor.init(key2, x)
#
# max_disp = 200 // 3 # randomly picked

key3, key4 = random.split(random.PRNGKey(0), 2)

model = FeaturePyrmaid()
x = random.uniform(key3, (15, 32, 32, 3))  # for AANet
init_pyramid = model.init(key4, x)

from flax.core import freeze, unfreeze

print('initialized parameter shapes:\n', jax.tree_map(jnp.shape, unfreeze(init_pyramid)))

