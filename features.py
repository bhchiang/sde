import jax.numpy as jnp
from flax import linen as nn
# from flax import optim
from typing import Any, Callable, Sequence, Tuple
from functools import partial
from jax import random
import numpy as onp
import jax

ModuleDef = Any

def _compute_fans(shape, in_axis=-2, out_axis=-1):
  receptive_field_size = onp.prod(shape) / shape[in_axis] / shape[out_axis]
  fan_in = shape[in_axis] * receptive_field_size
  fan_out = shape[out_axis] * receptive_field_size
  return fan_in, fan_out


def variance_scaling(scale, mode, distribution, in_axis=-2, out_axis=-1, dtype=jnp.float32):
  def init(key, shape, dtype=dtype):
    fan_in, fan_out = _compute_fans(shape, in_axis, out_axis)
    if mode == "fan_in": denominator = fan_in
    elif mode == "fan_out": denominator = fan_out
    elif mode == "fan_avg": denominator = (fan_in + fan_out) / 2
    else:
      raise ValueError(
        "invalid mode for variance scaling initializer: {}".format(mode))
    variance = jnp.array(scale / denominator, dtype=dtype)
    if distribution == "truncated_normal":
      # constant is stddev of standard normal truncated to (-2, 2)
      stddev = jnp.sqrt(variance) / jnp.array(.87962566103423978, dtype)
      return random.truncated_normal(key, -2, 2, shape, dtype) * stddev
    elif distribution == "normal":
      return random.normal(key, shape, dtype) * jnp.sqrt(variance)
    elif distribution == "uniform":
      return random.uniform(key, shape, dtype, -1) * onp.sqrt(3 * variance)
    else:
      raise ValueError("invalid distribution for variance scaling initializer")
  return init

kaiming_normal = partial(variance_scaling, 2.0, "fan_out", "truncated_normal")
#nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#  padding: [(0, 0), (_d, _d), (_d, _d), (0, 0)]

def dilated_conv3x3(x, features, strides=1, groups=1, dilation=1, name='dilated_conv3x3'):
    """3x3 convolution with padding"""

    """
    METHOD 1: pad the original and pass 'VALID' into padding
    """
    d = max(1, dilation)
    # pad_width = [(d, d),(d,d)]
    #             #* x.ndim
    # x = jnp.pad(x, pad_width, 'constant' ) # apply padding , (0, 0)
    # return nn.Conv(features, kernel_size=(3, 3), strides=(strides,strides), padding='VALID', kernel_dilation=(d, d), feature_group_count=groups, use_bias=False, name=name)(x)
    #
    """
        METHOD 2: pass padding = (dilation, dilation) directly
    """
    return nn.Conv(features, kernel_size=(3, 3), strides=(strides,strides), padding=((dilation,dilation)), kernel_dilation=(d, d), feature_group_count=groups, use_bias=False, name=name)(x)

    # return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
    #                  padding=dilation, groups=groups, bias=False, dilation=dilation)



def conv1x1(features, stride=1):
    """1x1 convolution"""
    #lhs: a rank `n+2` dimensional input array.
    # rhs: a rank `n+2` dimensional array of kernel weights.

    return nn.Conv(features=features, kernel_size=(1, 1), strides=(stride, stride), padding='VALID', use_bias=False)
    # return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
  """Bottleneck ResNet block."""
  expansion = 4
  __constants__ = ['downsample']
  features: int
  # norm_layer: ModuleDef
  strides: int = 1
  downsample: Any = None
  groups: int = 1
  dilation: int = 1
  base_width: int = 64
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
    # in_planes: int = 32
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


        # TODO: fix the sequential, and check inputs
        # x = nn.Conv(3, self.inplanes, kernel_size=7, stride=stride, padding=3, bias=False)
        # x =
        # self.conv1 = nn.Sequential(nn.Conv(3, self.inplanes, kernel_size=7, stride=stride, padding=3, bias=False),
        #                            nn.BatchNorm(self.inplanes),
        #                            nn.relu(inplace=True))  # H/3

        # self.layer1 = self.apply_layer(Bottleneck, self.in_channels, layers[0])  # H/3
        # self.layer2 = self.apply_layer(Bottleneck, self.in_channels * 2, layers[1], stride=2)  # H/6

        # block = DeformBottleneck if self.feature_mdconv else Bottleneck
        # block = Bottleneck # TODO: change this back to above
        #
        # self.layer3 = self.apply_layer(block, self.in_channels * 4, layers[2], stride=2)  # H/12

        # for m in self.modules():
        #     if isinstance(m, nn.Conv):  #DONE
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, (nn.BatchNorm, nn.GroupNorm)):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

        # # Zero-initialize the last BN in each residual branch,
        # # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        # if self.zero_init_residual:
        #     for m in self.modules():  #
        #         if isinstance(m, Bottleneck):
        #             nn.init.constant_(m.bn3.weight, 0) #
        #         elif isinstance(m, BasicBlock):
        #             nn.init.constant_(m.bn2.weight, 0)  #

    def apply_layer(self, x, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self.norm_layer
        downsample = None
        previous_dilation = self.dilation
        # dilation = self.dilation
        if dilate:
            self.dilation *= stride #TODO: local variable: dilation
            stride = 1

        # if stride != 1 or self.inplanes != planes * block.expansion:
        #     # TODO: fix this conv 1x1 gives error
        #     downsample = conv1x1(planes * block.expansion, stride)(x)
        #     downsample = norm_layer(downsample)
            # planes * block.expansion

            # downsample = nn.Sequential(
            #     conv1x1(self.inplanes, planes * block.expansion, stride),
            #     norm_layer(planes * block.expansion),
            # )

        # layers = []
        #x = block(----)(x)
        x = block(planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer)(x)
        # layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
        #                     self.base_width, previous_dilation, norm_layer))

        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            x = block(planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer)(x)
            # layers.append(block(self.inplanes, planes, groups=self.groups,
            #                     base_width=self.base_width, dilation=self.dilation,
            #                     norm_layer=norm_layer))

        return x
            #nn.Sequential(*layers)

    @nn.compact
    def __call__(self, x):  # TODO: call
        stride = 3

        #TODO: check the padding
        # nn.Conv2d(3, self.inplanes, kernel_size=7, stride=stride, padding=3, bias=False),

        # x = nn.Conv(self.inplanes, kernel_size=(7,7), strides=(stride,stride), padding=(3,3), use_bias=False, kernel_init=kaiming_normal(dtype=jnp.float64))(x)
        x = nn.BatchNorm(self.inplanes, scale_init=nn.initializers.ones, bias_init=nn.initializers.zeros)(x)
        x = nn.relu(x)  # H/3

        layers = [3, 4, 6]  # ResNet-40

        # TODO: enough inputs!?
        #x,  block, planes, blocks, stride=1, dilate=False):
        layer1 = self.apply_layer(x, Bottleneck, self.in_channels, layers[0])  # H/3

        layer2 = self.apply_layer(x, Bottleneck, self.in_channels * 2, layers[1], stride=2)  # H/6

        # block = DeformBottleneck if self.feature_mdconv else Bottleneck
        block = Bottleneck # TODO: change this back to above
        layer3 = self.apply_layer(x, block, self.in_channels * 4, layers[2], stride=2)  # H/12


        return [layer1, layer2, layer3]

# model.init
# model.apply


key1, key2 = random.split(random.PRNGKey(0), 2)
x = random.uniform(key1, (4,4))

    # inplanes: int
    # planes: int
    # stride: int = 1
    # downsample: Any = None
    # groups: int = 1
    # base_width: int = 64
    # dilation: int = 1

model = Bottleneck(32)
init_variables = model.init(key2, x)

print("hi")
print(init_variables)

from flax.core import freeze, unfreeze

print('initialized parameter shapes:\n', jax.tree_map(jnp.shape, unfreeze(init_variables)))
# print('output:\n', y)
