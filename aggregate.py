from deform import DeformableConv
from features import conv1x1
from flax import linen as nn
from IPython import embed
from jax import jit
from jax import numpy as jnp
from jax import random
import jax


class DeformSimpleBottleneck(nn.Module):
    planes: int

    md_conv_dilation: int = 2
    num_deform_groups: int = 2
    modulation: bool = True

    base_width: int = 64

    def setup(self):
        self.width = int(self.planes *
                         (self.base_width / 64.)) * self.num_deform_groups

    @nn.compact
    def __call__(self, x):
        identity = x
        out = conv1x1(self.width)(x)
        out = nn.BatchNorm(use_running_average=True)(out)
        out = nn.relu(out)

        out = DeformableConv(filters=self.width,
                             kernel_size=(3, 3),
                             kernel_dilation=(self.md_conv_dilation,
                                              self.md_conv_dilation),
                             num_deform_groups=self.num_deform_groups)(out)
        out = nn.BatchNorm(use_running_average=True)(out)
        out = nn.relu(out)

        out = conv1x1(self.planes)(out)
        out = nn.BatchNorm(use_running_average=True)(out)

        # TODO: Since DeformConv only supports padding = "VALID", do a hacky padding
        # to make the skip connection work for now.
        out = jnp.pad(out,
                      ((0, 0), (self.md_conv_dilation, self.md_conv_dilation),
                       (self.md_conv_dilation, self.md_conv_dilation), (0, 0)))
        out += identity
        out = nn.relu(out)

        return out


class _AdapativeAggregation(nn.Module):
    num_scales: int
    num_output_branches: int
    max_disp: int
    num_blocks: int = 1

    num_deform_groups: int = 2
    md_conv_dilation: int = 2

    @nn.compact
    def __call__(self, x):
        isa = []

        print("Intra-scale aggregation")
        for i in range(self.num_scales):
            num_candidates = self.max_disp // (2**i)
            print(f"i = {i}, num_candidates = {num_candidates}")
            _x = x[i]
            for j in range(self.num_blocks):
                _x = DeformSimpleBottleneck(
                    planes=num_candidates,
                    md_conv_dilation=self.md_conv_dilation,
                    num_deform_groups=self.num_deform_groups)(_x)
            isa.append(_x)

        print("Cross-scale aggregation")
        csa = []
        for i in range(self.num_output_branches):
            # Fuse aggregated cost volumes at all scales
            print(f"i = {i}")
            _csa = isa[i]  # Identity
            b, h, w, c = _csa.shape
            for j in range(self.num_scales):
                _isa = isa[j]
                print(f"j = {j}, _isa.shape = {_isa.shape}")
                if i == j:
                    continue
                elif i < j:
                    # Upsample j
                    _isa = jax.image.resize(_isa, (b, h, w, _isa.shape[-1]),
                                            method="bilinear")
                    # Align channels
                    _isa = conv1x1(features=c)(_isa)
                elif i > j:
                    # Downsample j
                    for k in range(i - j - 1):
                        # Retain # of j channels
                        _isa = nn.Conv(features=self.max_disp // (2**j),
                                       kernel_size=(3, 3),
                                       strides=(2, 2),
                                       padding=((1, 1), (1, 1)),
                                       use_bias=False)(_isa)
                        _isa = nn.BatchNorm(use_running_average=True)(_isa)
                        _isa = nn.leaky_relu(_isa, negative_slope=0.2)
                    # Final downsample, align channels
                    _isa = nn.Conv(features=self.max_disp // (2**i),
                                   kernel_size=(3, 3),
                                   strides=(2, 2),
                                   padding=((1, 1), (1, 1)),
                                   use_bias=False)(_isa)
                    _isa = nn.BatchNorm(use_running_average=False)(_isa)
                _csa += _isa
            print(f"_csa.shape = {_csa.shape}")
            csa.append(_csa)
        return csa


class AdapativeAggregation(nn.Module):
    @nn.compact
    def __call__(self, x):
        return x


# TODO: implement if necessary
AdapativeAggregation = _AdapativeAggregation

if __name__ == "__main__":
    x_k, m_k = random.split(random.PRNGKey(0), 2)
    _model = _AdapativeAggregation(num_scales=3,
                                   num_output_branches=3,
                                   max_disp=20)
    cost_volumes = [
        random.uniform(x_k, (15, 32, 32, 20)),
        random.uniform(x_k, (15, 16, 16, 10)),
        random.uniform(x_k, (15, 8, 8, 5)),
    ]
    # embed()

    _variables = _model.init(m_k, cost_volumes)
    y = jit(_model.apply)(_variables, cost_volumes, mutable=['batch_stats'])
    embed()
