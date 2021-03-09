from typing import Tuple, Union

import jax
from flax import linen as nn
from IPython import embed
from jax import numpy as jnp


class DeformableConv(nn.Module):
    """Deformable 2D convolution implementation.
    """

    filters: int
    kernel_size: Tuple
    strides: Tuple = (1, 1)
    kernel_dilation: Tuple = (1, 1)
    padding: Union[str, Tuple] = 'VALID'
    num_deform_groups: int = 1

    def setup(self):
        if self.filters % self.num_deform_groups != 0:
            raise ValueError(
                "\"filters\" mod \"num_deform_groups\" must be zero.")

        if self.padding != "VALID":
            raise NotImplementedError(
                f"Padding mode \"f{self.padding}\" has not been implemented yet."
            )

        self.filter_h, self.filter_w = self.kernel_size
        if self.filter_h % 2 == 0 or self.filter_w % 2 == 0:
            raise NotImplementedError(
                f"Even \"kernel_size\" is not supported.")

        # Multiply by 2 for x, y offsets
        self.offset_num = self.filter_h * self.filter_w * self.num_deform_groups * 2

        # Manual unwrapping to avoid tracing
        self.pad_y = self.filter_h // 2
        self.pad_x = self.filter_w // 2

        self.dilation_y, self.dilation_x = self.kernel_dilation
        self.dilated_filter_h = self.dilation_y * self.pad_y * 2 + 1
        self.dilated_filter_w = self.dilation_x * self.pad_x * 2 + 1

        self.dilated_pad_y = self.dilated_filter_h // 2
        self.dilated_pad_x = self.dilated_filter_w // 2

        self.stride_y, self.stride_x = self.strides

    @nn.compact
    def __call__(self, volume):
        """volume represents correlation between two 3D cost volume of disparity candidates.
        """
        # Generate offsets
        offsets = nn.Conv(features=self.offset_num,
                          kernel_size=self.kernel_size,
                          strides=self.strides,
                          padding=self.padding,
                          kernel_dilation=self.kernel_dilation)(volume)

        batch_size, in_h, in_w, channel_in = volume.shape
        _, out_h, out_w, *_ = offsets.shape

        offsets = jnp.reshape(
            offsets, (batch_size, out_h, out_w, -1, 2, self.num_deform_groups))

        offsets = jnp.reshape(
            offsets, (batch_size, out_h, out_w, -1, 2, self.num_deform_groups))

        # Convolution indices
        ys = jnp.arange(self.dilated_pad_y, in_h - self.dilated_pad_y,
                        self.stride_y)
        xs = jnp.arange(self.dilated_pad_x, in_w - self.dilated_pad_x,
                        self.stride_x)
        assert len(ys) == out_h
        assert len(xs) == out_w
        us, vs = jnp.meshgrid(xs, ys)

        # Kernel indices
        kernel_ys = jnp.arange(-self.dilated_pad_y, self.dilated_pad_y + 1,
                               self.dilation_y)
        kernel_xs = jnp.arange(-self.dilated_pad_x, self.dilated_pad_x + 1,
                               self.dilation_x)
        kernel_us, kernel_vs = jnp.meshgrid(kernel_xs, kernel_ys)

        def _wrap(_volume, _image_offsets):
            """
            _image_offsets = (out_h, out_w, filter_h * filter_w, 2)
            """
            def _retrieve(y, x, _kernel_offsets):
                """
                _kernel_offsets = (filter_h * filter_w, 2)
                """
                def _pixel(_y, _x, _pixel_offset):
                    """Retrieve offset pixel values
                    _pixel_offset = (2, )
                    """
                    dy, dx = _pixel_offset
                    return _volume[y + _y, x + _x]

                # embed()
                _kernel_offsets = jnp.reshape(
                    _kernel_offsets, (self.filter_h, self.filter_w, 2))
                return jax.vmap(jax.vmap(_pixel))(kernel_vs, kernel_us,
                                                  _kernel_offsets)

            # _retrieve(vs[10, 0], us[10, 0], _offsets[10, 0])
            pixels = jax.vmap(jax.vmap(_retrieve))(vs, us, _image_offsets)
            # embed()
            return pixels

        _volume = volume[0]
        _offsets = offsets[0, ..., 0]

        y = _wrap(_volume, _offsets)

        def __wrap(_volume, _group_offsets):
            # (2) Map over num_deform_groups dimension for offsets
            return jax.vmap(_wrap, in_axes=(None, -1))(_volume, _group_offsets)

        # (1) Map over batch dimension for volume, offsets
        return jax.vmap(__wrap)(volume, offsets)  # Batch
        return y
        # embed()

        # Depth-wise convolution

        # Add up
        # return y


x_k, m_k = jax.random.split(jax.random.PRNGKey(0), 2)
# N x H x W x C
# C = D (maximum disparity)
x = jax.random.uniform(x_k, (100, 64, 32, 10))

model = DeformableConv(filters=32,
                       kernel_size=(5, 5),
                       num_deform_groups=2,
                       kernel_dilation=(4, 2))
variables = model.init(m_k, x)


@jax.jit
def apply(variables, x):
    y = model.apply(variables, x)
    return y


y = apply(variables, x)
embed()
