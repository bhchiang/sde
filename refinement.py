from flax import linen as nn
import jax
import jax.numpy as jnp
from features import BasicBlock
from warp import disp_warp


def conv2d(x, out_channels, kernel_size=3, stride=1, dilation=1, groups=1):
    x = nn.Conv(out_channels,
                kernel_size=(kernel_size, kernel_size),
                strides=(stride, stride),
                padding=((dilation, dilation), (dilation, dilation)),
                kernel_dilation=(dilation, dilation),
                use_bias=False,
                feature_group_count=groups)(x)
    x = nn.BatchNorm(use_running_average=True)(x)
    x = nn.leaky_relu(x, 0.2)
    return x


class StereoDRNetRefinement(nn.Module):
    @nn.compact
    def __call__(self, low_disp, left_img, right_img):
        # assert low_disp.dim() == 3
        #TODO: unsqueeze??
        low_disp = jnp.expand_dims(
            low_disp, axis=3
        )  # low_disp = low_disp.unsqueeze(1)  # [B, 1, H, W] <- O.G. SHAPE; ours is [B,H,W,1(C)]
        b_disp, h_disp, w_disp, _ = low_disp.shape  #last val is channel=1
        _, h_img, w_img, _ = left_img.shape  #last is channels, 3
        scale_factor = w_img / w_disp  # W_img/W_disp
        # print(scale_factor)
        if scale_factor == 1.0:
            disp = low_disp
        else:
            disp = jax.image.resize(
                low_disp, shape=(b_disp, h_img, w_img, 1), method='bilinear'
            )  # F.interpolate(low_disp, size=left_img.size()[-2:], mode='bilinear', align_corners=False)
            disp = disp * scale_factor
            # print("disp range", disp.max(), disp.min())

        # Warp right image to left view with current disparity
        warped_right = disp_warp(right_img, disp)[0]  # [B, C, H, W]
        error = warped_right - left_img  # [B, C, H, W]
        # print("error range", error.max, error.min())

        concat1 = jnp.concatenate(
            [error, left_img], axis=3
        )  #torch.cat((error, left_img), dim=1)  # [B, 6, H, W]; along channels

        conv1 = conv2d(concat1, 16)  # [B, 16, H, W]; conv2d(in_channels, 16)
        conv2 = conv2d(
            disp, 16)  # [B, 16, H, W];   conv2d(1, 16)  # on low disparity
        concat2 = jnp.concatenate(
            [conv1, conv2], axis=3
        )  #torch.cat((conv1, conv2), dim=1)  # [B, 32, H, W] # along channels

        dilation_list = [1, 2, 4, 8, 1, 1]
        #dilated_blocks = []  # nn.ModuleList()
        # self.dilated_blocks = nn.Sequential(*self.dilated_blocks)
        # print(concat2.shape)
        for i, dilation in enumerate(dilation_list):
            if i == 0:
                out = BasicBlock(features=32, dilation=1)(concat2)
            else:
                out = BasicBlock(features=32, dilation=1)(out)
        # out = [B, 32, H, W]

        #out = (concat2)
        #in_channels, out_channels, kernel_size, stride=1, padding=0,
        residual_disp = nn.Conv(features=1,
                                kernel_size=(3, 3),
                                strides=(1, 1),
                                padding=((1, 1), (1, 1)))(out)  # [B, 1, H, W]

        disp = nn.relu(
            disp + residual_disp
        )  #F.relu(disp + residual_disp, inplace=True)  # [B, 1, H, W]
        # disp = jnp.squeeze(disp, axis=3)  #disp.squeeze(1)  # [B, H, W]

        return disp
