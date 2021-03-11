import jax.numpy as jnp
from flax import linen as nn
from typing import Any, Callable
from jax import random
import numpy as onp
import jax
import jax.ops as ops
from IPython import embed
# import features


class CostVolume(nn.Module):
    """Construct cost volume based on different similarity measures
        Args:
            max_disp: max disparity candidate
            feature_similarity: type of similarity measure
    """
    max_disp: int
    feature_similarity: str = 'correlation'

    @nn.compact
    def __call__(self, left_feature, right_feature):
        b, h, w, c = left_feature.shape  # TODO: changed from size to shape ... is ok?

        #TODO: are features just DeviceArrays? ie can just delcare array of zeros?
        cost_volume = jnp.zeros((b, h, w, self.max_disp))

        for i in range(self.max_disp):
            if i > 0:
                update_value = jnp.mean(left_feature[:, :, i:,:] *right_feature[:, :, :-i, :], axis=3)
                cost_volume = ops.index_update(cost_volume, jax.ops.index[:, :, i:, i], update_value)

            else:
                update_value = jnp.mean(left_feature * right_feature, axis=3)
                cost_volume = ops.index_update(cost_volume, jax.ops.index[:, :, :,i], update_value)

        return cost_volume


class CostVolumePyramid(nn.Module):
    max_disp: int
    feature_similarity: str = 'correlation'

    @nn.compact
    def __call__(self, left_feature_pyramid, right_feature_pyramid):
        num_scales = len(left_feature_pyramid)

        cost_volume_pyramid = []
        for s in range(num_scales):
            max_disp = self.max_disp // (2 ** s)
            cost_volume_module = CostVolume(max_disp, self.feature_similarity)
            cost_volume = cost_volume_module(left_feature_pyramid[s], right_feature_pyramid[s])
            cost_volume_pyramid.append(cost_volume)

        return cost_volume_pyramid  # H/3, H/6, H/12



# Testing in jitted context

# test cost volume and cost vol pyramid
# 1) init and apply model FeaturePyramidNetwork()
# key3, key4 = random.split(random.PRNGKey(0), 2)
# model = features.FeaturePyramidNetwork()  #inchannels
# x = random.uniform(key3, (15, 128, 128, 3))  # for AANet
# x2 = random.uniform(key3, (15, 64, 64, 3))
# x3 = random.uniform(key3, (15, 32, 32, 3))
# init_pyramid = model.init(key4, [x, x2, x3])
# print("finished getting features")
#
# @jax.jit
# def apply_feature(variables, _x):
#     return model.apply(variables, _x)
#
# feature_pyr = apply_feature(init_pyramid, [x,x2,x3])
#

# 2) Use features as input to cost volume pyramid
# pretending left and right features are same for simplicity
# key1, key2 = random.split(random.PRNGKey(0), 2)
# costModel = CostVolumePyramid(10) #random max disp=10
# init_cost = costModel.init(key2, feature_pyr, feature_pyr)
#
# @jax.jit
# def apply_cost(variables, left_feature, right_feature):
#     return costModel.apply(variables, left_feature, right_feature) # left feature, right feature
#
# cost_output = apply_cost(init_cost, feature_pyr, feature_pyr)
# print("finished cost pyramid")
