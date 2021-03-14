from flax import linen as nn
import features, cost, aggregate, disparity
from IPython import embed
from jax import random
import jax


class Model(nn.Module):
    max_disp: int

    def setup(self):
        self.feature_extractor = features.AANetFeature()
        self.fpn = features.FeaturePyramidNetwork()
        self.cost_volume_construction = cost.CostVolumePyramid(
            max_disp=self.max_disp)
        self.aggregation = aggregate._AdapativeAggregation(
            num_scales=3, num_output_branches=3, max_disp=self.max_disp)
        self.disparity_estimation = disparity.DisparityEstimation(
            max_disp=self.max_disp)

    def disparity_computation(self, aggregation):
        length = len(aggregation)
        disparity_pyramid = []
        for i in range(length):
            disp = self.disparity_estimation(aggregation[length - 1 - i])
            disparity_pyramid.append(disp)
        return disparity_pyramid

    def feature_extraction(self, img):
        feature = self.feature_extractor(img)
        feature = self.fpn(feature)
        return feature

    @nn.compact
    def __call__(self, left_img, right_img):
        left_feature = self.feature_extraction(left_img)
        print("Left feature done")
        right_feature = self.feature_extraction(right_img)
        print("Right feature done")
        cost_volume = self.cost_volume_construction(left_feature,
                                                    right_feature)
        print("Cost volume done")
        print([x.shape for x in cost_volume])
        aggregation = self.aggregation(cost_volume)
        print("Aggregation done")
        # embed()
        disparity_pyramid = self.disparity_computation(aggregation)
        print("Disparity computation done")
        # embed()
        return disparity_pyramid


if __name__ == "__main__":
    l_k, r_k, m_k = random.split(random.PRNGKey(0), 3)
    size = 432
    # size = 480
    batch_size = 1
    left_img = random.uniform(l_k, (batch_size, size, size, 3))
    right_img = random.uniform(r_k, (batch_size, size, size, 3))

    model = Model(max_disp=64)

    variables = model.init(m_k, left_img, right_img)

    @jax.jit
    def loss(variables, x):
        y = model.apply(variables, x)
        return y

    embed()
