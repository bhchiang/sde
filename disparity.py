from flax import linen as nn
import jax.numpy as jnp


class DisparityEstimation(nn.Module):
    max_disp: int

    @nn.compact
    def __call__(self, cost_volume):
        # Correlation -> match_similarity = True
        b, h, w, d = cost_volume.shape
        prob_volume = nn.softmax(cost_volume, axis=-1)
        max_disp = d
        disp_candidates = jnp.arange(max_disp)
        disp_candidates = jnp.reshape(disp_candidates, [1, 1, 1, max_disp])
        disp_candidates = jnp.tile(disp_candidates, [b, h, w, 1])
        return jnp.sum(disp_candidates * prob_volume, axis=-1)
