import jax.numpy as jnp


def smooth_l1_loss(x, y):
    return jnp.sum(jnp.abs(x - y))
