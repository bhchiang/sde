import jax
import jax.numpy as jnp


@jax.jit
def _epe(disp, gt_disp):
    return jnp.mean(jnp.abs(disp - gt_disp))


@jax.jit
def _1pixel(disp, gt_disp):
    e = jnp.abs(disp - gt_disp)
    total = jnp.size(disp)
    missed = jnp.sum(e > 1)
    return missed / total