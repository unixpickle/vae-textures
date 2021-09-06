import jax
import jax.numpy as jnp


@jax.jit
def gauss_to_uniform(g: jnp.ndarray):
    """
    Map normal random variable to uniform random variables.
    """
    return jax.scipy.stats.norm.cdf(g) * 2 - 1
