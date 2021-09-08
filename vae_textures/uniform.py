import jax
import jax.numpy as jnp


@jax.jit
def gauss_to_uniform(g: jnp.ndarray):
    """
    Map normal random variables to uniform random variables.
    """
    return jax.scipy.stats.norm.cdf(g) * 2 - 1


@jax.jit
def uniform_to_gauss(g: jnp.ndarray):
    """
    Map uniform random variable to normal random variables.
    """
    return jax.scipy.special.ndtri(jnp.clip((g + 1) / 2, -0.00001, 0.99999))
