from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
from jax.scipy.special import erf


def pred_xstart(x_t: jnp.ndarray, alpha: jnp.ndarray):
    # x_t = x_0*sqrt(alpha) + z*sqrt(1-alpha)
    # p(x_0 | x_t) = p(x_t | x_0) p(x_0) / p(x_t)
    # E[x_0 | x_t] = integral over x_0 of x_0 * p(x_0 | x_t)
    #              ~ integral over x_0 of x_0 * p(x_t | x_0) p(x_0)
    #              ~ integral over x_0 of x_0 * p(x_t | x_0)
    #              = integral over x_0 of x_0 * N(x_t; x_0*sqrt(alpha), (1-alpha))
    #              ~ integral over x_0 of x_0 * exp(-0.5 * (x_0*sqrt(alpha) - x_t)^2/(1-alpha))
    unnormalized_expectation = _normal_expectation_uniform(
        x_t, jnp.sqrt(alpha), 1 - alpha
    )
    normalizers = _normal_integral_uniform(x_t, jnp.sqrt(alpha), 1 - alpha)
    return unnormalized_expectation / normalizers


def _normal_expectation_uniform(x_t: jnp.ndarray, a: jnp.ndarray, k: jnp.ndarray):
    """
    Compute integral from x=-1 to x=1 of x*exp(-0.5*(a*x - x_t)^2/k) dx
    """
    return _definite_integral(
        f=lambda x: x * jnp.exp(-0.5 * (a * x - x_t) ** 2 / k),
    )


def _normal_integral_uniform(x_t: jnp.ndarray, a: jnp.ndarray, k: jnp.ndarray):
    """
    Compute integral from x=-1 to x=1 of exp(-0.5*(a*x - x_t)^2/k) dx
    """
    return _definite_integral(
        f=lambda x: jnp.exp(-0.5 * (a * x - x_t) ** 2 / k),
    )


def _definite_integral(f: Callable[[jnp.ndarray], jnp.ndarray]) -> jnp.ndarray:
    xs = jnp.linspace(-1, 1, num=200)
    ys = jax.vmap(f)(xs)
    return jnp.trapz(ys, xs, axis=0)


if __name__ == "__main__":
    xs = jnp.linspace(-10, 10, num=50)
    ys = pred_xstart(xs, jnp.array([0.5] * 50))
    import matplotlib.pyplot as plt

    plt.plot(xs, ys)
    plt.show()
