from typing import Callable

import jax
import jax.lax as lax
import jax.numpy as jnp
import numpy as np


@jax.jit
def sample_ddim(x_T: jnp.ndarray):
    """
    Map normal random variables to uniform random variables.
    """

    def sample_it(x_t, alphas):
        alpha_prev, alpha_next = alphas[0], alphas[1]
        x_0 = pred_x0(x_t, alpha_prev)
        eps = (x_t - jnp.sqrt(alpha_prev) * x_0) / jnp.sqrt(1 - alpha_prev)
        x_t = jnp.sqrt(alpha_next) * x_0 + jnp.sqrt(1 - alpha_next) * eps
        return x_t, None

    alphas = jnp.exp(-(jnp.linspace(0, 4, num=50)[::-1] ** 2))
    joined = jnp.stack([alphas[:-1], alphas[1:]], axis=-1)
    x_0, _ = lax.scan(sample_it, x_T, joined)
    return x_0


def pred_x0(x_t: jnp.ndarray, alpha: jnp.ndarray):
    """
    Compute the exact prediction E[q(x_0|x_t,alpha)] assuming that q(x_0) is
    the uniform distribution.
    """
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
    key = jax.random.PRNGKey(1337)
    xs = jax.random.normal(key, shape=(10000, 2))
    ys = sample_ddim(xs)

    import matplotlib.pyplot as plt

    plt.scatter(ys[:, 0], ys[:, 1])
    plt.show()
