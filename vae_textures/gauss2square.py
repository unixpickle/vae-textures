import jax
import jax.numpy as jnp


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
    Compute integral from x=-1 to x=1 of x * x*exp(-0.5*(a*x - x_t)^2/k) dx
    """
    one = jnp.ndarray(1)
    f1 = _normal_expectation_indef(one, x_t, a, k)
    f0 = _normal_expectation_indef(-one, x_t, a, k)
    return f1 - f0


def _normal_expectation_indef(
    x: jnp.ndarray, x_t: jnp.ndarray, a: jnp.ndarray, k: jnp.ndarray
):
    """
    Indefinite integral of x*exp(-0.5*(a*x - x_t)^2/k) dx
    """
    return -(
        jnp.sqrt(k * jnp.pi / 2)
        * x_t
        * jnp.scipy.special.erf(jnp.sqrt(0.5) * (x_t - a * x) / jnp.sqrt(k))
        + k * jnp.exp(-(0.5 * (x_t - a * x) ** 2) / k)
    ) / (a ** 2)


def _normal_integral_uniform(x_t: jnp.ndarray, a: jnp.ndarray, k: jnp.ndarray):
    """
    Compute integral from x=-1 to x=1 of exp(-0.5*(a*x - x_t)^2/k) dx
    """
    one = jnp.ndarray(1)
    f1 = _normal_integral_indef(one, x_t, a, k)
    f0 = _normal_integral_indef(-one, x_t, a, k)
    return f1 - f0


def _normal_integral_indef(
    x: jnp.ndarray, x_t: jnp.ndarray, a: jnp.ndarray, k: jnp.ndarray
):
    """
    Indefinite integral of exp(-0.5*(a*x - x_t)^2/k) dx
    """
    return (
        -(
            jnp.sqrt(jnp.pi / 2)
            * jnp.sqrt(k)
            * jnp.scipy.special.erf(jnp.sqrt(0.5) * (x_t - a * x) / jnp.sqrt(k))
        )
        / a
    )
