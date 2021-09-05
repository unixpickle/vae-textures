from typing import Any, Dict, Iterable, Tuple

import flax.linen as nn
from flax.training import train_state
import jax
import jax.numpy as jnp
import optax


class GaussianSIREN(nn.Module):
    num_out: int

    @nn.compact
    def __call__(self, xs):
        # SIREN layers: https://vsitzmann.github.io/siren/
        h = nn.Dense(128)(xs)
        h = jnp.sin(h)
        h = nn.Dense(128)(h)
        h = jnp.sin(h)
        # Regular dense layers.
        h = nn.Dense(128)(h)
        h = jnp.tanh(h)
        h = nn.Dense(self.num_out * 2)(h)
        return h[:, : self.num_out], h[:, self.num_out :]


class VAE(nn.Module):
    def setup(self):
        self.encoder = GaussianSIREN(2)
        self.decoder = GaussianSIREN(3)

    def __call__(self, coords_and_basis):
        """
        Compute the variational lower-bound for encoding and decoding xs.
        """
        xs = coords_and_basis[:, 0]

        # TODO: use basis for angle-preservation bonus

        mean, log_stddev = self.encoder(xs)
        latent_noise = jax.random.normal(
            self.make_rng("latent_noise"), shape=log_stddev.shape
        )
        latent_sample = mean + latent_noise * jnp.exp(log_stddev)
        dec_mean, dec_log_stddev = self.decoder(latent_sample)

        recon_loss = -jnp.mean(
            jax.vmap(
                lambda x, m, s: jnp.sum(jax.scipy.stats.norm.logpdf(x, loc=m, scale=s))
            )(xs, dec_mean, jnp.exp(dec_log_stddev))
        )
        kl_loss = jnp.mean(
            jax.vmap(lambda m, s: -0.5 * jnp.sum(1 + 2 * s - m ** 2 - jnp.exp(2 * s)))(
                mean, log_stddev
            )
        )
        return kl_loss + recon_loss


def train(data_iter: Iterable[jnp.ndarray]) -> Dict[str, Any]:
    vae = VAE()
    init_rng, noise_rng = jax.random.split(jax.random.PRNGKey(1234))
    var_dict = jax.jit(vae.init)(
        dict(params=init_rng, latent_noise=noise_rng), next(iter(data_iter))
    )
    state = train_state.TrainState.create(
        apply_fn=vae.apply,
        params=var_dict["params"],
        tx=optax.adam(1e-3),
    )
    losses = []
    for i, batch in enumerate(data_iter):
        loss, state, noise_rng = train_step(state, noise_rng, batch)
        losses.append(loss.tolist())
        if len(losses) == 100:
            print(f"step {i}: loss={sum(losses)/len(losses):.05f}")
            losses = []
    if len(losses):
        print(f"step {i}: loss={sum(losses)/len(losses):.05f}")
    return state.params


@jax.jit
def train_step(
    state: train_state.TrainState, rng: jax.random.PRNGKey, batch: jnp.ndarray
) -> Tuple[jnp.ndarray, train_state.TrainState, jax.random.PRNGKey]:
    rng, new_rng = jax.random.split(rng)
    loss, grads = jax.value_and_grad(
        lambda params: VAE().apply(
            dict(params=params), batch, rngs=dict(latent_noise=rng)
        )
    )(state.params)
    return loss, state.apply_gradients(grads=grads), new_rng
