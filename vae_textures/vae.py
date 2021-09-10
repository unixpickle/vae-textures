from functools import partial
from typing import Any, Dict, Iterable, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

from .uniform import gauss_to_uniform


class GaussianSIREN(nn.Module):
    num_out: int

    @nn.compact
    def __call__(self, xs):
        # SIREN layers: https://vsitzmann.github.io/siren/
        input_init = partial(
            nn.initializers.variance_scaling, 2.0 * 30.0 ** 2.0, "fan_in", "uniform"
        )
        mid_layer_init = partial(
            nn.initializers.variance_scaling, 2.0, "fan_in", "uniform"
        )
        h = nn.Dense(128, kernel_init=input_init(), bias_init=nn.initializers.normal())(
            xs
        )
        h = jnp.sin(h)
        h = nn.Dense(128, kernel_init=mid_layer_init())(h)
        h = jnp.sin(h)
        # Regular dense layers.
        h = nn.Dense(128)(h)
        h = jnp.tanh(h)
        h = nn.Dense(self.num_out * 2)(h)
        return h[:, : self.num_out], h[:, self.num_out :]


class VAE(nn.Module):
    ortho_coeff: float
    kl_coeff: float
    recon_mse: bool

    def setup(self):
        self.encoder = GaussianSIREN(2)
        self.decoder = GaussianSIREN(3)

    def __call__(
        self, coords_and_basis: jnp.ndarray
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        """
        Compute the variational lower-bound for encoding and decoding xs.
        """
        xs = coords_and_basis[:, 0]

        mean, log_stddev = self.encoder(xs)
        latent_noise = jax.random.normal(
            self.make_rng("latent_noise"), shape=log_stddev.shape
        )
        latent_sample = mean + latent_noise * jnp.exp(log_stddev)
        dec_mean, dec_log_stddev = self.decoder(latent_sample)

        if self.recon_mse:
            recon_loss = jnp.mean(jnp.sum((xs - dec_mean) ** 2, axis=-1))
        else:
            recon_loss = -jnp.mean(
                jax.vmap(
                    lambda x, m, s: jnp.sum(
                        jax.scipy.stats.norm.logpdf(x, loc=m, scale=s)
                    )
                )(xs, dec_mean, jnp.exp(dec_log_stddev))
            )

        kl_loss = jnp.mean(
            jax.vmap(lambda m, s: -0.5 * jnp.sum(1 + 2 * s - m ** 2 - jnp.exp(2 * s)))(
                mean, log_stddev
            )
        )

        info_dict = dict(
            recon_loss=recon_loss,
            kl_loss=kl_loss,
            mse=jnp.mean((dec_mean - xs) ** 2),
        )

        loss = recon_loss + self.kl_coeff * kl_loss

        if self.ortho_coeff:

            def uniform_point(xs):
                mean, _ = self.encoder(xs)
                return gauss_to_uniform(mean)

            _, diff_1 = jax.jvp(uniform_point, (xs,), (coords_and_basis[:, 1],))
            _, diff_2 = jax.jvp(uniform_point, (xs,), (coords_and_basis[:, 2],))
            ortho_loss = jnp.mean(jax.vmap(_ortho_loss)(diff_1, diff_2))
            info_dict["ortho_loss"] = ortho_loss
            loss = loss + self.ortho_coeff * ortho_loss

        return loss, info_dict


def _ortho_loss(v1, v2):
    matrix = jnp.stack([v1, v2])
    eigs = jnp.linalg.eigvalsh(matrix.T @ matrix)
    return jnp.abs(eigs[0] - eigs[1])
    # This objective is the actual condition number, but appears
    # to be relatively unstable.
    # return jnp.maximum(eigs[0], eigs[1]) / jnp.minimum(eigs[0], eigs[1])


def train(
    data_iter: Iterable[jnp.ndarray],
    lr: float = 1e-3,
    ortho_coeff: float = 0.0,
    kl_coeff: float = 1.0,
    recon_mse: bool = False,
    init_seed: int = 1234,
) -> Dict[str, Any]:
    vae = VAE(ortho_coeff=ortho_coeff, kl_coeff=kl_coeff, recon_mse=recon_mse)
    init_rng, noise_rng = jax.random.split(jax.random.PRNGKey(init_seed))
    var_dict = jax.jit(vae.init)(
        dict(params=init_rng, latent_noise=noise_rng), next(iter(data_iter))
    )
    state = train_state.TrainState.create(
        apply_fn=vae.apply,
        params=var_dict["params"],
        tx=optax.adam(lr),
    )
    step_fn = jax.jit(partial(train_step, vae))
    losses = []
    infos = []

    def print_step(i):
        keys = [f"loss={sum(losses)/len(losses):.05f}"]
        for k in infos[0].keys():
            vs = [x[k] for x in infos]
            keys.append(f"{k}={sum(vs)/len(vs):.05f}")
        print(f"step {i+1}: {' '.join(keys)}")
        losses.clear()
        infos.clear()

    for i, batch in enumerate(data_iter):
        noise_rng, cur_noise_rng = jax.random.split(noise_rng)
        (loss, info), state = step_fn(state, cur_noise_rng, batch)
        losses.append(loss.tolist())
        infos.append({k: v.tolist() for k, v in info.items()})
        if len(losses) == 100:
            print_step(i)
    if len(losses):
        print_step(i)
    return state.params


def train_step(
    vae: VAE, state: train_state.TrainState, rng: jax.random.PRNGKey, batch: jnp.ndarray
) -> Tuple[Tuple[jnp.ndarray, Dict[str, jnp.ndarray]], train_state.TrainState]:
    loss_aux, grads = jax.value_and_grad(
        lambda params: vae.apply(
            dict(params=params), batch, rngs=dict(latent_noise=rng)
        ),
        has_aux=True,
    )(state.params)
    return loss_aux, state.apply_gradients(grads=grads)
