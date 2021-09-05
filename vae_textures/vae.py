import flax.linen as nn
import jax
import jax.numpy as jnp


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
