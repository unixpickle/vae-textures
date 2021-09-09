import argparse
import pickle

import jax

from vae_textures.mesh import mesh_sampler, read_stl
from vae_textures.vae import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-iters", default=1000, type=int)
    parser.add_argument("--save-path", default="vae.pkl", type=str)
    parser.add_argument("--batch-size", default=100, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--ortho-coeff", default=0.0, type=float)
    parser.add_argument("--kl-coeff", default=1.0, type=float)
    parser.add_argument("--recon-mse", action="store_true")
    parser.add_argument("mesh_path", type=str)
    args = parser.parse_args()

    mesh = read_stl(args.mesh_path)
    raw_sampler = mesh_sampler(mesh)
    sampler = jax.jit(lambda x: raw_sampler(x, args.batch_size))

    def data_fn():
        key = jax.random.PRNGKey(1)
        for _ in range(args.num_iters + 1):
            use_key, key = jax.random.split(key)
            samples = sampler(use_key)
            yield samples

    params = train(
        data_fn(),
        lr=args.lr,
        ortho_coeff=args.ortho_coeff,
        kl_coeff=args.kl_coeff,
        recon_mse=args.recon_mse,
    )
    with open(args.save_path, "wb") as f:
        pickle.dump(params, f)


if __name__ == "__main__":
    main()
