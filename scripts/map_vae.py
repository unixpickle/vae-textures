import argparse
import pickle

import jax
import jax.numpy as jnp

from vae_textures.uniform import gauss_to_uniform
from vae_textures.mesh import read_stl, write_material_obj
from vae_textures.vae import GaussianSIREN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae-path", default="vae.pkl", type=str)
    parser.add_argument("mesh_in", type=str)
    parser.add_argument("obj_out", type=str)
    args = parser.parse_args()

    mesh = read_stl(args.mesh_in)
    with open(args.vae_path, "rb") as f:
        params = pickle.load(f)["encoder"]

    enc = jax.jit(lambda x: GaussianSIREN(2).apply(dict(params=params), x)[0])

    points = mesh.reshape([-1, 3])
    out_points = []
    for i in range(0, len(points), 100):
        gauss_points = enc(points[i : i + 100])
        uniform_points = gauss_to_uniform(gauss_points)
        out_points.append(jnp.clip((uniform_points + 1) / 2, 0, 1))
    uvs = jnp.concatenate(out_points, axis=0).reshape([-1, 3, 2])

    write_material_obj(args.obj_out, mesh, uvs)


if __name__ == "__main__":
    main()
