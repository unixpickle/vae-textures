import argparse
import pickle

from PIL import Image
import jax
import jax.numpy as jnp
import numpy as np

from vae_textures.uniform import uniform_to_gauss
from vae_textures.mesh import read_stl, write_material_obj
from vae_textures.vae import GaussianSIREN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae-path", default="vae.pkl", type=str)
    parser.add_argument("--resolution", default=256, type=int)
    parser.add_argument("mesh_in", type=str)
    parser.add_argument("img_out", type=str)
    args = parser.parse_args()

    mesh = read_stl(args.mesh_in)
    min_coord = jnp.min(mesh.reshape([-1, 3]), axis=0)
    max_coord = jnp.max(mesh.reshape([-1, 3]), axis=0)

    with open(args.vae_path, "rb") as f:
        params = pickle.load(f)["decoder"]

    dec = jax.jit(lambda x: GaussianSIREN(3).apply(dict(params=params), x)[0])

    stops = np.linspace(-0.999, 0.999, num=args.resolution)
    points = jnp.array([[x, y] for y in stops for x in stops])

    out_points = []
    for i in range(0, len(points), 100):
        spatial_point = dec(uniform_to_gauss(points[i : i + 100]))
        constrained = jnp.clip(
            (spatial_point - min_coord) / (max_coord - min_coord), 0, 1
        )
        out_points.append(constrained)
    colors = jnp.concatenate(out_points, axis=0).reshape(
        [args.resolution, args.resolution, 3]
    )
    colors = np.array((colors * 255).astype(jnp.uint8))
    Image.fromarray(colors).save(args.img_out)


if __name__ == "__main__":
    main()
