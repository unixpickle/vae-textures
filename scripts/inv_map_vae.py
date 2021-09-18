import argparse
import pickle
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
from vae_textures.mesh import read_stl
from vae_textures.uniform import uniform_to_gauss
from vae_textures.vae import GaussianSIREN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae-path", default="vae.pkl", type=str)
    parser.add_argument("--resolution", default=256, type=int)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--preserve-coords", action="store_true")
    parser.add_argument("--equirect-path", default=None, type=str)
    parser.add_argument("mesh_in", type=str)
    parser.add_argument("img_out", type=str)
    args = parser.parse_args()

    mesh = read_stl(args.mesh_in, normalize=not args.preserve_coords)

    if args.equirect_path is not None:
        color_fn = _equirect_color_func(args.equirect_path)
    else:
        color_fn = _rgb_color_func(mesh)

    with open(args.vae_path, "rb") as f:
        params = pickle.load(f)["decoder"]

    dec = jax.jit(lambda x: GaussianSIREN(3).apply(dict(params=params), x)[0])

    stops = np.linspace(-0.999, 0.999, num=args.resolution)
    points = jnp.array([[x, y] for y in stops for x in stops])

    out_colors = []
    for i in range(0, len(points), args.batch_size):
        spatial_point = dec(uniform_to_gauss(points[i : i + args.batch_size]))
        out_colors.append(color_fn(spatial_point))
    colors = np.array(jnp.concatenate(out_colors, axis=0)).reshape(
        [args.resolution, args.resolution, 3]
    )
    Image.fromarray(colors).save(args.img_out)


def _equirect_color_func(path: str) -> Callable[[jnp.ndarray], jnp.ndarray]:
    img = jnp.array(Image.open(path).convert("RGB"))

    def color_fn(point: jnp.ndarray) -> jnp.ndarray:
        normalized = point / jnp.sqrt(jnp.sum(point ** 2))
        lat = jnp.arctan2(normalized[0], normalized[1]) + jnp.pi
        lon = jnp.arctan2(
            jnp.sqrt(normalized[0] ** 2 + normalized[1] ** 2), normalized[2]
        )
        x = jnp.clip(
            (img.shape[1] - 1) * lat / (jnp.pi * 2), 0, img.shape[1] - 1
        ).astype(jnp.int32)
        y = jnp.clip((img.shape[0] - 1) * lon / jnp.pi, 0, img.shape[0] - 1).astype(
            jnp.int32
        )
        return img[y, x]

    return jax.jit(jax.vmap(color_fn))


def _rgb_color_func(mesh: jnp.ndarray) -> jnp.ndarray:
    min_coord = jnp.min(mesh.reshape([-1, 3]), axis=0)
    max_coord = jnp.max(mesh.reshape([-1, 3]), axis=0)

    def color_fn(points: jnp.ndarray) -> jnp.ndarray:
        return jnp.clip(
            255 * (points - min_coord) / (max_coord - min_coord), 0, 255
        ).astype(jnp.uint8)

    return jax.jit(color_fn)


if __name__ == "__main__":
    main()
