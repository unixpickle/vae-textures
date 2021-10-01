import argparse
import os
import pickle

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
from vae_textures.mesh import read_stl
from vae_textures.render import ray_cast, ray_grid
from vae_textures.uniform import gauss_to_uniform
from vae_textures.vae import GaussianSIREN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae-path", default="vae.pkl", type=str)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument(
        "--texture",
        type=str,
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "outputs",
            "checkerboard.jpg",
        ),
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--camera-x", default="1,0,0", type=str)
    parser.add_argument("--camera-y", default="0,1,0", type=str)
    parser.add_argument("--camera-depth", default="0,0,-1", type=str)
    parser.add_argument("--camera-origin", default="0,0,2", type=str)
    parser.add_argument("--preserve-coords", action="store_true")
    parser.add_argument("--ambient-light", default=0.1, type=float)
    parser.add_argument("mesh_in", type=str)
    parser.add_argument("img_out", type=str)
    args = parser.parse_args()

    mesh = read_stl(args.mesh_in, normalize=not args.preserve_coords)

    with open(args.vae_path, "rb") as f:
        params = pickle.load(f)["encoder"]

    enc = jax.jit(lambda x: GaussianSIREN(2).apply(dict(params=params), x)[0])
    texture_fn = image_color_fn(args.texture)

    def color_fn(x):
        gauss_points = enc(x)
        uniform_points = gauss_to_uniform(gauss_points)
        return texture_fn((uniform_points + 1) / 2)

    rays = ray_grid(
        parse_vector(args.camera_x),
        parse_vector(args.camera_y),
        parse_vector(args.camera_depth),
        args.resolution,
    )
    colors = ray_cast(
        mesh,
        parse_vector(args.camera_origin),
        rays,
        color_fn,
        batch_size=args.batch_size,
        ambient_light=args.ambient_light,
    )
    colors = (
        np.clip(np.array(colors) * 255 + 0.5, 0, 255)
        .astype(np.uint8)
        .reshape([args.resolution, args.resolution, 3])
    )
    Image.fromarray(colors).save(args.img_out)


def parse_vector(vec_str: str) -> jnp.ndarray:
    parts = [float(x.strip()) for x in vec_str.split(",")]
    return jnp.array(parts)


def image_color_fn(path: str) -> jnp.ndarray:
    img = jnp.array(Image.open(path).convert("RGB")).astype(jnp.float32) / 255.0

    def color_fn(point: jnp.ndarray) -> jnp.ndarray:
        point = jnp.clip(point, 0, 1)
        x = jnp.clip((img.shape[1] - 1) * point[0], 0, img.shape[1] - 1).astype(jnp.int32)
        y = jnp.clip((img.shape[0] - 1) * point[1], 0, img.shape[0] - 1).astype(jnp.int32)
        return img[y, x]

    return jax.jit(jax.vmap(color_fn))


if __name__ == "__main__":
    main()
