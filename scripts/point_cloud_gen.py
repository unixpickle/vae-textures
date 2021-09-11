import argparse
import pickle

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
from vae_textures.mesh import write_plain_obj
from vae_textures.uniform import uniform_to_gauss
from vae_textures.vae import GaussianSIREN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae-path", default="vae.pkl", type=str)
    parser.add_argument("--resolution", default=256, type=int)
    parser.add_argument("--radius", default=0.01, type=float)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("obj_out", type=str)
    args = parser.parse_args()

    with open(args.vae_path, "rb") as f:
        params = pickle.load(f)["decoder"]

    dec = jax.jit(lambda x: GaussianSIREN(3).apply(dict(params=params), x)[0])

    stops = np.linspace(-0.999, 0.999, num=args.resolution)
    points = jnp.array([[x, y] for y in stops for x in stops])

    out_points = []
    for i in range(0, len(points), args.batch_size):
        spatial_point = dec(uniform_to_gauss(points[i : i + args.batch_size]))
        out_points.append(spatial_point)

    out_points = jnp.concatenate(out_points)
    out_mesh = points_to_cloud_mesh(out_points, jnp.array(args.radius))
    write_plain_obj(args.obj_out, out_mesh)


def points_to_cloud_mesh(points: jnp.ndarray, radius: jnp.ndarray) -> jnp.ndarray:
    def point_to_mesh(point: jnp.ndarray) -> jnp.ndarray:
        thetas = jnp.array([0.0, jnp.pi * 2 / 3, jnp.pi * 4 / 3])
        base_points = (
            jnp.stack(
                [jnp.cos(thetas), jnp.sin(thetas), jnp.zeros_like(thetas)], axis=-1
            )
            * radius
            + point
        )
        tip_1 = point + jnp.array([0.0, 0.0, radius])
        tip_2 = point + jnp.array([0.0, 0.0, -radius])
        return jnp.stack(
            [
                jnp.stack([base_points[0], base_points[1], tip_1]),
                jnp.stack([base_points[1], base_points[2], tip_1]),
                jnp.stack([base_points[2], base_points[0], tip_1]),
                jnp.stack([base_points[1], base_points[0], tip_2]),
                jnp.stack([base_points[2], base_points[1], tip_2]),
                jnp.stack([base_points[0], base_points[2], tip_2]),
            ]
        )

    return jax.vmap(point_to_mesh)(points).reshape([-1, 3, 3])


if __name__ == "__main__":
    main()
