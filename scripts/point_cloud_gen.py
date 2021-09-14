import argparse
import pickle

import jax
import jax.numpy as jnp
import numpy as np
from vae_textures.mesh import write_plain_obj
from vae_textures.uniform import uniform_to_gauss
from vae_textures.vae import GaussianSIREN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae-path", default="vae.pkl", type=str)
    parser.add_argument("--resolution", default=256, type=int)
    parser.add_argument("--radius", default=0.01, type=float)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--seed", default=1, type=int)
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
    out_mesh = points_to_cloud_mesh(
        out_points, jnp.array(args.radius), jax.random.PRNGKey(args.seed)
    )
    write_plain_obj(args.obj_out, out_mesh)


def points_to_cloud_mesh(
    points: jnp.ndarray, radius: jnp.ndarray, rng: jax.random.PRNGKey
) -> jnp.ndarray:
    def point_to_mesh(
        point: jnp.ndarray, random_vec: jnp.ndarray, random_theta: jnp.ndarray
    ) -> jnp.ndarray:
        thetas = jnp.array([0.0, jnp.pi * 2 / 3, jnp.pi * 4 / 3])
        base_points = (
            jnp.stack(
                [jnp.cos(thetas), jnp.sin(thetas), jnp.zeros_like(thetas)], axis=-1
            )
            * radius
        )
        tip_1 = jnp.array([0.0, 0.0, radius])
        tip_2 = jnp.array([0.0, 0.0, -radius])

        rotation = random_rotation(random_vec, random_theta)
        base_points = base_points @ rotation + point
        tip_1 = tip_1 @ rotation + point
        tip_2 = tip_2 @ rotation + point

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

    meshes = jax.vmap(point_to_mesh)(
        points,
        jax.random.normal(rng, shape=(len(points), 3)),
        jax.random.uniform(rng, shape=(len(points),)) * jnp.pi * 2,
    )
    return meshes.reshape([-1, 3, 3])


def random_rotation(random_vec: jnp.ndarray, random_theta: jnp.ndarray) -> jnp.ndarray:
    v1 = random_vec / jnp.sqrt(jnp.sum(random_vec ** 2))
    v2 = jnp.array([-v1[1], v1[0], 0])
    v2 = v2 / jnp.sqrt(jnp.sum(v2 ** 2))
    v3 = jnp.array(
        [
            v1[1] * v2[2] - v2[1] * v1[2],
            -(v1[0] * v2[2] - v2[0] * v1[2]),
            v1[0] * v2[1] - v2[0] * v1[1],
        ]
    )
    return jnp.stack(
        [
            v1,
            v2 * jnp.cos(random_theta) - v3 * jnp.sin(random_theta),
            v2 * jnp.sin(random_theta) + v3 * jnp.cos(random_theta),
        ],
        axis=0,
    )


if __name__ == "__main__":
    main()
