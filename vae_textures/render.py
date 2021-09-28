from typing import Callable, Tuple

import jax
import jax.numpy as jnp


def ray_cast(
    mesh: jnp.ndarray,
    origin: jnp.ndarray,
    directions: jnp.ndarray,
    color_fn: Callable[[jnp.ndarray], jnp.ndarray],
    batch_size: int = 128,
    bg_color: jnp.ndarray = None,
) -> jnp.ndarray:
    """
    Render a mesh by casting an array of rays from an origin point, finding
    where they hit the mesh, and using the combination of a color texture
    function and the normal of the ray collision to color the resulting pixel.

    :param mesh: an [N x 3 x 3] triangle mesh.
    :param origin: the camera origin point.
    :param directions: an [R x 3] array of rays.
    :param color_func: a function which takes a batch of [B x 3] coordinates
                       and returns the [B x 3] RGB colors at those coordinates.
    :param batch_size: the number of rays to evaluate at once.
    :param bg_color: the background color for non-collision rays.
    :return: an [R x 3] array of RGB colors.
    """
    # Normalize the directions so that dot products are meaningful.
    directions = directions / jnp.sqrt(jnp.sum(directions ** 2, axis=-1, keepdims=True))
    if bg_color is None:
        bg_color = jnp.array([0, 0, 0])

    @jax.jit
    def ray_color(sub_directions: jnp.ndarray) -> jnp.ndarray:
        collides, points, normals = jax.vmap(lambda d: mesh_ray_collision(mesh, origin, d))(
            sub_directions
        )
        colors = color_fn(points)
        colors *= jnp.sum(jnp.abs(normals * sub_directions), axis=-1, keepdims=True)
        return jnp.where(collides[:, None], colors, bg_color.astype(colors.dtype))

    outputs = []
    for i in range(0, len(directions), batch_size):
        outputs.append(ray_color(directions[i : i + batch_size]))
    return jnp.concatenate(outputs, axis=0)


def ray_grid(
    x: jnp.ndarray,
    y: jnp.ndarray,
    z: jnp.ndarray,
    resolution: int,
):
    """
    Produce a grid of ray directions to use for ray_cast().

    :param x: the unit x direction for the grid.
    :param y: the unit y direction for the grid.
    :param z: the direction facing towards the rendering plane.
    """
    scales = jnp.linspace(-1.0, 1.0, num=resolution)
    x_vecs = x * scales[None, :, None]
    y_vecs = y * scales[:, None, None]
    results = (x_vecs + y_vecs + z).reshape([-1, x.shape[0]])
    return jax.vmap(_normalize)(results)


def mesh_ray_collision(mesh: jnp.ndarray, origin: jnp.ndarray, direction: jnp.ndarray):
    """
    Shoot a ray from an origin point in a given direction and find the first
    place it intersects a mesh, if anywhere.

    :param mesh: an [N x 3 x 3] triangle mesh.
    :param origin: 1 1-D array, the ray origin point.
    :param direction: a 1-D array, the ray direction.
    :return: a tuple (collides, point, normal), where collides is a 0-D boolean
             tensor and point is a 1-D coordinate for an intersection.
    """
    collides, positions, distances = jax.vmap(
        lambda t: _triangle_ray_collision(t, origin, direction)
    )(mesh)
    idx = jnp.argmin(jnp.where(collides, distances, jnp.inf))
    return (
        jnp.any(collides),
        positions[idx],
        _triangle_normal(mesh[idx]),
    )


def _triangle_ray_collision(
    t: jnp.ndarray, origin: jnp.ndarray, direction: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    # Adapted from https://github.com/unixpickle/model3d/blob/efe391aa6714a269aac3f09de525406a90fa51bc/model3d/primitives.go#L169
    normal = _triangle_normal(t)
    direction = _normalize(direction)
    collides = jnp.abs(normal @ direction) > 1e-8

    v1 = t[1] - t[0]
    v2 = t[2] - t[0]
    cross1 = _cross_product(direction, v2)
    det = cross1 @ v1

    collides = jnp.logical_and(collides, jnp.abs(det) > 1e-8)

    invDet = 1 / det
    o = origin - t[0]
    bary1 = invDet * (o @ cross1)
    collides = jnp.logical_and(collides, jnp.logical_and(bary1 >= 0, bary1 <= 1))

    cross2 = _cross_product(o, v1)
    bary2 = invDet * (direction @ cross2)
    collides = jnp.logical_and(collides, jnp.logical_and(bary2 >= 0, bary2 <= 1))

    bary0 = 1 - (bary1 + bary2)
    collision_point = bary1 * t[1] + bary2 * t[2] + bary0 * t[0]

    # Make sure this is in the positive part of the ray.
    scale = invDet * (v2 @ cross2)
    collides = jnp.logical_and(collides, scale > 0)

    return collides, collision_point, scale


def _triangle_normal(t: jnp.ndarray) -> jnp.ndarray:
    return _normalize(_cross_product(t[1] - t[0], t[2] - t[0]))


def _cross_product(v1: jnp.ndarray, v2: jnp.ndarray) -> jnp.ndarray:
    return jnp.stack(
        [
            v1[1] * v2[2] - v2[1] * v1[2],
            -(v1[0] * v2[2] - v2[0] * v1[2]),
            v1[0] * v2[1] - v2[0] * v1[1],
        ]
    )


def _normalize(v: jnp.ndarray):
    return v / jnp.sqrt(jnp.sum(v ** 2))
