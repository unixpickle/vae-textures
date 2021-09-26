import jax
import jax.numpy as jnp
from typing import Tuple


def mesh_ray_collision(mesh: jnp.ndarray, origin: jnp.ndarray, dir: jnp.ndarray):
    """
    Shoot a ray from an origin point in a given direction and find the first
    place it intersects a mesh, if anywhere.

    :param mesh: an [N x 3 x 3] triangle mesh.
    :param origin: 1 1-D array, the ray origin point.
    :param dir: a 1-D array, the ray direction.
    :return: a tuple (collides, point), where collides is a 0-D boolean tensor
             and point is a 1-D coordinate for an intersection.
    """
    collides, positions = jax.vmap(_triangle_ray_collision)(mesh, origin, dir)
    distances = jnp.where(
        collides, jnp.sum((positions - origin) ** 2, axis=-1), jnp.inf
    )
    return jnp.any(collides), positions[jnp.argmin(distances)]


def _triangle_ray_collision(
    t: jnp.ndarray, origin: jnp.ndarray, dir: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    # Adapted from https://github.com/unixpickle/model3d/blob/efe391aa6714a269aac3f09de525406a90fa51bc/model3d/primitives.go#L169
    normal = _triangle_normal(t)
    dir = _normalize(dir)
    collides = jnp.abs(normal @ dir) > 1e-8

    v1 = t[1] - t[0]
    v2 = t[2] - t[0]
    cross1 = _cross_product(dir, v2)
    det = cross1 @ v1

    collides = jnp.logical_and(collides, jnp.abs(det) > 1e-8)

    invDet = 1 / det
    o = origin - t[0]
    bary1 = invDet * (o @ cross1)
    collides = jnp.logical_and(collides, jnp.logical_and(bary1 >= 0, bary1 <= 1))

    cross2 = _cross_product(o, v1)
    bary2 = invDet * (dir @ cross2)
    collides = jnp.logical_and(collides, jnp.logical_and(bary2 >= 0, bary2 <= 1))

    bary3 = 1 - (bary1 + bary2)
    collision_point = bary1 * t[0] + bary2 * t[1] + bary3 * t[3]

    # Make sure this is in the positive part of the ray.
    scale = invDet * (v2 @ cross2)
    collides = jnp.logical_and(collides, scale > 0)

    return collides, collision_point


def _triangle_normal(t: jnp.ndarray) -> jnp.ndarray:
    return _normalize(_cross_product(t[1] - t[0], t[2] - t[0]))


def _cross_product(v1: jnp.ndarray, v2: jnp.ndarray) -> jnp.ndarray:
    return [
        v1[1] * v2[2] - v2[1] * v1[2],
        -(v1[0] * v2[2] - v2[0] * v1[2]),
        v1[0] * v2[1] - v2[0] * v1[1],
    ]


def _normalize(v: jnp.ndarray):
    return v / jnp.sqrt(jnp.sum(v ** 2))
