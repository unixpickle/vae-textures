import struct
from typing import Callable

import jax
import jax.numpy as jnp


def read_stl(path: str) -> jnp.ndarray:
    """
    Read a 3D model triangle mesh from an STL file.

    The resulting model is represented as an [N x 3 x 3] array of N triangles,
    each containing three vertices with three coordinates each.
    """
    with open(path, "rb") as f:
        f.read(80)
        num_tris = struct.unpack("<I", f.read(4))[0]
        tri_size = 4 * 4 * 3 + 2
        tri_data = f.read(tri_size * num_tris)
        assert len(tri_data) == tri_size * num_tris
    tris = jnp.array([x[3:-1] for x in struct.iter_unpack(f"<{'f'*12}h", tri_data)])
    return tris.reshape([-1, 3, 3])


def write_plain_obj(
    path: str,
    mesh: jnp.ndarray,
):
    """
    Write an obj file with a mesh, represented as an [N x 3 x 3] array.
    """
    vertex_strs = [
        f"v {x:.5f} {y:.5f} {z:.5f}" for x, y, z in mesh.reshape([-1, 3]).tolist()
    ]
    face_strs = [
        f"f {i*3+1}/{i*3+1} {i*3+2}/{i*3+2} {i*3+3}/{i*3+3}" for i in range(len(mesh))
    ]
    with open(path, "w") as f:
        f.write("\n".join(vertex_strs) + "\n")
        f.write("\n".join(face_strs) + "\n")


def write_material_obj(
    path: str,
    mesh: jnp.ndarray,
    uvs: jnp.ndarray,
    mat_file: str = "material.mtl",
    mat_name: str = "material",
):
    """
    Write an obj file with a mesh and corresponding UVs.

    The mesh is [N x 3 x 3] and the UV array is [N x 3 x 2].
    """
    vertex_strs = [
        f"v {x:.5f} {y:.5f} {z:.5f}" for x, y, z in mesh.reshape([-1, 3]).tolist()
    ]
    uv_strs = [f"vt {u:.5f} {v:.5f}" for u, v in uvs.reshape([-1, 2]).tolist()]
    face_strs = [
        f"f {i*3+1}/{i*3+1} {i*3+2}/{i*3+2} {i*3+3}/{i*3+3}" for i in range(len(mesh))
    ]
    with open(path, "w") as f:
        f.write(f"mtllib {mat_file}\n")
        f.write("\n".join(vertex_strs) + "\n")
        f.write("\n".join(uv_strs) + "\n")
        f.write(f"usemtl {mat_name}\n")
        f.write("\n".join(face_strs) + "\n")


def mesh_sampler(mesh: jnp.ndarray) -> Callable[[jax.random.PRNGKey, int], jnp.ndarray]:
    """
    Create a function that samples points on the surface of a mesh.

    Given a request for N samples, the result is of shape [N x 3 x 3], where
    result[:, 0] gives coordinates and result[:, 1:2] are two unit vectors
    tangent to the surface at that point.
    """

    def triangle_area(t: jnp.ndarray) -> jnp.ndarray:
        v1 = t[1] - t[0]
        v2 = t[2] - t[0]
        # Cross product.
        c1 = v1[1] * v2[2] - v2[1] * v1[2]
        c2 = -(v1[0] * v2[2] - v2[0] * v1[2])
        c3 = v1[0] * v2[1] - v2[0] * v1[1]
        return jnp.sqrt(c1 ** 2 + c2 ** 2 + c3 ** 2) / 2

    def triangle_basis(t: jnp.ndarray) -> jnp.ndarray:
        v1 = t[1] - t[0]
        v2 = t[2] - t[0]
        v1 = v1 / jnp.sqrt(jnp.dot(v1, v1))
        v2 = v2 - v1 * jnp.dot(v1, v2)
        v2 = v2 / jnp.sqrt(jnp.dot(v2, v2))
        return jnp.stack([v1, v2])

    def sample_tri(t: jnp.ndarray, r1: jnp.ndarray, r2: jnp.ndarray) -> jnp.ndarray:
        # https://stackoverflow.com/questions/4778147/sample-random-point-in-triangle
        r1 = jnp.sqrt(r1)
        return t[0] * (1 - r1) + t[1] * (r1 * (1 - r2)) + t[2] * (r1 * r2)

    areas = jax.vmap(triangle_area)(mesh)
    probs = jnp.cumsum(areas / jnp.sum(areas))
    bases = jax.vmap(triangle_basis)(mesh)

    def sample_fn(rng: jax.random.PRNGKey, n: int):
        rs = jax.random.uniform(rng, shape=(n, 3))
        indices = jnp.argmax(probs[None] > rs[:, :1], axis=-1)
        tris = mesh[indices]
        points = jax.vmap(sample_tri)(tris, rs[:, 1], rs[:, 2])
        sub_bases = bases[indices]
        return jnp.concatenate([points[:, None], sub_bases], axis=1)

    return sample_fn
