# vae-textures

This is an experiment with using variational autoencoders (VAEs) to perform [mesh parameterization](https://en.wikipedia.org/wiki/Mesh_parameterization). This was also my first project using [JAX](https://github.com/google/jax) and [Flax](https://github.com/google/flax), and I found them both quite intuitive and easy to use.

<p align="center">
    <img src="outputs/renders/torus_no_ortho.png" height="200">
</p>

# The plan

 * For a given 3D model, create a "surface dataset" with random points on the surface and their respective normals.
 * Train a VAE to generate points on the surface using a 2D Gaussian latent space. Possibly add a conformality term to the loss to encourage orthogonal directions on the surface to be orthogonal in (uniform) latent space.
 * Use the gaussian CDF to convert the above latents to the uniform distribution.
 * Apply the 3D -> 2D mapping from above to map the vertices of the original mesh to the unit square, and render the resulting model with some test 2D texture image.

Some immediately obvious limitations:

 * Some triangles will be messed up because of cuts/seams. In particular, the VAE will have to "cut up" the surface to place it into the latent space, and we won't know exactly where these cuts are when mapping texture coordinates to triangle vertices. As a result, some triangles will be messed up because one of their vertices likely has two far-away places in latent space where it could belong.
 * It will be difficult to force the mapping to be conformal. The VAE objective will mostly attempt to preserve areas (i.e. density), and ideally we care about conformality as well.

# Results

This was my first time using JAX. Nevertheless, I was able to get interesting results right out of the gate.

Initially, I trained VAEs with a Gaussian loss on the decoder, and played around with an orthogonality bonus. This resulted in textures like this one:

![Torus with bonus](outputs/renders/torus_ortho.png)

The above picture looks like a clean mapping, but something is wrong. To see why, let's sample from this VAE. In particular, we will map uniformly spaced points on the texture back to 3D space. In this case I'll use the mean prediction from the decoder, even though its output is a Gaussian distribution:

![A flat disk with a hole in the middle](outputs/renders/torus_sample.png)

It might be hard to tell from a single rendering, but this is just a flat disk with a hole in the middle. In particular, the VAE isn't encoding the z axis (i.e. thickness) at all, but rather just the x and y axes. I discovered that this caused by the Gaussian likelihood loss on the decoder. It is possible for the model to reduce this loss arbitrarily by shrinking the standard deviations of the x and y axes, so there is little incentive to actually capture every axis accurately.

To achieve better results, we can drop the Gaussian likelihood loss and instead use pure MSE for the decoder. This isn't super well-principled, and we now have to select a reasonable coefficient for the KL term of the VAE to balance the reconstruction accuracy with the quality of the latent distribution. I found good hyperparameters for the torus, but these won't necessarily generalize.

With the better reconstruction loss function, sampling the VAE gives the expected result:

![The surface of a torus, point cloud](outputs/renders/torus_sample_mse.png)

# Running

First, install the package with 

```
pip install -e .
```

My initial VAE experiments were run like so, via `scripts/train_vae.py`:

```shell
python scripts/train_vae.py --ortho-coeff 0.002 --num-iters 20000 models/torus.stl
```

In the above command, the `--ortho-loss` flag controls the coefficient of the orthogonality bonus. This will save a model checkpoint to `vae.pkl` after 20000 iterations, which only takes a minute or two on a laptop CPU.

The above will train a VAE with Gaussian reconstruction loss, which is not particularly useful as shown above. To instead use my best settings for the torus, try:

```shell
python scripts/train_vae.py --recon-loss-fn mse --kl-coeff 0.001 --batch-size 1024 --num-iters 20000 models/torus.stl
```

Once you have trained a VAE, you can export a 3D model with the resulting texture mapping like so:

```shell
python scripts/map_vae.py models/torus.stl outputs/torus_ortho.obj
```

Note that the resulting `.obj` file references a `material.mtl` file which should be in the same directory. I already include such a file with a checkerboard texture in [outputs/material.mtl](outputs/material.mtl).

You can also sample a point cloud from the VAE using `point_cloud_gen.py`:

```shell
python scripts/point_cloud_gen.py outputs/point_cloud.obj
```

Finally, you can sample a texture mapping such that points on the texture are normalized (x,y,z) coordinates encoded as RGB:

```shell
python scripts/inv_map_vae.py models/torus.stl outputs/rgb_texture.png
```
