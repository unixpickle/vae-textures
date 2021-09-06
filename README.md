# vae-textures

This is an experiment with using variational autoencoders (VAEs) to perform surface parametrization.

The plan:

 * For a given 3D model, create a "surface dataset" with random points on the surface and their respective normals.
 * Train a VAE to generate points on the surface using a 2D Gaussian latent space. Possibly add a conformality term to the loss to encourage orthogonal directions on the surface to be orthogonal in latent space.
 * Use the gaussian CDF to convert the above latents to the uniform distribution.
 * Apply the 3D -> 2D mapping from above to map the vertices of the original mesh to the unit square, and render the resulting model with some test 2D texture image.

Some immediately obvious limitations:

 * Some triangles will be messed up because of cuts/seams. In particular, the VAE will have to "cut up" the surface to place it into the latent space, and we won't know exactly where these cuts are when mapping texture coordinates to triangle vertices. As a result, some triangles will be messed up because one of their vertices likely has two far-away places in latent space where it could belong.
 * It will be difficult to force the mapping to be conformal. The VAE objective will mostly attempt to preserve areas (i.e. density), and ideally we care about conformality as well.
