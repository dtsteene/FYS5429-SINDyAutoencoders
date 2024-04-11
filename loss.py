"""
The loss functions
These might need to be revised considering how the the encoder/decoders are called
using flax. Could by a syntax more similar to encoder.apply(params, x) instead of encoder(x)
also then the differentiaion will be different. Might want to pass grad_z and grad_x as arguments
to the loss functions. 
"""

import jax.numpy as jnp
from jax import grad, jit, vmap
from jax.scipy.special import logsumexp


def loss_recon(x, decoder, encoder):
    """
    Reconstruction loss
    """
    return jnp.mean(jnp.linalg.norm(x - decoder(encoder(x)), axis=1)**2)


def loss_dynamics_x(x, dx_dt, encoder, decoder, theta, xi, mask):
    """
    Loss for the dynamics in x

    args : 
        x: input data
        dx_dt: time derivative of the input data
        encoder: the encoder model
        decoder: the decoder model
        theta: the dynamics matrix
        xi: the dynamics matrix
        mask: the mask for the dynamics matrix
    """
    grad_z = grad(encoder)

    return jnp.mean(jnp.linalg.norm(dx_dt - jnp.dot(grad_z(x), theta) @ mask*xi, axis=1)**2)


def loss_dynamics_z(x, dx_dt, encoder, decoder, theta, xi, mask):
    """
    Loss for the dynamics in z

    args:
        x: input data
        dx_dt: time derivative of the input data
        encoder: the encoder model
        decoder: the decoder model
        theta: the dynamics matrix
        xi: the dynamics matrix
        mask: the mask for the dynamics matrix
    """

    grad_x = grad(decoder)

    return jnp.mean(jnp.linalg.norm(jnp.dot(grad_x(x), dx_dt) - theta @ mask*xi, axis=1)**2)


def loss_regularization(xi):
    """
    Regularization loss
    """
    return jnp.linalg.norm(xi, ord=1)   


# %% [markdown]

#### Reconstruction Loss (`Lrecon`)

# $$ L_{ \text{recon} } = \frac{1}{m} \sum_{i=1}^{m}  ||x_i - \psi(\phi(x_i))||^2_2  $$

#### Dynamics in `x` Loss (`Ldx/dt`)
# $$ L_{dx/dt} = \frac{1}{m} \sum_{i=1}^{m} \left\| \dot{x}_i - (\nabla_z \psi(\phi(x_i))) \Theta(\phi(x_i))^T \Xi \right\|^2_2 $$

#### Dynamics in `z` Loss (`Ldz/dt`)
# $$ L_{dz/dt} = \frac{1}{m} \sum_{i=1}^{m} \left\| \nabla_x \phi(x_i) \dot{x}_i - \Theta(\phi(x_i))^T \Xi \right\|^2_2 $$

####  Regularization Loss (`Lreg`)
# $$ L_{\text{reg}} = \frac{1}{pd} \| \Xi \|_1 $$


