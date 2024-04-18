import jax.numpy as jnp
from jax import grad


def loss_recon(params, model, batch):
    """
    Reconstruction loss
    """
    x = batch['x']
    x_hat = model.apply(params, x)
    return jnp.mean(jnp.linalg.norm(x - x_hat, axis=1)**2)


def loss_dynamics_dz(params, encoder, batch, theta, xi, mask):
    """
    Loss for the dynamics in z
    """
    x = batch['x']
    dx_dt = batch['dx']

    def phi(x, params): return encoder.apply(params, x)
    grad_phi = grad(phi, argnums=1)

    return jnp.mean(jnp.linalg.norm(jnp.dot(grad_phi(params, x), dx_dt) - theta @ mask*xi, axis=1)**2)


def loss_regularization(xi):
    """
    Regularization loss
    """
    return jnp.linalg.norm(xi, ord=1)


# %% [markdown]

# Reconstruction Loss (`Lrecon`)

# $$ L_{ \text{recon} } = \frac{1}{m} \sum_{i=1}^{m}  ||x_i - \psi(\phi(x_i))||^2_2  $$

# Dynamics in `x` Loss (`Ldx/dt`)
# $$ L_{dx/dt} = \frac{1}{m} \sum_{i=1}^{m} \left\| \dot{x}_i - (\nabla_z \psi(\phi(x_i))) \Theta(\phi(x_i))^T \Xi \right\|^2_2 $$

# Dynamics in `z` Loss (`Ldz/dt`)
# $$ L_{dz/dt} = \frac{1}{m} \sum_{i=1}^{m} \left\| \nabla_x \phi(x_i) \dot{x}_i - \Theta(\phi(x_i))^T \Xi \right\|^2_2 $$

# Regularization Loss (`Lreg`)
# $$ L_{\text{reg}} = \frac{1}{pd} \| \Xi \|_1 $$
