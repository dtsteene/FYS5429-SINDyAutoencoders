import jax.numpy as jnp
from jax import grad


def loss_recon(params, model, x):
    """
    Reconstruction loss
    """
    x_hat = model.apply(params, x)
    return jnp.mean(jnp.linalg.norm(x - x_hat, axis=1)**2)


def loss_dynamics_dx(params, decoder, x, dx_dt, theta, xi, mask):
    """
    Loss for the dynamics in x
    """

    def psi(z, params): return decoder.apply(params, z)
    grad_psi = grad(psi, argnums=1)

    return jnp.mean(jnp.linalg.norm(jnp.dot(grad_psi(params, x), dx_dt) - theta @ mask*xi, axis=1)**2)


def loss_dynamics_dz(params, encoder, x, dx_dt, theta, xi, mask):
    """
    Loss for the dynamics in z
    """

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
# $$ L_{dx/dt} = \frac{1}{m} \sum_{i=1}^{m} \left\| \dot{x}_i - (\nabla_z \psi(z)) \Theta(z)^T \Xi \right\|^2_2 $$

# Dynamics in `z` Loss (`Ldz/dt`)
# $$ L_{dz/dt} = \frac{1}{m} \sum_{i=1}^{m} \left\| \nabla_x \phi(x_i) \dot{x}_i - \Theta(\phi(x_i))^T \Xi \right\|^2_2 $$
# $$ L_{dz/dt} = \frac{1}{m} \sum_{i=1}^{m} \left\| \nabla_x \phi(x_i) \dot{x}_i - \Theta(z)^T \Xi \right\|^2_2 $$

# Regularization Loss (`Lreg`)
# $$ L_{\text{reg}} =  \Xi \|_1 $$
