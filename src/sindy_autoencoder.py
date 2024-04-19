from flax import linen as nn
import jax
import jax.numpy as jnp
from jax import random, grad, jit, vmap, value_and_grad
import optax


class Encoder(nn.Module):
    input_dim: int
    latent_dim: int
    widths: list
    activation: nn.activation = nn.relu

    @nn.compact
    def __call__(self, x):
        for width in self.widths:
            x = nn.Dense(width)(x)
            x = self.activation(x)
        z = nn.Dense(self.latent_dim)(x)
        return z


class Decoder(nn.Module):
    input_dim: int
    latent_dim: int
    widths: list
    activation: nn.activation = nn.relu

    @nn.compact
    def __call__(self, z):
        for width in reversed(self.widths):
            z = nn.Dense(width)(z)
            z = self.activation(z)
        x_decode = nn.Dense(self.input_dim)(z)
        return x_decode


class Autoencoder(nn.Module):
    input_dim: int
    latent_dim: int
    widths: list
    activation: nn.activation = nn.relu

    def setup(self):
        self.encoder = Encoder(
            self.input_dim, self.latent_dim, self.widths, self.activation)
        self.decoder = Decoder(
            self.latent_dim, self.input_dim, self.widths, self.activation)

        self.sindy_coefficients = self.param('sindy_coefficients', lambda rng, shape: random.normal(
            rng, shape), (self.library_dim, self.latent_dim))

    def __call__(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat


class SindyAutoencoder(nn.Module):
    input_dim: int
    latent_dim: int
    library_dim: int
    widths: list
    activation: nn.activation = nn.relu

    def setup(self):
        self.encoder = Encoder(
            self.input_dim, self.latent_dim, self.widths, self.activation)
        self.decoder = Decoder(
            self.latent_dim, self.input_dim, self.widths, self.activation)

        self.sindy_coefficients = self.param('sindy_coefficients', lambda rng, shape: random.normal(
            rng, shape), (self.library_dim, self.latent_dim))

    def __call__(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z
