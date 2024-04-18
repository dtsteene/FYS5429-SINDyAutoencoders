# not using package style imports to avoid easier use in google colab
# import sys
# sys.path.append('UvAutils')
from UvAutils.data_utils import create_data_loaders, numpy_collate
from UvAutils.Basetrainer import TrainerModule
from autoencoder import Autoencoder

model = Autoencoder(input_dim=2, latent_dim=2, widths=[
                    32, 32], poly_order=2, include_sine=True, library_dim=4, model_order=2)

trainer = TrainerModule(model, create_data_loaders, numpy_collate)
