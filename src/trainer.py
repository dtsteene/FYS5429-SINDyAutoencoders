# not using package style imports to avoid easier use in google colab
from UvAutils.Basetrainer import TrainerModule
from sindy_autoencoder import SindyAutoencoder
from typing import Any, Callable, Dict, Tuple
import jax.numpy as jnp
from flax.training import train_state

# import our loss functions
from loss import loss_recon, loss_dynamics_dz, loss_regularization


class TrainState(train_state.TrainState):
    # A simple extension of TrainState to also include batch statistics
    # If a model has no batch statistics, it is None
    batch_stats: Any = None
    # You can further extend the TrainState by any additional part here
    # For example, rng to keep for init, dropout, etc.
    rng: Any = None
    mask: jnp.ndarray = None


def compute_masked_coefficients(coefficients, mask):
    return coefficients * mask


def update_mask(coefficients, threshold=0.1):
    return jnp.where(jnp.abs(coefficients) >= threshold, 1, 0)


class Trainer(TrainerModule):
    def __init__(self, model, train_loader, val_loader, optimizer, loss_fn, device, **kwargs):
        super().__init__(model, train_loader, val_loader,
                         optimizer, loss_fn, device, **kwargs)

    def create_functions(self) -> Tuple[Callable[[TrainState, Any], Tuple[TrainState, Dict]],
                                        Callable[[TrainState, Any], Tuple[TrainState, Dict]]]:
        """
        UVA docstring:

        'Creates and returns functions for the training and evaluation step. The
        functions take as input the training state and a batch from the train/
        val/test loader. Both functions are expected to return a dictionary of
        logging metrics, and the training function a new train state. This
        function needs to be overwritten by a subclass. The train_step and
        eval_step functions here are examples for the signature of the functions.'

        The main metric is of course the loss, but other metrics like accuracy, precision, recall, etc.
        can be added. So these funcsions need to calculate the loss of a state given a batch.

        NOTE: These funcions are to be jit-compiled and should adhere to the functional programming scheme
        of JAX!

        Returns:
            train_step: Function that executes a training
                step on a batch of data.
            eval_step: Function that executes an evaluation
                step on a batch of data.

        """
        def train_step(state: TrainState,
                       batch: Any):
            metrics = {}

            # get mask from state ?
            mask = state.mask
            optimizer = state.optimizer

            # optimizer.target accesess the paramaters which are being optimized.
            masked_coefficients = compute_masked_coefficients(
                optimizer.target.coefficients, mask)
            optimizer = optimizer.replace(
                target=optimizer.target.replace(coefficients=masked_coefficients))

            """
            THIS NEEDS  TO BE ADDED TO THE MAIN TRAINING LOOP FOR MASKING THE COEFFICIENTS

            # Update the mask every #OF ITERATIONS
            if epoch % # OF ITERATIONS == 0:
                mask = update_mask(optimizer.target.coefficients)
            """
            return state, metrics

        def eval_step(state: TrainState,
                      batch: Any):
            metrics = {}
            return metrics

        return train_step, eval_step


if __name__ == "__main__":
    # how one might use the trainer

    model_hparams = {'input_dim': 2,
                     'latent_dim': 2, 'widths': [60, 40, 20, 3], }
    optimizer_hparams = {'learning_rate': 1e-3}

    # 12 examples of 124 dimensions. Example input should not be created manualy like this, but rather fetched from a pytorch dataloader,
    # that way batchdim ect is always correct
    exmp_input = jnp.ones((12, 124))

    trainer = TrainerModule(
        SindyAutoencoder,  model_hparams, optimizer_hparams, exmp_input)
    # trainer.train(train_loader, val_loader, test_loader: Optional, num_epochs)  #data loader object should be created my jouval and daniel.
    # now everything is stored in logger, that supposedly can plot the matrics nicely for us?
    # trainer.logger.plot()? -this is just a wonky suggestion
