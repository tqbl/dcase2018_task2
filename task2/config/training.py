training_id = 'jul31_vgg13'
"""str: A string identifying this particular training instance."""

initial_seed = 1000
"""int: Fixed seed used prior to training."""

batch_size = 128
"""int: The number of samples in a mini batch."""

n_epochs = 40
"""int: The number of epochs to train the network for.

A value of -1 indicates an early stopping condition should be used.
"""

learning_rate = {'initial': 0.0005,
                 'decay': 0.90,
                 'decay_rate': 2.,
                 }
"""dict: Learning rate hyperparameters for SGD.

Keyword Args:
    initial (float): Initial learning rate.
    decay (float): Multiplicative factor for learning rate decay. A
        value of 1 indicates the learning rate should not be decayed.
    decay_rate (float): Number of epochs until learning rate is decayed.
"""
