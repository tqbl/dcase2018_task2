import numpy as np


class MixupGenerator():
    """Implementation of mixup [1]_ data augmentation.

    Args:
        x_train (np.ndarray): Array of training data.
        y_train (np.ndarray): Target values of the training data.
        sample_weight (np.ndarray): Weights for the training data.
        batch_size (int): Number of examples in a mini-batch.
        alpha (float): Parameter for sampling mixing weights.
        generator (ImageDataGenerator): Generator for preprocessing.

    Attributes:
        x_train (np.ndarray): Array of training data.
        y_train (np.ndarray): Target values of the training data.
        sample_weight (np.ndarray): Weights for the training data.
        batch_size (int): Number of examples in a mini-batch.
        alpha (float): Parameter for sampling mixing weights.
        generator (ImageDataGenerator): Generator for preprocessing.

    References:
        .. [1] Zhang, H. and Cisse, M. and Dauphin, Y.~N. and Lopez-Paz,
               “mixup: Beyond Empirical Risk Minimization,”
    """

    def __init__(self, x_train, y_train, sample_weight=None,
                 batch_size=32, alpha=1.0, generator=None):
        self.x_train = x_train
        self.y_train = y_train
        self.sample_weight = sample_weight
        self.batch_size = batch_size
        self.alpha = alpha
        self.generator = generator

    def __call__(self):
        batch_size = self.batch_size
        n_classes = self.y_train.shape[1]
        n_examples = np.sum(self.y_train, axis=0).astype(int)
        indexes = [np.where(self.y_train[:, label] == 1)[0]
                   for label in range(n_classes)]
        offsets = [0] * n_classes

        while True:
            # Choose which class each mini-batch example will belong to
            labels = np.random.choice(n_classes, size=(batch_size * 2,))
            batch_indexes = np.empty(batch_size * 2, dtype=int)

            for i, label in enumerate(labels):
                batch_indexes[i] = indexes[label][offsets[label]]

                offsets[label] += 1
                if offsets[label] >= n_examples[label]:
                    np.random.shuffle(indexes[label])
                    offsets[label] = 0

            x, y, sample_weight = self._generate(batch_indexes)

            yield x, y, sample_weight

    def _generate(self, indexes):
        # Generate mixing weights using beta distribution
        mixup_weights = np.random.beta(a=self.alpha, b=self.alpha,
                                       size=self.batch_size)

        # Mix training data and labels
        x = self._mixup(self.x_train, indexes,
                        mixup_weights[:, None, None, None])
        y = self._mixup(self.y_train, indexes, mixup_weights[:, None])

        # Mix sample weights if applicable
        sample_weight = self.sample_weight
        if sample_weight is not None:
            sample_weight = self._mixup(sample_weight, indexes, mixup_weights)

        # Apply preprocessing to training data
        if self.generator:
            for i in range(self.batch_size):
                x[i] = self.generator.random_transform(x[i])
                x[i] = self.generator.standardize(x[i])

        return x, y, sample_weight

    def _mixup(self, tensor, indexes, weights):
        t1 = tensor[indexes[:self.batch_size]]
        t2 = tensor[indexes[self.batch_size:]]
        return t1 * weights + t2 * (1 - weights)
