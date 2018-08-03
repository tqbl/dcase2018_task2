import os

import sklearn.metrics as metrics

from keras.callbacks import Callback
from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
import keras.utils

from mixup import MixupGenerator
import config as cfg
import convnet
import evaluation
import inference
import utils


def train(tr_x, tr_y, val_x, val_y, val_index, model_id='gcnn',
          fold=-1, sample_weight=None, class_weight=None):
    """Train a neural network using the given training set.

    Args:
        tr_x (np.ndarray): Array of training data.
        tr_y (np.ndarray): Target values of the training data.
        val_x (np.ndarray): Array of validation data.
        val_y (np.ndarray): Target values of the validation data.
        val_index (pd.Index): File names of validation data. Used to
            group chunks in order to compute clip-level predictions.
        model_id (str): The neural network architecture.
        fold (int): Fold number identifying validation set.
        sample_weight (float): Weights for the training examples.
        class_weight (float): Class-wise weights.
    """
    if model_id == 'gcnn':
        create_model = convnet.gcnn
    elif model_id == 'vgg13':
        create_model = convnet.vgg13
    elif model_id == 'crnn':
        create_model = convnet.crnn
    elif model_id == 'gcrnn':
        create_model = convnet.gcrnn

    # Create model and print summary
    model = create_model(input_shape=tr_x.shape[1:],
                         n_classes=tr_y.shape[1])
    _print_model_summary(model)

    # Use Adam SGD optimizer
    optimizer = Adam(lr=cfg.learning_rate['initial'])
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'],
                  )

    # Create the appropriate callbacks to use during training
    callbacks = _create_callbacks(fold)
    for callback in callbacks:
        callback.val_index = val_index

    # Set a large value for `n_epochs` if early stopping is used
    n_epochs = cfg.n_epochs
    if n_epochs < 0:
        n_epochs = 10000

    # Standardize validation data
    generator = utils.fit_scaler(tr_x)
    if len(val_x):
        validation_data = (generator.standardize(val_x), val_y)
    else:
        validation_data = None

    # Redefine generator for mixup data augmentation
    batch_size = cfg.batch_size
    generator = MixupGenerator(tr_x,
                               tr_y,
                               sample_weight=sample_weight,
                               batch_size=batch_size,
                               alpha=1.0,
                               generator=generator,
                               )

    return model.fit_generator(generator(),
                               steps_per_epoch=len(tr_x) // batch_size,
                               epochs=n_epochs,
                               callbacks=callbacks,
                               validation_data=validation_data,
                               class_weight=class_weight,
                               )


class Evaluator(Callback):
    """A base class for logging evaluation results."""

    def predict(self):
        """Predict target values of the validation data.

        The main utility of this function is to merge the predictions of
        chunks belonging to the same audio clip. The same is done for
        the ground truth target values so that dimensions match.

        Returns:
            tuple: Tuple containing:

                y_true (np.ndarray): Ground truth target values.
                y_pred (np.ndarray): Predicted target values.
        """
        x, y_true = self.validation_data[:2]

        y_pred = self.model.predict(x)
        y_true = inference.merge_predictions(y_true, self.val_index, 'first')
        y_pred = inference.merge_predictions(y_pred, self.val_index)
        return y_true.values, y_pred.values


class MAPLogger(Evaluator):
    """A callback for computing the mean average precision at k (MAP@k).

    At the end of each epoch, the MAP is computed and logged for the
    predictions of the validation dataset. It is assumed that the ground
    truths are single-label.

    Args:
        k (int): The maximum number of predicted elements.

    Attributes:
        k (int): The maximum number of predicted elements.
    """

    def __init__(self, k=3):
        super(MAPLogger, self).__init__()

        self.k = k

    def on_epoch_end(self, epoch, logs=None):
        """Compute the MAP of the validation set predictions."""
        y_true, y_pred = self.predict()
        map_k = evaluation.compute_map(y_true, y_pred, self.k)
        map_k_min = min(evaluation.compute_map(y_true, y_pred, self.k, True))

        # Log the computed value
        logs = logs or {}
        logs['val_map'] = map_k
        logs['val_map_min'] = map_k_min


class F1ScoreLogger(Evaluator):
    """A callback for computing the F1 score.

    At the end of each epoch, the F1 score is computed and logged for
    the predictions of the validation dataset.

    Args:
        threshold (float): Threshold used to binarize predictions.

    Attributes:
        threshold (float): Threshold used to binarize predictions.
    """

    def __init__(self, threshold=-1):
        super(F1ScoreLogger, self).__init__()

        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        """Compute the F1 score of the validation set predictions."""
        y_true, y_pred = self.predict()
        y_pred_b = inference.binarize_predictions(y_pred, self.threshold)
        f1_score = metrics.f1_score(y_true, y_pred_b, average='micro',
                                    labels=range(y_true.shape[1]))

        # Log the computed value
        logs = logs or {}
        logs['val_f1_score'] = f1_score


def _print_model_summary(model):
    """Print a summary of the model and also write the summary to disk.

    Args:
        model: The Keras model to summarize.
    """
    keras.utils.print_summary(model)
    with open(os.path.join(cfg.model_path, 'summary.txt'), 'w') as f:
        keras.utils.print_summary(model, print_fn=lambda s: f.write(s + '\n'))


def _create_callbacks(fold):
    """Create a list of training callbacks.

    The following callbacks are included in the list:
      * Several performance-logging callbacks.
      * A callback for logging results to a CSV file.
      * A callback for saving models.
      * A callback for using TensorBoard.
      * An optional callback for learning rate decay.
      * An optional callback for early stopping.

    Args:
        fold (int): Fold number identifying validation set.

    Returns:
        list: List of Keras callbacks.
    """
    # Create callbacks for computing various metrics and logging them
    callbacks = []
    if fold >= 0:
        callbacks += [MAPLogger(), F1ScoreLogger(cfg.threshold),
                      CSVLogger(cfg.history_path.format(fold))]

    # Create callback to save model after every epoch
    fold_dir = str(fold) if fold >= 0 else 'all'
    path = os.path.join(cfg.model_path, fold_dir,
                        'model.{epoch:02d}-{acc:.4f}.h5')
    callbacks.append(ModelCheckpoint(filepath=path, monitor='acc'))

    # Create callback for TensorBoard logs
    callbacks.append(TensorBoard(cfg.log_path.format(fold_dir),
                                 batch_size=cfg.batch_size))

    lr_decay = cfg.learning_rate['decay']
    if lr_decay < 1.:
        # Create callback to decay learning rate
        def _lr_schedule(epoch, lr):
            decay = epoch % cfg.learning_rate['decay_rate'] == 0
            return lr * lr_decay if decay else lr
        callbacks.append(LearningRateScheduler(schedule=_lr_schedule))

    if cfg.n_epochs == -1:
        # Create callback to use an early stopping condition
        callbacks.append(EarlyStopping(monitor='val_loss',
                                       min_delta=0,
                                       patience=5,
                                       ))

    return callbacks
