import numpy as np
import pandas as pd
import scipy.stats as stats

import utils


def binarize_predictions(y_pred, threshold=-1):
    """Convert prediction probabilities to binary values.

    This function is intended for audio tagging predictions. The
    predictions should be passed in a 2D array in which the first
    dimension is the sample axis and the second is the class axis.

    Args:
        y_pred (np.ndarray): 2D array of predictions.
        threshold (float or list): Threshold used to determine the
            binary values. If a list is given, it must specify a
            threshold for each class. If the value is -1, the label
            with the highest probability is selected.

    Returns:
        np.ndarray: Binarized prediction values.
    """
    if threshold > 0:
        return (y_pred > threshold).astype(int)

    return utils.to_categorical(np.argmax(y_pred, axis=1))


def merge_predictions(y_pred, index, op='gmean'):
    """Merge predictions of chunks belonging to the same audio clip.

    Args:
        y_pred (np.ndarray): 2D array of chunk-level predictions.
        index (pd.Index): Files names indicating how to group chunks.
        op (str): The operation to perform on grouped predictions.
            Either ``'first'``, ``'mean'``, or ``'gmean'``.

    Returns:
        pd.DataFrame: The merged predictions.
    """
    pred = pd.DataFrame(y_pred, index=index, columns=utils.LABELS)
    group = utils.group_by_name(pred)
    if op == 'first':
        pred = group.first()
    elif op == 'mean':
        pred = group.mean()
    elif op == 'gmean':
        # TODO: Improve performance as this operation is slow
        pred = group.agg(lambda x: stats.gmean(x + 1e-8))

    pred.index.name = index.name

    return pred
