import numpy as np
import pandas as pd
import sklearn.metrics as metrics

import inference
import utils


def evaluate_audio_tagging(y_true, y_pred, threshold=-1):
    """Evaluate audio tagging performance.

    Three types of scores are returned:

      * Class-wise
      * Macro-averaged
      * Micro-averaged

    The ground truth values and predictions should both be passed in a
    2D array in which the first dimension is the sample axis and the
    second is the class axis.

    Args:
        y_true (np.ndarray): 2D array of ground truth values.
        y_pred (np.ndarray): 2D array of predictions.
        threshold (number): Threshold used to binarize predictions.

    Returns:
        pd.DataFrame: Table of evaluation results.
    """
    y_pred_b = inference.binarize_predictions(y_pred, threshold)

    class_scores = compute_audio_tagging_scores(y_true, y_pred, y_pred_b).T
    macro_scores = np.mean(class_scores, axis=0, keepdims=True)
    micro_scores = compute_audio_tagging_scores(
        y_true, y_pred, y_pred_b, average='micro')

    # Create DataFrame of evaluation results
    data = np.concatenate((class_scores, macro_scores, micro_scores[None, :]))
    index = utils.LABELS + ['Macro Average', 'Micro Average']
    columns = ['MAP@3', 'F-score', 'Precision', 'Recall']
    return pd.DataFrame(data, pd.Index(index, name='Class'), columns)


def compute_audio_tagging_scores(y_true, y_pred, y_pred_b, average=None):
    """Compute prediction scores using several performance metrics.

    The following metrics are used:

      * MAP@3
      * F1 Score
      * Precision
      * Recall

    Args:
        y_true (np.ndarray): 2D array of ground truth values.
        y_pred (np.ndarray): 2D array of prediction probabilities.
        y_pred_b (np.ndarray): 2D array of binary predictions.
        average (str): The averaging method. Either ``'macro'``,
            ``'micro'``, or ``None``, where the latter is used to
            disable averaging.

    Returns:
        np.ndarray: Scores corresponding to the metrics used.
    """
    # Compute MAP@3
    map_3 = compute_map(y_true, y_pred, k=3, class_wise=average is None)

    # Compute precision and recall scores
    precision, recall, f1_score, _ = metrics.precision_recall_fscore_support(
        y_true, y_pred_b, average=average)

    return np.array([map_3, f1_score, precision, recall])


def compute_map(y_true, y_pred, k=3, class_wise=False):
    """Compute the mean average precision at k (MAP@k).

    Args:
        y_true (np.ndarray): 2D array of ground truth values.
        y_pred (np.ndarray): 2D array of predictions.
        k (int): The maximum number of predicted elements.
        class_wise (bool): Whether to compute a score for each class.

    Returns:
        float or np.ndarray: The mean average precision score(s) at k.

    Note:
        This function assumes the grounds truths are single-label.
    """
    if class_wise:
        nonzero = np.nonzero(y_true)[1]
        return np.array([compute_map(y_true[nonzero == i],
                                     y_pred[nonzero == i], k)
                         for i in range(y_true.shape[1])])

    # Compute how the true label ranks in terms of probability
    idx = y_pred.argsort()[:, ::-1].argsort()
    rank = idx[y_true.astype(bool)] + 1

    if len(rank) > len(y_true):
        raise Exception('Multi-label classification not supported')

    return np.sum(1 / rank[rank <= k]) / len(y_true)
