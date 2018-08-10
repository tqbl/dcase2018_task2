"""Predict labels for the test set using a second-level classifier.

This script trains a logistic regression classifier on the training set
meta-features created using the ``meta_features.py`` script. It then
generates predictions for either the training set or the test set. The
former refers to training and predicting each fold.

This script requires three command-line arguments:

  * train_path: Path to training features.
  * metadata_path: Path to training metadata.
  * output_path: Output file path.

It also takes an optional argument:

  * --test_path: Path to test features. If this is specified, the script
    will generate predictions for the test set and write them to a
    submission file. Otherwise, it will generate predictions for the
    training set on a fold-by-fold basis and write them to a csv file.
"""

import argparse
import sys

import h5py
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression

sys.path.append('task2')

import file_io as io
import utils as utils


def train(x, df):
    """Train a logistic regression classifier.

    Args:
        x (np.ndarray): Training data.
        df (pd.DataFrame): Training metadata.

    Returns:
        The trained classifier.
    """
    y = df.label.astype('category').cat.codes.values
    sample_weight = np.ones(len(x))
    sample_weight[df.manually_verified == 0] = 0.65

    clf = LogisticRegression(
        penalty='l2',
        tol=0.0001,
        C=1.0,
        random_state=1000,
        class_weight='balanced',
    )
    clf.fit(x, y, sample_weight=sample_weight)

    return clf


# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('train_path', help='path to training features')
parser.add_argument('metadata_path', help='path to training metadata')
parser.add_argument('output_path', help='output file path')
parser.add_argument('--test_path', help='path to test features')
args = parser.parse_args()

# Load training data
with h5py.File(args.train_path, 'r') as f:
    x_train = np.array(f['F'])

    df_train = pd.read_csv(args.metadata_path, index_col=0)
    y_train = df_train.label.astype('category').cat.codes.values

if args.test_path:
    # Load test data
    with h5py.File(args.test_path, 'r') as f:
        x_test = np.array(f['F'])

        index = pd.Index(f['names'], name='fname')

    # Train and predict the test data
    clf = train(x_train, df_train)
    y_pred = clf.predict_proba(x_test)

    # Write to a submission file.
    df_pred = pd.DataFrame(y_pred, index=index, columns=utils.LABELS)
    io.write_predictions(df_pred, args.output_path)
else:
    index = pd.Index([], name='fname')

    # Train and predict for each fold and concatenate the predictions
    y_preds = []
    for fold in range(5):
        mask = df_train.fold == fold
        index = index.append(df_train[mask].index)
        clf = train(x_train[~mask], df_train[~mask])
        y_preds.append(clf.predict_proba(x_train[mask]))
    y_pred = np.concatenate(y_preds)

    # Write to a CSV file
    df_pred = pd.DataFrame(y_pred, index=index, columns=utils.LABELS)
    df_pred = df_pred.loc[df_train.index]
    df_pred.to_csv(args.output_path)
