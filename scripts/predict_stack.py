"""Predict labels for the test set using a second-level classifier.

This script trains a logistic regression classifier on the training set
meta-features created using the ``meta_features.py`` script. It then
uses the classifier to predict the labels of the test set and writes the
predictions in a submission file.

This script requires four command-line arguments:

  * train_path: Path to training features.
  * test_path: Path to test features.
  * metadata_path: Path to training metadata.
  * output_path: Submission file path.
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


# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('train_path', help='path to training features')
parser.add_argument('test_path', help='path to test features')
parser.add_argument('metadata_path', help='path to training metadata')
parser.add_argument('output_path', help='submission file path')
args = parser.parse_args()

# Load training data
with h5py.File(args.train_path, 'r') as f:
    x_train = np.array(f['F'])

    df_train = pd.read_csv(args.metadata_path, index_col=0)
    y_train = df_train.label.astype('category').cat.codes.values

# Load test data
with h5py.File(args.test_path, 'r') as f:
    x_test = np.array(f['F'])

    index_test = pd.Index(f['names'], name='fname')

# Train logistic regression classifier
clf = LogisticRegression(
    penalty='l2',
    tol=0.0001,
    C=1.0,
    random_state=1000,
    class_weight='balanced',
)
sample_weight = np.ones(len(x_train))
sample_weight[df_train.manually_verified == 0] = 0.65
clf.fit(x_train, y_train, sample_weight=sample_weight)

# Predict test set data
y_pred = clf.predict_proba(x_test)

# Write predictions to a submission file
df_pred = pd.DataFrame(y_pred, index=index_test, columns=utils.LABELS)
io.write_predictions(df_pred, args.output_path)
