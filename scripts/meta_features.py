"""Generate meta-features for stacking.

After training a model on the cross-validation folds, the user can
generate predictions for the validation sets -- which constitute the
training set -- and the test set. This script generates features based
on these predictions. For example, if we have five models, and each
model outputs an N x K matrix of predictions, where N is the number of
predicted audio clips and K=41 is the number of classes, this script
concatenates these to produce an N x 5K matrix, i.e. N feature vectors.

This script requires three command-line arguments:

  * pred_path: Path to predictions directory.
  * pred_type: Either ``'training'`` or ``'test'``.
  * output_path: Output file path of meta-features.

It is assumed that the relevant predictions have already been generated
for each fold. This script merges the fold predictions into one.
"""

import argparse
import os.path

import h5py
import numpy as np
import pandas as pd


MODELS = [
    'jul28_pydub_gcnn',
    'jul28_pydub_gcnn_1s',
    'jul31_pydub_vgg13',
    'jul31_pydub_vgg13_1s',
    'jul28_pydub_crnn',
    'jul28_pydub_crnn_1s',
    'jul25_pydub_gcrnn',
    'jul30_pydub_gcrnn_1s',
]
"""The training IDs of the models to use."""


def merge_predictions(base_path, pred_type, n_folds=5):
    """Merge the predictions of the training folds.

    If the predictions are for the training set, they are collated. If
    they are for the test set, they are averaged.

    Args:
        base_path (str): Path to predictions directory.
        pred_type (str): Either ``'training'`` or ``'test'``.
        n_folds (int): Number of training folds.

    Returns:
        pd.DataFrame: The merged predictions.
    """
    dfs = []
    for i in range(n_folds):
        name = 'fold' if pred_type == 'training' else pred_type
        path = os.path.join(base_path, 'predictions_%s%d.csv' % (name, i))
        dfs.append(pd.read_csv(path, index_col=0))

    df = pd.concat(dfs)
    if pred_type == 'training':
        metadata_path = '/vol/vssp/msos/ti/dcase2018/task2/metadata/train.csv'
        df_train = pd.read_csv(metadata_path, index_col=0)
        return df.loc[df_train.index]
    if pred_type == 'test':
        return df.groupby(level=0).mean()


# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('pred_path', help='path to predictions directory')
parser.add_argument('pred_type', help='either "training" or "test"')
parser.add_argument('output_path', help='output file path')
args = parser.parse_args()

# Collect predictions for each model
feats = []
top_preds = []
for model in MODELS:
    path = os.path.join(args.pred_path, model)
    df = merge_predictions(path, args.pred_type)
    feats.append(df.values)

    top_preds.append(df.idxmax(axis=1).astype('category').cat.codes)

# Print correlation matrix
print(pd.concat(top_preds, axis=1).corr())

# Save meta-features to disk
feats = np.stack(feats, axis=1)
feats = np.reshape(feats, (feats.shape[0], -1))
with h5py.File(args.output_path, 'w') as f:
    f.create_dataset('F', data=feats)
    f.create_dataset('names', data=top_preds[0].index.values,
                     dtype=h5py.special_dtype(vlen=str))
