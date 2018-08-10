"""Relabel/promote training examples based on predicted labels.

This script is for pseudo-labeling non-verified examples. It can also
promote non-verified examples to verified if the predicted labels match
the ground truth labels. In both cases, the confidence of the prediction
must exceed a certain threshold.

This script requires three command-line arguments:

  * metadata_path: Path to metadata file containing ground truth.
  * pred_path: Path to training predictions.
  * output_path: Output file path.

It also takes optional arguments:

  * relabel_threshold: Confidence threshold for relabeling.
  * promote_threshold: Confidence threshold for promotion.
"""

import argparse

import h5py
import numpy as np
import pandas as pd


# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('metadata_path', help='path to metadata')
parser.add_argument('pred_path', help='path to predictions')
parser.add_argument('output_path', help='output file path')
parser.add_argument('--relabel_threshold', type=float, default=0,
                    help='confidence threshold for relabeling')
parser.add_argument('--promote_threshold', type=float, default=1.0,
                    help='confidence threshold for promotion')
args = parser.parse_args()

df_true = pd.read_csv(args.metadata_path, index_col=0)
df_pred = pd.read_csv(args.pred_path, index_col=0)
top_label = df_pred.idxmax(axis=1)
confidence = df_pred.max(axis=1)

# Determine which examples should be relabeled or promoted
relabel_mask = (df_true.manually_verified == 0) \
               & (top_label != df_true.label) \
               & (confidence > args.relabel_threshold)
promote_mask = (df_true.manually_verified == 0) \
               & (top_label == df_true.label) \
               & (confidence > args.promote_threshold)

df_true.loc[relabel_mask, 'label'] = top_label[relabel_mask]
print('%d examples relabeled' % sum(relabel_mask))

df_true.loc[promote_mask, 'manually_verified'] = 2 
print('%d examples promoted' % sum(promote_mask))

# Save as a new metadata file
df_true.to_csv(args.output_path)
