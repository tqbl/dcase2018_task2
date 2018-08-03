"""Split the training set into K folds.

This script requires three command-line arguments:

  * metadata_path: Path to training set metadata.
  * output_path: Output file path.
  * n_folds: Number of folds to use.

The output is a new metadata file that assigns each example to a fold.
"""

import argparse

import pandas as pd

from sklearn.model_selection import StratifiedKFold


# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('metadata_path', help='path to training set metadata')
parser.add_argument('output_path', help='output metadata file path')
parser.add_argument('--n_folds', type=int, default=5,
                    help='number of folds to use')
args = parser.parse_args()

# Create dummy labels to ensure each fold has a similar number of
# manually verified examples.
df = pd.read_csv(args.metadata_path, index_col=0)
labels = df.label + df.manually_verified.astype(str)

# Assign a fold number to each example
df['fold'] = -1
skf = StratifiedKFold(args.n_folds)
for i, (_, te) in enumerate(skf.split(df.index, labels)):
    df.iloc[te, 2] = i

print('Number of verified examples per fold:')
print([sum((df.fold == i) & (df.manually_verified == 1))
       for i in range(args.n_folds)])

# Save new metadata file to disk
df.to_csv(args.output_path)
