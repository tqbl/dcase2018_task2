import collections
import os.path

import config.paths as paths


Dataset = collections.namedtuple('Dataset',
                                 ['name',
                                  'path',
                                  'metadata_path',
                                  ])
"""Data structure encapsulating information about a dataset."""


_root_dataset_path = ('/vol/vssp/datasets/audio/dcase2018/task2')
"""str: Path to root directory containing input audio clips."""

training_set = Dataset(
    name='training',
    path=os.path.join(_root_dataset_path, 'audio_train'),
    metadata_path='metadata/training.csv',
)
"""Dataset instance for the training dataset."""

test_set = Dataset(
    name='test',
    path=os.path.join(_root_dataset_path, 'audio_test'),
    metadata_path='metadata/test.csv',
)
"""Dataset instance for the test dataset."""

preprocessed_training_set = Dataset(
    name='training',
    path=os.path.join(paths.preprocessing_path, 'training'),
    metadata_path=os.path.join(paths.preprocessing_path, 'training.csv'),
)
"""Dataset instance for the preprocessed training dataset."""

preprocessed_test_set = Dataset(
    name='test',
    path=os.path.join(paths.preprocessing_path, 'test'),
    metadata_path=os.path.join(paths.preprocessing_path, 'test.csv'),
)
"""Dataset instance for the preprocessed test dataset."""


def to_dataset(name, preprocessed=True):
    """Return the Dataset instance corresponding to the given name.

    Args:
        name (str): Name of dataset.
        preprocessed (bool): Whether to return the preprocessed instance.

    Returns:
        The Dataset instance corresponding to the given name.
    """
    if name == 'training':
        return preprocessed_training_set if preprocessed else training_set
    elif name == 'test':
        return preprocessed_test_set if preprocessed else test_set
    return None
