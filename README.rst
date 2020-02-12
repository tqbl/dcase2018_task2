Surrey CVSSP DCASE 2018 Task 2 System
=====================================

This is the source code for CVSSP's system used in `DCASE 2018 Task 2`__.

The accompanying technical report can be found `here`__.

__ http://dcase.community/challenge2018/task-general-purpose-audio-tagging
__ http://dcase.community/challenge2018/task-general-purpose-audio-tagging-results#Iqbal2018

Requirements
------------

This software requires Python 3. To install the dependencies, run::

    pipenv install

or::

    pip install -r requirements.txt

The main functionality of this software also requires the DCASE 2018 Task 2
datasets, which may be downloaded `here`__. After acquiring the datasets,
modify ``task2/config/dcase2018_task2.py`` accordingly.

For example::

    _root_dataset_path = ('/path/to/datasets')
    """str: Path to root directory containing input audio clips."""

    training_set = Dataset(
        name='training',
        path=os.path.join(_root_dataset_path, 'audio_train'),
        metadata_path='metadata/training.csv',
    )
    """Dataset instance for the training dataset."""

You will also want to change the work path in ``task2/config/paths.py``::

    work_path = '/path/to/workspace'
    """str: Path to parent directory containing program output."""

__ https://www.kaggle.com/c/freesound-audio-tagging/data

Usage
-----

In this section, the various commands are described. Using this software, the
user is able to apply preprocessing (silence removal), extract feature vectors,
train the network, generate predictions, and evaluate the predictions.

Preprocessing
^^^^^^^^^^^^^

Our implementation of preprocessing involves extracting the non-silent sections
of audio clips and saving these to disk separately. A new metadata file is then
created with entries corresponding to the new files.

To apply preprocessing, run::

    python task2/main.py preprocess <training/test>

Refer to ``task2/silence.py`` for the relevant code.

Feature Extraction
^^^^^^^^^^^^^^^^^^

To extract feature vectors, run::

    python task2/main.py extract <training/test> [--recompute]

If ``--recompute`` is enabled, the program will recompute existing feature
vectors. This implementaion extracts log-mel spectrogram features. See
``task2/config/logmel.py`` for tweaking the parameters.

Training
^^^^^^^^

To train a model, run::

    python task2/main.py train [--model MODEL] [--fold n] [--sample_weight x] [--class_weight]

The ``--model`` option can be one of the following:

* ``vgg13``
* ``gcnn``
* ``crnn``
* ``gcrnn``

The training set is assumed to be split into several folds, so the ``--fold``
option specifies which one to use as the validation set. If set to ``-1``, the
program trains on the entire dataset. The ``--sample_weight`` option allows
setting a sample weight to be used for unverified (noisy) examples. Finally,
setting the ``--class_weight`` flag indicates that examples should be weighted
based on the class that they belong to.

See ``task2/config/training.py`` for tweaking the parameters or
``task2/training.py`` for further modifications.

Prediction
^^^^^^^^^^

To generate predictions, run::

    python task2/main.py predict <training/test> [--fold n]

The ``--fold`` option specifies which fold-specific model to use.

See ``task2/config/predictions.py`` to modify which epochs are selected for
generating the predictions. By default, the top four models based on their MAP
score on the validation set are chosen.

Evaluation
^^^^^^^^^^

To evaluate the predictions, run::

    python task2/main.py evaluate <training/test> [--fold n]


Stacking
^^^^^^^^
Stacking is an ensembling technique that involves creating meta-features based
on the predictions of a number of base classifiers. These meta-features are
then used to train a second-level classifier and generate new predictions. We
provide scripts to do this.

To generate meta-features, run::

    python scripts/meta_features.py <pred_path> <pred_type> <output_path>

The argument ``pred_path`` refers to the parent directory in which the
predictions of the base classifiers are stored. ``pred_type`` must be either
``training`` or ``test``, depending on which dataset the meta-features are for.
``output_path`` specifies the path of the output HDF5 file.

To give an example, assume that the directory structure looks like this::

    workspace
    ├── predictions
    │   ├── classifier1
    │   ├── classifier2
    │   ├── classifier3

In this case, you might run::

    python scripts/meta_features.py workspace/predictions training training.h5
    python scripts/meta_features.py workspace/predictions test test.h5

For the time being, the script must be edited to select the classifiers.

To then generate predictions using a second-level classifier, run::

    python scripts/predict_stack.py --test_path test.h5 training.h5 <metadata_path> <output_path>

The argument ``metadata_path`` is the path to the training set metadata file.
See the script itself for more details.


Pseudo-labeling
^^^^^^^^^^^^^^^
To relabel or promote training examples, run::

    python scripts/relabel.py <metadata_path> <pred_path> <output_path> [--relabel_threshold t1] [--promote_threshold t2]

The argument ``metadata_path`` is the path to the training set metadata file
containing the original labels. ``pred_path`` is the path to the predictions
file used for pseudo-labeling. ``output_path`` is the path of the new metadata
file to be written. The threshold options allow constraining which examples are
relabeled or promoted.
