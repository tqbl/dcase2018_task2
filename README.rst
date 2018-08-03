Surrey CVSSP DCASE 2018 Task 2 System
=====================================

This is the source code for the system used in the DCASE 2018 Task 2 challenge.

Requirements
------------

This software requires Python 3. To install the dependencies, run::

    pipenv install

or::

    pip install -r requirements.txt

The main functionality of this software also requires the DCASE 2018 Task 2
datasets, which may be downloaded here_. After acquiring the datasets, modify
``task2/config/paths.py`` accordingly.

For example::

    _root_dataset_path = ('/path/to/datasets')
    """str: Path to root directory containing input audio clips."""

    training_set = Dataset(
        name='training',
        path=os.path.join(_root_dataset_path, 'audio_train'),
        metadata_path='metadata/training.csv',
    )
    """Dataset instance for the training dataset."""

You may also want to change the work path::

    work_path = '/path/to/workspace'
    """str: Path to parent directory containing program output."""

.. _here: https://www.kaggle.com/c/freesound-audio-tagging/data

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
vectors. See ``task2/config/logmel.py`` for tweaking the parameters.

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

    python task2/main.py evaluate <fold>
