import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

import config as cfg
import file_io as io
import utils


def main():
    """Execute a task based on the given command-line arguments.

    This function is the main entry-point of the program. It allows the
    user to extract features, train a model, generate predictions, or
    evaluate predictions using the command-line interface.
    """
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')

    parser_preprocess = subparsers.add_parser('preprocess')
    parser_preprocess.add_argument('dataset', choices=['training', 'test'])

    # Add sub-parser for feature extraction
    parser_extract = subparsers.add_parser('extract')
    parser_extract.add_argument('dataset', choices=['training', 'test'])
    parser_extract.add_argument('--recompute', action='store_true')

    # Add sub-parser for training
    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--model',
                              choices=['vgg13',
                                       'gcnn',
                                       'crnn',
                                       'gcrnn',
                                       ],
                              default='gcnn',
                              )
    parser_train.add_argument('--fold', type=int, default=-1)
    parser_train.add_argument('--class_weight', action='store_true')
    parser_train.add_argument('--sample_weight', type=float)

    # Add sub-parser for inference
    parser_predict = subparsers.add_parser('predict')
    parser_predict.add_argument('dataset', choices=['training', 'test'])
    parser_predict.add_argument('--fold', type=int, default=-1)

    # Add sub-parser for evaluation
    parser_evaluate = subparsers.add_parser('evaluate')
    parser_evaluate.add_argument('fold', type=int)

    args = parser.parse_args()
    if args.mode == 'preprocess':
        preprocess(cfg.to_dataset(args.dataset, preprocessed=False))
    elif args.mode == 'extract':
        extract(cfg.to_dataset(args.dataset), args.recompute)
    elif args.mode == 'train':
        train(args.model, args.fold, args.class_weight, args.sample_weight)
    elif args.mode == 'predict':
        predict(cfg.to_dataset(args.dataset), args.fold)
    elif args.mode == 'evaluate':
        evaluate_audio_tagging(args.fold)


def preprocess(dataset):
    """Apply preprocessing to the audio clips.

    Args:
        dataset: Dataset to apply preprocessing to.
    """
    import silence

    # Ensure output directory exists
    output_path = os.path.join(cfg.preprocessing_path, dataset.name)
    os.makedirs(output_path, exist_ok=True)

    # Split each audio clip based on silence
    file_names = []
    df = io.read_metadata(dataset.metadata_path)
    for name in tqdm(df.index):
        file_names += silence.split_audio(dataset.path, name, output_path)

    # Create new metadata DataFrame
    df = df.loc[[s[:8] + '.wav' for s in file_names]]
    df.index = file_names

    # Save metadata to disk
    df.to_csv(os.path.join(cfg.preprocessing_path, '%s.csv' % dataset.name))


def extract(dataset, recompute=False):
    """Extract feature vectors from the given dataset.

    Args:
        dataset: Dataset to extract features from.
        recompute (bool): Whether to recompute existing features.
    """
    import features

    # Use a logmel representation for feature extraction
    extractor = features.LogmelExtractor(cfg.sample_rate,
                                         cfg.n_window,
                                         cfg.hop_length,
                                         cfg.n_mels,
                                         )

    # Ensure output directory exists and set file path
    os.makedirs(cfg.extraction_path, exist_ok=True)
    output_path = os.path.join(cfg.extraction_path, dataset.name + '.h5')

    # Save free parameters to disk
    utils.log_parameters(cfg.logmel, os.path.join(cfg.extraction_path,
                                                  'parameters.json'))

    # Extract features for each audio clip in the dataset
    df = io.read_metadata(dataset.metadata_path)
    features.extract_dataset(dataset_path=dataset.path,
                             file_names=df.index.tolist(),
                             extractor=extractor,
                             output_path=output_path,
                             recompute=recompute,
                             )


def train(model, fold, use_class_weight, noisy_sample_weight):
    """Train the neural network model.

    Args:
        model (str): The neural network architecture.
        fold (int): The fold to use for validation.
        use_class_weight (bool): Whether to use class-wise weights.
        noisy_sample_weight (float): Examples that are not verified are
            weighted according to this value.

    Note:
        For reproducibility, the random seed is set to a fixed value.
    """
    import training

    # Try to create reproducible results
    np.random.seed(cfg.initial_seed)

    # Load (standardized) training data
    x, df = _load_data(cfg.to_dataset('training'))
    # Get one-hot representation of target values
    y = utils.to_categorical(df.label)

    # Split training data into training and validation
    if fold >= 0:
        mask = df.fold == fold
    else:
        mask = np.zeros(len(df), dtype=bool)
    val_mask = mask & (df.manually_verified == 1)

    tr_x = x[~mask]
    tr_y = y[~mask]
    val_x = x[val_mask]
    val_y = y[val_mask]
    val_index = df.index[val_mask]

    # Compute class weights based on number of class examples
    if use_class_weight:
        group = utils.group_by_name(df)
        n_examples = group.first().groupby('label').size().values
        class_weight = len(group) / (len(n_examples) * n_examples)
    else:
        class_weight = None

    # Assign a specific sample weight to unverified examples
    if noisy_sample_weight:
        sample_weight = df[~mask].manually_verified.values.astype(float)
        sample_weight[sample_weight == 0] = noisy_sample_weight
    else:
        sample_weight = None

    # Ensure output directories exist
    fold_dir = str(fold) if fold >= 0 else 'all'
    os.makedirs(os.path.join(cfg.model_path, fold_dir), exist_ok=True)
    os.makedirs(cfg.log_path.format(fold_dir), exist_ok=True)

    # Save free parameters to disk
    utils.log_parameters(cfg.training, os.path.join(cfg.model_path,
                                                    'parameters.json'))

    training.train(tr_x, tr_y, val_x, val_y, val_index, model, fold,
                   class_weight=class_weight, sample_weight=sample_weight)


def predict(dataset, fold):
    """Generate predictions for audio tagging.

    This function uses an ensemble of trained models to generate the
    predictions, with the averaging function being an arithmetic mean.
    Computed predictions are then saved to disk.

    Args:
        dataset: Dataset to generate predictions for.
        fold (int): The specific fold to generate predictions for. Only
            applicable for the training dataset.
    """
    import inference

    # Load (standardized) input data and associated metadata
    x, df = _load_data(dataset)
    dataset_name = dataset.name
    if dataset.name == 'training':
        dataset_name += str(fold)
        mask = df.fold == fold
        tr_x = x[~mask]
        x = x[mask]
        df = df[mask]
    else:
        tr_x, tr_df = _load_data(cfg.to_dataset('training'))
        if fold >= 0:
            dataset_name += str(fold)
            tr_x = tr_x[tr_df.fold != fold]

    generator = utils.fit_scaler(tr_x)
    x = generator.standardize(x)

    # Predict class probabilities for each model (epoch)
    preds = []
    for epoch in _determine_epochs(cfg.prediction_epochs, fold, n=4):
        pred = utils.timeit(
            lambda: _load_model(fold, epoch).predict(x),
            '[Epoch %d] Predicted class probabilities' % epoch)

        preds.append(inference.merge_predictions(pred, df.index))

    pred_mean = pd.concat(preds).groupby(level=0).mean()

    # Ensure output directory exists and set file path format
    os.makedirs(os.path.dirname(cfg.predictions_path), exist_ok=True)
    predictions_path = cfg.predictions_path.format('%s', dataset_name)

    # Save free parameters to disk
    utils.log_parameters({'prediction_epochs': cfg.prediction_epochs},
                         os.path.join(os.path.dirname(cfg.predictions_path),
                                      'parameters.json'))

    # Write predictions to disk
    pred_mean.to_csv(predictions_path % 'predictions')
    io.write_predictions(pred_mean, predictions_path % 'submission')


def evaluate_audio_tagging(fold):
    """Evaluate the audio tagging predictions and write results.

    Args:
        fold (int): The fold (validation set) to evaluate.
    """
    import evaluation

    # Load grouth truth data and predictions
    dataset = cfg.to_dataset('training')
    fold_str = 'training' + str(fold)
    df = io.read_metadata(dataset.metadata_path)
    df = utils.group_by_name(df[df.fold == fold]).first()
    y_true = utils.to_categorical(df.label)
    path = cfg.predictions_path.format('predictions', fold_str)
    y_pred = pd.read_csv(path, index_col=0).values

    # Mask out those that are not manually verified
    mask = y_true.manually_verified == 1
    y_pred = y_pred[mask]
    y_true = y_true[mask]

    # Evaluate audio tagging performance
    scores = evaluation.evaluate_audio_tagging(
        y_true, y_pred, threshold=cfg.threshold)

    # Ensure output directory exist and write results
    os.makedirs(os.path.dirname(cfg.results_path), exist_ok=True)
    output_path = cfg.results_path.format(fold_str)
    scores.to_csv(output_path)

    # Print scores to 3 decimal places
    pd.options.display.float_format = '{:,.3f}'.format
    print('\n' + str(scores))


def _load_data(dataset):
    """Load input data and the associated metadata for a dataset.

    Args:
        dataset: Structure encapsulating dataset information.

    Returns:
        tuple: Tuple containing:

            x (np.ndarray): The input data of the dataset.
            df (pd.DataFrame): The metadata of the dataset.
    """
    import features

    # Load feature vectors and reshape to 4D tensor
    features_path = os.path.join(cfg.extraction_path, dataset.name + '.h5')
    x, n_chunks = utils.timeit(lambda: features.load_features(features_path),
                               'Loaded features of %s dataset' % dataset.name)
    x = np.expand_dims(x, -1)
    assert x.ndim == 4

    # Load metadata and duplicate entries based on number of chunks
    df = io.read_metadata(dataset.metadata_path)
    df = df.loc[np.repeat(df.index, n_chunks)]

    return x, df


def _determine_epochs(spec, fold, n=5):
    """Return a list of epoch numbers based on the given argument.

    If `spec` is a list, this function simply returns the list.
    Otherwise, `spec` should be a string, in which case this function
    returns the top `n` epochs based on the training history file
    and the contents of `spec`. For example, if `spec` is ``'val_acc'``,
    the epochs that achieved the highest accuracy are returned.

    Args:
        spec (list or str): A list of epoch numbers or a string
            specifying how to select the epoch numbers.
        fold (int): Fold number, since determining the top epochs
            depends on the fold in question.
        n (int): Number of epochs to return (if applicable).

    Returns:
        list: The relevant epoch numbers.
    """
    if type(spec) is list:
        return spec

    fold_dir = str(fold) if fold >= 0 else 'all'
    path = cfg.history_path.format(fold_dir)
    history = io.read_training_history(path, ordering=spec)
    return (history.index.values + 1)[:n]


def _load_model(fold, epoch):
    """Load model based on specified fold and epoch number.

    Args:
        fold (int): Fold used to train the model.
        epoch (int): Epoch number of the model to load.

    Returns:
        An instance of a Keras model.
    """
    import keras.models

    from gated_conv import GatedConv

    fold_dir = str(fold) if fold >= 0 else 'all'
    model_path = glob.glob(os.path.join(cfg.model_path, fold_dir,
                                        '*.%.02d*.h5' % epoch))[0]

    custom_objects = {
        'GatedConv': GatedConv,
    }

    return keras.models.load_model(model_path, custom_objects)


if __name__ == '__main__':
    sys.exit(main())
