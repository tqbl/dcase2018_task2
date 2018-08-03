import os.path

from . import training


work_path = '/vol/vssp/msos/ti/dcase2018/task2'
"""str: Path to parent directory containing program output."""

preprocessing_path = os.path.join(work_path, 'split')
"""str: Path to the directory containing preprocessed audio files."""

extraction_path = os.path.join(work_path, 'features/logmel64')
"""str: Path to the directory containing extracted feature vectors."""

model_path = os.path.join(work_path, 'models', training.training_id)
"""str: Path to the output directory of saved models."""

log_path = os.path.join(work_path, 'logs', training.training_id, '{}')
"""str: Path to the directory of TensorBoard logs."""

history_path = os.path.join(log_path, 'history.csv')
"""str: Path to log file for training history."""

predictions_path = os.path.join(
    work_path, 'predictions', training.training_id, '{}_{}.csv')
"""str: Path to a model predictions file."""

results_path = os.path.join(
    work_path, 'results', training.training_id, '{}_results.csv')
"""str: Path to the file containing results."""
