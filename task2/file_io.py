import pandas as pd


def read_metadata(path):
    """Read from the specified metadata file.

    Args:
        path (str): Path of metadata file.

    Returns:
        pd.DataFrame: The parsed metadata.
    """
    return pd.read_csv(path, index_col=0)


def read_training_history(path, ordering=None):
    """Read training history from the specified CSV file.

    Args:
        path (str): Path of CSV file.
        ordering (str): Column name to order the entries with respect to
            or ``None`` if the entries should remain unordered.

    Returns:
        pd.DataFrame: The training history.
    """
    df = pd.read_csv(path, index_col=0)
    ascending = ordering not in ['val_acc', 'val_map']
    if ordering:
        df.sort_values(by=ordering, ascending=ascending, inplace=True)
    return df


def write_predictions(y_pred, output_path):
    """Write classification predictions to a CSV file.

    Args:
        y_pred (pd.DataFrame): Table of predictions.
        output_path (str): Output file path.
    """
    top_3 = y_pred.apply(lambda x: ' '.join(x.nlargest(3).index), axis=1)
    pd.Series(top_3, name='label').to_csv(output_path, header=True)
