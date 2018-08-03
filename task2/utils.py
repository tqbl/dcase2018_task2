import json
import time
import types

import numpy as np
import pandas as pd


LABELS = [
    'Acoustic_guitar',
    'Applause',
    'Bark',
    'Bass_drum',
    'Burping_or_eructation',
    'Bus',
    'Cello',
    'Chime',
    'Clarinet',
    'Computer_keyboard',
    'Cough',
    'Cowbell',
    'Double_bass',
    'Drawer_open_or_close',
    'Electric_piano',
    'Fart',
    'Finger_snapping',
    'Fireworks',
    'Flute',
    'Glockenspiel',
    'Gong',
    'Gunshot_or_gunfire',
    'Harmonica',
    'Hi-hat',
    'Keys_jangling',
    'Knock',
    'Laughter',
    'Meow',
    'Microwave_oven',
    'Oboe',
    'Saxophone',
    'Scissors',
    'Shatter',
    'Snare_drum',
    'Squeak',
    'Tambourine',
    'Tearing',
    'Telephone',
    'Trumpet',
    'Violin_or_fiddle',
    'Writing',
]


def to_categorical(y):
    """Encode labels as one-hot vectors.

    Args:
        y (pd.Series): Labels to be converted into categorical format.

    Returns:
        np.ndarray: Matrix of encoded labels.
    """
    return pd.get_dummies(y).values


def pad_truncate(x, length, pad_value=0):
    """Pad or truncate an array to a specified length.

    Args:
        x (array_like): Input array.
        length (int): Target length.
        pad_value (number): Padding value.

    Returns:
        array_like: The array padded/truncated to the specified length.
    """
    x_len = len(x)
    if x_len > length:
        x = x[:length]
    elif x_len < length:
        padding = np.full((length - x_len,) + x.shape[1:], pad_value)
        x = np.concatenate((x, padding))

    return x


def fit_scaler(x):
    """Fit an ImageDataGenerator to the given data.

    Args:
        x (np.ndarray): 4D array of data.

    Returns:
        keras.ImageDataGenerator: The fitted generator.
    """
    from keras.preprocessing.image import ImageDataGenerator

    generator = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
    )
    generator.fit(x)
    return generator


def group_by_name(data):
    """Group metadata entries based on original file names.

    Args:
        data (pd.Series or pd.DataFrame): The metadata to group.

    Returns:
        The relevant GroupBy object.
    """
    return data.groupby(lambda s: s[:8] + '.wav')


def timeit(callback, message):
    """Measure the time taken to execute the given callback.

    This function measures the amount of time it takes to execute the
    specified callback and prints a message afterwards regarding the
    time taken. The `message` parameter provides part of the message,
    e.g. if `message` is 'Executed', the printed message is 'Executed in
    1.234567 seconds'.

    Args:
        callback: Function to execute and time.
        message (str): Message to print after executing the callback.

    Returns:
        The return value of the callback.
    """
    # Record time prior to invoking callback
    onset = time.time()
    # Invoke callback function
    x = callback()

    print('%s in %f seconds' % (message, time.time() - onset))

    return x


def log_parameters(params, output_path):
    """Write the given parameters to a file in JSON format.

    Args:
        params (dict or module): Parameters to serialize. If `params` is
            a module, the relevant variables are serialized.
        output_path (str): Output file path.
    """
    if isinstance(params, types.ModuleType):
        params = {k: v for k, v in params.__dict__.items()
                  if not k.startswith('_')}
    elif not isinstance(params, dict):
        raise ValueError("'params' must be a dict or a module")

    with open(output_path, 'w') as f:
        json.dump(params, f, indent=2)
