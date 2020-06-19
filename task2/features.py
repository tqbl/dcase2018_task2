import os.path
import datetime as dt

import h5py
import librosa
import numpy as np
from tqdm import tqdm

import utils


def extract_dataset(dataset_path,
                    file_names,
                    extractor,
                    output_path,
                    recompute=False,
                    ):
    """Extract features from the audio clips in a dataset.

    Args:
        dataset_path (str): Path of directory containing dataset.
        file_names (array_like): List of file names for the audio clips.
        extractor: Class instance for feature extraction.
        output_path (str): File path of output HDF5 file.
        recompute (bool): Whether to extract features that already exist
            in the HDF5 file.
    """
    # Create/load the HDF5 file to store the feature vectors
    with h5py.File(output_path, 'a') as f:
        size = len(file_names)  # Size of dataset

        # Create/load feature vector dataset and timestamp dataset
        feats = f.require_dataset('F', (size,),
                                  dtype=h5py.special_dtype(vlen=float))
        timestamps = f.require_dataset('timestamps', (size,),
                                       dtype=h5py.special_dtype(vlen=bytes))

        # Record shape of reference feature vector. Used to infer the
        # original shape of a vector prior to flattening.
        feats.attrs['shape'] = extractor.output_shape(1)[1:]

        for i, name in enumerate(tqdm(file_names)):
            # Skip if existing feature vector should not be recomputed
            if timestamps[i] and not recompute:
                continue

            path = os.path.join(dataset_path, name)
            x, sample_rate = librosa.load(path, sr=None)
            if sample_rate is None:
                print('Warning: Skipping {}'.format(name))
                continue

            # Extract and save to dataset as flattened array
            feats[i] = extractor.extract(x, sample_rate).flatten()
            # Record timestamp in ISO format
            timestamps[i] = dt.datetime.now().isoformat()


def load_features(path, chunk_size=128, r_threshold=32):
    """Load feature vectors from the specified HDF5 file.

    Since the original feature vectors are of variable length, this
    function partitions them into chunks of length `chunk_size`. When
    they cannot be partitioned exactly, one of three things can happen:

      * If the length of the vector is less than the chunk size, the
        vector is simply padded with a fill value.
      * If the remainder, ``r``, is less than ``r_threshold``, the edges
        of the vector are truncated so that it can be partitioned.
      * If the remainder, ``r``, is greater than ``r_threshold``, the
        last chunk is the last `chunk_size` frames of the feature vector
        such that it overlaps with the penultimate chunk.

    Args:
        path (str): Path to the HDF5 file.
        chunk_size (int): Size of a chunk.
        r_threshold (int): Threshold for ``r`` (see above).

    Returns:
        np.ndarray: Array of feature vectors.
        list: Number of chunks for each audio clip.
    """
    chunks = []
    n_chunks = []
    with h5py.File(path, 'r') as f:
        feats = f['F']
        shape = feats.attrs['shape']
        for i, feat in enumerate(tqdm(feats)):
            # Reshape flat array to original shape
            feat = np.reshape(feat, (-1, *shape))

            if len(feat) == 0:
                n_chunks.append(0)
                continue

            # Split feature vector into chunks along time axis
            q = len(feat) // chunk_size
            r = len(feat) % chunk_size
            if not q and r:
                split = [utils.pad_truncate(feat, chunk_size,
                                            pad_value=np.min(feat))]
            elif r:
                r = len(feat) % chunk_size
                off = r // 2 if r < r_threshold else 0
                split = np.split(feat[off:q * chunk_size + off], q)
                if r >= r_threshold:
                    split.append(feat[-chunk_size:])
            else:
                split = np.split(feat, q)

            n_chunks.append(len(split))
            chunks += split

    return np.array(chunks), n_chunks


class LogmelExtractor(object):
    """Feature extractor for logmel representations.

    A logmel feature vector is a spectrogram representation that has
    been scaled using a Mel filterbank and a log nonlinearity.

    Args:
        sample_rate (number): Target resampling rate.
        n_window (int): Number of bins in each spectrogram frame.
        hop_length (int): Number of samples between frames.
        n_mels (int): Number of Mel bands.

    Attributes:
        sample_rate (number): Target resampling rate.
        n_window (int): Number of bins in each spectrogram frame.
        hop_length (int): Number of samples between frames.
        mel_fb (np.ndarray): Mel fitlerbank matrix.
    """

    def __init__(self,
                 sample_rate=16000,
                 n_window=1024,
                 hop_length=512,
                 n_mels=64,
                 ):
        self.sample_rate = sample_rate
        self.n_window = n_window
        self.hop_length = hop_length

        # Create Mel filterbank matrix
        self.mel_fb = librosa.filters.mel(sr=sample_rate,
                                          n_fft=n_window,
                                          n_mels=n_mels,
                                          )

    def output_shape(self, clip_duration):
        """Determine the shape of a logmel feature vector.

        Args:
            clip_duration (number): Duration of the input time-series
                signal given in seconds.

        Returns:
            tuple: The shape of a logmel feature vector.
        """
        n_samples = clip_duration * self.sample_rate
        n_frames = n_samples // self.hop_length + 1
        return (n_frames, self.mel_fb.shape[0])

    def extract(self, x, sample_rate):
        """Transform the given signal into a logmel feature vector.

        Args:
            x (np.ndarray): Input time-series signal.
            sample_rate (number): Sampling rate of signal.

        Returns:
            np.ndarray: The logmel feature vector.
        """
        # Resample to target sampling rate
        x = librosa.resample(x, sample_rate, self.sample_rate)

        # Compute short-time Fourier transform
        D = librosa.stft(x, n_fft=self.n_window, hop_length=self.hop_length)
        # Transform to Mel frequency scale
        S = np.dot(self.mel_fb, np.abs(D) ** 2).T
        # Apply log nonlinearity and return as float32
        return librosa.power_to_db(S, ref=np.max, top_db=None)
