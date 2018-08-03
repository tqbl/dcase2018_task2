import os.path

import librosa
import numpy as np

from pydub import AudioSegment
import pydub.silence as silence
from pydub.exceptions import CouldntDecodeError


def split_audio(dataset_path, name, output_path):
    """Split an audio clip into its non-silent parts.

    This function detects the non-silent sections of an audio clip based
    on the RMS energy of individual frames. If there is continuous
    silence longer than 500 ms, the non-silent frames on either side are
    considered to be from separate sections. These sections are saved to
    disk to be considered as separate clips.

    Args:
        dataset_path (str): Path of directory containing dataset.
        name (str): File name of audio clip to be split.
        output_path (str): Path of output directory.

    Returns:
        list: The output file names.
    """
    def _export_segments(segments):
        fnames = []
        for i, seg in enumerate(segments):
            fname = '{}_{}.wav'.format(os.path.splitext(name)[0], i)
            seg.export(os.path.join(output_path, fname), format='wav')
            fnames.append(fname)
        return fnames

    try:
        x = AudioSegment.from_wav(os.path.join(dataset_path, name))
    except CouldntDecodeError:
        x = AudioSegment.empty()

    # Skip audio samples that are shorter than 1 second
    if x.duration_seconds < 1.0:
        return _export_segments([x])

    # Determine silence threshold based on whether the audio signal
    # consists entirely of transients.
    x_array = x.get_array_of_samples()
    if _is_transients(x_array, x.frame_rate, 1024):
        threshold = -56
    else:
        threshold = -48

    segments = silence.split_on_silence(
        audio_segment=x,
        min_silence_len=500,
        silence_thresh=threshold,
        keep_silence=400,
    )

    # Export the original clip if no non-silent sections were found
    if len(segments) == 0:
        return _export_segments([x])

    # Discard sections that are unlikely to be important
    mean_time = np.mean([seg.duration_seconds for seg in segments])
    if mean_time > 1.5:
        segments = [seg for seg in segments
                    if seg.duration_seconds > 0.9]

    return _export_segments(segments)


def _is_transients(x, sample_rate, n_window=512):
    """Determine whether an audio signal contains transients only.

    Args:
        x (np.ndarray): Audio signal to analyze.
        sample_rate (number): Sampling rate of signal.
        n_window (int): Window size for computing the signal's envelope.

    Returns:
        bool: Whether the audio signal contains transients only.
    """
    envelope = _moving_average(np.abs(x), n=n_window)
    envelope = librosa.amplitude_to_db(envelope, ref=np.max)
    mask = (envelope > -30).astype(int)
    diff = np.diff(mask)
    start = np.where(diff == 1)[0]
    end = np.where(diff == -1)[0]

    if len(end) == 0:
        return True

    if mask[0] == 1:
        start = np.concatenate(([0], start))
    if len(start) > len(end):
        start = start[:-1]

    return max(end - start) / sample_rate < 0.5


def _moving_average(x, n=3):
    """Compute the moving average of a 1D array.

    Args:
        x (array_like): Input 1D array.
        n (int): Window size of moving average.

    Returns:
        np.ndarray: The averaged version of the array.
    """
    ret = np.cumsum(x, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
