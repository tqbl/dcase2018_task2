import os.path

import librosa
import numpy as np

from pydub import AudioSegment
import pydub.silence as silence
from pydub.exceptions import CouldntDecodeError


def split_audio(dataset_path,
                file_name,
                output_path,
                n_window=1024,
                default_threshold=-56,
                transients_threshold=-56,
                min_silence=500,
                keep_silence=500,
                ):
    """Split an audio clip into non-silent segments.

    This function detects the non-silent segments of an audio clip and
    saves them separately as WAV files in the specified directory.
    Silence is detected on a frame-by-frame basis by thresholding the
    RMS energy of each frame. A non-silent segment is defined to be the
    span of non-silent frames such that two such adjacent frames are
    less than `min_silence` ms apart. `keep_silence` ms of silence is
    also kept at the beginning and end of each segment.

    Args:
        dataset_path (str): Path of directory containing dataset.
        file_name (str): File name of audio clip to be split.
        output_path (str): Path of output directory.
        n_window (int): Number of samples in a frame.
        default_threshold (int): Default silence threshold (in dBFS).
        transients_threshold (int): Silence threshold for transient
            audio signals (in dBFS).
        min_silence (int): Minimum length of silence between segments.
        keep_silence (int): Amound of start/end silence to keep (in ms).

    Returns:
        list: The output file names.
    """
    def _export_segments(segments):
        fnames = []
        for i, seg in enumerate(segments):
            fname = '{}_{}.wav'.format(os.path.splitext(file_name)[0], i)
            seg.export(os.path.join(output_path, fname), format='wav')
            fnames.append(fname)
        return fnames

    try:
        x = AudioSegment.from_wav(os.path.join(dataset_path, file_name))
    except CouldntDecodeError:
        x = AudioSegment.empty()

    # Skip audio clips that are not longer than the padding
    # Padding refers to the silence that is kept for each segment
    padding = keep_silence * 2
    if x.duration_seconds <= padding / 1000:
        return _export_segments([x])

    # Determine silence threshold based on whether the audio signal
    # consists entirely of transients.
    if _is_transients(x.get_array_of_samples(), x.frame_rate, n_window):
        threshold = transients_threshold
    else:
        threshold = default_threshold

    segments = silence.split_on_silence(
        audio_segment=x,
        min_silence_len=min_silence,
        silence_thresh=threshold,
        keep_silence=keep_silence,
    )

    # Export the original clip if no non-silent segments were found
    if len(segments) == 0:
        return _export_segments([x])

    # Discard segments that are too short
    mean_time = np.mean([seg.duration_seconds for seg in segments])
    discard_threshold = 100 + padding
    if mean_time > discard_threshold + 500:
        segments = [seg for seg in segments
                    if seg.duration_seconds > discard_threshold]

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
