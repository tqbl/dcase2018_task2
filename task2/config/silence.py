n_window = 1024
"""int: Length of a frame used for silence detection."""

default_threshold = -48
"""int: Default threshold for silence."""

transients_threshold = -56
"""int: Threshold for transient audio signals."""

min_silence = 500
"""int: Minimum length of silence between two non-silent segments."""

keep_silence = 400
"""int: Amount of start/end silence to keep for each audio segment."""
