prediction_epochs = 'val_map'
"""Specification for which models (epochs) to select for prediction.

Either a list of epoch numbers or a string specifying the metric to be
used to select the top epochs.
"""

threshold = -1
"""number: Number for thresholding audio tagging predictions.

A value of -1 indicates that the most probable label should be selected
instead of selecting labels that surpass a certain threshold.
"""
