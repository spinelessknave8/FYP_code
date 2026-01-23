import numpy as np


def classify_anomaly(scores: np.ndarray, threshold: float):
    # return boolean: True if anomalous
    return scores > threshold
