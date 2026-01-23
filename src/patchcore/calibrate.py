import numpy as np
from ..osr.calibrate import calibrate_threshold


def calibrate_anomaly_threshold(scores: np.ndarray, accept_rate: float):
    return calibrate_threshold(scores, accept_rate)
