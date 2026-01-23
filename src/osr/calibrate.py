import numpy as np


def calibrate_threshold(scores: np.ndarray, accept_rate: float):
    # Accept lowest scores (distance or negative confidence)
    k = int(np.ceil(len(scores) * accept_rate))
    sorted_scores = np.sort(scores)
    tau = sorted_scores[k - 1]
    return float(tau)
