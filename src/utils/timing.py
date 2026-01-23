import time
import numpy as np


def measure_time(fn, num_warmup: int, num_runs: int) -> dict:
    for _ in range(num_warmup):
        fn()
    times = []
    for _ in range(num_runs):
        t0 = time.time()
        fn()
        t1 = time.time()
        times.append((t1 - t0) * 1000.0)
    return {"mean_ms": float(np.mean(times)), "std_ms": float(np.std(times))}
