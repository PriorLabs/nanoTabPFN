import h5py
import random
import torch
import numpy as np

from pfns.bar_distribution import get_bucket_limits

def set_randomness_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_default_device():
    device = 'cpu'
    if torch.backends.mps.is_available(): device = 'mps'
    if torch.cuda.is_available(): device = 'cuda'
    return device

def make_global_bucket_edges(filename, n_buckets=100, device=get_default_device(), max_y=5_000_000):
    with h5py.File(filename, "r") as f:
        y = f["y"]
        num_tables, num_datapoints = y.shape
        total = num_tables * num_datapoints

        if max_y >= total:
            ys_concat = y[...].reshape(-1)
        else:
            full_rows = max_y // num_datapoints
            rem =  max_y % num_datapoints

            parts = []
            if full_rows > 0:
                parts.append(y[:full_rows, :].reshape(-1))
            if rem > 0:
                parts.append(y[full_rows, :rem].reshape(-1))
            ys_concat = np.concatenate(parts, axis=0) if parts else np.empty((0,), dtype=y.dtype)

    if ys_concat.size < n_buckets:
        raise ValueError(f"Too few target samples ({ys_concat.size}) to compute {n_buckets} buckets.")

    ys_tensor = torch.tensor(ys_concat, dtype=torch.float32, device=device)
    global_bucket_edges = get_bucket_limits(n_buckets, ys=ys_tensor).to(device)
    return global_bucket_edges
