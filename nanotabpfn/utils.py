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

        ys_normalised = []
        num_tables_to_use = min(num_tables, max_y // num_datapoints)

        for i in range(num_tables_to_use):
            y_i = np.array(y[i, :], dtype=np.float32)
            y_i_mean = np.mean(y_i)
            y_i_std = np.std(y_i) + 1e-8
            y_i_normalised = (y_i - y_i_mean) / y_i_std
            ys_normalised.append(y_i_normalised)

    ys_concat = np.concatenate(ys_normalised, axis=0)

    if ys_concat.size < n_buckets:
        raise ValueError(f"Too few target samples ({ys_concat.size}) to compute {n_buckets} buckets.")

    ys_tensor = torch.tensor(ys_concat, dtype=torch.float32, device=device)
    global_bucket_edges = get_bucket_limits(n_buckets, ys=ys_tensor).to(device)
    return global_bucket_edges
