import numpy as np
import time
from pathlib import Path

def get_valid_pairs(grid=25, min_dist=10, ordered=True):
    cache = Path(f"valid_pairs_{grid}_{min_dist}_{'ordered' if ordered else 'unordered'}.npz")

    if cache.exists():
        data = np.load(cache)
        return data["p1s"], data["p2s"]

    coords = np.array([(r, c) for r in range(grid) for c in range(grid)])
    dist = np.abs(coords[:, None, :] - coords[None, :, :]).sum(axis=2)
    mask = np.triu(dist >= min_dist, k=1)
    r_idx, c_idx = np.where(mask)
    p1s, p2s = coords[r_idx], coords[c_idx]

    if ordered:
        p1s = np.concatenate([p1s, p2s])
        p2s = np.concatenate([p2s, p1s[:len(p1s)//2]])

    # np.savez(cache, p1s=p1s, p2s=p2s)
    return p1s, p2s

def _main():
    p1s, p2s = get_valid_pairs(ordered=False)
    print(f"p1s: {p1s.shape}, p2s: {p2s.shape}")
    p1s, p2s = get_valid_pairs(ordered=True)
    print(f"p1s: {p1s.shape}, p2s: {p2s.shape}")

if __name__ == "__main__":
    _main()