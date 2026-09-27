from __future__ import annotations

import math
from typing import Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Signal / smoke helpers
# ---------------------------------------------------------------------------

def create_signal_map(grid_size: int, signal_cells: Sequence[tuple[int, int]]) -> np.ndarray:
    """Binary signal map [grid_size, grid_size] float32 in {0, 1}."""
    signal_map = np.zeros((grid_size, grid_size), dtype=np.float32)
    for r, c in signal_cells:
        signal_map[int(r), int(c)] = 1.0
    return signal_map


def compute_smoke_map(signal_map: np.ndarray, decay: float = 0.3) -> np.ndarray:
    """Single-signal smoke: exp(-decay * min_manhattan_dist_to_nearest_signal_cell)."""
    H, W = signal_map.shape
    signal_cells = np.argwhere(signal_map >= 0.5)
    if len(signal_cells) == 0:
        return np.zeros((H, W), dtype=np.float32)
    rr = np.arange(H, dtype=np.float32)[:, None, None]
    cc = np.arange(W, dtype=np.float32)[None, :, None]
    fr = signal_cells[:, 0].astype(np.float32)[None, None, :]
    fc = signal_cells[:, 1].astype(np.float32)[None, None, :]
    d = np.abs(rr - fr) + np.abs(cc - fc)
    d_min = d.min(axis=-1)
    return np.exp(-decay * d_min).astype(np.float32)


def compute_smoke_map_two(
    signal_rc1: tuple[int, int] | np.ndarray,
    signal_rc2: tuple[int, int] | np.ndarray,
    grid_size: int,
    intensity1: float = 1.0,
    intensity2: float = 1.0,
    decay: float = 0.3,
) -> np.ndarray:
    """Voronoi two-source smoke: each cell takes smoke from its nearer source.

    smoke(r,c) = intensity_k * exp(-decay * dist_k)  where k = argmin_k(dist_k).
    """
    G = grid_size
    rr = np.arange(G, dtype=np.float32)[:, None]
    cc = np.arange(G, dtype=np.float32)[None, :]
    r1, c1 = float(signal_rc1[0]), float(signal_rc1[1])
    r2, c2 = float(signal_rc2[0]), float(signal_rc2[1])
    dist1 = np.abs(rr - r1) + np.abs(cc - c1)
    dist2 = np.abs(rr - r2) + np.abs(cc - c2)
    smoke1 = float(intensity1) * np.exp(-decay * dist1)
    smoke2 = float(intensity2) * np.exp(-decay * dist2)
    return np.where(dist1 <= dist2, smoke1, smoke2).astype(np.float32)


# ---------------------------------------------------------------------------
# Area / view-mask helpers
# ---------------------------------------------------------------------------

def get_local_mask(center: tuple[int, int], radius: int, grid_size: int) -> np.ndarray:
    """L1 (Manhattan) ball around center; shape [grid_size, grid_size] bool."""
    mask = np.zeros((grid_size, grid_size), dtype=bool)
    r0, c0 = int(center[0]), int(center[1])
    rad = int(radius)
    for r in range(grid_size):
        for c in range(grid_size):
            if abs(r - r0) + abs(c - c0) <= rad:
                mask[r, c] = True
    return mask


def get_area_tile_mask(area_row: int, area_col: int, n: int, grid_size: int) -> np.ndarray:

    mask = np.zeros((grid_size, grid_size), dtype=bool)
    r0, c0 = int(area_row) * n, int(area_col) * n
    mask[r0 : r0 + n, c0 : c0 + n] = True
    return mask


def _view_mask_for_area(
    area_r: int,
    area_c: int,
    *,
    view_side: int,
    grid_size: int,
    use_l1_view_mask: bool,
    l1_view_radius: int,
) -> np.ndarray:
    """View footprint for a single area; matches RobotExploration2Env._area_view_masks."""
    n = int(view_side)
    if use_l1_view_mask:
        tr = int(area_r) * n
        tc = int(area_c) * n
        return get_local_mask((tr, tc), l1_view_radius, grid_size)
    return get_area_tile_mask(area_r, area_c, n, grid_size)


# ---------------------------------------------------------------------------
# Start-position helper
# ---------------------------------------------------------------------------

def generate_fixed_start_positions(M: int, num_areas_per_side: int) -> list[tuple[int, int]]:
    """M default starting area indices spread across a K*K area grid (K = num_areas_per_side)."""
    if M <= 0:
        raise ValueError("M must be positive.")
    K = int(num_areas_per_side)
    if K < 1:
        raise ValueError("num_areas_per_side must be >= 1.")
    low, high = 0, K - 1
    center = (low + high) // 2
    if M == 1:
        return [(center, center)]
    if M == 2:
        return [(low, low), (high, high)]
    if M == 3:
        return [(low, low), (low, high), (high, center)]
    if M == 4:
        return [(low, low), (low, high), (high, low), (high, high)]
    k_grid = math.ceil(math.sqrt(M))
    xs = np.linspace(low, high, num=k_grid, dtype=int)
    ys = np.linspace(low, high, num=k_grid, dtype=int)
    seen: set[tuple[int, int]] = set()
    unique: list[tuple[int, int]] = []
    for x in xs:
        for y in ys:
            p = (int(x), int(y))
            if p not in seen:
                seen.add(p)
                unique.append(p)
    if len(unique) < M:
        raise ValueError(
            f"Could not generate {M} unique start areas for K={K}."
        )
    return unique[:M]


# ---------------------------------------------------------------------------
# Valid signal-pair precomputation
# ---------------------------------------------------------------------------

def precompute_valid_signal_pairs(
    grid_size: int, min_signal_dist: int
) -> tuple[np.ndarray, np.ndarray]:
    """All ordered pairs of cell coordinates with Manhattan distance >= min_signal_dist.

    Returns (signal_locs, distractor_locs) each shape [N, 2] (row, col).
    Both orderings are included so ``rng.integers(0, N)`` gives an unbiased
    random pair without extra shuffling.
    """
    coords = np.array(
        [(r, c) for r in range(grid_size) for c in range(grid_size)], dtype=np.int32
    )
    dist = np.abs(coords[:, None, :] - coords[None, :, :]).sum(axis=2)
    mask = np.triu(dist >= min_signal_dist, k=1)
    r_idx, c_idx = np.where(mask)
    p1s_half = coords[r_idx]   # [M, 2]
    p2s_half = coords[c_idx]   # [M, 2]
    # Include both orderings so any draw from [0, 2M) is uniform over ordered pairs
    signal_locs = np.concatenate([p1s_half, p2s_half], axis=0)
    distractor_locs = np.concatenate([p2s_half, p1s_half], axis=0)
    return signal_locs, distractor_locs
