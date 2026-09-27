import numpy as np
import time
import matplotlib.pyplot as plt
from pathlib import Path

CACHE = Path("valid_pairs.npz")
G = 25
rows, cols = np.mgrid[:G, :G].astype(np.float32)

def batch_smoke_maps(p1s, p2s, intensities1, intensities2, decay=0.1, noise_sigma=0.07):
    r = rows[None, :, :]
    c = cols[None, :, :]
    r1 = p1s[:, 0, None, None].astype(np.float32)
    c1 = p1s[:, 1, None, None].astype(np.float32)
    r2 = p2s[:, 0, None, None].astype(np.float32)
    c2 = p2s[:, 1, None, None].astype(np.float32)
    dist1 = np.abs(r - r1) + np.abs(c - c1)
    dist2 = np.abs(r - r2) + np.abs(c - c2)
    i1 = intensities1[:, None, None].astype(np.float32)
    i2 = intensities2[:, None, None].astype(np.float32)
    smoke1 = i1 * np.exp(-decay * dist1)
    smoke2 = i2 * np.exp(-decay * dist2)
    smoke = np.where(dist1 <= dist2, smoke1, smoke2)

    noise = np.random.normal(0, noise_sigma, smoke.shape).astype(np.float32)
    smoke = np.maximum(0, smoke + noise)

    return smoke

# ======================
# LOAD
# ======================
start = time.perf_counter()
data = np.load(CACHE)
p1s, p2s = data["p1s"], data["p2s"]
load_time = time.perf_counter() - start
print(f"Load cache       : {load_time*1000:.2f} ms  ({len(p1s):,} pairs available)")

# ======================
# WARM UP (untimed)
# ======================
_idx = np.random.randint(0, len(p1s), size=5)
_i1 = np.random.uniform(0.5, 1.5, size=5).astype(np.float32)
_i2 = np.random.uniform(0.5, 1.5, size=5).astype(np.float32)
_ = batch_smoke_maps(p1s[_idx], p2s[_idx], _i1, _i2)

# ======================
# SAMPLE
# ======================
N = 100

start = time.perf_counter()
idx = np.random.randint(0, len(p1s), size=N)
sp1 = p1s[idx]
sp2 = p2s[idx]
i1 = np.ones(N, dtype=np.float32)
i2 = np.full(N, 0.7, dtype=np.float32)
sample_time = time.perf_counter() - start
print(f"Sample {N} pairs  : {sample_time*1000:.2f} ms")

# ======================
# COMPUTE
# ======================
start = time.perf_counter()
maps = batch_smoke_maps(sp1, sp2, i1, i2, decay=0.1, noise_sigma=0.07)
compute_time = time.perf_counter() - start
print(f"Compute {N} maps  : {compute_time*1000:.2f} ms  (output shape: {maps.shape})")

# ======================
# SUMMARY
# ======================
total = load_time + sample_time + compute_time
print(f"\n--- total ---")
print(f"Load             : {load_time*1000:.2f} ms  ({load_time/total*100:.1f}%)")
print(f"Sample           : {sample_time*1000:.2f} ms  ({sample_time/total*100:.1f}%)")
print(f"Compute          : {compute_time*1000:.2f} ms  ({compute_time/total*100:.1f}%)")
print(f"Total            : {total*1000:.2f} ms")
print(f"Per map          : {total/N*1000:.4f} ms")

# ======================
# PLOT LAST 5 MAPS
# ======================
fig, axes = plt.subplots(1, 5, figsize=(20, 4))

for i, ax in enumerate(axes):
    map_idx = N - 5 + i
    ax.imshow(maps[map_idx], cmap='plasma', origin='upper',
              vmin=0, vmax=maps[map_idx].max())

    ax.text(sp1[map_idx, 1], sp1[map_idx, 0], 'X',
            color='red', fontsize=6 + i1[map_idx] * 10,
            ha='center', va='center', fontweight='bold')
    ax.text(sp2[map_idx, 1], sp2[map_idx, 0], 'X',
            color='blue', fontsize=6 + i2[map_idx] * 10,
            ha='center', va='center', fontweight='bold')

    ax.set_title(
        f"map {map_idx}\n"
        f"F1={sp1[map_idx].tolist()} i={i1[map_idx]:.2f}\n"
        f"F2={sp2[map_idx].tolist()} i={i2[map_idx]:.2f}",
        fontsize=8
    )
    ax.set_xticks(np.arange(-0.5, G, 5))
    ax.set_yticks(np.arange(-0.5, G, 5))
    ax.grid(which='major', linestyle='-', linewidth=0.8)
    ax.tick_params(labelbottom=False, labelleft=False)

plt.suptitle("Last 5 smoke maps  (red X = fire 1, blue X = fire 2, larger X = hotter)", fontsize=10)
plt.tight_layout()
plt.savefig("smoke_maps_check.png", dpi=150, bbox_inches='tight')
print("\nPlot saved to smoke_maps_check.png")