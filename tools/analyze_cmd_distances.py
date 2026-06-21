"""tools/analyze_cmd_distances.py — diagnostic: how distinguishable are
commands in the prior dataset's cmd space?

Loads checkpoints/amp_priors_cmdcond.npz, picks reference cmd queries
the user cares about (e.g., "straight forward" vs "straight + barely
turning"), and reports the K-nearest-neighbor structure under several
distance metrics. Helps decide:

  - Whether per-dim normalization is needed (it is — dims have wildly
    different scales)
  - What K (neighborhood size) to use for cmd-matched prior sampling
  - Whether pathological overlaps exist (e.g., distinct intents that
    end up in each other's neighborhoods)

Run from project root:
    PYTHONPATH=. .venv/Scripts/python.exe tools/analyze_cmd_distances.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

PRIOR_PATH = Path("checkpoints/amp_priors_cmdcond.npz")

# These are the cmd_sample_ranges from envs/hexapod_env_jax.py:284 (paper_stance).
# vx/vy are stored as [-1.0, 1.0] placeholders in the env but the actual
# sampler narrows them — see the data-driven range below.
CMD_NAMES = ["vx", "vy", "wz", "pitch", "roll", "dh", "dw", "sx", "sy"]


def fmt_cmd(c: np.ndarray) -> str:
    """One-line readable cmd vector."""
    parts = []
    for i, name in enumerate(CMD_NAMES):
        if abs(c[i]) > 1e-4:
            parts.append(f"{name}={c[i]:+.3f}")
    return "(" + ", ".join(parts) + ")" if parts else "(zero)"


def per_dim_scale(cmds: np.ndarray) -> np.ndarray:
    """Returns the half-range of each cmd dim observed in the dataset.
    Used to per-dim-normalize before Euclidean distance, so each dim
    contributes proportionally rather than being dominated by the
    largest-scale dim."""
    half = (cmds.max(axis=0) - cmds.min(axis=0)) / 2.0
    half = np.where(half < 1e-6, 1.0, half)   # avoid div-by-zero on zero dims
    return half


def knn_query(cmds: np.ndarray, query: np.ndarray, k: int,
              metric: str = "norm") -> tuple[np.ndarray, np.ndarray]:
    """Returns (sorted_indices, sorted_distances) for the K nearest
    cmds in the dataset to `query` under the given metric.

    metric:
      "raw"  — plain Euclidean (biased toward large-scale dims)
      "norm" — per-dim-normalized Euclidean (recommended)
    """
    if metric == "raw":
        d = np.linalg.norm(cmds - query[None, :], axis=-1)
    elif metric == "norm":
        scale = per_dim_scale(cmds)
        d = np.linalg.norm((cmds - query[None, :]) / scale, axis=-1)
    else:
        raise ValueError(f"unknown metric {metric!r}")
    idx = np.argpartition(d, k)[:k]
    idx = idx[np.argsort(d[idx])]
    return idx, d[idx]


def report_query(cmds: np.ndarray, label: str, query: np.ndarray, k: int = 10):
    """Print a query + its K-NN under raw and normalized metrics."""
    print(f"\n{'='*70}\nQuery: {label}")
    print(f"  query cmd: {fmt_cmd(query)}")
    for metric in ["raw", "norm"]:
        idx, dist = knn_query(cmds, query, k, metric=metric)
        print(f"\n  Top-{k} nearest priors by {metric.upper()} distance:")
        for j in range(k):
            tag = "  <-- CLOSEST" if j == 0 else ""
            print(f"    rank{j+1:>2d}  d={dist[j]:6.3f}  cmd={fmt_cmd(cmds[idx[j]])}{tag}")


def main():
    if not PRIOR_PATH.exists():
        raise SystemExit(f"missing {PRIOR_PATH} — generate priors first.")
    print(f"loading {PRIOR_PATH} ...")
    npz = np.load(PRIOR_PATH)
    cmds = npz["cmds_t"].astype(np.float32)        # (N, 9)
    print(f"  N transitions = {cmds.shape[0]:,}")
    print(f"  cmd dim       = {cmds.shape[1]}")

    # Per-dim summary
    print("\nDataset cmd distribution (per dim):")
    print(f"  {'dim':<8}{'min':>8}{'max':>8}{'mean':>8}{'std':>8}")
    for i, name in enumerate(CMD_NAMES):
        c = cmds[:, i]
        print(f"  {name:<8}{c.min():+8.3f}{c.max():+8.3f}{c.mean():+8.3f}{c.std():8.3f}")

    print("\nPer-dim normalization scale (half-range):")
    s = per_dim_scale(cmds)
    for i, name in enumerate(CMD_NAMES):
        print(f"  {name:<8}{s[i]:.3f}")

    # The user's specific test cases.
    queries = {
        "Straight forward, neutral stance":
            np.array([0.17, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
        "Straight forward, BARELY turning (wz=0.1)":
            np.array([0.17, 0, 0.1, 0, 0, 0, 0, 0, 0], dtype=np.float32),
        "Straight forward, MORE turning (wz=0.5)":
            np.array([0.17, 0, 0.5, 0, 0, 0, 0, 0, 0], dtype=np.float32),
        "Straight forward, raised body (dh=-0.020)":
            np.array([0.17, 0, 0, 0, 0, -0.020, 0, 0, 0], dtype=np.float32),
        "Straight forward, wider stance (dw=+0.015)":
            np.array([0.17, 0, 0, 0, 0, 0, +0.015, 0, 0], dtype=np.float32),
        "Spin in place (vx=0, wz=1.0)":
            np.array([0, 0, 1.0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
        "Strafe left (vy=0.15)":
            np.array([0, 0.15, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
        "Stand still (zero cmd)":
            np.zeros(9, dtype=np.float32),
    }

    for label, q in queries.items():
        report_query(cmds, label, q, k=8)

    # Cross-test: how close are the "barely turning" priors to "straight" priors?
    # If the K=10 neighbors of "straight forward" include "barely turning"
    # cases, then K=10 doesn't actually enforce filtering at that resolution.
    print(f"\n{'='*70}")
    print("CROSS-TEST: do 'straight forward' and 'barely turning' overlap?")
    q_straight = queries["Straight forward, neutral stance"]
    q_barely   = queries["Straight forward, BARELY turning (wz=0.1)"]
    for k in [5, 20, 100, 500]:
        idx_s, _ = knn_query(cmds, q_straight, k, "norm")
        idx_b, _ = knn_query(cmds, q_barely,   k, "norm")
        overlap = len(set(idx_s) & set(idx_b))
        print(f"  K={k:<4d}  shared neighbors between 'straight' & 'barely turning': "
              f"{overlap}/{k}  ({100*overlap/k:.0f}%)")

    print(f"\n{'='*70}")
    print("CROSS-TEST: do 'straight forward neutral' and 'straight raised' overlap?")
    q_neutral = queries["Straight forward, neutral stance"]
    q_raised  = queries["Straight forward, raised body (dh=-0.020)"]
    for k in [5, 20, 100, 500]:
        idx_n, _ = knn_query(cmds, q_neutral, k, "norm")
        idx_r, _ = knn_query(cmds, q_raised,  k, "norm")
        overlap = len(set(idx_n) & set(idx_r))
        print(f"  K={k:<4d}  shared neighbors between 'neutral' & 'raised': "
              f"{overlap}/{k}  ({100*overlap/k:.0f}%)")

    print("\nDone.")


if __name__ == "__main__":
    main()
