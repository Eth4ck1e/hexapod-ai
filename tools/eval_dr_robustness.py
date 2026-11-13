"""tools/eval_dr_robustness.py — how well do policies survive randomized dynamics?

Batched evaluation in the JAX env (GPU in WSL; CPU works for tiny sizes):
for each `--dr-scale`, N episodes with paper_stance cmds are rolled out under
per-episode dynamics randomization (envs.hexapod_env_jax.DR_RANGES_FULL ×
scale) for every policy AND for the pure scaffold (zero residual). The same
episode keys are used for every policy, so the cmd + dynamics draws are
paired across rows.

    PYTHONPATH=. python tools/eval_dr_robustness.py --model models/mini_hexapod_mjx.xml \
        --policy v1=checkpoints/amp_to_mini_hexapod_v1/iter10/final/params.pkl \
        --policy v1dr=checkpoints/amp_to_mini_hexapod_v1dr/iter10/final/params.pkl \
        --dr-scales 0 0.5 1.0 1.5 --envs 512 --steps 1000

Reports per (policy, dr_scale): fall rate, any-termination rate, mean
linear/angular tracking reward over alive steps, mean alive steps.
Deterministic actions (tanh of the policy mean, Brax's mode()).
"""
from __future__ import annotations

import argparse
import csv
import pickle
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))   # watch_demo_jax imports chain_train bare

import jax
import jax.numpy as jnp
import numpy as np

from envs import hexapod_env_jax as hx


def load_policy(path: str):
    with open(path, "rb") as f:
        loaded = pickle.load(f)
    normalizer, policy = loaded[0], loaded[1]
    obs_dim = int(jax.tree_util.tree_leaves(normalizer)[0].shape[-1])
    return normalizer, policy, obs_dim


def rollout_fn(params: hx.EnvParams, n_steps: int, infer):
    """infer(obs_batch) -> action_batch, or None for the pure scaffold."""
    def one(keys):
        st = jax.vmap(lambda k: hx.reset(params, k, gait_scale=0.0))(keys)
        n = keys.shape[0]

        def body(carry, _):
            st, alive = carry
            act = jnp.zeros((n, 18), jnp.float32) if infer is None else infer(st.obs)
            st = jax.vmap(lambda s, a: hx.step_foot(params, s, a))(st, act)
            out = dict(
                r_lin=st.metrics["r_lin_track"] * alive,
                r_ang=st.metrics["r_ang_track"] * alive,
                fell=jnp.logical_and(st.metrics["fell"], alive),
                done=jnp.logical_and(st.done, alive),
                alive=alive,
            )
            alive = jnp.logical_and(alive, jnp.logical_not(st.done))
            return (st, alive), out

        (st, _), outs = jax.lax.scan(body, (st, jnp.ones(n, bool)), None, length=n_steps)
        alive_steps = outs["alive"].sum(0).astype(jnp.float32)              # (n,)
        return dict(
            fall_rate=outs["fell"].any(0).mean(),
            done_rate=outs["done"].any(0).mean(),
            r_lin=(outs["r_lin"].sum(0) / jnp.maximum(alive_steps, 1)).mean(),
            r_ang=(outs["r_ang"].sum(0) / jnp.maximum(alive_steps, 1)).mean(),
            alive_steps=alive_steps.mean(),
            dr_scales=st.dr_scales,
        )
    return jax.jit(one)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="models/mini_hexapod_mjx.xml")
    ap.add_argument("--policy", action="append", default=[], metavar="NAME=params.pkl")
    ap.add_argument("--no-scaffold", action="store_true", help="skip the pure-scaffold row")
    ap.add_argument("--dr-scales", type=float, nargs="+", default=[0.0, 0.5, 1.0, 1.5])
    ap.add_argument("--envs", type=int, default=512)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--csv", type=Path, default=None)
    args = ap.parse_args()

    policies = []
    if not args.no_scaffold:
        policies.append(("scaffold", None))
    for spec in args.policy:
        name, _, path = spec.partition("=")
        if not path:
            name, path = Path(spec).parts[-4] if len(Path(spec).parts) >= 4 else spec, spec
        from watch_demo_jax import _build_inference_fn   # needs brax; lazy so scaffold-only runs anywhere
        norm, pol, obs_dim = load_policy(path)
        raw = _build_inference_fn(obs_dim, 18, deterministic=True)
        infer = (lambda norm, pol, raw: (lambda obs: jnp.tanh(raw(norm, pol, obs, None))))(norm, pol, raw)
        policies.append((name, infer))
        print(f"loaded {name}: {path} (obs {obs_dim})")

    keys = jax.random.split(jax.random.PRNGKey(args.seed), args.envs)
    rows = []
    print(f"\n{'policy':>10} {'dr':>5} {'fall%':>7} {'done%':>7} {'r_lin':>7} {'r_ang':>7} {'alive':>7}")
    for dr in args.dr_scales:
        params = hx.make_env_params(args.model, cmd_mask_name="paper_stance", dr_scale=dr)
        env_obs = int(hx.reset(params, keys[0]).obs.shape[-1])
        for name, infer in policies:
            t0 = time.time()
            res = rollout_fn(params, args.steps, infer)(keys)
            res = {k: np.asarray(v) for k, v in res.items()}
            row = dict(policy=name, dr_scale=dr, fall_rate=float(res["fall_rate"]),
                       done_rate=float(res["done_rate"]), r_lin=float(res["r_lin"]),
                       r_ang=float(res["r_ang"]), alive_steps=float(res["alive_steps"]))
            rows.append(row)
            print(f"{name:>10} {dr:5.2f} {100*row['fall_rate']:7.1f} {100*row['done_rate']:7.1f} "
                  f"{row['r_lin']:7.3f} {row['r_ang']:7.3f} {row['alive_steps']:7.0f}   ({time.time()-t0:.0f}s)")
        if dr > 0:
            sc = res["dr_scales"]
            print(f"           dr draws: " + "  ".join(f"{f}[{sc[:, i].min():.2f},{sc[:, i].max():.2f}]"
                                                  for i, f in enumerate(hx.DR_FACTORS)))
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"wrote {args.csv}")
    del env_obs


if __name__ == "__main__":
    main()
