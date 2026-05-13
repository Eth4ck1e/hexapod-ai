"""Quick tfevents summarizer: pull scalar trajectories for the keys we care
about across segments. No tensorboard install required (pip-installs just
tensorboard for the EventAccumulator)."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

KEYS_OF_INTEREST = [
    "eval/episode_reward",
    "eval/episode_tracking_reward",
    "eval/episode_action_rate_pen",
    "eval/episode_z_vel_pen",
    "eval/episode_joint_torque_pen",
    "eval/episode_joint_vel_limit_pen",
    "eval/episode_joint_torque_limit_pen",
    "eval/episode_n_contact",
    "eval/episode_fell",
    "eval/episode_no_progress",
    "eval/episode_amp_style_reward",
    "eval/episode_foot_dev_total",
    "eval/episode_short_lifts",
    "eval/avg_episode_length",
    "disc/d_prior_mean",
    "disc/d_policy_mean",
    "disc/loss_final",
    "training/sps",
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logdir", type=str)
    ap.add_argument("--list-keys", action="store_true",
                    help="just list all scalar keys present")
    args = ap.parse_args()

    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("tensorboard not installed; pip install tensorboard")
        sys.exit(1)

    ea = EventAccumulator(args.logdir, size_guidance={"scalars": 0})
    ea.Reload()
    keys = ea.Tags().get("scalars", [])

    if args.list_keys:
        for k in sorted(keys):
            print(k)
        return

    print(f"Found {len(keys)} scalar tags. Pulling trajectories of interest:\n")
    for key in KEYS_OF_INTEREST:
        if key not in keys:
            # try fuzzy match
            matches = [k for k in keys if key.split("/")[-1] in k]
            if not matches:
                continue
            key_used = matches[0]
        else:
            key_used = key
        events = ea.Scalars(key_used)
        if not events:
            continue
        print(f"== {key_used} ==")
        # Print first, every-Nth, last to keep output manageable
        n = len(events)
        idxs = sorted(set(list(range(min(5, n))) +
                          [int(i*n/8) for i in range(8)] +
                          list(range(max(0, n-5), n))))
        for i in idxs:
            e = events[i]
            print(f"  step={e.step:>10}  val={e.value:+.4f}")
        print()

if __name__ == "__main__":
    main()
