"""tools/plot_training_log.py — eval-reward curves from AMP-PPO stdout logs.

Parses the `[EVAL ]` lines that scripts/train_jax_amp.py prints at the end
of every PPO segment (total reward + AMP style component vs env steps) and
plots one curve per log. Presentation-friendly defaults.

    python tools/plot_training_log.py logs/stdout/amp_mini_hexapod_v1.log \
        --label "mini v1 (nominal)" logs/stdout/amp_mini_hexapod_v1dr.log \
        --label "mini v1dr (dynamics DR)" --out media/plots/mini_training.png
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

EVAL_RE = re.compile(r"\[EVAL \]\s+step=\s*([\d,]+)\s+reward=\s*([+-]?[\d.]+|\+?nan)\s+style=\s*([+-]?[\d.]+|\+?nan)")


def parse(path: Path):
    steps, rew, sty = [], [], []
    for line in path.read_text(errors="replace").splitlines():
        m = EVAL_RE.search(line)
        if m:
            steps.append(int(m.group(1).replace(",", "")) / 1e6)
            rew.append(float(m.group(2)))
            sty.append(float(m.group(3)))
    return steps, rew, sty


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+", type=Path)
    ap.add_argument("--label", action="append", default=[], help="one per log, in order")
    ap.add_argument("--out", type=Path, default=None, help="PNG path (default: show)")
    ap.add_argument("--title", default="AMP-PPO eval reward per 50M-step segment")
    ap.add_argument("--max-reward", type=float, default=None,
                    help="draw the theoretical per-episode max (e.g. 2000 task + 500 style)")
    ap.add_argument("--step-offset", action="append", type=float, default=[],
                    help="shift a log's x axis by this many M steps (continuation runs); one per log")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg" if args.out else matplotlib.get_backend())
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    for i, log in enumerate(args.logs):
        steps, rew, sty = parse(log)
        if not steps:
            print(f"no [EVAL] lines in {log}"); continue
        off = args.step_offset[i] if i < len(args.step_offset) else 0.0
        x = [s + off for s in steps]
        label = args.label[i] if i < len(args.label) else log.stem
        ax1.plot(x, rew, marker="o", label=label)
        ax2.plot(x, sty, marker="o", label=label)
        print(f"{label}: {len(steps)} evals, reward first {rew[0]:+.0f} best {max(rew):+.0f} last {rew[-1]:+.0f}")
    if args.max_reward:
        ax1.axhline(args.max_reward, ls="--", c="gray", lw=1, label=f"max {args.max_reward:g}")
    ax1.set_ylabel("eval episode reward"); ax1.grid(alpha=0.3); ax1.legend()
    ax1.set_title(args.title)
    ax2.set_ylabel("AMP style reward"); ax2.set_xlabel("env steps (millions)"); ax2.grid(alpha=0.3)
    fig.tight_layout()
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=150)
        print(f"wrote {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
