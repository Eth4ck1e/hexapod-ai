---
vault_tag: legged-rl-budgets
created: 2026-05-06T07:24:00+00:00
source: user-prompt
---

Light tier. Survey training-budget standards in legged-locomotion reinforcement learning papers from 2023-2026, with priority weighting toward hexapod and quadruped work using PPO and/or AMP (Adversarial Motion Priors).

For each paper: extract total environment steps trained, num_envs × iterations × rollout_length breakdown if reported, wall-clock training hours, hardware used (GPU model + count), and any reported sample-efficiency metrics. Note papers that report convergence at notably less or notably more than the median.

Output should help me decide: for my own hexapod project running Brax PPO+AMP on a single RTX 5060 Ti at ~240k it/s effective throughput, what total-step budget is reasonable to target? Currently planning 100M-500M steps per session in a multi-session curriculum. Is that under-trained, on-trend, or over-trained vs published practice?

Particular attention: distinguish "stage 0 / BC pretrain", "main RL run", and "fine-tune / DR" budgets, since those are often reported separately. The final report should include a comparison table and a clear recommendation for my budget question.
