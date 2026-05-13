# Search plan — legged-rl-budgets (light tier)

Light tier targets 15-25 curated sources via 1-2 fetcher waves. Academic APIs are normally skipped at light tier, but this query is *about* academic papers — searches must target arXiv / Semantic Scholar / paper PDFs, not generic web. Lens D (period-pinned) does not apply (no time_periods in decomposition).

## Search queries

| # | Atomic item | Search query | Type | Lens |
|---|---|---|---|---|
| 1 | Sub-Q1, Q2 (quadruped PPO budgets) | `Isaac Gym quadruped PPO training environment steps wall-clock` | web | breadth |
| 2 | Sub-Q1, Q2 (hexapod PPO+AMP) | `hexapod PPO adversarial motion priors training iterations` | web | breadth |
| 3 | Entity: ANYmal | `ANYmal reinforcement learning training time PPO env steps` | web | breadth |
| 4 | Entity: ANYmal | `Hwangbo ANYmal Science Robotics training duration sim2real` | web | depth |
| 5 | Entity: PPO | `quadruped locomotion num_envs rollout_length PPO Isaac` | web | depth |
| 6 | Entity: AMP | `AMP adversarial motion priors quadruped training cost discriminator` | web | depth |
| 7 | Sub-Q4 (single-GPU) | `Brax MJX legged locomotion RTX single GPU training steps` | web | breadth |
| 8 | Entity: Brax | `Brax PPO quadruped 5060 4090 3090 training time` | web | breadth |
| 9 | Sub-Q7 (BC pretrain) | `behavioral cloning pretrain locomotion PPO scaffold steps` | web | breadth |
| 10 | Sub-Q7 (DR fine-tune) | `domain randomization legged locomotion fine-tune budget steps` | web | breadth |
| 11 | Sub-Q6 (efficient outliers) | `sample efficient PPO quadruped less than 100 million steps` | web | breadth |
| 12 | Sub-Q6 (heavy outliers) | `quadruped curriculum locomotion 10 billion env steps PPO` | web | breadth |
| 13 | Adversarial — sample efficiency claims | `PPO quadruped sample efficiency overstated criticism` | web | adversarial |
| 14 | Adversarial — large-budget norms | `legged RL training budget excessive compute critique` | web | adversarial |
| 15 | Hexapod-specific 2024-2026 | `arxiv hexapod locomotion PPO 2024 2025 training` | web | depth |
| 16 | Hexapod-specific (PhantomX, Bittle) | `PhantomX Bittle reinforcement learning training time arxiv` | web | breadth |
| 17 | Generic legged 2025-2026 | `arxiv quadruped reinforcement learning 2025 budget compute` | web | depth |
| 18 | AMP follow-on | `AMP legged locomotion follow-up extension 2024 2025` | web | depth |
| 19 | Quadruped Sim-to-Real | `RMA rapid motor adaptation quadruped training steps` | web | depth |
| 20 | Massively-parallel benchmark | `IsaacGym 4096 envs quadruped PPO benchmark training steps` | web | depth |

## Coverage check vs coverage-matrix

Every coverage-matrix row maps to ≥1 query above:
- "training-budget standards", "total environment steps", "num_envs × iterations × rollout_length", "wall-clock training hours" → queries 1, 2, 3, 5, 7, 8 (multiple)
- "hardware (GPU model + count)" → queries 7, 8, 20
- "sample-efficiency metrics", "convergence at notably less / more than median" → queries 11, 12, 13
- "stage 0 / BC pretrain" → query 9
- "main RL run" → queries 1-8
- "fine-tune / DR" → query 10
- "hexapod" priority → queries 2, 15, 16
- "quadruped" priority → queries 1, 3, 4, 5, 12, 17, 19, 20
- "PPO" → most queries
- "AMP" → queries 6, 18; cross-referenced in 2

Adversarial queries: #13, #14 (≥2; under the standard 5-minimum but appropriate for light tier on a non-contested topic).

## Plan

1. Execute the 20 searches via WebSearch in parallel batches.
2. Deduplicate; aim for 30-45 candidate URLs.
3. Spawn 4 hyperresearch-fetcher subagents in parallel, 6-9 URLs each, non-overlapping.
4. Coverage check; spawn 2 wave-2 fetchers if gaps.
5. Cap at ~25 substantive vault notes per the light tier ceiling.
