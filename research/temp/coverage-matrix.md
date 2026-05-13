## Coverage Matrix — query phrase → atomic item mapping

| Query phrase (verbatim) | Mapped atomic item(s) | Scope check | Gap? |
|---|---|---|---|
| "training-budget standards" | Sub-Q1, Sub-Q7 (stage decomposition); Required section #1, #2, #3 | OK — covered both holistically (total budgets) and stage-decomposed | No |
| "legged-locomotion reinforcement learning papers" | Entities: Hexapod RL papers, Quadruped RL papers; scope_conditions[0,1] | OK — full scope retained | No |
| "from 2023-2026" | time_horizons[0] | OK | No |
| "priority weighting toward hexapod and quadruped" | Entities (hexapod, quadruped); scope_conditions[2] | OK — priority documented | No |
| "PPO" | Entity: PPO algorithm | OK | No |
| "AMP (Adversarial Motion Priors)" | Entity: AMP algorithm | OK — acronym expanded inline as user did | No |
| "total environment steps trained" | Sub-Q1; Comparison table column | OK | No |
| "num_envs × iterations × rollout_length breakdown" | Sub-Q2; Comparison table column | OK | No |
| "wall-clock training hours" | Sub-Q3; Comparison table column | OK | No |
| "hardware used (GPU model + count)" | Sub-Q4; Required section #4 (Hardware Context); Comparison table column | OK | No |
| "sample-efficiency metrics" | Sub-Q5; Comparison table column | OK | No |
| "papers that report convergence at notably less or notably more than the median" | Sub-Q6; Required section #5 (Sample-Efficiency Outliers) | OK — explicitly reserved as own section | No |
| "stage 0 / BC pretrain" | Entity: BC pretrain (Stage 0); Required section #3 | OK | No |
| "main RL run" | Entity: Main RL run; Required section #3 | OK | No |
| "fine-tune / DR" | Entity: Fine-tune/DR; Required section #3 (DR = Domain Randomization, expanded) | OK | No |
| "Brax PPO+AMP on a single RTX 5060 Ti at ~240k it/s" | Sub-Q8; Required section #4 + #6 | OK — user's hardware context preserved | No |
| "100M-500M steps per session in a multi-session curriculum" | Sub-Q9; Required section #6 | OK | No |
| "under-trained, on-trend, or over-trained" | Sub-Q9; Required section #6 — committed recommendation | OK — explicit verdict required | No |
| "comparison table" | required_formats[0]; Required section #2 | OK | No |
| "clear recommendation" | required_formats[1]; Required section #6 | OK | No |
| "Light tier" | pipeline_tier="light" | OK — user explicit | No |

**Result: zero Gap? = YES rows. Decomposition covers every named phrase in the verbatim query at full scope.**
