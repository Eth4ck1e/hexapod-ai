# Scaffold — legged-rl-budgets

**Private orchestrator planning document. MUST NOT appear in the final report.**

## User Prompt (VERBATIM — gospel)

> Light tier. Survey training-budget standards in legged-locomotion reinforcement learning papers from 2023-2026, with priority weighting toward hexapod and quadruped work using PPO and/or AMP (Adversarial Motion Priors).
>
> For each paper: extract total environment steps trained, num_envs × iterations × rollout_length breakdown if reported, wall-clock training hours, hardware used (GPU model + count), and any reported sample-efficiency metrics. Note papers that report convergence at notably less or notably more than the median.
>
> Output should help me decide: for my own hexapod project running Brax PPO+AMP on a single RTX 5060 Ti at ~240k it/s effective throughput, what total-step budget is reasonable to target? Currently planning 100M-500M steps per session in a multi-session curriculum. Is that under-trained, on-trend, or over-trained vs published practice?
>
> Particular attention: distinguish "stage 0 / BC pretrain", "main RL run", and "fine-tune / DR" budgets, since those are often reported separately. The final report should include a comparison table and a clear recommendation for my budget question.

## Run config

- **vault_tag**: `legged-rl-budgets`
- **query_file_path**: `research/query-legged-rl-budgets.md`
- **modality**: `compare` (proportionate per-paper depth + a committed recommendation for the user's specific budget question)
- **wrapper requirements**: none — direct user prompt, no `prompt.txt`, no `wrapper_contract.json`. Final report at default path: `research/notes/final_report_legged-rl-budgets.md`.

## Modality classification rationale

User asks for:
1. A comparison table across multiple papers (suggests `collect`/`compare`)
2. A specific recommendation answering "is 100M-500M under-trained, on-trend, or over-trained?" (suggests `compare` with committed verdict)
3. Sample-efficiency outliers (papers above/below median) — comparative judgment

The recommendation aspect is the load-bearing output. Pure enumeration without a verdict would underserve the prompt. → `compare`.

## Tier rationale

**Confirmed `light` + `structured` + `wikilink` after Step 1.**

User explicitly stated "Light tier" in the prompt. Confirmed by step 1 because:
- Tightly scoped subdomain (legged-locomotion RL, 2023-2026)
- Structured-data extraction task with finite per-paper schema (8 fields per paper)
- Decision-bounded recommendation question — not a defended argumentative thesis
- Comparison table is the load-bearing output; argumentative density is unnecessary
- 9 sub-questions but all factually answerable from paper text

`response_format = structured` because the deliverable is breadth-first survey + table + a single committed recommendation, not a 5000-10000 word thesis. `citation_style = wikilink` because no wrapper_contract.json overrides the default.

## Wrapper requirements

None. No prompt.txt, no wrapper_contract.json. Output goes to `research/notes/final_report_legged-rl-budgets.md`.

## Domain context (for the orchestrator's awareness only — NOT for the final report)

User's project (from CLAUDE.md): PhantomX hexapod, Brax PPO + planned AMP integration, RTX 5060 Ti single-GPU at ~240k it/s effective. Currently 100M-500M steps per training session in a multi-session curriculum. Recently discovered the trained PPO policy reward-hacks via "hovering" (feet 7mm above ground, n_contact=0), which is why they're pivoting to AMP.

The user's anchor paper "Learning Natural and Robust Hexapod Locomotion" (Chen et al, SJTU) reports 50k iterations × 1000 steps × 4096 envs ≈ 200B steps on a 3090Ti for 35 hours. That's a known data point for the survey to contextualize.
