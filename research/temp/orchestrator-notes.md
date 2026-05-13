# Orchestrator notes — legged-rl-budgets

## Wave 1 results so far (batches 1, 3, 4 returned; batch 2 still running)

### Batch 1 — Anchor hexapod + Brax/MJX (returned)
- **Liu et al SJTU hexapod motion priors (arxiv 2511.03167)**: 4096 envs × 1000 steps × 50,000 episodes = ~204.8B step ceiling (early termination → effective lower); 35h on single RTX 3090Ti; IsaacGym; AMP prior = 8.6 sec of TO tripod gait
- **Versatile Hexapod Skills (arxiv 2412.10628)**: 10h teacher RL + 5h student distillation on RTX TITAN; total step count NOT reported; teacher-student pipeline
- **Brax/MJX Humanoid (arxiv 2407.05148)**: 200M steps, 8192 envs, 56 min on single RTX 4090 — directly comparable infra to user's stack
- **MuJoCo Playground (arxiv 2502.08844)**: per-task budgets (verbatim from appendix): Go1JoystickFlat/Rough 100M; Go1Handstand/Footstand 200M; Go1Getup 400M (32768 envs); G1Joystick 400M; T1Joystick 100M; Go1Backflip 50M. Wall-clock under 10-30 min on A100 or 2× RTX 4090. Throughput on A100: Go1 flat 417k SPS, Spot 408k SPS.
- **Bonus discovery: Rudin et al "Walk in Minutes" (arxiv 2109.11978)** — 4096 robots, batch 98304, 1500 policy updates, <20 min on i9-11900k + RTX A6000

### Batch 3 — AMP variants (returned)
- **Bipedal walking on Quadruped via AMP (2407.02282)**: 500 envs, 26000 iterations, 15.88h on RTX 4070; total env steps NOT explicitly stated
- **CAMP multi-skill AMP (2509.21810)**: 4096 envs, ~7h on RTX 4060Ti; total step count truncated in HTML; CAMP = Conditional AMP; 43-dim state, [1024,512] discriminator
- **Walk + Fly AMP (2309.12784)**: 4096 envs, ~2h on Quadro RTX 6000; PPO mini-batch 32768
- **Escontrela AMP (2203.15103)**: budget data not accessible (PDF only)
- **Discriminator update frequency NOT reported in any AMP paper** — consistent gap across the corpus

### Batch 4 — BC pretrain + sample-efficiency outliers (returned)
- **PostBC (2512.16911)**: manipulation-only, 4 envs, 2M gradient steps; NOT a locomotion paper
- **ResFiT residual off-policy (2509.19301)**: ~200× sample-eff claim CONFIRMED VERBATIM ("converging at 200k vs 40M steps") but narrowly scoped — manipulation, off-policy vs on-policy residual; NOT locomotion-from-scratch
- **Athletic Loco-Manipulation (2502.10894)**: 4096 envs, horizon=24/96, MIT Supercloud; total step count NOT reported; explicit 3-stage split (UAN sim-to-real → WBC pretrain → task fine-tune)
- **Systematic Sim-to-Real diverse legged (2509.06342)**: ETH PACE — 4096 envs × 24 steps × 30,000 iterations = **~2.95B env steps**; RTX 3080 for system ID
- **Curriculum Quadruped Jumping (2401.16337)**: best stage-decomposed budget — Stage I 3000 iter (~295M, 1.4h), Stage II 10000 iter (~983M, 4.1h), Stage III 10000 iter (~983M, 4.8h) → **~2.26B total, ~10.3h on single RTX 3090**

### Batch 2 — Foundational quadruped RL (still running async)

### Pre-final-report tentative finding

The "100M-500M per session in a multi-session curriculum" plan maps cleanly to published practice:
- 100M-200M = single Brax/MJX flat-ground budget
- 400M = harder tasks (getup, G1, handstand) in MuJoCo Playground
- Multi-session totals of 1-5B align with 4096-env IsaacGym norms (Rudin/PACE/Curriculum-Jumping = 0.15-2.95B per published run)
- ANYTHING in this range is on-trend; <50M would be under-trained, >10B would be excessive

The user's RTX 5060 Ti at 240k it/s effective is comparable to MuJoCo Playground's A100 throughput (~400k SPS) divided by ~2 — i.e., Brax/MJX wall-clock for 200M steps would be ~30-60 min on the 5060 Ti.
