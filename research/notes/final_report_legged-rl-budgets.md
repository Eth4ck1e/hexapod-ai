# Legged-Locomotion RL Training Budgets, 2023-2026 — Survey & Recommendation

## 1. Overview — Legged-Locomotion RL Training Budgets, 2023-2026

Across roughly two dozen 2019-2026 legged-locomotion RL papers, "training budget" splits along three orthogonal axes: **total environment steps**, **wall-clock time on a stated GPU**, and **whether budgets are decomposed by stage** (BC pretrain / main RL / DR or fine-tune). Reporting practice is uneven. Most papers report two of those three axes, very few report all three, and several modern AMP papers omit total step counts entirely while quoting wall-clock and `num_envs` only.

A few cross-cutting observations frame the rest of this report:

- **The dominant template is `num_envs ≈ 4096`, `unroll_length ≈ 24`, `iterations ≈ 1500-30000`**, popularized by Rudin et al. "Walk in Minutes" [[210911978-learning-to-walk-in-minutes-using-massively-parallel-deep-reinforcemen-2]] and codified in the `leggedrobotics/legged_gym` config defaults [[legged_gymlegged_gymenvsbaselegged_robot_configpy-at-master-leggedroboticslegged]] (`num_envs=4096`, `num_steps_per_env=24`, `max_iterations=1500` → ~147M steps as the canonical baseline).
- **Single-task locomotion converges around 100-200M steps**; recovery-from-failure tasks (Go1Getup, G1Joystick) and longer curricula push to 400M-3B [[mujoco-playground]] [[curriculum-based-reinforcement-learning-for-quadrupedal-jumping-a-reference-free]] [[towards-bridging-the-gap-systematic-sim-to-real-transfer-for-diverse-legged-robo]].
- **Hexapod papers are sparse but congruent.** The Liu et al. SJTU paper [[learning-natural-and-robust-hexapod-locomotion-over-complex-terrains-via-motion]] reports 4096 envs × 1000 steps × 50,000 episodes (~204.8B step ceiling, lower with early termination) in 35 hours on a single RTX 3090Ti. Berkeley/DeepMind hexapod skills [[versatile-locomotion-skills-for-hexapod-robots]] report 10h teacher RL + 5h student distillation on RTX TITAN, no step count.
- **AMP-augmented PPO costs roughly the same as vanilla PPO** at a given env count. Walk-Fly AMP [[learning-to-walk-and-fly-with-adversarial-motion-priors]] runs 4096 envs in ~2h on a Quadro RTX 6000; CAMP [[learning-multi-skill-legged-locomotion-using-conditional-adversarial-motion-prio]] runs 4096 envs in ~7h on a 4060Ti; bipedal AMP [[learning-bipedal-walking-on-a-quadruped-robot-via-adversarial-motion-priors]] runs 500 envs × 26000 iter in 15.88h on a 4070. None of these AMP papers report discriminator update frequency.
- **Modern Brax/MJX numbers collapse wall-clock dramatically without changing total step count.** Brax humanoid [[learning-velocity-based-humanoid-locomotion-massively-parallel-learning-with-bra]]: 200M steps × 8192 envs in **56 min on a single RTX 4090**. MuJoCo Playground [[mujoco-playground]] hits 400-417k SPS for Go1 quadruped on a single A100, training Go1Joystick in under 10 minutes.
- **Stage decomposition is inconsistently reported.** The most explicit stage-decomposed budget is Bin Yang et al. quadrupedal jumping [[curriculum-based-reinforcement-learning-for-quadrupedal-jumping-a-reference-free]]: Stage I 295M / Stage II 983M / Stage III 983M, each from the previous stage's checkpoint. Most other papers either report a single PPO run or fold DR randomization into the main run rather than as a separate fine-tune phase.

This report extracts those numbers paper-by-paper (Section 2), regroups them by training stage (Section 3), separates single-GPU norms from multi-GPU norms (Section 4), highlights the budget outliers above and below the median (Section 5), and commits to a recommendation for the user's RTX 5060 Ti hexapod project (Section 6).

## 2. Comparison Table — Per-Paper Budget Extraction

The table below extracts every legged-locomotion RL paper in the corpus where at least one budget axis was explicitly reported. **NOT REPORTED** is used verbatim where a paper omits a field; derived numbers are tagged "(derived)". Robot column abbreviations: A1/Go1/Go2 = Unitree quadrupeds; B2 = Unitree quadruped + Z1 arm; ANYmal = ETH Zurich quadruped; Digit = Agility Robotics biped; G1/T1 = Unitree humanoids.

| Paper (year) | Robot | Algo | num_envs | iters / episodes | rollout | Total env steps | Wall-clock | GPU | Stage breakdown |
|---|---|---|---|---|---|---|---|---|---|
| Liu et al. SJTU hexapod (2025) [[learning-natural-and-robust-hexapod-locomotion-over-complex-terrains-via-motion]] | PhantomX-class hexapod | PPO + AMP | 4096 | 50,000 episodes | 1000 steps/episode | ~204.8B ceiling (less w/ early termination, derived) | 35 h | RTX 3090Ti (1×) | Single PPO+AMP run; DR mixed in throughout |
| Berkeley/DeepMind hexapod (2024) [[versatile-locomotion-skills-for-hexapod-robots]] | SpiderPi hexapod | PPO + BC distill | NOT REPORTED | NOT REPORTED | NOT REPORTED | NOT REPORTED | 10h teacher + 5h student = ~15h | RTX TITAN (1×) | Two-stage: teacher RL ~10h, student distillation ~5h |
| Liu et al. bipedal-AMP on quadruped (2024) [[learning-bipedal-walking-on-a-quadruped-robot-via-adversarial-motion-priors]] | Unitree A1 (biped mode) | PPO + AMP + teacher-student | 500 | 26,000 | NOT REPORTED | NOT REPORTED | 15.88 h | RTX 4070 (1×) | Single AMP+PPO run, no separate BC pretrain |
| CAMP multi-skill AMP (2025) [[learning-multi-skill-legged-locomotion-using-conditional-adversarial-motion-prio]] | Unitree Go2 | PPO + Conditional AMP | 4096 | NOT REPORTED (HTML truncation) | NOT REPORTED | "[X] million simulated time steps" — value truncated | ~7 h | RTX 4060Ti (1×) | Single conditional-AMP+PPO run |
| Walk + Fly AMP (2023) [[learning-to-walk-and-fly-with-adversarial-motion-priors]] | iRonCub aerial humanoid (23-DOF) | PPO + AMP | 4096 | NOT REPORTED | NOT REPORTED | NOT REPORTED | ~2 h | Quadro RTX 6000 (1×) | Single 2h training, no pretrain or DR fine-tune stage |
| Escontrela et al. AMP (2022) [[220315103-adversarial-motion-priors-make-good-substitutes-for-complex-reward-fun]] [[adversarial-motion-priors-make-good-substitutes-for-complex-reward-functions-ale]] | quadruped (sim-to-real) | PPO + AMP | NOT REPORTED (project page only) | NOT REPORTED | NOT REPORTED | NOT REPORTED | NOT REPORTED | NOT REPORTED | Mocap prior (~few sec German Shepherd) |
| MuJoCo Playground Go1JoystickFlat (2025) [[mujoco-playground]] [[250208844-mujoco-playground]] | Unitree Go1 | Brax PPO | 8192 | NOT REPORTED | 20 | 100M | < 10 min | A100 (1×) | Flat first then rough-terrain DR fine-tune (separate phases) |
| MuJoCo Playground Go1Handstand/Footstand (2025) [[mujoco-playground]] | Unitree Go1 | Brax PPO | 8192 | NOT REPORTED | 20 | 200M | < 30 min (1× A100, est.) | A100 (1×) | Single PPO run |
| MuJoCo Playground Go1Getup (2025) [[mujoco-playground]] | Unitree Go1 | Brax PPO | 32768 | NOT REPORTED | 32 | 400M | NOT REPORTED | A100 (1×) | Single PPO run |
| MuJoCo Playground G1Joystick (2025) [[mujoco-playground]] | Unitree G1 humanoid | Brax PPO | 32768 | NOT REPORTED | 32 | 400M | < 30 min | 2× RTX 4090 | Flat-ground first, rough-terrain fine-tune |
| MuJoCo Playground Go1Backflip (2025) [[mujoco-playground]] | Unitree Go1 | Brax PPO | NOT REPORTED | NOT REPORTED | NOT REPORTED | 50M | NOT REPORTED | A100 (1×) | Single PPO run |
| Brax humanoid REEM-C (Thibault et al., 2024) [[learning-velocity-based-humanoid-locomotion-massively-parallel-learning-with-bra]] [[240705148-learning-velocity-based-humanoid-locomotion-massively-parallel-learnin]] | REEM-C 12-DOF biped | Brax PPO | 8192 | NOT REPORTED | NOT REPORTED | 200M | ~56 min | RTX 4090 (1×) | Single PPO run, DR integrated |
| Rudin et al. "Walk in Minutes" (2022) [[210911978-learning-to-walk-in-minutes-using-massively-parallel-deep-reinforcemen-2]] [[210911978-learning-to-walk-in-minutes-using-massively-parallel-deep-reinforcemen]] | ANYmal | PPO (Isaac Gym) | 4096 | 1500 updates | batch 98,304 | ~147M (derived) | < 20 min rough; < 4 min flat | RTX A6000 (1×) | Single PPO; ablation shows 2048-4096 envs is the sweet spot |
| `legged_gym` defaults (2021-2024) [[legged_gymlegged_gymenvsbaselegged_robot_configpy-at-master-leggedroboticslegged]] | ANYmal (config) | PPO | 4096 | 1500 | 24 | ~147M (derived) | NOT REPORTED | NOT REPORTED (Isaac Gym) | Single PPO; canonical reference template |
| Curriculum quadruped jumping (2024) [[curriculum-based-reinforcement-learning-for-quadrupedal-jumping-a-reference-free]] | Unitree Go1 | PPO (Isaac Gym + RSL-RL) | 4096 | 3k / 10k / 10k (3 stages) | 24 | ~295M / ~983M / ~983M = ~2.26B (derived) | 1.4h + 4.1h + 4.8h = ~10.3 h | RTX 3090 (1×) | Explicit 3-stage curriculum; each stage from prev checkpoint |
| ETH PACE diverse legged (2025) [[towards-bridging-the-gap-systematic-sim-to-real-transfer-for-diverse-legged-robo]] | ANYmal/Tytan/Minimal + 10 more | PPO (Isaac Gym/Lab) | 4096 | 30,000 | 24 | ~2.95B (derived) | 1-24 h CMA-ES per robot (sysID) | RTX 3080 (1×) for sysID | sysID first (CMA-ES), then PPO with NO dynamics randomization |
| Athletic loco-manipulation (2025) [[bridging-the-sim-to-real-gap-for-athletic-loco-manipulation]] | Unitree B2 + Z1 arm (19-DOF) | PPO (RSL-RL) | 4096 | NOT REPORTED | 24 (pre-train) / 96 (UAN) | NOT REPORTED | NOT REPORTED | MIT Supercloud cluster | UAN sim-to-real → WBC pretrain → task fine-tune (3 stages) |
| Real-world humanoid (Radosavovic, 2023) [[real-world-humanoid-locomotion-with-reinforcement-learning]] [[230303381-real-world-humanoid-locomotion-with-reinforcement-learning]] | Agility Digit (30-DoF) | PPO + teacher-student | "thousands" | NOT REPORTED in main paper | NOT REPORTED | NOT REPORTED | NOT REPORTED | 4× A100 | Two-stage teacher (full state) → student (PPO + imitation) |
| Hwangbo et al. (2019) [[190108652-learning-agile-and-dynamic-motor-skills-for-legged-robots-2]] | ANYmal | TRPO + actuator net | 30 | NOT REPORTED | NOT REPORTED | ~250M (derived) | 4 h locomotion / 11 h recovery | 1 desktop GPU + 1 CPU | Two-stage: actuator net pretrain (<4 min, 1M+ samples) → RL |
| Lee et al. (2020) [[201011251-learning-quadrupedal-locomotion-over-challenging-terrain-2]] | ANYmal | TRPO + teacher-student | NOT REPORTED | 4000 (teacher) / 1000 (student) | batch 20,000 | ~80M teacher + ~20M student = ~100M (derived) | ~12h teacher + ~4h student = ~16 h | i7-8700K + RTX 2080 (1×) | Teacher (privileged) → student (proprioception) |
| RMA (2021) [[210704034-rma-rapid-motor-adaptation-for-legged-robots-2]] | Unitree A1 | PPO + supervised distill | NOT REPORTED (RaiSim) | 15,000 (Phase 1) + 1000 (Phase 2) | batch 80,000 | 1.2B (Phase 1) + 80M (Phase 2) = ~1.28B | ~24h + ~3h = ~27 h | 1 GPU desktop | Phase 1: PPO base policy. Phase 2: adaptation distill |
| Isaac Gym ANYmal benchmark (2021) [[210810470-isaac-gym-high-performance-gpu-based-physics-simulation-for-robot-lear]] | ANYmal | PPO | 4096 | NOT REPORTED | NOT REPORTED | NOT REPORTED | < 2 min | A100 (1×) | Throughput benchmark only |
| Barkour (2023) [[230514654-barkour-benchmarking-animal-level-agility-with-quadruped-robots]] | Google Barkour | PPO + Transformer distill | NOT REPORTED | NOT REPORTED | NOT REPORTED | NOT REPORTED | NOT REPORTED | NOT REPORTED | Specialist RL → Locomotion-Transformer distillation |
| ResFiT residual off-policy (manipulation) (2025) [[residual-off-policy-rl-for-finetuning-behavior-cloning-policies]] [[250919301-residual-off-policy-rl-for-finetuning-behavior-cloning-policies]] | bimanual humanoid (manipulation) | Off-policy residual on frozen BC | 4 (sim) | NOT REPORTED | NOT REPORTED | 200k (sim) / 134 rollouts (real) | NOT REPORTED / ~15 min real | NOT REPORTED | Reference for BC+RL sample-efficiency claim only |
| PostBC pretraining (manipulation) (2025) [[posterior-behavioral-cloning-pretraining-bc-policies-for-efficient-rl-finetuning]] | manipulation (Robomimic, Libero) | Diffusion BC + DSRL | 4 | NOT REPORTED | NOT REPORTED | 2M gradient steps | NOT REPORTED | NOT REPORTED | BC pretrain → RL fine-tune |
| Walk in the Park (2022) [[220807860-a-walk-in-the-park-learning-to-walk-in-20-minutes-with-model-free-rein]] | quadruped (real-world DroQ) | Off-policy SAC variant | 1 (real) | NOT REPORTED | NOT REPORTED | NOT REPORTED | 20 min real-world | N/A (real-world) | Real-world from-scratch reference point |

**Notes on the table.** "Derived" totals use the standard formula `num_envs × iterations × rollout`. Where wall-clock spans a curriculum (e.g., the jumping paper), each stage is itemized. The two manipulation papers (ResFiT and PostBC) are included for context in the BC-pretrain / sample-efficiency discussion (Sections 3 and 5) only — they are not legged-locomotion training-budget reference points.

## 3. Stage-Decomposed Budgets — BC Pretrain, Main RL, Fine-Tune / DR

Few legged-locomotion papers explicitly decompose their budgets across "Stage 0 BC pretrain", "main RL run", and "fine-tune / domain randomization (DR)". Most either report a single PPO run with DR randomization mixed in throughout, or describe a teacher-student pipeline in which the second stage is supervised distillation rather than a fine-tune. Below we group the corpus by which decomposition pattern they exhibit.

### 3.1 Single-stage PPO with DR mixed in (no explicit pretrain or fine-tune)

This is by far the dominant pattern. Examples:

- **Liu et al. SJTU hexapod** [[learning-natural-and-robust-hexapod-locomotion-over-complex-terrains-via-motion]]: "We randomize dynamic parameters for both robots and environments to reflect differences between real and simulated conditions." Single 35h PPO+AMP run, no separate BC pretrain or DR fine-tune.
- **Bipedal AMP on quadruped** [[learning-bipedal-walking-on-a-quadruped-robot-via-adversarial-motion-priors]]: "We trained 500 parallel agents on different types of terrains... in overall 26000 iterations and cost 15.88 hours in total." No pretrain phase mentioned.
- **Walk-Fly AMP** [[learning-to-walk-and-fly-with-adversarial-motion-priors]]: "The training requires ~2 hours on an NVIDIA Quadro RTX 6000." Single run, no stage breakdown.
- **CAMP** [[learning-multi-skill-legged-locomotion-using-conditional-adversarial-motion-prio]]: 4096 envs, ~7h, no separate pretrain or DR fine-tune stage.
- **Brax humanoid REEM-C** [[learning-velocity-based-humanoid-locomotion-massively-parallel-learning-with-bra]]: "200M training steps with 8192 parallel environments using PPO... completed training in approximately 56 minutes." Single PPO with DR integrated.
- **Rudin "Walk in Minutes"** [[210911978-learning-to-walk-in-minutes-using-massively-parallel-deep-reinforcemen-2]]: 1500 policy updates, single run.

### 3.2 Two-stage teacher-student (RL + supervised distillation)

Common in older quadruped papers and some hexapod work. Stage 1 is RL with privileged information; Stage 2 is a supervised "student" trained to imitate the teacher using only proprioceptive observations. The student stage is short relative to the teacher.

- **Lee et al. 2020 ANYmal** [[201011251-learning-quadrupedal-locomotion-over-challenging-terrain-2]]: ~12h teacher (4000 iters × batch 20,000 = ~80M derived) + ~4h student (1000 iters × 20,000 = ~20M derived) = ~16h on i7-8700K + RTX 2080.
- **RMA** [[210704034-rma-rapid-motor-adaptation-for-legged-robots-2]]: Phase 1 PPO 15,000 iters × batch 80,000 = **1.2B steps in ~24h**. Phase 2 distillation 1000 iters × 80,000 = **80M steps in ~3h**. Total ~1.28B / ~27h on a single desktop GPU. The Phase 2 supervised distillation is roughly 1/15 the cost of Phase 1.
- **Berkeley/DeepMind hexapod** [[versatile-locomotion-skills-for-hexapod-robots]]: "10 hours for the teacher policy and 5 hours for the student policy on an RTX TITAN GPU" — student is half the teacher cost.
- **Real-world humanoid (Digit)** [[real-world-humanoid-locomotion-with-reinforcement-learning]]: teacher-student with imitation-loss weighting "annealed to zero at the mid-point of the training horizon."

The common ratio is roughly **teacher : student ≈ 2:1 to 3:1 in wall-clock**, with the student stage being supervised (cheap, deterministic) and the teacher being the bulk of the compute.

### 3.3 Multi-stage curriculum (each stage from previous checkpoint)

These pipelines explicitly increment difficulty and budget per stage; this is the closest analog to the user's "multi-session curriculum" plan.

- **Curriculum quadruped jumping (Bin Yang et al.)** [[curriculum-based-reinforcement-learning-for-quadrupedal-jumping-a-reference-free]]: Stage I (jump-in-place) 3000 iter ≈ **295M steps, ~1.4h**. Stage II (long-distance jump) 10,000 iter ≈ **983M, ~4.1h**. Stage III (jump with obstacles) 10,000 iter ≈ **983M, ~4.8h**. Total ≈ **2.26B / ~10.3h on a single RTX 3090**. The paper explicitly validates that skipping Stage I causes Stage II to collapse to a "standing behavior."
- **MuJoCo Playground locomotion curriculum** [[mujoco-playground]]: "We firstly train the policy in flat ground with restricted command ranges within 5 minutes (2× RTX 4090). and finetune it in rough terrain with wider ranges." This is the cleanest published example of a flat-then-rough two-stage curriculum.
- **Athletic loco-manipulation** [[bridging-the-sim-to-real-gap-for-athletic-loco-manipulation]]: Three explicit stages — Unsupervised Actuator Net (UAN) sim-to-real → WBC pretrain → task-specific RL fine-tune. The WBC base policy is reused across multiple task fine-tunes, amortizing pretraining cost.

### 3.4 BC pretrain ("Stage 0") for legged locomotion — almost nonexistent in the corpus

A striking gap: **no surveyed legged-locomotion paper reports a separate BC ("behavior cloning") pretrain stage from a hand-engineered or trajectory-optimization (TO) reference gait, followed by RL fine-tuning.** The closest is:

- AMP itself [[210402180-amp-adversarial-motion-priors-for-stylized-physics-based-character-con]] [[adversarial-motion-priors-make-good-substitutes-for-complex-reward-functions-ale]] uses prior data as a discriminator-shaped reward, NOT as a pretrain initializer.
- Hwangbo et al. 2019 [[190108652-learning-agile-and-dynamic-motor-skills-for-legged-robots-2]] has a Stage 0, but it pretrains the **actuator network** (hardware modeling) with self-supervised learning, not the policy.
- The two manipulation papers in the corpus that **do** report explicit BC-pretrain + RL-finetune budgets — PostBC [[posterior-behavioral-cloning-pretraining-bc-policies-for-efficient-rl-finetuning]] (2M gradient steps RL fine-tune) and ResFiT [[residual-off-policy-rl-for-finetuning-behavior-cloning-policies]] (200k env steps to converge in sim, 134 rollouts ≈15 min real-world) — show that BC+RL sample efficiency is a manipulation-domain story, not a locomotion-domain story.

The user's BC-pretrain stage from their own scaffold gait (`scripts/pretrain_bc_jax.py`) is therefore **methodologically novel within the published locomotion corpus**: most locomotion practitioners either go random-init → AMP, or random-init → teacher-student.

### 3.5 Sim-to-real DR fine-tune as a separate budget

DR (Domain Randomization) is overwhelmingly **applied throughout the main run** rather than as a separate fine-tune stage. The exceptions:

- **MuJoCo Playground**'s flat-then-rough split (above) is the cleanest two-phase DR-fine-tune example.
- **PACE** [[towards-bridging-the-gap-systematic-sim-to-real-transfer-for-diverse-legged-robo]] **eliminates DR entirely** by replacing it with accurate actuator system identification (CMA-ES on RTX 3080) — "Because pace identifies the dynamics end-to-end, we do not use dynamics randomization." Their PPO run is a clean 30,000 iter × 4096 envs × 24 rollout = ~2.95B steps with NO randomization.

**Summary.** When stage decomposition exists in legged locomotion, it is most commonly **teacher RL + student distillation (~2:1 wall-clock)** or **multi-stage curriculum (each stage from prev checkpoint, with budgets growing 1×-3.3× per stage)**. Pure BC-pretrain → RL-fine-tune as a stage decomposition is rare in locomotion and largely confined to manipulation work.

## 4. Hardware Context — Single-GPU vs Multi-GPU Norms

The corpus splits cleanly into single-GPU and multi-GPU work. Single-GPU is the overwhelming majority for 2023-2026 legged-locomotion RL.

### 4.1 Single-GPU norms

| Paper | GPU | Notes |
|---|---|---|
| Liu SJTU hexapod [[learning-natural-and-robust-hexapod-locomotion-over-complex-terrains-via-motion]] | RTX 3090Ti | 4096 envs, 35h |
| Berkeley hexapod [[versatile-locomotion-skills-for-hexapod-robots]] | RTX TITAN | 15h total (teacher + student) |
| Bipedal AMP [[learning-bipedal-walking-on-a-quadruped-robot-via-adversarial-motion-priors]] | RTX 4070 | 500 envs, 15.88h — small env count |
| CAMP [[learning-multi-skill-legged-locomotion-using-conditional-adversarial-motion-prio]] | RTX 4060Ti | 4096 envs, ~7h |
| Walk-Fly AMP [[learning-to-walk-and-fly-with-adversarial-motion-priors]] | Quadro RTX 6000 | 4096 envs, ~2h |
| Brax humanoid [[learning-velocity-based-humanoid-locomotion-massively-parallel-learning-with-bra]] | RTX 4090 | 8192 envs, 56 min, 200M steps |
| Rudin "Walk in Minutes" [[210911978-learning-to-walk-in-minutes-using-massively-parallel-deep-reinforcemen-2]] | RTX A6000 | 4096 envs, <20 min |
| Curriculum jumping [[curriculum-based-reinforcement-learning-for-quadrupedal-jumping-a-reference-free]] | RTX 3090 | 4096 envs, 10.3h, 2.26B steps |
| MuJoCo Playground (quadruped) [[mujoco-playground]] | A100 | 8192 envs, < 10 min, 100M steps |
| RMA [[210704034-rma-rapid-motor-adaptation-for-legged-robots-2]] | "1 desktop GPU" (CPU sim) | 1.28B steps, 27h — CPU-bound |
| Lee 2020 [[201011251-learning-quadrupedal-locomotion-over-challenging-terrain-2]] | RTX 2080 (CPU sim) | 100M derived, 16h — CPU-bound |

The 2023-2026 single-GPU consumer-class baseline is **a 4060Ti through 4090, training 4096-8192 envs to 100M-200M steps in 1-15 hours**, with the longer wall-clock numbers reserved for AMP variants and rough-terrain curricula.

### 4.2 Multi-GPU norms

Genuinely multi-GPU legged-locomotion work is rare in this corpus:

- **Real-world humanoid (Digit)** [[real-world-humanoid-locomotion-with-reinforcement-learning]] uses **4× A100** for "thousands of randomized environments" — a humanoid-specific compute scale, with the main paper not reporting step counts.
- **MuJoCo Playground humanoids** [[mujoco-playground]] use **2× RTX 4090** for Berkeley Humanoid / G1 / T1 (under 15-30 min wall-clock for each).
- **Athletic loco-manipulation** [[bridging-the-sim-to-real-gap-for-athletic-loco-manipulation]] runs on the MIT Supercloud HPC cluster but doesn't report a step count or per-GPU breakdown.

For quadruped and hexapod work specifically, **multi-GPU is essentially absent from the published corpus**. The 4096-env Isaac Gym / MJX setup saturates a single high-end GPU; scaling beyond that hits PPO's well-known on-policy diminishing returns. Rudin et al. explicitly verify this: "From the third plot we can conclude that increasing the number of robots is beneficial for both final performance and training time, but there is an upper limit on this number after which an on-policy algorithm cannot learn effectively... using 2048 to 4096 robots with a batch size of 98304" [[210911978-learning-to-walk-in-minutes-using-massively-parallel-deep-reinforcemen-2]].

### 4.3 GPU throughput reference points

| Backend | GPU | Throughput | Source |
|---|---|---|---|
| Isaac Gym, ANYmal | A100 (1×) | ~540K steps/s | [[210810470-isaac-gym-high-performance-gpu-based-physics-simulation-for-robot-lear]] |
| MJX, Go1 flat | A100 (1×) | ~417K SPS | [[mujoco-playground]] |
| MJX, Spot flat | A100 (1×) | ~405K SPS | [[mujoco-playground]] |
| MJX, Berkeley Humanoid flat | A100 (1×) | ~120K SPS | [[mujoco-playground]] |
| MJX, Berkeley Humanoid rough | A100 (1×) | ~30K SPS | [[mujoco-playground]] |
| RaiSim CPU (Hwangbo 2019) | desktop CPU | ~500K steps/s with actuator nets | [[190108652-learning-agile-and-dynamic-motor-skills-for-legged-robots-2]] |

The user's reported **240k it/s effective on a 5060 Ti via WSL2** is consistent with the public throughput data: 240k SPS sits roughly between the 120k SPS Berkeley Humanoid flat number and the 405-417k SPS Go1/Spot quadruped numbers on an A100. That is expected. The 5060 Ti is roughly half an A100's compute, and a 6-leg hexapod has more contacts than a Go1 quadruped. **A 200M-step Brax PPO run on 240k SPS effective throughput finishes in ~14 minutes; a 500M-step run finishes in ~35 minutes.** That's the same band as MuJoCo Playground's published wall-clocks for Go1Joystick and Go1Handstand.

## 5. Sample-Efficiency Outliers — Papers Above and Below the Median

Across the corpus, the **median total step count for a single-task locomotion training run is roughly 100M-200M**. Around that median:

### 5.1 Below the median — sample-efficient outliers

- **Rudin "Walk in Minutes"** [[210911978-learning-to-walk-in-minutes-using-massively-parallel-deep-reinforcemen-2]] — ~147M steps in <20 min on a single RTX A6000 for a rough-terrain ANYmal policy. The headline outlier on wall-clock; sample count is in-band, but the throughput unlocks a wall-clock regime that prior work (Lee 2020 16h, RMA 27h, Miki 120h) couldn't reach.
- **MuJoCo Playground Go1Backflip** [[mujoco-playground]] — **50M steps**, the lowest in the corpus for a published locomotion task. Single-skill, narrow distribution.
- **Brax humanoid REEM-C** [[learning-velocity-based-humanoid-locomotion-massively-parallel-learning-with-bra]] — 200M steps in 56 min on a single RTX 4090. This is the most directly hardware-comparable paper to the user's setup (Brax+MJX, single consumer GPU) and converges humanoid locomotion at the lower end of the budget band.
- **Walk in the Park** [[220807860-a-walk-in-the-park-learning-to-walk-in-20-minutes-with-model-free-rein]] — quadruped walking learned in **20 minutes of real-world experience** with off-policy SAC variants. This is a sample-efficiency outlier in absolute terms but uses a fundamentally different algorithm class (off-policy + 1 real robot vs on-policy + 4096 sim envs); not directly comparable for sim-budget purposes.

### 5.2 Above the median — large-budget outliers

- **Liu SJTU hexapod** [[learning-natural-and-robust-hexapod-locomotion-over-complex-terrains-via-motion]] — ~204.8B step ceiling (lower with early termination), 35h on a single 3090Ti. The largest published hexapod budget in the corpus, but the early-termination caveat makes the effective step count substantially lower.
- **ETH PACE** [[towards-bridging-the-gap-systematic-sim-to-real-transfer-for-diverse-legged-robo]] — ~2.95B derived steps over 30,000 iter for diverse-robot transfer, biggest derived total in the corpus among single-task papers.
- **Curriculum jumping** [[curriculum-based-reinforcement-learning-for-quadrupedal-jumping-a-reference-free]] — ~2.26B steps across a 3-stage curriculum, ~10.3h wall-clock.
- **RMA Phase 1** [[210704034-rma-rapid-motor-adaptation-for-legged-robots-2]] — 1.2B steps in 24h, but this number is **CPU-bound** (RaiSim); a modern Isaac Gym run hits the same ~1B threshold in roughly 1-2 hours on a single GPU.
- **Go1Getup / G1Joystick** [[mujoco-playground]] — 400M steps each, 4× the Go1JoystickFlat baseline. The increment is task-driven (recovery from arbitrary failure states, full humanoid joystick) rather than wall-clock-driven.

### 5.3 Outlier *claim*: ResFiT's "200× sample efficiency"

The largest sample-efficiency claim in the corpus comes from ResFiT [[residual-off-policy-rl-for-finetuning-behavior-cloning-policies]]: "**our approach converging at 200k steps versus 40M steps, we see a ~200x boost in sample efficiency**." It is critical to read the scope carefully: this is a **manipulation** result (BoxCleanup task), comparing **off-policy residual RL to on-policy PPO residual RL on top of a frozen BC base**. It is **not** a comparison between residual fine-tuning and from-scratch PPO, and the absolute numbers (200k vs 40M) refer to manipulation-task convergence, not locomotion. Cited as a methodological signal for the user's BC+RL pipeline, **not** as a transferable locomotion budget reduction.

### 5.4 Median band, by task type

Aggregating the outliers and the per-paper totals from Section 2:

| Task type | Typical total env steps | Typical wall-clock (single consumer GPU) |
|---|---|---|
| Single-task quadruped joystick (flat or rough) | 100M-200M | 10 min - 2 h |
| AMP-augmented quadruped or hexapod | ~150M-1B (often unreported) | 2-35 h |
| Recovery / handstand / getup | 200M-400M | 30 min - 1 h |
| Multi-stage curriculum (jump, manipulation+locomotion) | 1B-3B aggregated | 8-15 h |
| Outlier real-robot (off-policy) | <1M actual | ~20 min |

## 6. Recommendation — Budget Target for a Single RTX 5060 Ti at 240k it/s

**Verdict: the user's planned 100M-500M steps per session in a multi-session curriculum is firmly ON-TREND with published practice for hexapod / quadruped PPO+AMP work — neither under-trained nor over-trained.** Below we lay out the exact reasoning, pinned to the table.

### 6.1 Per-session budget (100M-500M)

The user's per-session window of **100M-500M** maps almost identically to the per-task budgets in MuJoCo Playground [[mujoco-playground]]:

- 100M = Go1JoystickFlatTerrain / Go1JoystickRoughTerrain / T1Joystick (the canonical quadruped flat-terrain budget)
- 200M = Go1Handstand / Go1Footstand / Brax humanoid REEM-C [[learning-velocity-based-humanoid-locomotion-massively-parallel-learning-with-bra]] (more-dynamic skill or humanoid)
- 400M = Go1Getup / G1Joystick (recovery-class tasks that explore failure-state distributions)
- 50M = Go1Backflip (narrow specialist skill)

A hexapod walking/turning policy with AMP guidance falls between Go1JoystickRough and Go1Getup in difficulty: more contacts than a quadruped (6 feet vs 4 = more contact-pair complexity, which the user has already mitigated by trimming `contype/conaffinity` to feet↔floor only), but less recovery-from-failure scope than Go1Getup. **A 200M-step base run is the right anchor** for a single AMP-guided session, with up to 400M for harder skills (rough terrain, payload).

### 6.2 Multi-session curriculum total

Stacked across a multi-session curriculum, total budgets in the corpus that match or exceed the user's plan:

- Curriculum jumping [[curriculum-based-reinforcement-learning-for-quadrupedal-jumping-a-reference-free]]: **2.26B aggregated** over 3 stages, **10.3h on a single RTX 3090**.
- ETH PACE [[towards-bridging-the-gap-systematic-sim-to-real-transfer-for-diverse-legged-robo]]: ~2.95B derived in a single Isaac Lab PPO run.
- RMA [[210704034-rma-rapid-motor-adaptation-for-legged-robots-2]]: 1.28B over two stages on a single GPU.
- Liu SJTU hexapod [[learning-natural-and-robust-hexapod-locomotion-over-complex-terrains-via-motion]]: ~200B ceiling (effective lower); 35h on a 3090Ti.

A user budget of **100M-500M per session × 4-10 sessions = 400M-5B aggregate** sits squarely in this published band — comparable to ETH PACE (2.95B) and the curriculum-jumping pipeline (2.26B), and well below the Liu hexapod ceiling.

**Anywhere in 400M-5B aggregate is on-trend.** Below ~50M aggregate would be under-trained for a hexapod-with-AMP project; above ~10B aggregate would be excessive without a clear novel-task justification.

### 6.3 Wall-clock check on the RTX 5060 Ti

At the user's stated **240k it/s effective**:

- 100M steps → ~7 minutes
- 200M steps → ~14 minutes
- 500M steps → ~35 minutes
- 1B steps → ~70 minutes
- A multi-session 5B aggregate → ~6 hours of wall-clock training across the curriculum

That sits between Brax humanoid's 56 minutes (200M on RTX 4090, [[learning-velocity-based-humanoid-locomotion-massively-parallel-learning-with-bra]]) and MuJoCo Playground's <10 min Go1Joystick (100M on A100, [[mujoco-playground]]), as expected from a 5060 Ti's intermediate position in the consumer-GPU hierarchy. **Wall-clock per session is comfortable; the user is not memory-bound or compute-bound at this budget.**

### 6.4 AMP-specific overhead

The discriminator forward+backward pass is **roughly free relative to the PPO actor-critic update** for the 2-layer MLP discriminators all surveyed AMP papers use: [1024, 512] in Liu SJTU hexapod [[learning-natural-and-robust-hexapod-locomotion-over-complex-terrains-via-motion]] and CAMP [[learning-multi-skill-legged-locomotion-using-conditional-adversarial-motion-prio]], analogous in bipedal AMP [[learning-bipedal-walking-on-a-quadruped-robot-via-adversarial-motion-priors]] and walk-fly AMP [[learning-to-walk-and-fly-with-adversarial-motion-priors]]. Wall-clock overhead is implicit in the published AMP numbers (CAMP's 7h vs Brax humanoid's 56 min reflects discriminator AND larger curriculum AND slower GPU, not just AMP). **Do not budget extra steps for AMP; budget the same step count and let the discriminator update piggyback on each PPO iteration.**

### 6.5 Step-count vs reward-signal trade-off (project-specific)

The user's project context — a recently discovered hovering reward exploit, planned pivot to AMP — points to a different binding constraint: **for the next 1-2 sessions the limit is reward-signal quality, not step count**. AMP fixes the structural problem (no signal requiring foot contact) that more steps cannot fix. Concretely: a 100M-step AMP run with a properly tuned discriminator and TO-generated prior should outperform a 1B-step pure-PPO run with the current "no contact penalties" reward. The Liu SJTU hexapod prior dataset of just 8.6 seconds of TO tripod gait [[learning-natural-and-robust-hexapod-locomotion-over-complex-terrains-via-motion]] and the Escontrela dog-mocap prior of "a few seconds" [[adversarial-motion-priors-make-good-substitutes-for-complex-reward-functions-ale]] both demonstrate that **prior dataset size is not where the budget belongs**.

### 6.6 Concrete recommendation

For the user's RTX 5060 Ti hexapod project running Brax PPO+AMP at ~240k SPS:

1. **Per-session target: 200M steps as the default**, 100M for quick iteration runs, 400M for harder skills. (Anchored to MuJoCo Playground per-task budgets [[mujoco-playground]] and Brax humanoid [[learning-velocity-based-humanoid-locomotion-massively-parallel-learning-with-bra]].)
2. **Multi-session curriculum aggregate: 1-3B total** across 4-8 staged sessions (BC-init → AMP flat → AMP rough → DR fine-tune). Anchored to the curriculum-jumping budget of 2.26B [[curriculum-based-reinforcement-learning-for-quadrupedal-jumping-a-reference-free]] and ETH PACE 2.95B [[towards-bridging-the-gap-systematic-sim-to-real-transfer-for-diverse-legged-robo]].
3. **Wall-clock per session ≈ 14-35 minutes**; full curriculum ≈ 4-7 hours; well within an overnight or weekend slot.
4. **Do not chase the Liu SJTU 200B ceiling** [[learning-natural-and-robust-hexapod-locomotion-over-complex-terrains-via-motion]] — that number is inflated by early-termination, and Liu's 35h on a 3090Ti is not throughput-comparable to the user's MJX setup.
5. **AMP step overhead: zero added budget**; the discriminator runs on the same iterations as PPO.
6. **The 100M-500M-per-session plan is on-trend.** It is neither under-trained nor over-trained relative to published practice. The right single-session anchor for a hexapod AMP run is **200M**, with the option to push to 400M for the rough-terrain or DR stage.
