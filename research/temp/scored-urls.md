# URL queue — light tier (20 URLs, 4 batches)

Light tier skips formal utility scoring (per skill 2.3). URLs grouped thematically into non-overlapping batches.

## Batch 1 — Anchor hexapod + Brax/MJX (directly comparable to user's setup)

1. https://arxiv.org/abs/2511.03167 — Liu et al, "Learning Natural and Robust Hexapod Locomotion over Complex Terrains via Motion Priors" (user's anchor paper, hexapod + AMP)
2. https://arxiv.org/abs/2412.10628 — "Versatile Locomotion Skills for Hexapod Robots" (Berkeley, IROS 2024)
3. https://arxiv.org/abs/2407.05148 — "Learning Velocity-based Humanoid Locomotion: Massively Parallel Learning with Brax and MJX" (200M steps, 8192 envs, RTX 4090, 56 min)
4. https://arxiv.org/abs/2502.08844 — "MuJoCo Playground" (60-100M PPO step quadruped benchmarks on A100)
5. https://playground.mujoco.org/assets/playground_technical_report.pdf — MuJoCo Playground technical report (extended hyperparameters)

## Batch 2 — Foundational quadruped RL (training budgets in canonical work)

6. https://www.science.org/doi/10.1126/scirobotics.aau5872 — Hwangbo et al, "Learning agile and dynamic motor skills for legged robots" (ANYmal Science Robotics)
7. https://www.science.org/doi/10.1126/scirobotics.abc5986 — Lee et al, "Learning quadrupedal locomotion over challenging terrain" (Science Robotics)
8. https://arxiv.org/abs/2107.04034 — Kumar et al, "RMA: Rapid Motor Adaptation for Legged Robots" (canonical sim-to-real)
9. https://arxiv.org/abs/2108.10470 — Makoviychuk et al, "Isaac Gym: High Performance GPU-Based Physics Simulation For Robot Learning" (training-throughput baseline)
10. https://github.com/leggedrobotics/legged_gym — leggedrobotics/legged_gym (default training configs for quadrupeds)

## Batch 3 — AMP variants (user's planned algorithm)

11. https://arxiv.org/abs/2203.15103 — Escontrela et al, "Adversarial Motion Priors Make Good Substitutes for Complex Reward Functions" (foundational AMP for legged hardware, IROS 2022)
12. https://arxiv.org/abs/2407.02282 — "Learning Bipedal Walking on a Quadruped Robot via Adversarial Motion Priors"
13. https://arxiv.org/abs/2509.21810 — "Learning Multi-Skill Legged Locomotion Using Conditional Adversarial Motion Priors"
14. https://arxiv.org/abs/2309.12784 — "Learning to Walk and Fly with Adversarial Motion Priors"
15. https://escontrela.me/amp_in_real/ — Escontrela project page (training notes, hardware demo videos)

## Batch 4 — BC pretrain + sample efficiency outliers + DR fine-tune

16. https://arxiv.org/abs/2512.16911 — "Posterior Behavioral Cloning: Pretraining BC Policies for Efficient RL Finetuning" (sample-efficiency claims)
17. https://arxiv.org/abs/2509.19301 — "Residual Off-Policy RL for Finetuning Behavior Cloning Policies" (200× sample efficiency claim)
18. https://arxiv.org/abs/2502.10894 — "Bridging the Sim-to-Real Gap for Athletic Loco-Manipulation" (DR + curriculum on quadruped)
19. https://arxiv.org/abs/2509.06342 — "Towards bridging the gap: Systematic sim-to-real transfer for diverse legged robots"
20. https://arxiv.org/abs/2401.16337 — "Curriculum-Based Reinforcement Learning for Quadrupedal Jumping: A Reference-free Design"

## Coverage check

All 20 URLs map to ≥1 atomic item. Hexapod (priority): 2 papers (Liu, Versatile Hexapod). Quadruped: 9+ papers. AMP: 5 papers (foundational + 4 variants). BC pretrain: 2 papers. DR fine-tune: 2 papers. Sample-efficiency outliers: 2 papers. Frameworks: 4 (Isaac Gym, Brax/MJX, Playground, legged_gym).
