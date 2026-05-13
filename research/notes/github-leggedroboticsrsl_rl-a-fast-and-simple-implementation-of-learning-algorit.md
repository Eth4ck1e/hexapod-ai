---
title: 'GitHub - leggedrobotics/rsl_rl: A fast and simple implementation of learning
  algorithms for robotics. · GitHub'
id: github-leggedroboticsrsl_rl-a-fast-and-simple-implementation-of-learning-algorit
tags:
- legged-rl-budgets
created: '2026-05-06T07:45:00.824823Z'
updated: '2026-05-06T07:45:13.855505Z'
source: https://github.com/leggedrobotics/rsl_rl
source_domain: github.com
fetched_at: '2026-05-06T07:45:00.824823Z'
fetch_provider: builtin
status: draft
type: note
tier: ground_truth
content_type: code
deprecated: false
summary: 'leggedrobotics/rsl_rl GitHub repo — GPU-accelerated PPO library used by
  legged_gym, Isaac Lab, MuJoCo Playground. Used in the Rudin 2021 legged_gym pipeline
  and is the de facto PPO backend for most ETH-lineage quadruped papers. Supports
  multi-GPU. The library is the implementation vehicle for training configs that define
  num_envs, rollout length, and iteration counts in Isaac Gym/Isaac Lab era papers.
  Note: library README sparse on specific training defaults (those live in the environment
  repos like legged_gym or Isaac Lab configs).'
---

*Suggested by [[github-leggedroboticslegged_gym-isaac-gym-environments-for-legged-robots-github]] — RSL-RL PPO library used in legged_gym and many subsequent papers, has training config defaults*

GitHub - leggedrobotics/rsl_rl: A fast and simple implementation of learning algorithms for robotics. · GitHub
Skip to content
You signed in with another tab or window.
Reload
to refresh your session.
You signed out in another tab or window.
Reload
to refresh your session.
You switched accounts on another tab or window.
Reload
to refresh your session.
Dismiss alert
leggedrobotics
/
rsl_rl
Public
Notifications
You must be signed in to change notification settings
Fork
587
Star
2.6k
main
Branches
Tags
Go to file
Code
Open more actions menu
Folders and files
Name
Name
Last commit message
Last commit date
Latest commit
History
132 Commits
132 Commits
.github
.github
docs
docs
licenses/
dependencies
licenses/
dependencies
rsl_rl
rsl_rl
tests
tests
.gitignore
.gitignore
.pre-commit-config.yaml
.pre-commit-config.yaml
CITATION.cff
CITATION.cff
CONTRIBUTING.md
CONTRIBUTING.md
CONTRIBUTORS.md
CONTRIBUTORS.md
LICENSE
LICENSE
README.md
README.md
pyproject.toml
pyproject.toml
ruff.toml
ruff.toml
setup.py
setup.py
View all files
Repository files navigation
RSL-RL
RSL-RL
is a GPU-accelerated, lightweight learning library for robotics research. Its compact design allows
researchers to prototype and test new ideas without the overhead of modifying large, complex libraries. RSL-RL can also
be used out-of-the-box by installing it via
PyPI
, supports multi-GPU training,
and features common algorithms for robot learning.
Key Features
Minimal, readable codebase
with clear extension points for rapid prototyping.
Robotics-first methods
including PPO and Student-Teacher Distillation.
High-throughput training
with native Multi-GPU support.
Proven performance
in numerous research publications.
Learning Environments
RSL-RL is currently used by the following robot learning libraries:
Isaac Lab
(built on top of NVIDIA Isaac Sim)
Legged Gym
(built on top of NVIDIA Isaac Gym)
mjlab
(built on top of MuJoCo Warp)
MuJoCo Playground
(built on top of MuJoCo MJX and Warp)
Installation
Before installing RSL-RL, ensure that Python
3.9+
is available. It is recommended to install the library in a virtual
environment (e.g. using
venv
or
conda
), which is often already created by the used environment library (e.g.
Isaac Lab). If so, make sure to activate it before installing RSL-RL.
Installing RSL-RL as a dependency
pip install rsl-rl-lib
Installing RSL-RL for development
git clone https://github.com/leggedrobotics/rsl_rl
cd
rsl_rl
pip install -e
.
Citation
If you use RSL-RL in your research, please cite the
paper
:
@article{schwarke2025rslrl,
  title={RSL-RL: A Learning Library for Robotics Research},
  author={Schwarke, Clemens and Mittal, Mayank and Rudin, Nikita and Hoeller, David and Hutter, Marco},
  journal={arXiv preprint arXiv:2509.10771},
  year={2025}
}
About
A fast and simple implementation of learning algorithms for robotics.
leggedrobotics.github.io/rsl_rl/
Resources
Readme
License
View license
Contributing
Contributing
Uh oh!
There was an error while loading.
Please reload this page
.
Activity
Custom properties
Stars
2.6k
stars
Watchers
36
watching
Forks
587
forks
Report repository
Releases
27
v5.2.0
Latest
Apr 23, 2026
+ 26 releases
Uh oh!
There was an error while loading.
Please reload this page
.
Contributors
Uh oh!
There was an error while loading.
Please reload this page
.
Languages
Python
100.0%
You can’t perform that action at this time.