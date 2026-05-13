---
title: Home
id: home
tags:
- legged-rl-budgets
created: '2026-05-06T07:37:01.821247Z'
updated: '2026-05-06T07:38:40.783723Z'
source: https://leggedrobotics.github.io/legged_gym/
source_domain: leggedrobotics.github.io
fetched_at: '2026-05-06T07:37:01.821247Z'
fetch_provider: builtin
status: deprecated
type: note
tier: unknown
content_type: unknown
deprecated: false
---

*Suggested by [[github-leggedroboticslegged_gym-isaac-gym-environments-for-legged-robots-github]] — legged_gym project webpage with training details*

Home
Learning To Walk in Minutes
Using Massively Parallel Deep Reinforcement Learning
Nikita Rudin, David Hoeller, Philipp Reist and Marco Hutter
Robotic Systems Lab, ETH Zurich & NVIDIA
We present and study a training set-up that achieves fast policy generation for real-world robotic tasks by usin massive parallelism on a single workstation GPU. In addition, we present a novel game-inspired curriculum that is well suited for training with thousands of simulated robots in parallel. We evaluate the approach by training multiple
legged robots
to walk on challenging terrain
The parallel approach allows training policies for flat terrain in under four minutes, and in twenty minutes for uneven terrain.
Finally, we transfer the polic
y
to the real ANYmal C robot to validate the approach.
The work was accepted to CoRL 2021.
Full paper
. The code is available
here
.
Additional Quadrupedal and Bipedal robots
ANYmal B
A1
Cassie
Game Inspired Curriculum
The robots progress through the terrains based on their success. When a robot solves its assigned terrain by walking past the borders, it moves to the next more complex terrain. However, if it did not walk far enough, it's level is reduced. this creates an automatic curriculum, which adapts the difficulty for each terrain type individually.
For example, after 500 iterations (top) the policy is able to cross sloped terrains and to go down stairs, but climbing stairs and traversing obstacles requires more training iterations.
After 1000 iterations (bottom), the robots have reached the most challenging level for all terrain types and are spread across the map.
Sim-to-Real Deployment