---
title: '[2109.11978] Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement
  Learning'
id: 210911978-learning-to-walk-in-minutes-using-massively-parallel-deep-reinforcemen
tags:
- legged-rl-budgets
- quadruped
- ppo
- training-budget
- isaac-gym
created: '2026-05-06T07:31:32.558613Z'
updated: '2026-05-06T07:37:54.848699Z'
source: https://arxiv.org/abs/2109.11978
source_domain: arxiv.org
fetched_at: '2026-05-06T07:31:32.558613Z'
fetch_provider: builtin
status: draft
type: note
tier: institutional
content_type: paper
deprecated: false
summary: 'Rudin et al. (ETH Zurich/NVIDIA, CoRL 2022): Foundational massively-parallel
  RL training paper for legged locomotion. Trains ANYmal quadruped in IsaacGym on
  single GPU. Key numbers: 4096 parallel robots, batch size 98,304, 1500 policy updates,
  in under 20 minutes for rough terrain (under 4 min for flat) on NVIDIA RTX A6000.
  Total env steps implied = 4096 x n_steps x 1500 where n_steps varies by config.
  Paper shows near-linear throughput scaling up to ~4000 robots; diminishing returns
  beyond. Uses IsaacGym end-to-end GPU pipeline. Ablations show 2048-4096 robots optimal
  for the tradeoff. Trained for 1500 iterations total. This paper establishes the
  IsaacGym paradigm that all subsequent legged RL papers (including Liu et al. and
  Berkeley hexapod) build on.'
---

*Suggested by [[learning-natural-and-robust-hexapod-locomotion-over-complex-terrains-via-motion]] — cited as baseline for IsaacGym massively parallel training*

*Suggested by [[learning-bipedal-walking-on-a-quadruped-robot-via-adversarial-motion-priors]] — Rudin Learning to Walk in Minutes - foundational massively parallel RL paper cited by both bipedal AMP and CAMP papers*

*Suggested by [[github-leggedroboticslegged_gym-isaac-gym-environments-for-legged-robots-github]] — paper cited in legged_gym README - Rudin et al walk-these-ways*

[2109.11978] Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning
Computer Science > Robotics
arXiv:2109.11978
(cs)
[Submitted on 24 Sep 2021 (
v1
), last revised 19 Aug 2022 (this version, v3)]
Title:
Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning
Authors:
Nikita Rudin
,
David Hoeller
,
Philipp Reist
,
Marco Hutter
View a PDF of the paper titled Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning, by Nikita Rudin and 3 other authors
View PDF
Abstract:
In this work, we present and study a training set-up that achieves fast policy generation for real-world robotic tasks by using massive parallelism on a single workstation GPU. We analyze and discuss the impact of different training algorithm components in the massively parallel regime on the final policy performance and training times. In addition, we present a novel game-inspired curriculum that is well suited for training with thousands of simulated robots in parallel. We evaluate the approach by training the quadrupedal robot ANYmal to walk on challenging terrain. The parallel approach allows training policies for flat terrain in under four minutes, and in twenty minutes for uneven terrain. This represents a speedup of multiple orders of magnitude compared to previous work. Finally, we transfer the policies to the real robot to validate the approach. We open-source our training code to help accelerate further research in the field of learned legged locomotion.
Comments:
CoRL 2021 Project website: :
this https URL
Video:
this https URL
Subjects:
Robotics (cs.RO)
; Machine Learning (cs.LG)
Cite as:
arXiv:2109.11978
[cs.RO]
(or
arXiv:2109.11978v3
[cs.RO]
for this version)
https://doi.org/10.48550/arXiv.2109.11978
Focus to learn more
arXiv-issued DOI via DataCite
Submission history
From: Nikita Rudin [
view email
]
[v1]
Fri, 24 Sep 2021 14:04:19 UTC (48,407 KB)
[v2]
Sat, 30 Oct 2021 14:35:58 UTC (48,402 KB)
[v3]
Fri, 19 Aug 2022 07:52:32 UTC (48,402 KB)
Full-text links:
Access Paper:
View a PDF of the paper titled Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning, by Nikita Rudin and 3 other authors
View PDF
TeX Source
view license
Current browse context:
cs.RO
< prev
|
next >
new
|
recent
|
2021-09
Change to browse by:
cs
cs.LG
References & Citations
NASA ADS
Google Scholar
Semantic Scholar
DBLP
- CS Bibliography
listing
|
bibtex
Marco Hutter
export BibTeX citation
Loading...
BibTeX formatted citation
×
loading...
Data provided by:
Bookmark
Bibliographic Tools
Bibliographic and Citation Tools
Bibliographic Explorer Toggle
Bibliographic Explorer
(
What is the Explorer?
)
Connected Papers Toggle
Connected Papers
(
What is Connected Papers?
)
Litmaps Toggle
Litmaps
(
What is Litmaps?
)
scite.ai Toggle
scite Smart Citations
(
What are Smart Citations?
)
Code, Data, Media
Code, Data and Media Associated with this Article
alphaXiv Toggle
alphaXiv
(
What is alphaXiv?
)
Links to Code Toggle
CatalyzeX Code Finder for Papers
(
What is CatalyzeX?
)
DagsHub Toggle
DagsHub
(
What is DagsHub?
)
GotitPub Toggle
Gotit.pub
(
What is GotitPub?
)
Huggingface Toggle
Hugging Face
(
What is Huggingface?
)
Links to Code Toggle
Papers with Code
(
What is Papers with Code?
)
ScienceCast Toggle
ScienceCast
(
What is ScienceCast?
)
Demos
Demos
Replicate Toggle
Replicate
(
What is Replicate?
)
Spaces Toggle
Hugging Face Spaces
(
What is Spaces?
)
Spaces Toggle
TXYZ.AI
(
What is TXYZ.AI?
)
Related Papers
Recommenders and Search Tools
Link to Influence Flower
Influence Flower
(
What are Influence Flowers?
)
Core recommender toggle
CORE Recommender
(
What is CORE?
)
Author
Venue
Institution
Topic
About arXivLabs
arXivLabs: experimental projects with community collaborators
arXivLabs is a framework that allows collaborators to develop and share new arXiv features directly on our website.
Both individuals and organizations that work with arXivLabs have embraced and accepted our values of openness, community, excellence, and user data privacy. arXiv is committed to these values and only works with partners that adhere to them.
Have an idea for a project that will add value for arXiv's community?
Learn more about arXivLabs
.
Which authors of this paper are endorsers?
|
Disable MathJax
(
What is MathJax?
)