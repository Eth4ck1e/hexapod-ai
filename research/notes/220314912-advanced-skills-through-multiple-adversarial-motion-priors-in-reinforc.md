---
title: '[2203.14912] Advanced Skills through Multiple Adversarial Motion Priors in
  Reinforcement Learning'
id: 220314912-advanced-skills-through-multiple-adversarial-motion-priors-in-reinforc
tags:
- amp
- multi-discriminator
- quadruped
created: '2026-05-10T18:34:11.561090Z'
updated: '2026-05-10T18:35:15.422521Z'
source: https://arxiv.org/abs/2203.14912
source_domain: arxiv.org
fetched_at: '2026-05-10T18:34:11.560090Z'
fetch_provider: builtin
status: review
type: note
tier: institutional
content_type: paper
deprecated: false
summary: 'Vollenweider et al. (Hutter lab, ICRA 2023) — the canonical MULTIPLE-AMP
  paper for legged robots. Trains N separate AMP discriminators, one per discrete
  switchable style (walk, duck, bipedal stand-up). A one-hot style-selector input
  gates which disc''s reward is active per timestep; only style-matched prior samples
  are fed to the corresponding disc. Validated on real wheeled-legged ANYmal-X. Mechanism
  vs vanilla AMP: cmd-conditional via discrete disc selection + per-disc data partitioning.
  Strongest robotics-side analog for hexapod-style cmd-conditional AMP.'
---

[2203.14912] Advanced Skills through Multiple Adversarial Motion Priors in Reinforcement Learning
Computer Science > Robotics
arXiv:2203.14912
(cs)
[Submitted on 23 Mar 2022]
Title:
Advanced Skills through Multiple Adversarial Motion Priors in Reinforcement Learning
Authors:
Eric Vollenweider
,
Marko Bjelonic
,
Victor Klemm
,
Nikita Rudin
,
Joonho Lee
,
Marco Hutter
View a PDF of the paper titled Advanced Skills through Multiple Adversarial Motion Priors in Reinforcement Learning, by Eric Vollenweider and 5 other authors
View PDF
Abstract:
In recent years, reinforcement learning (RL) has shown outstanding performance for locomotion control of highly articulated robotic systems. Such approaches typically involve tedious reward function tuning to achieve the desired motion style. Imitation learning approaches such as adversarial motion priors aim to reduce this problem by encouraging a pre-defined motion style. In this work, we present an approach to augment the concept of adversarial motion prior-based RL to allow for multiple, discretely switchable styles. We show that multiple styles and skills can be learned simultaneously without notable performance differences, even in combination with motion data-free skills. Our approach is validated in several real-world experiments with a wheeled-legged quadruped robot showing skills learned from existing RL controllers and trajectory optimization, such as ducking and walking, and novel skills such as switching between a quadrupedal and humanoid configuration. For the latter skill, the robot is required to stand up, navigate on two wheels, and sit down. Instead of tuning the sit-down motion, we verify that a reverse playback of the stand-up movement helps the robot discover feasible sit-down behaviors and avoids tedious reward function tuning.
Subjects:
Robotics (cs.RO)
; Artificial Intelligence (cs.AI); Machine Learning (cs.LG)
Cite as:
arXiv:2203.14912
[cs.RO]
(or
arXiv:2203.14912v1
[cs.RO]
for this version)
https://doi.org/10.48550/arXiv.2203.14912
Focus to learn more
arXiv-issued DOI via DataCite
Submission history
From: Marko Bjelonic [
view email
]
[v1]
Wed, 23 Mar 2022 09:24:06 UTC (4,950 KB)
Full-text links:
Access Paper:
View a PDF of the paper titled Advanced Skills through Multiple Adversarial Motion Priors in Reinforcement Learning, by Eric Vollenweider and 5 other authors
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
2022-03
Change to browse by:
cs
cs.AI
cs.LG
References & Citations
NASA ADS
Google Scholar
Semantic Scholar
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