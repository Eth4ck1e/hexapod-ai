---
title: '[2104.02180] AMP: Adversarial Motion Priors for Stylized Physics-Based Character
  Control'
id: 210402180-amp-adversarial-motion-priors-for-stylized-physics-based-character-con
tags:
- legged-rl-budgets
- amp
- foundational
- character-animation
created: '2026-05-06T07:33:01.586857Z'
updated: '2026-05-06T07:35:46.512890Z'
source: https://arxiv.org/abs/2104.02180
source_domain: arxiv.org
fetched_at: '2026-05-06T07:33:01.586857Z'
fetch_provider: builtin
status: draft
type: note
tier: institutional
content_type: paper
deprecated: false
summary: 'Peng et al. 2021 (SIGGRAPH, arXiv 2104.02180). Original AMP paper for physics-based
  character animation (not robotics). Introduces the adversarial motion prior framework:
  discriminator distinguishes reference dataset transitions from policy transitions,
  provides style reward. Supports large, unstructured motion datasets without clip
  selection. Training is for simulated humanoid/character agents. No explicit robot
  deployment or hardware transfer. This is the foundational paper all legged-robot
  AMP papers build on. Abs page only — full training budget tables not visible.'
---

*Suggested by [[learning-natural-and-robust-hexapod-locomotion-over-complex-terrains-via-motion]] — Peng AMP paper is the foundation discriminator formulation used by Liu et al.*

*Suggested by [[learning-bipedal-walking-on-a-quadruped-robot-via-adversarial-motion-priors]] — Peng et al AMP original paper - foundational for all AMP work, cited by all 3 AMP papers in batch*

[2104.02180] AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control
Computer Science > Graphics
arXiv:2104.02180
(cs)
[Submitted on 5 Apr 2021 (
v1
), last revised 12 May 2022 (this version, v2)]
Title:
AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control
Authors:
Xue Bin Peng
,
Ze Ma
,
Pieter Abbeel
,
Sergey Levine
,
Angjoo Kanazawa
View a PDF of the paper titled AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control, by Xue Bin Peng and 4 other authors
View PDF
Abstract:
Synthesizing graceful and life-like behaviors for physically simulated characters has been a fundamental challenge in computer animation. Data-driven methods that leverage motion tracking are a prominent class of techniques for producing high fidelity motions for a wide range of behaviors. However, the effectiveness of these tracking-based methods often hinges on carefully designed objective functions, and when applied to large and diverse motion datasets, these methods require significant additional machinery to select the appropriate motion for the character to track in a given scenario. In this work, we propose to obviate the need to manually design imitation objectives and mechanisms for motion selection by utilizing a fully automated approach based on adversarial imitation learning. High-level task objectives that the character should perform can be specified by relatively simple reward functions, while the low-level style of the character's behaviors can be specified by a dataset of unstructured motion clips, without any explicit clip selection or sequencing. These motion clips are used to train an adversarial motion prior, which specifies style-rewards for training the character through reinforcement learning (RL). The adversarial RL procedure automatically selects which motion to perform, dynamically interpolating and generalizing from the dataset. Our system produces high-quality motions that are comparable to those achieved by state-of-the-art tracking-based techniques, while also being able to easily accommodate large datasets of unstructured motion clips. Composition of disparate skills emerges automatically from the motion prior, without requiring a high-level motion planner or other task-specific annotations of the motion clips. We demonstrate the effectiveness of our framework on a diverse cast of complex simulated characters and a challenging suite of motor control tasks.
Subjects:
Graphics (cs.GR)
; Machine Learning (cs.LG)
Cite as:
arXiv:2104.02180
[cs.GR]
(or
arXiv:2104.02180v2
[cs.GR]
for this version)
https://doi.org/10.48550/arXiv.2104.02180
Focus to learn more
arXiv-issued DOI via DataCite
Related DOI
:
https://doi.org/10.1145/3450626.3459670
Focus to learn more
DOI(s) linking to related resources
Submission history
From: Xue Bin Peng [
view email
]
[v1]
Mon, 5 Apr 2021 22:43:14 UTC (10,498 KB)
[v2]
Thu, 12 May 2022 04:38:30 UTC (10,498 KB)
Full-text links:
Access Paper:
View a PDF of the paper titled AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control, by Xue Bin Peng and 4 other authors
View PDF
TeX Source
view license
Current browse context:
cs.GR
< prev
|
next >
new
|
recent
|
2021-04
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
Xue Bin Peng
Ze Ma
Pieter Abbeel
Sergey Levine
Angjoo Kanazawa
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