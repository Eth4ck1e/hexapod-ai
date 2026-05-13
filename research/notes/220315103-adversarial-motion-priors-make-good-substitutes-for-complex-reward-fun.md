---
title: '[2203.15103] Adversarial Motion Priors Make Good Substitutes for Complex Reward
  Functions'
id: 220315103-adversarial-motion-priors-make-good-substitutes-for-complex-reward-fun
tags:
- legged-rl-budgets
- amp
- quadruped
- iros2022
created: '2026-05-06T07:30:43.366723Z'
updated: '2026-05-06T07:34:41.365565Z'
source: https://arxiv.org/abs/2203.15103
source_domain: arxiv.org
fetched_at: '2026-05-06T07:30:43.366723Z'
fetch_provider: builtin
status: draft
type: note
tier: institutional
content_type: paper
deprecated: false
summary: Abstract/abs page for Escontrela et al. IROS 2022 'Adversarial Motion Priors
  Make Good Substitutes for Complex Reward Functions' — the foundational AMP-for-hardware
  paper. Proposes substituting complex reward functions with GAN-based style rewards
  learned from a few seconds of motion capture data (German Shepherd). Demonstrates
  transfer to a real quadrupedal robot without complex reward engineering. 8 pages,
  6 figures, 3 tables. Training budget details not on abs page — requires PDF.
---

[2203.15103] Adversarial Motion Priors Make Good Substitutes for Complex Reward Functions
Computer Science > Artificial Intelligence
arXiv:2203.15103
(cs)
[Submitted on 28 Mar 2022]
Title:
Adversarial Motion Priors Make Good Substitutes for Complex Reward Functions
Authors:
Alejandro Escontrela
,
Xue Bin Peng
,
Wenhao Yu
,
Tingnan Zhang
,
Atil Iscen
,
Ken Goldberg
,
Pieter Abbeel
View a PDF of the paper titled Adversarial Motion Priors Make Good Substitutes for Complex Reward Functions, by Alejandro Escontrela and 6 other authors
View PDF
Abstract:
Training a high-dimensional simulated agent with an under-specified reward function often leads the agent to learn physically infeasible strategies that are ineffective when deployed in the real world. To mitigate these unnatural behaviors, reinforcement learning practitioners often utilize complex reward functions that encourage physically plausible behaviors. However, a tedious labor-intensive tuning process is often required to create hand-designed rewards which might not easily generalize across platforms and tasks. We propose substituting complex reward functions with "style rewards" learned from a dataset of motion capture demonstrations. A learned style reward can be combined with an arbitrary task reward to train policies that perform tasks using naturalistic strategies. These natural strategies can also facilitate transfer to the real world. We build upon Adversarial Motion Priors -- an approach from the computer graphics domain that encodes a style reward from a dataset of reference motions -- to demonstrate that an adversarial approach to training policies can produce behaviors that transfer to a real quadrupedal robot without requiring complex reward functions. We also demonstrate that an effective style reward can be learned from a few seconds of motion capture data gathered from a German Shepherd and leads to energy-efficient locomotion strategies with natural gait transitions.
Comments:
8 pages, 6 figures, 3 tables
Subjects:
Artificial Intelligence (cs.AI)
; Robotics (cs.RO)
Cite as:
arXiv:2203.15103
[cs.AI]
(or
arXiv:2203.15103v1
[cs.AI]
for this version)
https://doi.org/10.48550/arXiv.2203.15103
Focus to learn more
arXiv-issued DOI via DataCite
Submission history
From: Alejandro Escontrela [
view email
]
[v1]
Mon, 28 Mar 2022 21:17:36 UTC (40,579 KB)
Full-text links:
Access Paper:
View a PDF of the paper titled Adversarial Motion Priors Make Good Substitutes for Complex Reward Functions, by Alejandro Escontrela and 6 other authors
View PDF
TeX Source
view license
Current browse context:
cs.AI
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
cs.RO
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