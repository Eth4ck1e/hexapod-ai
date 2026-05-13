---
title: '[2201.08117] Learning robust perceptive locomotion for quadrupedal robots
  in the wild'
id: 220108117-learning-robust-perceptive-locomotion-for-quadrupedal-robots-in-the-wi
tags:
- legged-rl-budgets
created: '2026-05-06T07:39:38.054815Z'
updated: '2026-05-06T07:40:49.884105Z'
source: https://arxiv.org/abs/2201.08117
source_domain: arxiv.org
fetched_at: '2026-05-06T07:39:38.054815Z'
fetch_provider: builtin
status: deprecated
type: note
tier: institutional
content_type: paper
deprecated: false
summary: 'Miki et al. 2022 (ETH Zurich, Science Robotics) — Learning robust perceptive
  locomotion for ANYmal quadruped using attention-based recurrent encoder combining
  proprioception and exteroception (depth camera). Abstract only (753 words) — no
  budget details. The paper is cited in PACE (ETH 2025) as SOTA locomotion work requiring
  complex, high-dimensional (10+) reward functions. Completed a 1-hour hike in the
  Alps and led 4 ANYmals through 1700m of tunnel/urban/cave courses. Relevance: represents
  prior-generation reward-engineering approach (10+ hand-crafted terms) that PACE''s
  4-term reward aims to replace. No training budget accessible from abstract.'
---

*Suggested by [[towards-bridging-the-gap-systematic-sim-to-real-transfer-for-diverse-legged-robo]] — Miki 2022 cited as SOTA locomotion with complex reward terms (miki2022learning)*

[2201.08117] Learning robust perceptive locomotion for quadrupedal robots in the wild
Computer Science > Robotics
arXiv:2201.08117
(cs)
[Submitted on 20 Jan 2022]
Title:
Learning robust perceptive locomotion for quadrupedal robots in the wild
Authors:
Takahiro Miki
,
Joonho Lee
,
Jemin Hwangbo
,
Lorenz Wellhausen
,
Vladlen Koltun
,
Marco Hutter
View a PDF of the paper titled Learning robust perceptive locomotion for quadrupedal robots in the wild, by Takahiro Miki and 5 other authors
View PDF
Abstract:
Legged robots that can operate autonomously in remote and hazardous environments will greatly increase opportunities for exploration into under-explored areas. Exteroceptive perception is crucial for fast and energy-efficient locomotion: perceiving the terrain before making contact with it enables planning and adaptation of the gait ahead of time to maintain speed and stability. However, utilizing exteroceptive perception robustly for locomotion has remained a grand challenge in robotics. Snow, vegetation, and water visually appear as obstacles on which the robot cannot step~-- or are missing altogether due to high reflectance. Additionally, depth perception can degrade due to difficult lighting, dust, fog, reflective or transparent surfaces, sensor occlusion, and more. For this reason, the most robust and general solutions to legged locomotion to date rely solely on proprioception. This severely limits locomotion speed, because the robot has to physically feel out the terrain before adapting its gait accordingly. Here we present a robust and general solution to integrating exteroceptive and proprioceptive perception for legged locomotion. We leverage an attention-based recurrent encoder that integrates proprioceptive and exteroceptive input. The encoder is trained end-to-end and learns to seamlessly combine the different perception modalities without resorting to heuristics. The result is a legged locomotion controller with high robustness and speed. The controller was tested in a variety of challenging natural and urban environments over multiple seasons and completed an hour-long hike in the Alps in the time recommended for human hikers.
Subjects:
Robotics (cs.RO)
Cite as:
arXiv:2201.08117
[cs.RO]
(or
arXiv:2201.08117v1
[cs.RO]
for this version)
https://doi.org/10.48550/arXiv.2201.08117
Focus to learn more
arXiv-issued DOI via DataCite
Journal reference:
Science Robotics, 19 Jan 2022, Vol 7, Issue 62
Related DOI
:
https://doi.org/10.1126/scirobotics.abk2822
Focus to learn more
DOI(s) linking to related resources
Submission history
From: Takahiro Miki [
view email
]
[v1]
Thu, 20 Jan 2022 11:27:47 UTC (32,344 KB)
Full-text links:
Access Paper:
View a PDF of the paper titled Learning robust perceptive locomotion for quadrupedal robots in the wild, by Takahiro Miki and 5 other authors
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
2022-01
Change to browse by:
cs
References & Citations
NASA ADS
Google Scholar
Semantic Scholar
DBLP
- CS Bibliography
listing
|
bibtex
Takahiro Miki
Joonho Lee
Jemin Hwangbo
Lorenz Wellhausen
Vladlen Koltun
…
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