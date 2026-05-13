---
title: '[2305.14654] Barkour: Benchmarking Animal-level Agility with Quadruped Robots'
id: 230514654-barkour-benchmarking-animal-level-agility-with-quadruped-robots
tags:
- legged-rl-budgets
- quadruped
- training-budget
created: '2026-05-06T07:36:36.619269Z'
updated: '2026-05-06T07:38:45.025354Z'
source: https://arxiv.org/abs/2305.14654
source_domain: arxiv.org
fetched_at: '2026-05-06T07:36:36.618269Z'
fetch_provider: builtin
status: draft
type: note
tier: institutional
content_type: paper
deprecated: false
summary: 'Caluwaerts et al. (Google, May 2023, arXiv 2305.14654): Barkour benchmark
  for quadruped agility. Uses on-policy RL to train specialist locomotion skills and
  distills into Locomotion-Transformer generalist. Abstract page only — no explicit
  training budget numbers visible. Relevant as the foundational Brax/MJX quadruped
  reference cited by Thibault et al. and MuJoCo Playground.'
---

*Suggested by [[learning-velocity-based-humanoid-locomotion-massively-parallel-learning-with-bra]] — Barkour paper cited as Brax/MJX quadruped training reference*

[2305.14654] Barkour: Benchmarking Animal-level Agility with Quadruped Robots
Computer Science > Robotics
arXiv:2305.14654
(cs)
[Submitted on 24 May 2023]
Title:
Barkour: Benchmarking Animal-level Agility with Quadruped Robots
Authors:
Ken Caluwaerts
,
Atil Iscen
,
J. Chase Kew
,
Wenhao Yu
,
Tingnan Zhang
,
Daniel Freeman
,
Kuang-Huei Lee
,
Lisa Lee
,
Stefano Saliceti
,
Vincent Zhuang
,
Nathan Batchelor
,
Steven Bohez
,
Federico Casarini
,
Jose Enrique Chen
,
Omar Cortes
,
Erwin Coumans
,
Adil Dostmohamed
,
Gabriel Dulac-Arnold
,
Alejandro Escontrela
,
Erik Frey
,
Roland Hafner
,
Deepali Jain
,
Bauyrjan Jyenis
,
Yuheng Kuang
,
Edward Lee
,
Linda Luu
,
Ofir Nachum
,
Ken Oslund
,
Jason Powell
,
Diego Reyes
,
Francesco Romano
,
Feresteh Sadeghi
,
Ron Sloat
,
Baruch Tabanpour
,
Daniel Zheng
,
Michael Neunert
,
Raia Hadsell
,
Nicolas Heess
,
Francesco Nori
,
Jeff Seto
,
Carolina Parada
,
Vikas Sindhwani
,
Vincent Vanhoucke
,
Jie Tan
View a PDF of the paper titled Barkour: Benchmarking Animal-level Agility with Quadruped Robots, by Ken Caluwaerts and 43 other authors
View PDF
Abstract:
Animals have evolved various agile locomotion strategies, such as sprinting, leaping, and jumping. There is a growing interest in developing legged robots that move like their biological counterparts and show various agile skills to navigate complex environments quickly. Despite the interest, the field lacks systematic benchmarks to measure the performance of control policies and hardware in agility. We introduce the Barkour benchmark, an obstacle course to quantify agility for legged robots. Inspired by dog agility competitions, it consists of diverse obstacles and a time based scoring mechanism. This encourages researchers to develop controllers that not only move fast, but do so in a controllable and versatile way. To set strong baselines, we present two methods for tackling the benchmark. In the first approach, we train specialist locomotion skills using on-policy reinforcement learning methods and combine them with a high-level navigation controller. In the second approach, we distill the specialist skills into a Transformer-based generalist locomotion policy, named Locomotion-Transformer, that can handle various terrains and adjust the robot's gait based on the perceived environment and robot states. Using a custom-built quadruped robot, we demonstrate that our method can complete the course at half the speed of a dog. We hope that our work represents a step towards creating controllers that enable robots to reach animal-level agility.
Comments:
17 pages, 19 figures
Subjects:
Robotics (cs.RO)
; Artificial Intelligence (cs.AI)
Cite as:
arXiv:2305.14654
[cs.RO]
(or
arXiv:2305.14654v1
[cs.RO]
for this version)
https://doi.org/10.48550/arXiv.2305.14654
Focus to learn more
arXiv-issued DOI via DataCite
Submission history
From: Atil Iscen [
view email
]
[v1]
Wed, 24 May 2023 02:49:43 UTC (10,159 KB)
Full-text links:
Access Paper:
View a PDF of the paper titled Barkour: Benchmarking Animal-level Agility with Quadruped Robots, by Ken Caluwaerts and 43 other authors
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
2023-05
Change to browse by:
cs
cs.AI
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