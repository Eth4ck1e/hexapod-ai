---
title: '[1901.08652] Learning agile and dynamic motor skills for legged robots'
id: 190108652-learning-agile-and-dynamic-motor-skills-for-legged-robots
tags:
- legged-rl-budgets
- canonical-anchor
created: '2026-05-06T07:30:38.559581Z'
updated: '2026-05-06T07:56:23.585982Z'
source: https://arxiv.org/abs/1901.08652
source_domain: arxiv.org
fetched_at: '2026-05-06T07:30:38.558580Z'
fetch_provider: builtin
status: draft
type: note
tier: institutional
content_type: paper
deprecated: false
summary: Hwangbo et al. 2019 (Science Robotics) - abstract/arxiv index page only;
  full text in -2 ar5iv note. ANYmal quadruped, TRPO-based, single CPU+GPU desktop,
  trained ~250M steps (quarter of a billion transitions) in ~4 hours for locomotion,
  ~11 hours for fall recovery. Simulator generates ~900k timesteps/sec. 30 randomized
  ANYmal models for domain rand. Actuator network trained via supervised learning.
  No Isaac Gym - custom C++ simulator.
---

[1901.08652] Learning agile and dynamic motor skills for legged robots
Computer Science > Robotics
arXiv:1901.08652
(cs)
[Submitted on 24 Jan 2019]
Title:
Learning agile and dynamic motor skills for legged robots
Authors:
Jemin Hwangbo
,
Joonho Lee
,
Alexey Dosovitskiy
,
Dario Bellicoso
,
Vassilios Tsounis
,
Vladlen Koltun
,
Marco Hutter
View a PDF of the paper titled Learning agile and dynamic motor skills for legged robots, by Jemin Hwangbo and 6 other authors
View PDF
Abstract:
Legged robots pose one of the greatest challenges in robotics. Dynamic and agile maneuvers of animals cannot be imitated by existing methods that are crafted by humans. A compelling alternative is reinforcement learning, which requires minimal craftsmanship and promotes the natural evolution of a control policy. However, so far, reinforcement learning research for legged robots is mainly limited to simulation, and only few and comparably simple examples have been deployed on real systems. The primary reason is that training with real robots, particularly with dynamically balancing systems, is complicated and expensive. In the present work, we introduce a method for training a neural network policy in simulation and transferring it to a state-of-the-art legged system, thereby leveraging fast, automated, and cost-effective data generation schemes. The approach is applied to the ANYmal robot, a sophisticated medium-dog-sized quadrupedal system. Using policies trained in simulation, the quadrupedal machine achieves locomotion skills that go beyond what had been achieved with prior methods: ANYmal is capable of precisely and energy-efficiently following high-level body velocity commands, running faster than before, and recovering from falling even in complex configurations.
Subjects:
Robotics (cs.RO)
; Machine Learning (cs.LG); Machine Learning (stat.ML)
Cite as:
arXiv:1901.08652
[cs.RO]
(or
arXiv:1901.08652v1
[cs.RO]
for this version)
https://doi.org/10.48550/arXiv.1901.08652
Focus to learn more
arXiv-issued DOI via DataCite
Journal reference:
Science Robotics 4.26 (2019): eaau5872
Related DOI
:
https://doi.org/10.1126/scirobotics.aau5872
Focus to learn more
DOI(s) linking to related resources
Submission history
From: Jemin Hwangbo [
view email
]
[v1]
Thu, 24 Jan 2019 21:50:29 UTC (7,457 KB)
Full-text links:
Access Paper:
View a PDF of the paper titled Learning agile and dynamic motor skills for legged robots, by Jemin Hwangbo and 6 other authors
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
2019-01
Change to browse by:
cs
cs.LG
stat
stat.ML
References & Citations
NASA ADS
Google Scholar
Semantic Scholar
DBLP
- CS Bibliography
listing
|
bibtex
Jemin Hwangbo
Joonho Lee
Alexey Dosovitskiy
Dario Bellicoso
Vassilios Tsounis
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