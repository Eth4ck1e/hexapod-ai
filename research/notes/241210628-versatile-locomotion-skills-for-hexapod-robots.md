---
title: '[2412.10628] Versatile Locomotion Skills for Hexapod Robots'
id: 241210628-versatile-locomotion-skills-for-hexapod-robots
tags:
- legged-rl-budgets
- hexapod
- training-budget
created: '2026-05-06T07:30:26.314384Z'
updated: '2026-05-06T07:39:19.944650Z'
source: https://arxiv.org/abs/2412.10628
source_domain: arxiv.org
fetched_at: '2026-05-06T07:30:26.313383Z'
fetch_provider: builtin
status: draft
type: note
tier: institutional
content_type: paper
deprecated: false
summary: ArXiv abs for Berkeley/DeepMind Versatile Hexapod (2412.10628). Full content
  in companion HTML note versatile-locomotion-skills-for-hexapod-robots. IROS 2024.
---

[2412.10628] Versatile Locomotion Skills for Hexapod Robots
Computer Science > Robotics
arXiv:2412.10628
(cs)
[Submitted on 14 Dec 2024]
Title:
Versatile Locomotion Skills for Hexapod Robots
Authors:
Tomson Qu
,
Dichen Li
,
Avideh Zakhor
,
Wenhao Yu
,
Tingnan Zhang
View a PDF of the paper titled Versatile Locomotion Skills for Hexapod Robots, by Tomson Qu and 4 other authors
View PDF
HTML (experimental)
Abstract:
Hexapod robots are potentially suitable for carrying out tasks in cluttered environments since they are stable, compact, and light weight. They also have multi-joint legs and variable height bodies that make them good candidates for tasks such as stairs climbing and squeezing under objects in a typical home environment or an attic. Expanding on our previous work on joist climbing in attics, we train a legged hexapod equipped with a depth camera and visual inertial odometry (VIO) to perform three tasks: climbing stairs, avoiding obstacles, and squeezing under obstacles such as a table. Our policies are trained with simulation data only and can be deployed on lowcost hardware not requiring real-time joint state feedback. We train our model in a teacher-student model with 2 phases: In phase 1, we use reinforcement learning with access to privileged information such as height maps and joint feedback. In phase 2, we use supervised learning to distill the model into one with access to only onboard observations, consisting of egocentric depth images and robot pose captured by a tracking VIO camera. By manipulating available privileged information, constructing simulation terrains, and refining reward functions during phase 1 training, we are able to train the robots with skills that are robust in non-ideal physical environments. We demonstrate successful sim-to-real transfer and achieve high success rates across all three tasks in physical experiments.
Subjects:
Robotics (cs.RO)
Cite as:
arXiv:2412.10628
[cs.RO]
(or
arXiv:2412.10628v1
[cs.RO]
for this version)
https://doi.org/10.48550/arXiv.2412.10628
Focus to learn more
arXiv-issued DOI via DataCite
Submission history
From: Tomson Qu [
view email
]
[v1]
Sat, 14 Dec 2024 00:40:13 UTC (24,064 KB)
Full-text links:
Access Paper:
View a PDF of the paper titled Versatile Locomotion Skills for Hexapod Robots, by Tomson Qu and 4 other authors
View PDF
HTML (experimental)
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
2024-12
Change to browse by:
cs
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