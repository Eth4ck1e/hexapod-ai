---
title: '[2305.03286] Composite Motion Learning with Task Control'
id: 230503286-composite-motion-learning-with-task-control
tags:
- amp
- multi-discriminator
- character-animation
created: '2026-05-10T18:33:39.952533Z'
updated: '2026-05-10T18:35:08.948663Z'
source: https://arxiv.org/abs/2305.03286
source_domain: arxiv.org
fetched_at: '2026-05-10T18:33:39.952533Z'
fetch_provider: builtin
status: review
type: note
tier: institutional
content_type: paper
deprecated: false
summary: 'Composite Motion Learning (Xu, Shang, Zordan, Karamouzas; SIGGRAPH 2023)
  uses MULTIPLE discriminators in a GAN-like setup, one per body-part / motion source.
  Each discriminator independently scores a subset of the body''s motion against its
  own reference subset; a multi-objective adaptive weighting balances disc rewards
  across sources + task rewards. Mechanism vs vanilla AMP: keeps one policy but partitions
  DISCRIMINATORS not data, letting the policy compose decoupled per-part references
  (e.g. lower-body run + upper-body wave) without manual composite mocap. Cmd-conditioning
  happens via per-part disc gating, not per-sample filter.'
---

[2305.03286] Composite Motion Learning with Task Control
Computer Science > Graphics
arXiv:2305.03286
(cs)
[Submitted on 5 May 2023]
Title:
Composite Motion Learning with Task Control
Authors:
Pei Xu
,
Xiumin Shang
,
Victor Zordan
,
Ioannis Karamouzas
View a PDF of the paper titled Composite Motion Learning with Task Control, by Pei Xu and 3 other authors
View PDF
Abstract:
We present a deep learning method for composite and task-driven motion control for physically simulated characters. In contrast to existing data-driven approaches using reinforcement learning that imitate full-body motions, we learn decoupled motions for specific body parts from multiple reference motions simultaneously and directly by leveraging the use of multiple discriminators in a GAN-like setup. In this process, there is no need of any manual work to produce composite reference motions for learning. Instead, the control policy explores by itself how the composite motions can be combined automatically. We further account for multiple task-specific rewards and train a single, multi-objective control policy. To this end, we propose a novel framework for multi-objective learning that adaptively balances the learning of disparate motions from multiple sources and multiple goal-directed control objectives. In addition, as composite motions are typically augmentations of simpler behaviors, we introduce a sample-efficient method for training composite control policies in an incremental manner, where we reuse a pre-trained policy as the meta policy and train a cooperative policy that adapts the meta one for new composite tasks. We show the applicability of our approach on a variety of challenging multi-objective tasks involving both composite motion imitation and multiple goal-directed control.
Comments:
SIGGRAPH 2023. Code:
this https URL
. Video:
this https URL
Subjects:
Graphics (cs.GR)
; Artificial Intelligence (cs.AI); Machine Learning (cs.LG)
Cite as:
arXiv:2305.03286
[cs.GR]
(or
arXiv:2305.03286v1
[cs.GR]
for this version)
https://doi.org/10.48550/arXiv.2305.03286
Focus to learn more
arXiv-issued DOI via DataCite
Journal reference:
ACM Transactions on Graphics (August 2023)
Related DOI
:
https://doi.org/10.1145/3592447
Focus to learn more
DOI(s) linking to related resources
Submission history
From: Pei Xu [
view email
]
[v1]
Fri, 5 May 2023 05:02:41 UTC (22,831 KB)
Full-text links:
Access Paper:
View a PDF of the paper titled Composite Motion Learning with Task Control, by Pei Xu and 3 other authors
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
2023-05
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