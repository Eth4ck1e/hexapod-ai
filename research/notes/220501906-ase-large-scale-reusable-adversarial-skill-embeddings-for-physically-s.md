---
title: '[2205.01906] ASE: Large-Scale Reusable Adversarial Skill Embeddings for Physically
  Simulated Characters'
id: 220501906-ase-large-scale-reusable-adversarial-skill-embeddings-for-physically-s
tags:
- amp
- skill-conditional
- character-animation
created: '2026-05-10T18:33:37.575357Z'
updated: '2026-05-10T18:34:57.130586Z'
source: https://arxiv.org/abs/2205.01906
source_domain: arxiv.org
fetched_at: '2026-05-10T18:33:37.574357Z'
fetch_provider: builtin
status: review
type: note
tier: institutional
content_type: paper
deprecated: false
summary: 'ASE (Peng et al. SIGGRAPH 2022) trains a continuous LATENT skill embedding
  z (sampled from a prior distribution) jointly with the policy and an AMP-style discriminator.
  The discriminator is conditioned on (s_t, s_t+1, z); positive prior samples are
  paired with their associated z via an encoder, so the disc enforces that motion
  produced under z matches the prior subset that maps to z. Mechanism vs vanilla AMP:
  same disc loss but with a learned skill latent as extra disc input — UNSUPERVISED
  skill clustering, no explicit cmd or class label.'
---

[2205.01906] ASE: Large-Scale Reusable Adversarial Skill Embeddings for Physically Simulated Characters
Computer Science > Graphics
arXiv:2205.01906
(cs)
[Submitted on 4 May 2022 (
v1
), last revised 5 May 2022 (this version, v2)]
Title:
ASE: Large-Scale Reusable Adversarial Skill Embeddings for Physically Simulated Characters
Authors:
Xue Bin Peng
,
Yunrong Guo
,
Lina Halper
,
Sergey Levine
,
Sanja Fidler
View a PDF of the paper titled ASE: Large-Scale Reusable Adversarial Skill Embeddings for Physically Simulated Characters, by Xue Bin Peng and 4 other authors
View PDF
Abstract:
The incredible feats of athleticism demonstrated by humans are made possible in part by a vast repertoire of general-purpose motor skills, acquired through years of practice and experience. These skills not only enable humans to perform complex tasks, but also provide powerful priors for guiding their behaviors when learning new tasks. This is in stark contrast to what is common practice in physics-based character animation, where control policies are most typically trained from scratch for each task. In this work, we present a large-scale data-driven framework for learning versatile and reusable skill embeddings for physically simulated characters. Our approach combines techniques from adversarial imitation learning and unsupervised reinforcement learning to develop skill embeddings that produce life-like behaviors, while also providing an easy to control representation for use on new downstream tasks. Our models can be trained using large datasets of unstructured motion clips, without requiring any task-specific annotation or segmentation of the motion data. By leveraging a massively parallel GPU-based simulator, we are able to train skill embeddings using over a decade of simulated experiences, enabling our model to learn a rich and versatile repertoire of skills. We show that a single pre-trained model can be effectively applied to perform a diverse set of new tasks. Our system also allows users to specify tasks through simple reward functions, and the skill embedding then enables the character to automatically synthesize complex and naturalistic strategies in order to achieve the task objectives.
Subjects:
Graphics (cs.GR)
; Artificial Intelligence (cs.AI); Machine Learning (cs.LG)
Cite as:
arXiv:2205.01906
[cs.GR]
(or
arXiv:2205.01906v2
[cs.GR]
for this version)
https://doi.org/10.48550/arXiv.2205.01906
Focus to learn more
arXiv-issued DOI via DataCite
Related DOI
:
https://doi.org/10.1145/3528223.3530110
Focus to learn more
DOI(s) linking to related resources
Submission history
From: Xue Bin Peng [
view email
]
[v1]
Wed, 4 May 2022 06:13:28 UTC (23,462 KB)
[v2]
Thu, 5 May 2022 17:25:14 UTC (24,252 KB)
Full-text links:
Access Paper:
View a PDF of the paper titled ASE: Large-Scale Reusable Adversarial Skill Embeddings for Physically Simulated Characters, by Xue Bin Peng and 4 other authors
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
2022-05
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