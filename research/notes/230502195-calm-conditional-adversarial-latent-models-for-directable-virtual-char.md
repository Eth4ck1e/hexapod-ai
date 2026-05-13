---
title: '[2305.02195] CALM: Conditional Adversarial Latent Models for Directable Virtual
  Characters'
id: 230502195-calm-conditional-adversarial-latent-models-for-directable-virtual-char
tags:
- amp
- skill-conditional
- character-animation
created: '2026-05-10T18:34:34.330987Z'
updated: '2026-05-10T18:35:26.290406Z'
source: https://arxiv.org/abs/2305.02195
source_domain: arxiv.org
fetched_at: '2026-05-10T18:34:34.330987Z'
fetch_provider: builtin
status: review
type: note
tier: institutional
content_type: paper
deprecated: false
summary: 'CALM (Tessler, Peng et al. SIGGRAPH 2023) jointly trains a control policy,
  an AMP-style discriminator, and a MOTION ENCODER E(s_t, s_t+1) -> z. The encoder
  maps reference clips to a latent skill space; the policy is conditioned on z; the
  disc gets z as input AND positive samples are paired with their encoded z. Result:
  the latent z is semantic and directable — at runtime the user picks a clip, encodes
  it, and the policy reproduces that style. Mechanism vs vanilla AMP: still uses cmd-as-disc-input
  (z is encoder output, not raw cmd), no prior-batch filtering by cmd similarity.
  ASE-family with a learned encoder for direct controllability.'
---

[2305.02195] CALM: Conditional Adversarial Latent Models for Directable Virtual Characters
Computer Science > Computer Vision and Pattern Recognition
arXiv:2305.02195
(cs)
[Submitted on 2 May 2023]
Title:
CALM: Conditional Adversarial Latent Models for Directable Virtual Characters
Authors:
Chen Tessler
,
Yoni Kasten
,
Yunrong Guo
,
Shie Mannor
,
Gal Chechik
,
Xue Bin Peng
View a PDF of the paper titled CALM: Conditional Adversarial Latent Models for Directable Virtual Characters, by Chen Tessler and 5 other authors
View PDF
Abstract:
In this work, we present Conditional Adversarial Latent Models (CALM), an approach for generating diverse and directable behaviors for user-controlled interactive virtual characters. Using imitation learning, CALM learns a representation of movement that captures the complexity and diversity of human motion, and enables direct control over character movements. The approach jointly learns a control policy and a motion encoder that reconstructs key characteristics of a given motion without merely replicating it. The results show that CALM learns a semantic motion representation, enabling control over the generated motions and style-conditioning for higher-level task training. Once trained, the character can be controlled using intuitive interfaces, akin to those found in video games.
Comments:
Accepted to SIGGRAPH 2023
Subjects:
Computer Vision and Pattern Recognition (cs.CV)
; Artificial Intelligence (cs.AI); Robotics (cs.RO)
Cite as:
arXiv:2305.02195
[cs.CV]
(or
arXiv:2305.02195v1
[cs.CV]
for this version)
https://doi.org/10.48550/arXiv.2305.02195
Focus to learn more
arXiv-issued DOI via DataCite
Related DOI
:
https://doi.org/10.1145/3588432.3591541
Focus to learn more
DOI(s) linking to related resources
Submission history
From: Chen Tessler [
view email
]
[v1]
Tue, 2 May 2023 09:01:44 UTC (37,487 KB)
Full-text links:
Access Paper:
View a PDF of the paper titled CALM: Conditional Adversarial Latent Models for Directable Virtual Characters, by Chen Tessler and 5 other authors
View PDF
TeX Source
view license
Current browse context:
cs.CV
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