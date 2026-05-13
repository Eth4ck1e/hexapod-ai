---
title: '[2309.11351] C$\cdot$ASE: Learning Conditional Adversarial Skill Embeddings
  for Physics-based Characters'
id: 230911351-ccdotase-learning-conditional-adversarial-skill-embeddings-for-physics
tags:
- amp
- skill-conditional
- character-animation
created: '2026-05-10T18:33:38.807020Z'
updated: '2026-05-10T18:35:02.862408Z'
source: https://arxiv.org/abs/2309.11351
source_domain: arxiv.org
fetched_at: '2026-05-10T18:33:38.807020Z'
fetch_provider: builtin
status: review
type: note
tier: institutional
content_type: paper
deprecated: false
summary: 'CASE (Dou et al. SIGGRAPH Asia 2023) extends ASE for HETEROGENEOUS skill
  datasets. Key mechanism: explicit prior-data PARTITIONING — the heterogeneous mocap
  is divided into homogeneous subsets, and a discrete one-hot skill code z conditions
  both the policy and the discriminator. Crucially, during training the disc only
  sees prior samples DRAWN FROM THE SAME SUBSET as the current z — closest paper to
  data-level prior filtering by command. Adds focal skill sampling, residual forces,
  feature masking. Closest precursor to what we want for cmd-conditional disc filtering.'
---

[2309.11351] C$\cdot$ASE: Learning Conditional Adversarial Skill Embeddings for Physics-based Characters
Computer Science > Graphics
arXiv:2309.11351
(cs)
[Submitted on 20 Sep 2023]
Title:
C$\cdot$ASE: Learning Conditional Adversarial Skill Embeddings for Physics-based Characters
Authors:
Zhiyang Dou
,
Xuelin Chen
,
Qingnan Fan
,
Taku Komura
,
Wenping Wang
View a PDF of the paper titled C$\cdot$ASE: Learning Conditional Adversarial Skill Embeddings for Physics-based Characters, by Zhiyang Dou and 4 other authors
View PDF
Abstract:
We present C$\cdot$ASE, an efficient and effective framework that learns conditional Adversarial Skill Embeddings for physics-based characters. Our physically simulated character can learn a diverse repertoire of skills while providing controllability in the form of direct manipulation of the skills to be performed. C$\cdot$ASE divides the heterogeneous skill motions into distinct subsets containing homogeneous samples for training a low-level conditional model to learn conditional behavior distribution. The skill-conditioned imitation learning naturally offers explicit control over the character's skills after training. The training course incorporates the focal skill sampling, skeletal residual forces, and element-wise feature masking to balance diverse skills of varying complexities, mitigate dynamics mismatch to master agile motions and capture more general behavior characteristics, respectively. Once trained, the conditional model can produce highly diverse and realistic skills, outperforming state-of-the-art models, and can be repurposed in various downstream tasks. In particular, the explicit skill control handle allows a high-level policy or user to direct the character with desired skill specifications, which we demonstrate is advantageous for interactive character animation.
Comments:
SIGGRAPH Asia 2023
Subjects:
Graphics (cs.GR)
; Artificial Intelligence (cs.AI); Machine Learning (cs.LG)
Cite as:
arXiv:2309.11351
[cs.GR]
(or
arXiv:2309.11351v1
[cs.GR]
for this version)
https://doi.org/10.48550/arXiv.2309.11351
Focus to learn more
arXiv-issued DOI via DataCite
Related DOI
:
https://doi.org/10.1145/3610548.3618205
Focus to learn more
DOI(s) linking to related resources
Submission history
From: Zhiyang Dou [
view email
]
[v1]
Wed, 20 Sep 2023 14:34:45 UTC (10,032 KB)
Full-text links:
Access Paper:
View a PDF of the paper titled C$\cdot$ASE: Learning Conditional Adversarial Skill Embeddings for Physics-based Characters, by Zhiyang Dou and 4 other authors
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
2023-09
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