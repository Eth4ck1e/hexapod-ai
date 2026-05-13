---
title: 'Posterior Behavioral Cloning: Pretraining BC Policies for Efficient RL Finetuning'
id: posterior-behavioral-cloning-pretraining-bc-policies-for-efficient-rl-finetuning
tags:
- legged-rl-budgets
- bc-pretrain
- rl-finetuning
- manipulation
- sample-efficiency
created: '2026-05-06T07:30:49.627280Z'
updated: '2026-05-06T07:35:41.560583Z'
source: https://arxiv.org/html/2512.16911v1
source_domain: arxiv.org
fetched_at: '2026-05-06T07:30:49.626280Z'
fetch_provider: builtin
status: draft
type: note
tier: institutional
content_type: paper
deprecated: false
summary: 'Berkeley/Stanford 2025 paper proposing Posterior Behavioral Cloning (PostBC),
  a pretraining method designed to improve RL finetuning efficiency. Standard BC can
  overcommit to observed demonstrations and fail to cover the demonstrator''s action
  distribution in low-data-density regions, crippling downstream RL by starving it
  of meaningful reward signal. PostBC instead trains a diffusion-model policy to fit
  the *posterior* distribution of demonstrator behavior, preserving high-confidence
  imitation in well-covered regions while maintaining higher entropy elsewhere. Theoretical
  guarantees: PostBC covers the demonstrator''s action distribution (BC provably can
  fail this), incurs no pretrained performance penalty vs BC, and achieves near-optimal
  sampling cost. Experiments on Robomimic (Lift/Can/Square) and Libero use DSRL (diffusion-based
  RL) for finetuning, with 4 parallel environments. Total gradient steps for RL finetuning:
  2,000,000. BC pretraining dataset sizes: 5–30 trajectories (Robomimic) to give room
  for RL to improve. NOT a locomotion paper — focused on manipulation tasks. No locomotion,
  hexapod, or quadruped content. No GPU type reported, no wall-clock times. Relevant
  to the pipeline as a theory of why PostBC initialization improves RL finetuning
  sample efficiency, but specific step budgets are manipulation-task-sized.'
---

Posterior Behavioral Cloning: Pretraining BC Policies for Efficient RL Finetuning
\newtoggle
arxiv
\toggletrue
arxiv
Posterior Behavioral Cloning: Pretraining BC Policies for Efficient RL Finetuning
Andrew Wagenmaker
UC Berkeley
&Perry Dong
Stanford
&Raymond Tsao
UC Berkeley
&Chelsea Finn
Stanford
&Sergey Levine
UC Berkeley
Correspondance to:
ajwagen@berkeley.edu
.
Abstract
Standard practice across domains from robotics to language is to first pretrain a policy on a large-scale demonstration dataset, and then finetune this policy, typically with reinforcement learning (RL), in order to improve performance on deployment domains. This finetuning step has proved critical in achieving human or super-human performance, yet while much attention has been given to developing more effective finetuning algorithms, little attention has been given to ensuring the pretrained policy is an effective initialization for RL finetuning. In this work we seek to understand how the pretrained policy affects finetuning performance, and how to pretrain policies in order to ensure they are effective initializations for finetuning. We first show theoretically that standard behavioral cloning (
Bc
)—which trains a policy to directly match the actions played by the demonstrator—can fail to ensure coverage over the demonstrator’s actions, a minimal condition necessary for effective RL finetuning.
We then show that if, instead of exactly fitting the observed demonstrations, we train a policy to model the
posterior
distribution of the demonstrator’s behavior given the demonstration dataset, we
do
obtain a policy that ensures coverage over the demonstrator’s actions, enabling more effective finetuning. Furthermore, this policy—which we refer to as the
posterior behavioral cloning
(
PostBc
) policy—achieves this while ensuring pretrained performance is no worse than that of the
Bc
policy.
We then show that
PostBc
is practically implementable with modern generative models in robotic control domains—relying only on standard supervised learning—and leads to significantly improved RL finetuning performance on both realistic robotic control benchmarks and real-world robotic manipulation tasks, as compared to standard behavioral cloning.
1
Introduction
Across domains—from language, to vision, to robotics—a common paradigm has emerged for training highly effective “policies”: collect a large set of demonstrations, “pretrain” a policy via behavioral cloning (
Bc
) to mimic these demonstrations, then “finetune” the pretrained policy on a deployment domain of interest. While pretraining can endow the policy with generally useful abilities, the finetuning step has proved critical in obtaining effective performance, enabling human value alignment and reasoning capabilities in language domains
(Ouyang
et al.
,
2022
; Bai
et al.
,
2022
; Team
et al.
,
2025
; Guo
et al.
,
2025
)
, and improving task solving precision and generalization to unseen tasks in robotic domains
(Nakamoto
et al.
,
2024
; Chen
et al.
,
2025
; Kim
et al.
,
2025
; Wagenmaker
et al.
,
2025
)
. In particular, reinforcement learning (RL)-based finetuning—where the pretrained policy is deployed in a setting of interest and its behavior updated based on the outcomes of these online rollouts—is especially crucial in improving the performance of a pretrained policy.
Critical to achieving successful RL-based finetuning performance in many domains—particularly in settings when policy deployment is costly and time-consuming, such as robotic control—is sample efficiency; effectively modifying the behavior of the pretrained model using as few deployment rollouts as possible.
While significant attention has been given to developing more efficient finetuning algorithms, this ignores a primary ingredient in the RL finetuning process: the pretrained policy itself. Though generally more effective pretrained policies are the preferred initialization for finetuning
(Guo
et al.
,
2025
; Yue
et al.
,
2025
)
,
it is not well understood how pretraining impacts finetuning performance beyond this, and how we might pretrain policies to enable more efficient RL finetuning.
Figure 1
:
(a) We consider the setting where we are given demonstration data for some tasks of interest. (b) Standard
Bc
pretraining fits the behaviors in the demonstrations, leading to effective performance in regions with high demonstration data density, yet can overcommit to the observed behaviors in regions with low data density. (c) This leads to ineffective RL finetuning, since rollouts from the
Bc
policy provide little meaningful reward signal in such low data density regions, which is typically necessary to enable effective improvement. (d) In contrast, we propose
posterior behavioral cloning
(
PostBc
), which instead of directly mimicking the demonstrations, trains a generative policy to fit the
posterior distribution
of the demonstrator’s behavior. This endows the pretrained policy with a wider distribution of actions in regions of low demonstrator data density, while in regions of high data density it reduces to approximately the standard
Bc
policy. (e) This wider action distribution in low data density regions allows for collection of diverse observations with more informative reward signal, enabling more effective RL finetuning, while in regions of high data density performance converges to that of the demonstrator.
In this work we seek to understand the role of the pretrained policy in RL finetuning, and how we might pretrain policies that (a) enable efficient RL finetuning, and (b) before finetuning, perform no worse than the policy pretrained with standard
Bc
.
We propose a novel pretraining approach—
posterior behavioral cloning
(
PostBc
)—which, rather than fitting the empirical distribution of demonstrations as standard
Bc
does, instead fits the
posterior
distribution over the demonstrator’s behavior.
That is, assuming a uniform prior over the demonstrator’s behavior and viewing the demonstration data as samples from the demonstrator’s behavioral distribution, we seek to train a policy that models the posterior distribution of the demonstrator’s behavior given these observations.
This enables the pretrained policy to take into account its potential uncertainty about the demonstrator’s behavior, and adjust the entropy of its action distribution based on this uncertainty. In states where it is uncertain about the demonstrator’s actions,
PostBc
samples from a high-entropy distribution, allowing for a more diverse set of actions that may enable further policy improvement, while in states where it is certain about the demonstrator’s actions, it samples from a low-entropy distribution, simply mimicking what it knows to be the (correct) demonstrator behavior (see
Figure
1
).
Theoretically, we show that
PostBc
leads to provable improvements over standard
Bc
in terms of the potential for downstream RL performance. In particular, we focus on the ability of the pretrained policy to cover the demonstrator policy’s actions—whether it samples all actions the demonstrator policy might sample—which, for finetuning approaches that rely on rolling out the pretrained policy, is a prerequisite to ensure finetuning can even match the performance of the demonstrator.
We show that standard
Bc
can provably fail to cover the demonstrator’s distribution, while
PostBc
does
cover the demonstrator’s distribution, incurs no suboptimality in the performance of the pretrained policy as compared to the standard
Bc
policy, and achieves a near-optimal sampling cost out of all policy estimators which have pretrained performance no worse than the
Bc
policy’s.
Inspired by this, we develop a practical approach to approximating the posterior of the demonstrator in continuous action domains, and instantiate
PostBc
with modern generative models—diffusion models—on robotic control tasks. Our instantiation relies only on pretraining with scalable supervised learning objectives—no RL is required in pretraining—and can be incorporated into existing
Bc
training pipelines with minimal modification.
We demonstrate experimentally that
PostBc
pretraining can lead to significant performance gains in terms of the efficiency and effectiveness of RL finetuning, as compared to running RL finetuning on a policy pretrained with standard
Bc
, and achieves these gains without decreasing the performance of the pretrained policy itself.
We show that this holds for a variety of finetuning algorithms—both policy-gradient-style algorithms, and algorithms which explicitly refine or filter the distribution of the pretrained policy—enabling effective RL finetuning across a variety of challenging robotic tasks in both simulation and the real world.
2
Conclusion
In this work, we have proposed a novel approach to pretraining policies from demonstrations that ensures the pretrained performance is no worse than that of the
Bc
policy, while expanding the action distribution to enable more effective RL finetuning. We have shown that this approach does indeed lead to improved RL finetuning performance in practice, scaling to real-world robotic settings. We believe this work motivates a variety of interesting questions for future work.
•
Our demonstrator action coverage condition introduced in
LABEL:sec:act_coverage
is a
necessary
condition, in some cases, for RL finetuning to reach the performance of the demonstrator policy, as
LABEL:prop:bc_fails
shows. In general, however, demonstrator action coverage does not give a guarantee about the sample complexity of the downstream RL finetuning. Can we derive a non-trivial
sufficient
condition that ensures efficient RL finetuning without the aid of exploration approaches typically absent in practice (such as optimism), and how can we pretrain policies to ensure they meet such a sufficient condition?
•
We have focused on pretraining only with supervised learning. While this is the most scalable approach, and the most commonly used approach in practice, is this a limiting factor in obtaining an effective initialization for online RL finetuning, and could we pretrain using other approaches as well (for example, offline RL)?
•
While we have primarily considered applications to robotic control, our approach could also be applied in language domains. Does pretraining (or SFT finetuning) of language models with our approach lead to improved performance in downstream RL finetuning?
Acknowledgments
This research was partly supported by RAI, ONR N00014-25-1-2060, and NSF IIS-2150826. The work of CF was partially supported by an NSF CAREER award.
References
P. Abbeel and A. Y. Ng (2004)
Apprenticeship learning via inverse reinforcement learning
.
In
Proceedings of the twenty-first international conference on Machine learning
,
pp. 1
.
Cited by:
Appendix A
.
Y. Bai, A. Jones, K. Ndousse, A. Askell, A. Chen, N. DasSarma, D. Drain, S. Fort, D. Ganguli, T. Henighan,
et al.
(2022)
Training a helpful and harmless assistant with reinforcement learning from human feedback
.
arXiv preprint arXiv:2204.05862
.
Cited by:
§1
.
P. J. Ball, L. Smith, I. Kostrikov, and S. Levine (2023)
Efficient online reinforcement learning with offline data
.
In
International Conference on Machine Learning
,
pp. 1577–1594
.
Cited by:
Appendix A
.
J. Chae, S. Han, W. Jung, M. Cho, S. Choi, and Y. Sung (2022)
Robust imitation learning against variations in environment dynamics
.
In
International Conference on Machine Learning
,
pp. 2828–2852
.
Cited by:
Appendix A
.
Y. Chen, S. Tian, S. Liu, Y. Zhou, H. Li, and D. Zhao (2025)
Conrft: a reinforced fine-tuning method for vla models via consistency policy
.
arXiv preprint arXiv:2502.05450
.
Cited by:
§1
.
S. Dasari and A. Gupta (2021)
Transformers for one-shot visual imitation
.
In
Conference on Robot Learning
,
pp. 2071–2084
.
Cited by:
Appendix A
.
S. Dasari, O. Mees, S. Zhao, M. K. Srirama, and S. Levine (2024)
The ingredients for robotic diffusion transformers
.
arXiv preprint arXiv:2410.10088
.
Cited by:
§D.2
.
S. Desai, I. Durugkar, H. Karnan, G. Warnell, J. Hanna, and P. Stone (2020)
An imitation from observation approach to transfer learning with dynamics mismatch
.
Advances in Neural Information Processing Systems
33
,
pp. 3917–3929
.
Cited by:
Appendix A
.
J. Devlin, M. Chang, K. Lee, and K. Toutanova (2019)
Bert: pre-training of deep bidirectional transformers for language understanding
.
In
Proceedings of the 2019 conference of the North American chapter of the association for computational linguistics: human language technologies, volume 1 (long and short papers)
,
pp. 4171–4186
.
Cited by:
§D.2
.
Y. Duan, M. Andrychowicz, B. Stadie, O. Jonathan Ho, J. Schneider, I. Sutskever, P. Abbeel, and W. Zaremba (2017)
One-shot imitation learning
.
Advances in neural information processing systems
30
.
Cited by:
Appendix A
.
Y. Duan, J. Schulman, X. Chen, P. L. Bartlett, I. Sutskever, and P. Abbeel (2016)
Rl
2
: fast reinforcement learning via slow reinforcement learning
.
arXiv preprint arXiv:1611.02779
.
Cited by:
Appendix A
.
C. Finn, P. Abbeel, and S. Levine (2017a)
Model-agnostic meta-learning for fast adaptation of deep networks
.
In
International conference on machine learning
,
pp. 1126–1135
.
Cited by:
Appendix A
.
C. Finn, K. Xu, and S. Levine (2018)
Probabilistic model-agnostic meta-learning
.
Advances in neural information processing systems
31
.
Cited by:
Appendix A
.
C. Finn, T. Yu, T. Zhang, P. Abbeel, and S. Levine (2017b)
One-shot visual imitation learning via meta-learning
.
In
Conference on robot learning
,
pp. 357–368
.
Cited by:
Appendix A
.
D. J. Foster, S. M. Kakade, J. Qian, and A. Rakhlin (2021)
The statistical complexity of interactive decision making
.
arXiv preprint arXiv:2112.13487
.
Cited by:
§B.4
.
J. Fu, K. Luo, and S. Levine (2017)
Learning robust rewards with adversarial inverse reinforcement learning
.
arXiv preprint arXiv:1710.11248
.
Cited by:
Appendix A
.
C. Gao, Y. Jiang, and F. Chen (2023)
Transferring hierarchical structures with dual meta imitation learning
.
In
Conference on Robot Learning
,
pp. 762–773
.
Cited by:
Appendix A
.
D. Garg, S. Chakraborty, C. Cundy, J. Song, and S. Ermon (2021)
Iq-learn: inverse soft-q learning for imitation
.
Advances in Neural Information Processing Systems
34
,
pp. 4028–4039
.
Cited by:
Appendix A
.
D. Ghosh, A. Ajay, P. Agrawal, and S. Levine (2022)
Offline rl policies should be trained to be adaptive
.
In
International Conference on Machine Learning
,
pp. 7513–7530
.
Cited by:
Appendix A
.
V. Giammarino, J. Queeney, and I. C. Paschalidis (2025)
Visually robust adversarial imitation learning from videos with contrastive learning
.
In
2025 IEEE International Conference on Robotics and Automation (ICRA)
,
pp. 15642–15648
.
Cited by:
Appendix A
.
D. Guo, D. Yang, H. Zhang, J. Song, R. Zhang, R. Xu, Q. Zhu, S. Ma, P. Wang, X. Bi,
et al.
(2025)
Deepseek-r1: incentivizing reasoning capability in llms via reinforcement learning
.
arXiv preprint arXiv:2501.12948
.
Cited by:
§1
,
§1
.
J. Ho and S. Ermon (2016)
Generative adversarial imitation learning
.
Advances in neural information processing systems
29
.
Cited by:
Appendix A
.
S. James, M. Bloesch, and A. J. Davison (2018)
Task-embedded control networks for few-shot imitation learning
.
In
Conference on robot learning
,
pp. 783–795
.
Cited by:
Appendix A
.
M. J. Kim, C. Finn, and P. Liang (2025)
Fine-tuning vision-language-action models: optimizing speed and success
.
arXiv preprint arXiv:2502.19645
.
Cited by:
§1
.
I. Kostrikov, K. K. Agrawal, D. Dwibedi, S. Levine, and J. Tompson (2018)
Discriminator-actor-critic: addressing sample inefficiency and reward bias in adversarial imitation learning
.
arXiv preprint arXiv:1809.02925
.
Cited by:
Appendix A
.
I. Kostrikov, O. Nachum, and J. Tompson (2019)
Imitation learning via off-policy distribution matching
.
arXiv preprint arXiv:1912.05032
.
Cited by:
Appendix A
.
A. Kumar, A. Singh, F. Ebert, M. Nakamoto, Y. Yang, C. Finn, and S. Levine (2022)
Pre-training for robots: offline rl enables learning new tasks from a handful of trials
.
arXiv preprint arXiv:2210.05178
.
Cited by:
Appendix A
.
S. Lee, Y. Seo, K. Lee, P. Abbeel, and J. Shin (2022)
Offline-to-online reinforcement learning via balanced replay and pessimistic q-ensemble
.
In
Conference on Robot Learning
,
pp. 1702–1712
.
Cited by:
Appendix A
.
Z. Li, T. Xu, Z. Qin, Y. Yu, and Z. Luo (2023)
Imitation learning from imperfection: theoretical justifications and algorithms
.
Advances in Neural Information Processing Systems
36
,
pp. 18404–18443
.
Cited by:
Appendix A
.
M. Nakamoto, O. Mees, A. Kumar, and S. Levine (2024)
Steering your generalists: improving robotic foundation models via value guidance
.
arXiv preprint arXiv:2410.13816
.
Cited by:
§1
.
M. Nakamoto, S. Zhai, A. Singh, M. Sobol Mark, Y. Ma, C. Finn, A. Kumar, and S. Levine (2023)
Cal-ql: calibrated offline rl pre-training for efficient online fine-tuning
.
Advances in Neural Information Processing Systems
36
,
pp. 62244–62269
.
Cited by:
Appendix A
.
A. Y. Ng, S. Russell,
et al.
(2000)
Algorithms for inverse reinforcement learning.
.
In
Icml
,
Vol.
1
,
pp. 2
.
Cited by:
Appendix A
.
T. Ni, H. Sikchi, Y. Wang, T. Gupta, L. Lee, and B. Eysenbach (2021)
F-irl: inverse reinforcement learning via state marginal matching
.
In
Conference on Robot Learning
,
pp. 529–551
.
Cited by:
Appendix A
.
L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray,
et al.
(2022)
Training language models to follow instructions with human feedback
.
Advances in neural information processing systems
35
,
pp. 27730–27744
.
Cited by:
§1
.
N. Rajaraman, L. Yang, J. Jiao, and K. Ramchandran (2020)
Toward the fundamental limits of imitation learning
.
Advances in Neural Information Processing Systems
33
,
pp. 2914–2924
.
Cited by:
§B.3
,
§B.3
,
§B.3
,
Lemma 4
,
Lemma 4
,
footnote 1
.
A. Z. Ren, J. Lidard, L. L. Ankile, A. Simeonov, P. Agrawal, A. Majumdar, B. Burchfiel, H. Dai, and M. Simchowitz (2024)
Diffusion policy policy optimization
.
arXiv preprint arXiv:2409.00588
.
Cited by:
§D.1
.
V. Tangkaratt, N. Charoenphakdee, and M. Sugiyama (2020)
Robust imitation learning from noisy demonstrations
.
arXiv preprint arXiv:2010.10181
.
Cited by:
Appendix A
.
K. Team, A. Du, B. Gao, B. Xing, C. Jiang, C. Chen, C. Li, C. Xiao, C. Du, C. Liao,
et al.
(2025)
Kimi k1. 5: scaling reinforcement learning with llms
.
arXiv preprint arXiv:2501.12599
.
Cited by:
§1
.
I. Uchendu, T. Xiao, Y. Lu, B. Zhu, M. Yan, J. Simon, M. Bennice, C. Fu, C. Ma, J. Jiao,
et al.
(2023)
Jump-start reinforcement learning
.
In
International Conference on Machine Learning
,
pp. 34556–34583
.
Cited by:
Appendix A
.
A. Wagenmaker, M. Nakamoto, Y. Zhang, S. Park, W. Yagoub, A. Nagabandi, A. Gupta, and S. Levine (2025)
Steering your diffusion policy with latent space reinforcement learning
.
arXiv preprint arXiv:2506.15799
.
Cited by:
§D.2
,
§1
.
H. R. Walke, K. Black, T. Z. Zhao, Q. Vuong, C. Zheng, P. Hansen-Estruch, A. W. He, V. Myers, M. J. Kim, M. Du,
et al.
(2023)
Bridgedata v2: a dataset for robot learning at scale
.
In
Conference on Robot Learning
,
pp. 1723–1736
.
Cited by:
§D.3
.
J. X. Wang, Z. Kurth-Nelson, D. Tirumala, H. Soyer, J. Z. Leibo, R. Munos, C. Blundell, D. Kumaran, and M. Botvinick (2016)
Learning to reinforcement learn
.
arXiv preprint arXiv:1611.05763
.
Cited by:
Appendix A
.
Y. Wang, C. Xu, and B. Du (2021)
Robust adversarial imitation learning via adaptively-selected demonstrations.
.
In
IJCAI
,
pp. 3155–3161
.
Cited by:
Appendix A
.
H. Xu, X. Zhan, H. Yin, and H. Qin (2022)
Discriminator-weighted offline imitation learning from suboptimal demonstrations
.
In
International Conference on Machine Learning
,
pp. 24725–24742
.
Cited by:
Appendix A
.
S. Yue, X. Hua, J. Ren, S. Lin, J. Zhang, and Y. Zhang (2024)
OLLIE: imitation learning from offline pretraining to online finetuning
.
arXiv preprint arXiv:2405.17477
.
Cited by:
Appendix A
.
Y. Yue, Z. Chen, R. Lu, A. Zhao, Z. Wang, S. Song, and G. Huang (2025)
Does reinforcement learning really incentivize reasoning capacity in llms beyond the base model?
.
arXiv preprint arXiv:2504.13837
.
Cited by:
§1
.
H. Zhang, W. Xu, and H. Yu (2023)
Policy expansion for bridging offline-to-online reinforcement learning
.
arXiv preprint arXiv:2302.00935
.
Cited by:
Appendix A
.
H. Zheng, X. Luo, P. Wei, X. Song, D. Li, and J. Jiang (2023)
Adaptive policy learning for offline-to-online reinforcement learning
.
In
Proceedings of the AAAI Conference on Artificial Intelligence
,
Vol.
37
,
pp. 11372–11380
.
Cited by:
Appendix A
.
B. D. Ziebart, A. L. Maas, J. A. Bagnell, A. K. Dey,
et al.
(2008)
Maximum entropy inverse reinforcement learning.
.
In
Aaai
,
Vol.
8
,
pp. 1433–1438
.
Cited by:
Appendix A
.
Appendix A
Additional Related Work
Other approaches for pretraining from demonstrations.
While our primary focus is on behavioral cloning (as noted, the workhorse of most modern applications) other approaches to pretraining from demonstrations exist.
Bc
is only one possible instantiation of
imitation learning
; other approaches to imitation learning include inverse RL
(Ng
et al.
,
2000
; Abbeel and Ng,
2004
; Ziebart
et al.
,
2008
)
, methods that aim to learn a policy matching the state distribution of the demonstrator, such as adversarial imitation learning
(Ho and Ermon,
2016
; Kostrikov
et al.
,
2018
; Fu
et al.
,
2017
; Kostrikov
et al.
,
2019
; Ni
et al.
,
2021
; Garg
et al.
,
2021
; Xu
et al.
,
2022
; Li
et al.
,
2023
; Yue
et al.
,
2024
)
, and robust imitation learning
(Chae
et al.
,
2022
; Desai
et al.
,
2020
; Tangkaratt
et al.
,
2020
; Wang
et al.
,
2021
; Giammarino
et al.
,
2025
)
.
The majority of these works, however, either assume access to additional data sources (e.g. suboptimal trajectories), or require online environment access and are therefore not truly offline pretraining approaches, which is the focus of this work. Furthermore, none of these works explicitly consider the role of pretraining in enabling efficient RL finetuning.
Meta-learning directly aims learn an initialization that can be quickly adapted to a new task. While instantiations of meta-learning for
imitation learning exist
(Duan
et al.
,
2017
; Finn
et al.
,
2017b
; James
et al.
,
2018
; Dasari and Gupta,
2021
; Gao
et al.
,
2023
)
, our setting differs fundamentally from the meta-imitation learning setting. Meta-imitation learning assumes access to demonstration data from
more than one task
, and attempts to learn an initialization that will allow for quickly adapting to demonstrations from a
new
task. In contrast, our goal is to obtain an approach able to learn on a
single
task
(though we also consider the multi-task setting), and we aim to find an initialization that allows for improvement on the
same
task, while preserving pretrained performance on this task. Furthermore, rather than learning from new
demonstrations
, as meta-imitation learning does, we aim to learn from (potentially suboptimal) data collected online and that is labeled with rewards.
Reinforcement learning-based pretraining.
In the RL literature, two lines of work bear some resemblance to ours as well. The
offline-to-online RL
setting aims to train policies with RL on offline datasets that can then be improved with further online interaction
(Lee
et al.
,
2022
; Ghosh
et al.
,
2022
; Kumar
et al.
,
2022
; Zhang
et al.
,
2023
; Uchendu
et al.
,
2023
; Zheng
et al.
,
2023
; Ball
et al.
,
2023
; Nakamoto
et al.
,
2023
)
, and the
meta-RL
setting aims to meta-learn a policy on some set of tasks which can then be quickly adapted to a new task
(Wang
et al.
,
2016
; Duan
et al.
,
2016
; Finn
et al.
,
2017a
,
2018
)
.
While similar to our work in that these works also aim to learn behaviors that can be efficiently improved online, the settings differ significantly in that the offline- or meta-pretraining typically requires reward labels (rather than unlabeled demonstrations) and are performed with RL (rather than BC)—in contrast, we study how BC-like pretraining (as noted, the workhorse of most modern applications) can enable efficient online adaptation.
Appendix B
Proofs
B.1
BC Policy Fails to Cover Demonstrator Actions
Proof of
LABEL:prop:bc_fails
.
Let
ℳ
1
\mathcal{M}^{1}
and
ℳ
2
\mathcal{M}^{2}
denote multi-armed bandits with 3 arms and reward functions
r
1
r^{1}
and
r
2
r^{2}
:
r
1
​
(
a
1
)
=
0
,
r
1
​
(
a
2
)
=
1
,
r
1
​
(
a
3
)
=
0
\displaystyle r^{1}(a_{1})=0,r^{1}(a_{2})=1,r^{1}(a_{3})=0
r
2
​
(
a
1
)
=
0
,
r
2
​
(
a
2
)
=
0
,
r
2
​
(
a
3
)
=
1
.
\displaystyle r^{2}(a_{1})=0,r^{2}(a_{2})=0,r^{2}(a_{3})=1.
Let
π
β
​
(
a
1
)
=
1
−
4
​
ϵ
\pi^{\beta}(a_{1})=1-4\epsilon
,
π
β
​
(
a
2
)
=
2
​
ϵ
\pi^{\beta}(a_{2})=2\epsilon
,
π
β
​
(
a
3
)
=
2
​
ϵ
\pi^{\beta}(a_{3})=2\epsilon
.
By construction of
π
^
bc
\widehat{\pi}^{\mathrm{bc}}
, if
T
​
(
a
2
)
=
0
T(a_{2})=0
then we will have
π
^
bc
​
(
a
2
)
=
0
\widehat{\pi}^{\mathrm{bc}}(a_{2})=0
, and if
T
​
(
a
3
)
=
0
T(a_{3})=0
we will have
π
^
bc
​
(
a
3
)
=
0
\widehat{\pi}^{\mathrm{bc}}(a_{3})=0
.
By the definition of both
ℳ
1
\mathcal{M}^{1}
and
ℳ
2
\mathcal{M}^{2}
, we have
ℙ
ℳ
i
​
[
T
​
(
a
2
)
=
0
,
T
​
(
a
3
)
=
0
]
=
(
1
−
4
​
ϵ
)
T
.
\displaystyle\mathbb{P}^{\mathcal{M}^{i}}[T(a_{2})=0,T(a_{3})=0]=(1-4\epsilon)^{T}.
As we have assumed that
T
≤
1
20
​
ϵ
T\leq\frac{1}{20\epsilon}
and
ϵ
∈
(
0
,
1
/
8
]
\epsilon\in(0,1/8]
, some calculation shows that we can lower bound this as
1
/
2
1/2
. Note that for both
ℳ
1
\mathcal{M}^{1}
and
ℳ
2
\mathcal{M}^{2}
, we have
𝒥
​
(
π
β
)
=
2
​
ϵ
\mathcal{J}(\pi^{\beta})=2\epsilon
, while for policies
π
^
bc
\widehat{\pi}^{\mathrm{bc}}
that only play
a
1
a_{1}
, we have
𝒥
​
(
π
^
bc
)
=
0
\mathcal{J}(\widehat{\pi}^{\mathrm{bc}})=0
. This proves the first part of the result.
For the second part, note that the optimal policy on
ℳ
1
\mathcal{M}^{1}
plays only
a
2
a_{2}
and has expected reward of 1, while the optimal policy on
ℳ
2
\mathcal{M}^{2}
plays only
a
2
a_{2}
and has expected reward of 1. Let
π
^
\widehat{\pi}
denote an estimate of the optimal policy and
𝔼
ℳ
i
,
π
^
bc
​
[
⋅
]
\mathbb{E}^{\mathcal{M}^{i},\widehat{\pi}^{\mathrm{bc}}}[\cdot]
the expectation induced by playing the policy
π
^
bc
\widehat{\pi}^{\mathrm{bc}}
from the first part on instance
ℳ
i
\mathcal{M}^{i}
. Then:
min
π
^
⁡
max
i
∈
{
1
,
2
}
⁡
𝔼
ℳ
i
,
π
^
bc
​
[
max
π
⁡
𝒥
ℳ
i
​
(
π
)
−
𝒥
ℳ
i
​
(
π
^
)
]
\displaystyle\min_{\widehat{\pi}}\max_{i\in\{1,2\}}\mathbb{E}^{\mathcal{M}^{i},\widehat{\pi}^{\mathrm{bc}}}[\max_{\pi}\mathcal{J}^{\mathcal{M}^{i}}(\pi)-\mathcal{J}^{\mathcal{M}^{i}}(\widehat{\pi})]
=
min
π
^
⁡
max
i
∈
{
1
,
2
}
⁡
𝔼
ℳ
i
,
π
^
bc
​
[
1
−
π
^
​
(
a
1
+
i
)
]
.
\displaystyle=\min_{\widehat{\pi}}\max_{i\in\{1,2\}}\mathbb{E}^{\mathcal{M}^{i},\widehat{\pi}^{\mathrm{bc}}}[1-\widehat{\pi}(a_{1+i})].
Note that
1
−
π
^
​
(
a
2
)
=
π
^
​
(
a
1
)
+
π
^
​
(
a
3
)
≥
π
^
​
(
a
3
)
1-\widehat{\pi}(a_{2})=\widehat{\pi}(a_{1})+\widehat{\pi}(a_{3})\geq\widehat{\pi}(a_{3})
. Thus we can lower bound the above as
≥
min
π
^
⁡
max
⁡
{
𝔼
ℳ
1
,
π
^
bc
​
[
π
^
​
(
a
3
)
]
,
𝔼
ℳ
2
,
π
^
bc
​
[
1
−
π
^
​
(
a
3
)
]
}
\displaystyle\geq\min_{\widehat{\pi}}\max\{\mathbb{E}^{\mathcal{M}^{1},\widehat{\pi}^{\mathrm{bc}}}[\widehat{\pi}(a_{3})],\mathbb{E}^{\mathcal{M}^{2},\widehat{\pi}^{\mathrm{bc}}}[1-\widehat{\pi}(a_{3})]\}
≥
min
π
^
⁡
1
2
​
(
𝔼
ℳ
1
,
π
^
bc
​
[
π
^
​
(
a
3
)
]
+
𝔼
ℳ
2
,
π
^
bc
​
[
1
−
π
^
​
(
a
3
)
]
)
\displaystyle\geq\min_{\widehat{\pi}}\frac{1}{2}\left(\mathbb{E}^{\mathcal{M}^{1},\widehat{\pi}^{\mathrm{bc}}}[\widehat{\pi}(a_{3})]+\mathbb{E}^{\mathcal{M}^{2},\widehat{\pi}^{\mathrm{bc}}}[1-\widehat{\pi}(a_{3})]\right)
≥
1
2
−
1
2
​
min
π
^
⁡
|
𝔼
ℳ
1
,
π
^
bc
​
[
π
^
​
(
a
3
)
]
−
𝔼
ℳ
2
,
π
^
bc
​
[
π
^
​
(
a
3
)
]
|
.
\displaystyle\geq\frac{1}{2}-\frac{1}{2}\min_{\widehat{\pi}}\left|\mathbb{E}^{\mathcal{M}^{1},\widehat{\pi}^{\mathrm{bc}}}[\widehat{\pi}(a_{3})]-\mathbb{E}^{\mathcal{M}^{2},\widehat{\pi}^{\mathrm{bc}}}[\widehat{\pi}(a_{3})]\right|.
We can bound
|
𝔼
ℳ
1
,
π
^
bc
​
[
π
^
​
(
a
3
)
]
−
𝔼
ℳ
2
,
π
^
bc
​
[
π
^
​
(
a
3
)
]
|
≤
TV
​
(
ℙ
ℳ
1
,
π
^
bc
,
ℙ
ℳ
2
,
π
^
bc
)
.
\displaystyle\left|\mathbb{E}^{\mathcal{M}^{1},\widehat{\pi}^{\mathrm{bc}}}[\widehat{\pi}(a_{3})]-\mathbb{E}^{\mathcal{M}^{2},\widehat{\pi}^{\mathrm{bc}}}[\widehat{\pi}(a_{3})]\right|\leq\mathrm{TV}(\mathbb{P}^{\mathcal{M}^{1},\widehat{\pi}^{\mathrm{bc}}},\mathbb{P}^{\mathcal{M}^{2},\widehat{\pi}^{\mathrm{bc}}}).
Since
ℳ
1
\mathcal{M}^{1}
and
ℳ
2
\mathcal{M}^{2}
only differ on
a
2
a_{2}
and
a
3
a_{3}
, and since
π
^
bc
​
(
a
2
)
=
π
^
bc
​
(
a
3
)
=
0
\widehat{\pi}^{\mathrm{bc}}(a_{2})=\widehat{\pi}^{\mathrm{bc}}(a_{3})=0
, we have
TV
​
(
ℙ
ℳ
1
,
π
^
bc
,
ℙ
ℳ
2
,
π
^
bc
)
=
0
\mathrm{TV}(\mathbb{P}^{\mathcal{M}^{1},\widehat{\pi}^{\mathrm{bc}}},\mathbb{P}^{\mathcal{M}^{2},\widehat{\pi}^{\mathrm{bc}}})=0
. Thus, we conclude that
min
π
^
⁡
max
i
∈
{
1
,
2
}
⁡
𝔼
ℳ
i
,
π
^
bc
​
[
max
π
⁡
𝒥
ℳ
i
​
(
π
)
−
𝒥
ℳ
i
​
(
π
^
)
]
≥
1
2
.
\displaystyle\min_{\widehat{\pi}}\max_{i\in\{1,2\}}\mathbb{E}^{\mathcal{M}^{i},\widehat{\pi}^{\mathrm{bc}}}[\max_{\pi}\mathcal{J}^{\mathcal{M}^{i}}(\pi)-\mathcal{J}^{\mathcal{M}^{i}}(\widehat{\pi})]\geq\frac{1}{2}.
This proves the second part of the result.
∎
B.2
Uniform Noise Fails
Proof of
LABEL:prop:unif_fails
.
Construction.
Let
ℳ
\mathcal{M}
be the MDP with state space
{
s
~
1
,
…
,
s
~
k
,
s
1
,
s
2
}
\{\widetilde{s}_{1},\ldots,\widetilde{s}_{k},s_{1},s_{2}\}
, actions
{
a
1
,
a
2
}
\{a_{1},a_{2}\}
, horizon
H
≥
2
H\geq 2
with initial state distribution:
P
0
​
(
s
1
)
=
1
/
2
,
P
0
​
(
s
~
1
)
=
2
−
2
+
2
−
k
,
P
0
​
(
s
~
i
)
=
2
−
i
−
1
,
i
≥
2
,
\displaystyle P_{0}(s_{1})=1/2,\quad P_{0}(\widetilde{s}_{1})=2^{-2}+2^{-k},\quad P_{0}(\widetilde{s}_{i})=2^{-i-1},i\geq 2,
transition function, for all
h
∈
[
H
]
h\in[H]
:
P
h
​
(
s
~
i
∣
s
~
i
,
a
)
=
1
,
∀
a
∈
𝒜
,
P
h
​
(
s
1
∣
s
1
,
a
1
)
=
1
,
\displaystyle P_{h}(\widetilde{s}_{i}\mid\widetilde{s}_{i},a)=1,\forall a\in\mathcal{A},\quad P_{h}(s_{1}\mid s_{1},a_{1})=1,
P
h
​
(
s
2
∣
s
1
,
a
2
)
=
1
,
P
h
​
(
s
2
∣
s
2
,
a
)
=
1
,
∀
a
∈
𝒜
,
\displaystyle P_{h}(s_{2}\mid s_{1},a_{2})=1,\quad P_{h}(s_{2}\mid s_{2},a)=1,\forall a\in\mathcal{A},
and reward that is 0 everywhere except
r
1
​
(
s
~
i
,
a
1
)
=
r
H
​
(
s
1
,
a
1
)
=
1
,
r
1
​
(
s
~
i
,
a
2
)
=
1
−
2
​
Δ
,
\displaystyle r_{1}(\widetilde{s}_{i},a_{1})=r_{H}(s_{1},a_{1})=1,\quad r_{1}(\widetilde{s}_{i},a_{2})=1-2\Delta,
for some
Δ
>
0
\Delta>0
to be specified.
We consider
π
β
\pi^{\beta}
defined as
π
h
β
​
(
a
1
∣
s
~
i
)
=
π
h
β
​
(
a
2
∣
s
~
i
)
=
1
2
,
π
h
β
​
(
a
1
∣
s
1
)
=
1
.
\displaystyle\pi^{\beta}_{h}(a_{1}\mid\widetilde{s}_{i})=\pi^{\beta}_{h}(a_{2}\mid\widetilde{s}_{i})=\frac{1}{2},\quad\pi^{\beta}_{h}(a_{1}\mid s_{1})=1.
Let
ϵ
:=
H
2
​
S
​
log
⁡
T
T
+
ξ
\epsilon:=\frac{H^{2}S\log T}{T}+\xi
, and set
Δ
←
2
​
ϵ
\Delta\leftarrow 2\epsilon
.
Upper bound on
α
\alpha
.
Note that
𝒥
​
(
π
β
)
=
1
−
1
2
​
Δ
\mathcal{J}(\pi^{\beta})=1-\frac{1}{2}\Delta
, and that the value of the optimal policy
π
⋆
\pi^{\star}
is
𝒥
​
(
π
⋆
)
=
max
π
⁡
𝒥
​
(
π
)
=
1
\mathcal{J}(\pi^{\star})=\max_{\pi}\mathcal{J}(\pi)=1
. Let
π
~
u
,
α
\widetilde{\pi}^{\mathrm{u,\alpha}}
denote the policy that, on all
s
~
i
\widetilde{s}_{i}
plays
π
⋆
\pi^{\star}
, and on other states plays
π
⋆
\pi^{\star}
with probability
1
−
α
1-\alpha
, and otherwise plays
unif
​
(
𝒜
)
\mathrm{unif}(\mathcal{A})
. Note then that, regardless of the value of
π
^
bc
\widehat{\pi}^{\mathrm{bc}}
, we have that
𝒥
​
(
π
~
u
,
α
)
≥
𝒥
​
(
π
^
u
,
α
)
\mathcal{J}(\widetilde{\pi}^{\mathrm{u,\alpha}})\geq\mathcal{J}(\widehat{\pi}^{\mathrm{u,\alpha}})
. Thus,
𝒥
​
(
π
β
)
−
𝔼
​
[
𝒥
​
(
π
^
u
,
α
)
]
≥
𝒥
​
(
π
β
)
−
𝒥
​
(
π
~
u
,
α
)
\displaystyle\mathcal{J}(\pi^{\beta})-\mathbb{E}[\mathcal{J}(\widehat{\pi}^{\mathrm{u,\alpha}})]\geq\mathcal{J}(\pi^{\beta})-\mathcal{J}(\widetilde{\pi}^{\mathrm{u,\alpha}})
If we are in
s
1
s_{1}
at
h
=
2
h=2
, the only way we can receive any reward on the episode is if we take action
a
1
a_{1}
for the last
H
−
1
H-1
steps, and we then receive a reward of
1
1
.
Under
π
~
u
,
α
\widetilde{\pi}^{\mathrm{u,\alpha}}
, we take
a
1
a_{1}
at each step with probability
1
−
α
+
α
/
A
1-\alpha+\alpha/A
, so our probability of getting a reward of
1
1
is
(
1
−
α
+
α
/
A
)
H
−
1
(1-\alpha+\alpha/A)^{H-1}
. Note that in contrast
π
β
\pi^{\beta}
will always play
a
1
a_{1}
and receive a reward of 1 in this situation.
If we are in
s
~
i
\widetilde{s}_{i}
at
h
=
2
h=2
for any
i
i
, then
π
β
\pi^{\beta}
will incur a loss of
Δ
\Delta
more than
π
~
u
,
α
\widetilde{\pi}^{\mathrm{u,\alpha}}
.
Thus, we can lower bound
𝒥
​
(
π
β
)
−
𝒥
​
(
π
~
u
,
α
)
≥
−
1
2
​
Δ
+
1
2
⋅
(
1
−
(
1
−
α
+
α
/
A
)
H
−
1
)
\displaystyle\mathcal{J}(\pi^{\beta})-\mathcal{J}(\widetilde{\pi}^{\mathrm{u,\alpha}})\geq-\frac{1}{2}\Delta+\frac{1}{2}\cdot(1-(1-\alpha+\alpha/A)^{H-1})
By assumption we have that
1
2
​
Δ
=
ϵ
\frac{1}{2}\Delta=\epsilon
. Thus, if we want
𝒥
​
(
π
β
)
−
𝔼
​
[
𝒥
​
(
π
^
u
,
α
)
]
≤
ϵ
\mathcal{J}(\pi^{\beta})-\mathbb{E}[\mathcal{J}(\widehat{\pi}^{\mathrm{u,\alpha}})]\leq\epsilon
, we need
1
2
⋅
(
1
−
(
1
−
α
+
α
/
A
)
H
−
1
)
≤
2
​
ϵ
.
\displaystyle\frac{1}{2}\cdot(1-(1-\alpha+\alpha/A)^{H-1})\leq 2\epsilon.
Rearranging this, we have
1
−
4
​
ϵ
≤
(
1
−
α
+
α
/
A
)
H
−
1
⇔
1
H
−
1
​
log
⁡
(
1
−
4
​
ϵ
)
≤
log
⁡
(
1
−
α
+
α
/
A
)
.
\displaystyle 1-4\epsilon\leq(1-\alpha+\alpha/A)^{H-1}\iff\frac{1}{H-1}\log\left(1-4\epsilon\right)\leq\log(1-\alpha+\alpha/A).
From the Taylor decomposition of
log
⁡
(
1
−
x
)
\log(1-x)
, we see that
log
⁡
(
1
−
α
+
α
/
A
)
≤
−
(
1
−
1
/
A
)
​
α
\log(1-\alpha+\alpha/A)\leq-(1-1/A)\alpha
. Furthermore, we can lower bound
log
⁡
(
1
−
4
​
ϵ
)
≥
−
8
​
ϵ
\displaystyle\log(1-4\epsilon)\geq-8\epsilon
as long as
ϵ
≤
1
/
2
\epsilon\leq 1/2
. Altogether, then, we have
−
8
​
ϵ
H
−
1
≤
−
(
1
−
1
/
A
)
​
α
⟹
α
≤
8
​
ϵ
(
H
−
1
)
​
(
1
−
1
/
A
)
⟹
α
≤
32
​
ϵ
\displaystyle\frac{-8\epsilon}{H-1}\leq-(1-1/A)\alpha\implies\alpha\leq\frac{8\epsilon}{(H-1)(1-1/A)}\implies\alpha\leq 32\epsilon
where the last inequality follows since
H
≥
2
,
A
=
2
H\geq 2,A=2
.
Upper bound on
γ
\gamma
.
Let
i
T
:=
arg
​
max
i
⁡
{
2
−
i
−
1
∣
2
−
i
−
1
≤
1
/
T
}
i_{T}:=\operatorname*{arg\,max}_{i}\{2^{-i-1}\mid 2^{-i-1}\leq 1/T\}
, so that
1
/
2
​
T
≤
P
0
​
(
s
~
i
T
)
≤
1
/
T
1/2T\leq P_{0}(\widetilde{s}_{i_{T}})\leq 1/T
, and note that such an
s
~
i
T
\widetilde{s}_{i_{T}}
exists by construction. Let
ℰ
\mathcal{E}
be the event
ℰ
:=
{
T
1
​
(
s
~
i
T
)
=
T
1
​
(
s
~
i
T
,
a
2
)
=
1
}
\mathcal{E}:=\{T_{1}(\widetilde{s}_{i_{T}})=T_{1}(\widetilde{s}_{i_{T}},a_{2})=1\}
.
We have
ℙ
​
[
ℰ
]
\displaystyle\mathbb{P}[\mathcal{E}]
=
ℙ
​
[
T
1
​
(
s
~
i
T
,
a
2
)
=
1
∣
T
1
​
(
s
~
i
T
)
=
1
]
​
ℙ
​
[
T
1
​
(
s
~
i
T
)
=
1
]
\displaystyle=\mathbb{P}[T_{1}(\widetilde{s}_{i_{T}},a_{2})=1\mid T_{1}(\widetilde{s}_{i_{T}})=1]\mathbb{P}[T_{1}(\widetilde{s}_{i_{T}})=1]
=
1
2
⋅
T
​
P
0
​
(
s
~
i
T
)
​
(
1
−
P
0
​
(
s
~
i
T
)
)
T
−
1
\displaystyle=\frac{1}{2}\cdot TP_{0}(\widetilde{s}_{i_{T}})(1-P_{0}(\widetilde{s}_{i_{T}}))^{T-1}
=
1
2
⋅
T
⋅
1
2
​
T
⋅
(
1
−
1
T
)
T
−
1
\displaystyle=\frac{1}{2}\cdot T\cdot\frac{1}{2T}\cdot(1-\frac{1}{T})^{T-1}
≥
1
4
​
e
.
\displaystyle\geq\frac{1}{4e}.
Note that on the event
ℰ
\mathcal{E}
, we have
π
^
1
bc
​
(
a
1
∣
s
~
i
T
)
=
0
\widehat{\pi}^{\mathrm{bc}}_{1}(a_{1}\mid\widetilde{s}_{i_{T}})=0
, but
π
1
β
​
(
a
1
∣
s
~
i
T
)
=
1
/
2
\pi^{\beta}_{1}(a_{1}\mid\widetilde{s}_{i_{T}})=1/2
.
Thus,
π
^
1
u
,
α
​
(
a
1
∣
s
~
i
T
)
=
α
/
A
≤
32
​
ϵ
/
A
=
64
​
ϵ
/
A
⋅
π
1
β
​
(
a
1
∣
s
~
i
T
)
\displaystyle\widehat{\pi}^{\mathrm{u,\alpha}}_{1}(a_{1}\mid\widetilde{s}_{i_{T}})=\alpha/A\leq 32\epsilon/A=64\epsilon/A\cdot\pi^{\beta}_{1}(a_{1}\mid\widetilde{s}_{i_{T}})
where we have used the bound on
α
\alpha
shown above. Thus, on
ℰ
\mathcal{E}
, we will only have that
π
^
u
,
α
\widehat{\pi}^{\mathrm{u,\alpha}}
achieves demonstrator action coverage for
γ
≤
64
​
ϵ
/
A
\gamma\leq 64\epsilon/A
. Since
ℰ
\mathcal{E}
occurs with probability at least
1
/
4
​
e
1/4e
, it follows that if we want to guarantee
π
^
u
,
α
\widehat{\pi}^{\mathrm{u,\alpha}}
achieves demonstrator action coverage with probability at least
1
−
δ
1-\delta
for
δ
<
1
/
4
​
e
\delta<1/4e
, we must have
γ
≤
64
​
ϵ
/
A
\gamma\leq 64\epsilon/A
.
Note as well that, since
π
^
1
bc
​
(
a
2
∣
s
~
i
T
)
=
1
\widehat{\pi}^{\mathrm{bc}}_{1}(a_{2}\mid\widetilde{s}_{i_{T}})=1
, any policy in the support of
π
^
bc
\widehat{\pi}^{\mathrm{bc}}
will be suboptimal by a factor of at least
P
0
​
(
s
~
i
T
)
⋅
2
​
Δ
≥
Δ
/
T
P_{0}(\widetilde{s}_{i_{T}})\cdot 2\Delta\geq\Delta/T
.
∎
B.3
Analysis of Posterior Demonstrator Policy
Throughout this section we denote
π
~
h
​
(
a
∣
s
)
:=
{
(
1
−
α
)
⋅
T
h
​
(
s
,
a
)
T
h
​
(
s
)
+
α
⋅
T
h
​
(
s
,
a
)
+
λ
/
A
T
h
​
(
s
)
+
λ
T
h
​
(
s
)
>
0
unif
​
(
𝒜
)
T
h
​
(
s
)
=
0
\displaystyle\widetilde{\pi}_{h}(a\mid s):=\begin{cases}(1-\alpha)\cdot\frac{T_{h}(s,a)}{T_{h}(s)}+\alpha\cdot\frac{T_{h}(s,a)+\lambda/A}{T_{h}(s)+\lambda}&T_{h}(s)>0\\
\mathrm{unif}(\mathcal{A})&T_{h}(s)=0\end{cases}
for some
α
∈
[
0
,
1
]
\alpha\in[0,1]
.
We also denote
w
h
π
​
(
s
,
a
)
:=
ℙ
π
​
[
s
h
=
s
,
a
h
=
a
]
w_{h}^{\pi}(s,a):=\mathbb{P}^{\pi}[s_{h}=s,a_{h}=a]
.
Q
h
π
​
(
s
,
a
)
:=
𝔼
π
​
[
∑
h
′
≥
h
r
h
′
​
(
s
h
′
,
a
h
′
)
∣
s
h
=
s
,
a
h
=
a
]
Q_{h}^{\pi}(s,a):=\mathbb{E}^{\pi}[\sum_{h^{\prime}\geq h}r_{h^{\prime}}(s_{h^{\prime}},a_{h^{\prime}})\mid s_{h}=s,a_{h}=a]
denotes the standard
Q
Q
-function.
𝒥
​
(
π
;
r
)
\mathcal{J}(\pi;r)
denotes the expected return of policy
π
\pi
for reward
r
r
.
Lemma 1
.
As long as
δ
≤
0.9
\delta\leq 0.9
and
λ
≥
A
\lambda\geq A
, we have
ℙ
​
[
π
~
h
​
(
a
∣
s
)
≥
α
⋅
min
⁡
{
π
h
β
​
(
a
∣
s
)
64
​
log
⁡
S
​
H
/
δ
,
1
2
​
λ
}
,
∀
a
∈
𝒜
,
s
∈
𝒮
,
h
∈
[
H
]
]
≥
1
−
δ
.
\displaystyle\mathbb{P}\left[\widetilde{\pi}_{h}(a\mid s)\geq\alpha\cdot\min\left\{\frac{\pi^{\beta}_{h}(a\mid s)}{64\log SH/\delta},\frac{1}{2\lambda}\right\},\forall a\in\mathcal{A},s\in\mathcal{S},h\in[H]\right]\geq 1-\delta.
Proof.
Consider some
(
s
,
h
)
(s,h)
. By Bernstein’s inequality, if
T
h
​
(
s
)
>
0
T_{h}(s)>0
, we have that with probability at least
1
−
δ
1-\delta
,
T
h
​
(
s
,
a
)
T
h
​
(
s
)
≥
π
h
β
​
(
a
∣
s
)
−
2
​
π
h
β
​
(
a
∣
s
)
​
log
⁡
1
/
δ
T
h
​
(
s
)
−
2
​
log
⁡
1
/
δ
3
​
T
h
​
(
s
)
.
\displaystyle\frac{T_{h}(s,a)}{T_{h}(s)}\geq\pi^{\beta}_{h}(a\mid s)-\sqrt{\frac{2\pi^{\beta}_{h}(a\mid s)\log 1/\delta}{T_{h}(s)}}-\frac{2\log 1/\delta}{3T_{h}(s)}.
(1)
From some algebra, we see that as long as
T
h
​
(
s
)
≥
32
​
log
⁡
1
/
δ
π
h
β
​
(
a
∣
s
)
T_{h}(s)\geq\frac{32\log 1/\delta}{\pi^{\beta}_{h}(a\mid s)}
, we have that
T
h
​
(
s
,
a
)
T
h
​
(
s
)
≥
1
2
​
π
h
β
​
(
a
∣
s
)
\frac{T_{h}(s,a)}{T_{h}(s)}\geq\frac{1}{2}\pi^{\beta}_{h}(a\mid s)
.
By the definition of
π
~
\widetilde{\pi}
, under the good event of
(
1
)
we can then lower bound
π
~
h
​
(
a
∣
s
)
\displaystyle\widetilde{\pi}_{h}(a\mid s)
≥
{
α
1
+
λ
/
T
h
​
(
s
)
⋅
1
2
​
π
h
β
​
(
a
∣
s
)
T
h
​
(
s
)
≥
32
​
log
⁡
1
/
δ
π
h
β
​
(
a
∣
s
)
α
​
λ
/
A
T
h
​
(
s
)
+
A
o.w.
\displaystyle\geq\begin{cases}\frac{\alpha}{1+\lambda/T_{h}(s)}\cdot\frac{1}{2}\pi^{\beta}_{h}(a\mid s)&T_{h}(s)\geq\frac{32\log 1/\delta}{\pi^{\beta}_{h}(a\mid s)}\\
\frac{\alpha\lambda/A}{T_{h}(s)+A}&\text{o.w.}\end{cases}
≥
{
α
⋅
32
​
log
⁡
1
/
δ
32
​
log
⁡
1
/
δ
+
λ
⋅
π
h
β
​
(
a
∣
s
)
⋅
1
2
​
π
h
β
​
(
a
∣
s
)
N
h
​
(
s
)
≥
32
​
log
⁡
1
/
δ
π
h
β
​
(
a
∣
s
)
α
​
λ
/
A
⋅
π
h
β
​
(
a
∣
s
)
32
​
log
⁡
1
/
δ
+
λ
⋅
π
h
β
​
(
a
∣
s
)
o.w.
\displaystyle\geq\begin{cases}\frac{\alpha\cdot 32\log 1/\delta}{32\log 1/\delta+\lambda\cdot\pi^{\beta}_{h}(a\mid s)}\cdot\frac{1}{2}\pi^{\beta}_{h}(a\mid s)&N_{h}(s)\geq\frac{32\log 1/\delta}{\pi^{\beta}_{h}(a\mid s)}\\
\frac{\alpha\lambda/A\cdot\pi^{\beta}_{h}(a\mid s)}{32\log 1/\delta+\lambda\cdot\pi^{\beta}_{h}(a\mid s)}&\text{o.w.}\end{cases}
≥
(
a
)
​
α
⋅
π
h
β
​
(
a
∣
s
)
32
​
log
⁡
1
/
δ
+
λ
⋅
π
h
β
​
(
a
∣
s
)
\displaystyle\overset{(a)}{\geq}\frac{\alpha\cdot\pi^{\beta}_{h}(a\mid s)}{32\log 1/\delta+\lambda\cdot\pi^{\beta}_{h}(a\mid s)}
≥
α
⋅
min
⁡
{
π
h
β
​
(
a
∣
s
)
64
​
log
⁡
1
/
δ
,
1
2
​
λ
}
\displaystyle\geq\alpha\cdot\min\left\{\frac{\pi^{\beta}_{h}(a\mid s)}{64\log 1/\delta},\frac{1}{2\lambda}\right\}
where
(
a
)
(a)
follows as long as
δ
≤
0.9
\delta\leq 0.9
and
λ
≥
A
\lambda\geq A
.
In the case when
T
h
​
(
s
)
=
0
T_{h}(s)=0
we have
π
~
h
​
(
a
∣
s
)
=
1
/
A
≥
1
/
λ
\widetilde{\pi}_{h}(a\mid s)=1/A\geq 1/\lambda
, so this lower bound still holds.
Taking a union bound over arms proves the result.
∎
Lemma 2
.
As long as
λ
≥
4
​
log
⁡
(
H
​
T
)
\lambda\geq 4\log(HT)
, we have
𝔼
​
[
𝒥
​
(
π
^
bc
)
−
𝒥
​
(
π
~
)
]
≲
(
1
+
α
​
H
)
⋅
H
2
​
S
​
log
⁡
T
T
+
α
⋅
H
2
​
S
​
λ
T
.
\displaystyle\mathbb{E}[\mathcal{J}(\widehat{\pi}^{\mathrm{bc}})-\mathcal{J}(\widetilde{\pi})]\lesssim(1+\alpha H)\cdot\frac{H^{2}S\log T}{T}+\alpha\cdot\frac{H^{2}S\lambda}{T}.
Proof.
By the Performance-Difference Lemma we have:
𝒥
​
(
π
^
bc
)
−
𝒥
​
(
π
~
)
\displaystyle\mathcal{J}(\widehat{\pi}^{\mathrm{bc}})-\mathcal{J}(\widetilde{\pi})
=
∑
h
=
1
H
∑
s
∈
𝒮
w
h
π
^
bc
​
(
s
)
⋅
(
𝔼
a
∼
π
^
h
bc
​
(
s
)
​
[
Q
h
π
~
​
(
s
,
a
)
]
−
𝔼
a
∼
π
~
h
​
(
s
)
​
[
Q
h
π
~
​
(
s
,
a
)
]
)
\displaystyle=\sum_{h=1}^{H}\sum_{s\in\mathcal{S}}w_{h}^{\widehat{\pi}^{\mathrm{bc}}}(s)\cdot\left(\mathbb{E}_{a\sim\widehat{\pi}^{\mathrm{bc}}_{h}(s)}[Q_{h}^{\widetilde{\pi}}(s,a)]-\mathbb{E}_{a\sim\widetilde{\pi}_{h}(s)}[Q_{h}^{\widetilde{\pi}}(s,a)]\right)
≤
∑
h
=
1
H
∑
s
∈
𝒮
w
h
π
^
bc
​
(
s
)
⋅
|
𝔼
a
∼
π
^
h
bc
​
(
s
)
​
[
Q
h
π
~
​
(
s
,
a
)
]
−
𝔼
a
∼
π
~
h
​
(
s
)
​
[
Q
h
π
~
​
(
s
,
a
)
]
|
.
\displaystyle\leq\sum_{h=1}^{H}\sum_{s\in\mathcal{S}}w_{h}^{\widehat{\pi}^{\mathrm{bc}}}(s)\cdot\left|\mathbb{E}_{a\sim\widehat{\pi}^{\mathrm{bc}}_{h}(s)}[Q_{h}^{\widetilde{\pi}}(s,a)]-\mathbb{E}_{a\sim\widetilde{\pi}_{h}(s)}[Q_{h}^{\widetilde{\pi}}(s,a)]\right|.
(2)
For
(
s
,
h
)
(s,h)
with
N
h
​
(
s
)
>
0
N_{h}(s)>0
, we have
|
𝔼
a
∼
π
^
h
bc
​
(
s
)
​
[
Q
h
π
~
​
(
s
,
a
)
]
−
𝔼
a
∼
π
~
h
​
(
s
)
​
[
Q
h
π
~
​
(
s
,
a
)
]
|
\displaystyle\left|\mathbb{E}_{a\sim\widehat{\pi}^{\mathrm{bc}}_{h}(s)}[Q_{h}^{\widetilde{\pi}}(s,a)]-\mathbb{E}_{a\sim\widetilde{\pi}_{h}(s)}[Q_{h}^{\widetilde{\pi}}(s,a)]\right|
≤
∑
a
∈
𝒜
H
⋅
|
π
^
h
bc
(
a
∣
s
)
−
π
~
h
(
a
∣
s
)
|
,
\displaystyle\leq\sum_{a\in\mathcal{A}}H\cdot|\widehat{\pi}^{\mathrm{bc}}_{h}(a\mid s)-\widetilde{\pi}_{h}(a\mid s)|,
where we have used that
Q
h
π
^
post
​
(
s
,
a
)
∈
[
0
,
H
]
Q_{h}^{\widehat{\pi}^{\mathrm{post}}}(s,a)\in[0,H]
. Then, using the definition of
π
^
bc
\widehat{\pi}^{\mathrm{bc}}
and
π
~
\widetilde{\pi}
we can bound this as
≤
∑
a
∈
𝒜
α
​
H
⋅
|
T
h
​
(
s
,
a
)
T
h
​
(
s
)
−
T
h
​
(
s
,
a
)
+
λ
/
A
T
h
​
(
s
)
+
λ
|
\displaystyle\leq\sum_{a\in\mathcal{A}}\alpha H\cdot\left|\frac{T_{h}(s,a)}{T_{h}(s)}-\frac{T_{h}(s,a)+\lambda/A}{T_{h}(s)+\lambda}\right|
=
∑
a
∈
𝒜
α
​
λ
​
H
A
⋅
|
A
​
T
h
​
(
s
,
a
)
−
T
h
​
(
s
)
T
h
​
(
s
)
​
(
T
h
​
(
s
)
+
λ
)
|
\displaystyle=\sum_{a\in\mathcal{A}}\frac{\alpha\lambda H}{A}\cdot\left|\frac{AT_{h}(s,a)-T_{h}(s)}{T_{h}(s)(T_{h}(s)+\lambda)}\right|
≤
∑
a
∈
𝒜
α
​
λ
​
H
A
⋅
A
​
T
h
​
(
s
,
a
)
+
T
h
​
(
s
)
T
h
​
(
s
)
​
(
T
h
​
(
s
)
+
λ
)
\displaystyle\leq\sum_{a\in\mathcal{A}}\frac{\alpha\lambda H}{A}\cdot\frac{AT_{h}(s,a)+T_{h}(s)}{T_{h}(s)(T_{h}(s)+\lambda)}
=
2
​
α
​
λ
​
H
T
h
​
(
s
)
+
λ
.
\displaystyle=\frac{2\alpha\lambda H}{T_{h}(s)+\lambda}.
Since
𝔼
a
∼
π
^
h
bc
​
(
s
)
​
[
Q
h
π
~
​
(
s
,
a
)
]
−
𝔼
a
∼
π
~
h
​
(
s
)
​
[
Q
h
π
~
​
(
s
,
a
)
]
=
0
\mathbb{E}_{a\sim\widehat{\pi}^{\mathrm{bc}}_{h}(s)}[Q_{h}^{\widetilde{\pi}}(s,a)]-\mathbb{E}_{a\sim\widetilde{\pi}_{h}(s)}[Q_{h}^{\widetilde{\pi}}(s,a)]=0
by construction when
T
h
​
(
s
)
=
0
T_{h}(s)=0
, we then have
(
2
)
≤
∑
h
=
1
H
∑
s
∈
𝒮
w
h
π
^
bc
​
(
s
)
⋅
2
​
α
​
λ
​
H
T
h
​
(
s
)
+
λ
.
\displaystyle\text{(\ref{eq:post_regret_decomp1})}\leq\sum_{h=1}^{H}\sum_{s\in\mathcal{S}}w_{h}^{\widehat{\pi}^{\mathrm{bc}}}(s)\cdot\frac{2\alpha\lambda H}{T_{h}(s)+\lambda}.
Let
ℰ
\mathcal{E}
denote the good event from
Lemma
3
with
δ
=
S
T
\delta=\frac{S}{T}
. Then as long as
λ
≥
4
​
log
⁡
(
H
​
T
)
\lambda\geq 4\log(HT)
we can bound the above as
≤
∑
h
=
1
H
∑
s
∈
𝒮
w
h
π
^
bc
​
(
s
)
⋅
2
​
α
​
λ
​
H
T
h
​
(
s
)
+
λ
​
𝕀
​
{
ℰ
}
+
2
​
H
2
⋅
𝕀
​
{
ℰ
c
}
\displaystyle\leq\sum_{h=1}^{H}\sum_{s\in\mathcal{S}}w_{h}^{\widehat{\pi}^{\mathrm{bc}}}(s)\cdot\frac{2\alpha\lambda H}{T_{h}(s)+\lambda}\mathbb{I}\{\mathcal{E}\}+2H^{2}\cdot\mathbb{I}\{\mathcal{E}^{c}\}
≤
∑
h
=
1
H
∑
s
∈
𝒮
w
h
π
^
bc
​
(
s
)
⋅
4
​
α
​
λ
​
H
w
h
π
β
​
(
s
)
⋅
T
+
λ
+
2
​
H
2
⋅
𝕀
​
{
ℰ
c
}
.
\displaystyle\leq\sum_{h=1}^{H}\sum_{s\in\mathcal{S}}w_{h}^{\widehat{\pi}^{\mathrm{bc}}}(s)\cdot\frac{4\alpha\lambda H}{w_{h}^{\pi^{\beta}}(s)\cdot T+\lambda}+2H^{2}\cdot\mathbb{I}\{\mathcal{E}^{c}\}.
Let
r
~
\widetilde{r}
denote the reward function:
r
~
h
​
(
s
,
a
)
:=
λ
w
h
π
β
​
(
s
)
⋅
T
+
λ
\displaystyle\widetilde{r}_{h}(s,a):=\frac{\lambda}{w_{h}^{\pi^{\beta}}(s)\cdot T+\lambda}
and note that
r
~
∈
[
0
,
1
]
\widetilde{r}\in[0,1]
, and
∑
h
=
1
H
∑
s
∈
𝒮
w
h
π
^
bc
​
(
s
)
⋅
4
​
α
​
λ
​
H
w
h
π
β
​
(
s
)
⋅
T
+
λ
=
4
​
α
​
H
⋅
𝒥
​
(
π
^
bc
;
r
~
)
.
\displaystyle\sum_{h=1}^{H}\sum_{s\in\mathcal{S}}w_{h}^{\widehat{\pi}^{\mathrm{bc}}}(s)\cdot\frac{4\alpha\lambda H}{w_{h}^{\pi^{\beta}}(s)\cdot T+\lambda}=4\alpha H\cdot\mathcal{J}(\widehat{\pi}^{\mathrm{bc}};\widetilde{r}).
By Theorem 4.4 of
Rajaraman
et al.
(
2020
)
, we have
1
1
1
Note that Theorem 4.4 of
Rajaraman
et al.
(
2020
)
shows an inequality in the opposite direction of what we show here: they bound
𝒥
​
(
π
β
;
r
~
)
−
𝔼
​
[
𝒥
​
(
π
^
bc
;
r
~
)
]
\mathcal{J}(\pi^{\beta};\widetilde{r})-\mathbb{E}[\mathcal{J}(\widehat{\pi}^{\mathrm{bc}};\widetilde{r})]
instead of
𝔼
​
[
𝒥
​
(
π
^
bc
;
r
~
)
]
−
𝒥
​
(
π
β
;
r
~
)
\mathbb{E}[\mathcal{J}(\widehat{\pi}^{\mathrm{bc}};\widetilde{r})]-\mathcal{J}(\pi^{\beta};\widetilde{r})
. However, we see that the only place in their proof where their argument relied on this ordering is in Lemma A.8. We show in
Lemma
4
that a reverse version of their Lemma A.8 holds, allowing us to instead bound
𝔼
​
[
𝒥
​
(
π
^
bc
;
r
~
)
]
−
𝒥
​
(
π
β
;
r
~
)
\mathbb{E}[\mathcal{J}(\widehat{\pi}^{\mathrm{bc}};\widetilde{r})]-\mathcal{J}(\pi^{\beta};\widetilde{r})
.
𝔼
​
[
𝒥
​
(
π
^
bc
;
r
~
)
]
\displaystyle\mathbb{E}[\mathcal{J}(\widehat{\pi}^{\mathrm{bc}};\widetilde{r})]
≲
𝒥
​
(
π
β
;
r
~
)
+
H
2
​
S
​
log
⁡
T
T
\displaystyle\lesssim\mathcal{J}(\pi^{\beta};\widetilde{r})+\frac{H^{2}S\log T}{T}
=
∑
h
=
1
H
∑
s
∈
𝒮
w
h
π
β
​
(
s
)
⋅
λ
w
h
π
β
​
(
s
)
⋅
T
+
λ
+
H
2
​
S
​
log
⁡
T
T
\displaystyle=\sum_{h=1}^{H}\sum_{s\in\mathcal{S}}w_{h}^{\pi^{\beta}}(s)\cdot\frac{\lambda}{w_{h}^{\pi^{\beta}}(s)\cdot T+\lambda}+\frac{H^{2}S\log T}{T}
≤
H
​
S
​
λ
T
+
H
2
​
S
​
log
⁡
T
T
.
\displaystyle\leq\frac{HS\lambda}{T}+\frac{H^{2}S\log T}{T}.
Noting that
𝔼
​
[
2
​
H
2
⋅
𝕀
​
{
ℰ
c
}
]
≤
2
​
H
2
​
δ
≤
2
​
H
2
​
S
T
\mathbb{E}[2H^{2}\cdot\mathbb{I}\{\mathcal{E}^{c}\}]\leq 2H^{2}\delta\leq\frac{2H^{2}S}{T}
completes the proof.
∎
Lemma 3
.
With probability at least
1
−
δ
1-\delta
, for all
(
s
,
h
)
(s,h)
, we have
T
h
​
(
s
)
+
λ
≥
1
2
​
w
h
π
β
​
(
s
)
⋅
T
+
1
2
​
λ
\displaystyle T_{h}(s)+\lambda\geq\frac{1}{2}w_{h}^{\pi^{\beta}}(s)\cdot T+\frac{1}{2}\lambda
as long as
λ
≥
4
​
log
⁡
S
​
H
δ
\lambda\geq 4\log\frac{SH}{\delta}
.
Proof.
Consider some
(
s
,
h
)
(s,h)
and note that
𝔼
​
[
T
h
​
(
s
)
/
T
]
=
w
h
π
β
​
(
s
)
\mathbb{E}[T_{h}(s)/T]=w_{h}^{\pi^{\beta}}(s)
. By Bernstein’s inequality, we have with probability
1
−
δ
/
S
​
H
1-\delta/SH
:
T
h
​
(
s
)
≥
w
h
π
β
​
(
s
)
⋅
T
−
2
​
w
h
π
β
​
(
s
)
⋅
T
⋅
log
⁡
S
​
H
δ
−
2
3
​
log
⁡
S
​
H
δ
.
\displaystyle T_{h}(s)\geq w_{h}^{\pi^{\beta}}(s)\cdot T-\sqrt{2w_{h}^{\pi^{\beta}}(s)\cdot T\cdot\log\frac{SH}{\delta}}-\frac{2}{3}\log\frac{SH}{\delta}.
We would then like to show that
w
h
π
β
​
(
s
)
⋅
T
−
2
​
w
h
π
β
​
(
s
)
⋅
T
⋅
log
⁡
S
​
H
δ
−
2
3
​
log
⁡
S
​
H
δ
+
λ
≥
1
2
​
(
w
h
π
β
​
(
s
)
⋅
T
+
λ
)
\displaystyle w_{h}^{\pi^{\beta}}(s)\cdot T-\sqrt{2w_{h}^{\pi^{\beta}}(s)\cdot T\cdot\log\frac{SH}{\delta}}-\frac{2}{3}\log\frac{SH}{\delta}+\lambda\geq\frac{1}{2}(w_{h}^{\pi^{\beta}}(s)\cdot T+\lambda)
⇔
1
2
​
w
h
π
β
​
(
s
)
⋅
T
+
1
2
​
λ
≥
2
​
w
h
π
β
​
(
s
)
⋅
T
⋅
log
⁡
S
​
H
δ
+
2
3
​
log
⁡
S
​
H
δ
\displaystyle\iff\frac{1}{2}w_{h}^{\pi^{\beta}}(s)\cdot T+\frac{1}{2}\lambda\geq\sqrt{2w_{h}^{\pi^{\beta}}(s)\cdot T\cdot\log\frac{SH}{\delta}}+\frac{2}{3}\log\frac{SH}{\delta}
As we have assumed
λ
≥
4
​
log
⁡
S
​
H
δ
\lambda\geq 4\log\frac{SH}{\delta}
, it suffices to show
1
2
​
w
h
π
β
​
(
s
)
⋅
T
+
log
⁡
S
​
H
δ
≥
2
​
w
h
π
β
​
(
s
)
⋅
T
⋅
log
⁡
S
​
H
δ
.
\displaystyle\frac{1}{2}w_{h}^{\pi^{\beta}}(s)\cdot T+\log\frac{SH}{\delta}\geq\sqrt{2w_{h}^{\pi^{\beta}}(s)\cdot T\cdot\log\frac{SH}{\delta}}.
However, this is true by the AM-GM inequality. A union bound proves the result.
∎
Lemma 4
(Reversed version of Lemma A.8 of
Rajaraman
et al.
(
2020
)
)
.
Adopting the notation from
Rajaraman
et al.
(
2020
)
, we have
𝔼
​
[
Pr
π
first
​
[
ℰ
]
]
≤
S
​
H
​
log
⁡
N
N
\displaystyle\mathbb{E}[\mathrm{Pr}_{\pi^{\mathrm{first}}}[\mathcal{E}]]\leq\frac{SH\log N}{N}
for
ℰ
c
\mathcal{E}^{c}
the event that within a trajectory, the policy only visits states for which
T
h
​
(
s
)
>
0
T_{h}(s)>0
.
Proof.
Let
ℰ
s
,
h
\mathcal{E}_{s,h}
denote the event that the state
s
s
is visited at step
h
h
and
T
h
​
(
s
)
=
0
T_{h}(s)=0
, and
ℰ
h
:=
∪
s
∈
𝒮
ℰ
s
,
h
\mathcal{E}_{h}:=\cup_{s\in\mathcal{S}}\mathcal{E}_{s,h}
. Then, by simple set inclusions, we have:
ℰ
\displaystyle\mathcal{E}
=
⋃
h
∈
[
H
]
⋃
s
∈
𝒮
ℰ
s
,
h
=
⋃
h
∈
[
H
]
⋃
s
∈
𝒮
(
ℰ
s
,
h
∩
⋂
h
′
<
h
ℰ
h
′
c
)
.
\displaystyle=\bigcup_{h\in[H]}\bigcup_{s\in\mathcal{S}}\mathcal{E}_{s,h}=\bigcup_{h\in[H]}\bigcup_{s\in\mathcal{S}}\bigg(\mathcal{E}_{s,h}\cap\bigcap_{h^{\prime}<h}\mathcal{E}_{h^{\prime}}^{c}\bigg).
By a union bound it follows that
𝔼
​
[
Pr
π
first
​
[
ℰ
]
]
\displaystyle\mathbb{E}[\mathrm{Pr}_{\pi^{\mathrm{first}}}[\mathcal{E}]]
≤
∑
h
∈
[
H
]
∑
s
∈
𝒮
𝔼
​
[
Pr
π
first
​
[
ℰ
s
,
h
∩
⋂
h
′
<
h
ℰ
h
′
c
]
]
.
\displaystyle\leq\sum_{h\in[H]}\sum_{s\in\mathcal{S}}\mathbb{E}[\mathrm{Pr}_{\pi^{\mathrm{first}}}[\mathcal{E}_{s,h}\cap\bigcap_{h^{\prime}<h}\mathcal{E}_{h^{\prime}}^{c}]].
Now note that
Pr
π
first
​
[
ℰ
s
,
h
∩
⋂
h
′
<
h
ℰ
h
′
c
]
\displaystyle\mathrm{Pr}_{\pi^{\mathrm{first}}}[\mathcal{E}_{s,h}\cap\bigcap_{h^{\prime}<h}\mathcal{E}_{h^{\prime}}^{c}]
=
Pr
π
first
​
[
ℰ
s
,
h
∣
⋂
h
′
<
h
ℰ
h
′
c
]
​
Pr
π
first
​
[
⋂
h
′
<
h
ℰ
h
′
c
]
\displaystyle=\mathrm{Pr}_{\pi^{\mathrm{first}}}[\mathcal{E}_{s,h}\mid\bigcap_{h^{\prime}<h}\mathcal{E}_{h^{\prime}}^{c}]\mathrm{Pr}_{\pi^{\mathrm{first}}}[\bigcap_{h^{\prime}<h}\mathcal{E}_{h^{\prime}}^{c}]
=
Pr
π
first
​
[
ℰ
s
,
h
∣
⋂
h
′
<
h
ℰ
h
′
c
]
​
Pr
π
first
​
[
ℰ
h
−
1
c
∣
⋂
h
′
<
h
−
1
ℰ
h
′
c
]
​
Pr
π
first
​
[
⋂
h
′
<
h
−
1
ℰ
h
′
c
]
\displaystyle=\mathrm{Pr}_{\pi^{\mathrm{first}}}[\mathcal{E}_{s,h}\mid\bigcap_{h^{\prime}<h}\mathcal{E}_{h^{\prime}}^{c}]\mathrm{Pr}_{\pi^{\mathrm{first}}}[\mathcal{E}_{h-1}^{c}\mid\bigcap_{h^{\prime}<h-1}\mathcal{E}_{h^{\prime}}^{c}]\mathrm{Pr}_{\pi^{\mathrm{first}}}[\bigcap_{h^{\prime}<h-1}\mathcal{E}_{h^{\prime}}^{c}]
⋮
\displaystyle\vdots
=
Pr
π
first
​
[
ℰ
s
,
h
∣
⋂
h
′
<
h
ℰ
h
′
c
]
⋅
∏
h
′
<
h
Pr
π
first
​
[
ℰ
h
′
c
∣
⋂
h
′′
<
h
′
ℰ
h
′′
c
]
.
\displaystyle=\mathrm{Pr}_{\pi^{\mathrm{first}}}[\mathcal{E}_{s,h}\mid\bigcap_{h^{\prime}<h}\mathcal{E}_{h^{\prime}}^{c}]\cdot\prod_{h^{\prime}<h}\mathrm{Pr}_{\pi^{\mathrm{first}}}[\mathcal{E}^{c}_{h^{\prime}}\mid\bigcap_{h^{\prime\prime}<h^{\prime}}\mathcal{E}^{c}_{h^{\prime\prime}}].
If the event
⋂
h
′
<
h
ℰ
h
′
c
\bigcap_{h^{\prime}<h}\mathcal{E}_{h^{\prime}}^{c}
holds, then up to step
h
h
no states are encountered for which
T
h
′
​
(
s
)
=
0
T_{h^{\prime}}(s)=0
. Thus, on such states,
π
first
\pi^{\mathrm{first}}
and
π
orc
−
first
\pi^{\mathrm{orc-first}}
will behave identically. It follows that
𝔼
​
[
Pr
π
first
​
[
ℰ
s
,
h
∣
⋂
h
′
<
h
ℰ
h
′
c
]
]
=
𝔼
​
[
Pr
π
orc
−
first
​
[
ℰ
s
,
h
∣
⋂
h
′
<
h
ℰ
h
′
c
]
]
\mathbb{E}[\mathrm{Pr}_{\pi^{\mathrm{first}}}[\mathcal{E}_{s,h}\mid\bigcap_{h^{\prime}<h}\mathcal{E}_{h^{\prime}}^{c}]]=\mathbb{E}[\mathrm{Pr}_{\pi^{\mathrm{orc-first}}}[\mathcal{E}_{s,h}\mid\bigcap_{h^{\prime}<h}\mathcal{E}_{h^{\prime}}^{c}]]
. By a similar argument, we have
Pr
π
orc
−
first
​
[
ℰ
h
′
c
∣
⋂
h
′′
<
h
′
ℰ
h
′′
c
]
=
Pr
π
first
​
[
ℰ
h
′
c
∣
⋂
h
′′
<
h
′
ℰ
h
′′
c
]
\mathrm{Pr}_{\pi^{\mathrm{orc-first}}}[\mathcal{E}^{c}_{h^{\prime}}\mid\bigcap_{h^{\prime\prime}<h^{\prime}}\mathcal{E}^{c}_{h^{\prime\prime}}]=\mathrm{Pr}_{\pi^{\mathrm{first}}}[\mathcal{E}^{c}_{h^{\prime}}\mid\bigcap_{h^{\prime\prime}<h^{\prime}}\mathcal{E}^{c}_{h^{\prime\prime}}]
for each
h
′
<
h
h^{\prime}<h
.
Thus,
Pr
π
first
​
[
ℰ
s
,
h
∩
⋂
h
′
<
h
ℰ
h
′
c
]
=
Pr
π
orc
−
first
​
[
ℰ
s
,
h
∩
⋂
h
′
<
h
ℰ
h
′
c
]
.
\displaystyle\mathrm{Pr}_{\pi^{\mathrm{first}}}[\mathcal{E}_{s,h}\cap\bigcap_{h^{\prime}<h}\mathcal{E}_{h^{\prime}}^{c}]=\mathrm{Pr}_{\pi^{\mathrm{orc-first}}}[\mathcal{E}_{s,h}\cap\bigcap_{h^{\prime}<h}\mathcal{E}_{h^{\prime}}^{c}].
It follows that
𝔼
​
[
Pr
π
first
​
[
ℰ
]
]
\displaystyle\mathbb{E}[\mathrm{Pr}_{\pi^{\mathrm{first}}}[\mathcal{E}]]
≤
∑
h
∈
[
H
]
∑
s
∈
𝒮
𝔼
​
[
Pr
π
orc
−
first
​
[
ℰ
s
,
h
∩
⋂
h
′
<
h
ℰ
h
′
c
]
]
≤
∑
h
∈
[
H
]
∑
s
∈
𝒮
𝔼
​
[
Pr
π
orc
−
first
​
[
ℰ
s
,
h
]
]
.
\displaystyle\leq\sum_{h\in[H]}\sum_{s\in\mathcal{S}}\mathbb{E}[\mathrm{Pr}_{\pi^{\mathrm{orc-first}}}[\mathcal{E}_{s,h}\cap\bigcap_{h^{\prime}<h}\mathcal{E}_{h^{\prime}}^{c}]]\leq\sum_{h\in[H]}\sum_{s\in\mathcal{S}}\mathbb{E}[\mathrm{Pr}_{\pi^{\mathrm{orc-first}}}[\mathcal{E}_{s,h}]].
From here the proof follows identically to the proof of Lemma A.8 of
Rajaraman
et al.
(
2020
)
.
∎
Proof of
LABEL:thm:main
.
Set
λ
=
max
⁡
{
A
,
4
​
log
⁡
(
H
​
T
)
}
\lambda=\max\{A,4\log(HT)\}
and
α
=
1
max
⁡
{
A
,
H
,
log
⁡
(
H
​
T
)
}
\alpha=\frac{1}{\max\{A,H,\log(HT)\}}
.
We have
𝒥
​
(
π
β
)
−
𝔼
​
[
𝒥
​
(
π
^
bc
)
]
+
𝔼
​
[
𝒥
​
(
π
^
bc
)
]
−
𝔼
​
[
𝒥
​
(
π
~
)
]
≲
H
2
​
S
​
log
⁡
T
T
+
(
1
+
α
​
H
)
⋅
H
2
​
S
​
log
⁡
T
T
+
α
⋅
H
2
​
S
​
λ
T
\displaystyle\mathcal{J}(\pi^{\beta})-\mathbb{E}[\mathcal{J}(\widehat{\pi}^{\mathrm{bc}})]+\mathbb{E}[\mathcal{J}(\widehat{\pi}^{\mathrm{bc}})]-\mathbb{E}[\mathcal{J}(\widetilde{\pi})]\lesssim\frac{H^{2}S\log T}{T}+(1+\alpha H)\cdot\frac{H^{2}S\log T}{T}+\alpha\cdot\frac{H^{2}S\lambda}{T}
where we bound
𝒥
​
(
π
β
)
−
𝔼
​
[
𝒥
​
(
π
^
bc
)
]
\mathcal{J}(\pi^{\beta})-\mathbb{E}[\mathcal{J}(\widehat{\pi}^{\mathrm{bc}})]
by Theorem 4.4 of
Rajaraman
et al.
(
2020
)
, and
𝔼
​
[
𝒥
​
(
π
^
bc
)
]
−
𝔼
​
[
𝒥
​
(
π
~
)
]
\mathbb{E}[\mathcal{J}(\widehat{\pi}^{\mathrm{bc}})]-\mathbb{E}[\mathcal{J}(\widetilde{\pi})]
by
Lemma
2
since
λ
≥
4
​
log
⁡
(
H
​
T
)
\lambda\geq 4\log(HT)
. By our choice of
α
=
1
max
⁡
{
A
,
H
,
log
⁡
(
H
​
T
)
}
\alpha=\frac{1}{\max\{A,H,\log(HT)\}}
, we can bound all of this as
≲
H
2
​
S
​
log
⁡
T
T
.
\displaystyle\lesssim\frac{H^{2}S\log T}{T}.
This proves the suboptimality guarantee. To show that
π
~
\widetilde{\pi}
achieves demonstrator action coverage, we apply
Lemma
1
using our values of
λ
\lambda
and
α
\alpha
.
∎
B.4
Optimality of Posterior Demonstrator Policy
Let
ℳ
\mathcal{M}
denote a multi-armed bandit with
A
>
1
A>1
actions where
r
​
(
a
1
)
=
1
r(a_{1})=1
and
r
​
(
a
i
)
=
0
r(a_{i})=0
for
i
>
1
i>1
. Let
π
β
,
i
\pi^{\beta,i}
denote the policy defined as
π
β
,
i
​
(
a
)
=
{
1
−
α
a
=
1
α
a
=
i
0
o.w.
\displaystyle\pi^{\beta,i}(a)=\begin{cases}1-\alpha&a=1\\
\alpha&a=i\\
0&\text{o.w.}\end{cases}
for
i
>
1
i>1
and
α
\alpha
some value we will set, and
π
β
,
1
​
(
1
)
=
1
\pi^{\beta,1}(1)=1
.
We let
ℳ
i
=
(
ℳ
,
π
β
,
i
)
\mathcal{M}^{i}=(\mathcal{M},\pi^{\beta,i})
the instance-demonstrator pair,
𝔼
i
​
[
⋅
]
\mathbb{E}^{i}[\cdot]
the expectation on this instance,
ℙ
i
\mathbb{P}^{i}
the distribution on this instance, and
ℙ
i
,
T
=
⊗
t
=
1
T
ℙ
i
\mathbb{P}^{i,T}=\otimes_{t=1}^{T}\mathbb{P}^{i}
.
Lemma 5
.
Consider the instance constructed above. Then we have that, for
j
≠
i
j\neq i
:
ℙ
i
​
[
π
^
​
(
i
)
≥
γ
⋅
α
]
≤
2
⋅
ℙ
j
​
[
π
^
​
(
i
)
≥
γ
⋅
α
]
+
T
⋅
α
.
\displaystyle\mathbb{P}^{i}[\widehat{\pi}(i)\geq\gamma\cdot\alpha]\leq 2\cdot\mathbb{P}^{j}[\widehat{\pi}(i)\geq\gamma\cdot\alpha]+T\cdot\alpha.
Proof.
This follows from Lemma A.11 of
Foster
et al.
(
2021
)
, which immediately gives that:
ℙ
i
[
{
π
^
(
i
)
≥
γ
⋅
α
]
≤
2
⋅
ℙ
j
[
π
^
(
i
)
≥
γ
⋅
α
]
+
D
H
2
(
ℙ
i
,
T
,
ℙ
j
,
T
)
,
\displaystyle\mathbb{P}^{i}[\{\widehat{\pi}(i)\geq\gamma\cdot\alpha]\leq 2\cdot\mathbb{P}^{j}[\widehat{\pi}(i)\geq\gamma\cdot\alpha]+D_{\mathrm{H}}^{2}(\mathbb{P}^{i,T},\mathbb{P}^{j,T}),
where
D
H
​
(
⋅
,
⋅
)
D_{\mathrm{H}}(\cdot,\cdot)
denotes the Hellinger distance.
Since the squared Hellinger distance is subadditive we have
D
H
2
​
(
ℙ
i
,
T
,
ℙ
j
,
T
)
≤
T
⋅
D
H
2
​
(
ℙ
i
,
ℙ
j
)
.
\displaystyle D_{\mathrm{H}}^{2}(\mathbb{P}^{i,T},\mathbb{P}^{j,T})\leq T\cdot D_{\mathrm{H}}^{2}(\mathbb{P}^{i},\mathbb{P}^{j}).
By elementary calculations we see that
D
H
2
​
(
ℙ
i
,
ℙ
j
)
=
α
D_{\mathrm{H}}^{2}(\mathbb{P}^{i},\mathbb{P}^{j})=\alpha
, which proves the result.
∎
Theorem 1
(Full version of
LABEL:thm:main_lb
)
.
Let
π
^
\widehat{\pi}
achieve demonstrator action coverage with some parameter
γ
\gamma
for each
ℳ
i
,
i
∈
[
A
]
\mathcal{M}^{i},i\in[A]
, and some
δ
∈
(
0
,
1
/
4
]
\delta\in(0,1/4]
, and assume that
𝒥
​
(
π
β
,
i
)
−
𝔼
i
​
[
𝒥
​
(
π
^
)
]
≤
ξ
,
∀
i
≥
1
\displaystyle\mathcal{J}(\pi^{\beta,i})-\mathbb{E}^{i}[\mathcal{J}(\widehat{\pi})]\leq\xi,\quad\forall i\geq 1
for some
ξ
>
0
\xi>0
. Then if
T
≤
1
4
​
α
T\leq\frac{1}{4\alpha}
, it must be the case that
γ
≤
ξ
2
​
A
​
α
.
\displaystyle\gamma\leq\frac{\xi}{2A\alpha}.
In particular, setting
ξ
=
c
⋅
log
⁡
T
T
\xi=c\cdot\frac{\log T}{T}
and if
α
=
1
2
​
T
\alpha=\frac{1}{2T}
, we have
γ
≤
c
⋅
log
⁡
T
A
.
\displaystyle\gamma\leq c\cdot\frac{\log T}{A}.
Proof.
Our goal is to find the maximum value of
γ
\gamma
such that our constraint on the optimality of
π
^
\widehat{\pi}
is met, for each
ℳ
i
\mathcal{M}^{i}
. In particular, this can be upper bounded as
max
π
^
,
γ
⁡
γ
s.t.
ℙ
i
​
[
{
π
^
​
(
a
)
≥
γ
⋅
π
β
​
(
a
)
,
∀
a
∈
𝒜
}
]
≥
1
−
δ
,
𝒥
​
(
π
β
,
i
)
−
𝔼
i
​
[
𝒥
​
(
π
^
)
]
≤
ξ
,
∀
i
≥
1
.
\displaystyle\max_{\widehat{\pi},\gamma}\gamma\quad\text{s.t.}\quad\mathbb{P}^{i}[\{\widehat{\pi}(a)\geq\gamma\cdot\pi^{\beta}(a),\forall a\in\mathcal{A}\}]\geq 1-\delta,\ \mathcal{J}(\pi^{\beta,i})-\mathbb{E}^{i}[\mathcal{J}(\widehat{\pi})]\leq\xi,\ \forall i\geq 1.
(3)
Note that for
ℳ
i
,
i
≥
1
\mathcal{M}^{i},i\geq 1
, the event
{
π
^
​
(
a
)
≥
γ
⋅
π
β
,
i
​
(
a
)
,
∀
a
∈
𝒜
}
\{\widehat{\pi}(a)\geq\gamma\cdot\pi^{\beta,i}(a),\forall a\in\mathcal{A}\}
is a subset of the event
{
π
^
​
(
i
)
≥
γ
⋅
α
}
\{\widehat{\pi}(i)\geq\gamma\cdot\alpha\}
. This allows us to bound
(
3
)
as
max
π
^
,
γ
⁡
γ
s.t.
ℙ
i
​
[
π
^
​
(
i
)
≥
γ
⋅
α
]
≥
1
−
δ
,
𝒥
​
(
π
β
,
i
)
−
𝔼
i
​
[
𝒥
​
(
π
^
)
]
≤
ξ
,
∀
i
≥
1
.
\displaystyle\max_{\widehat{\pi},\gamma}\gamma\quad\text{s.t.}\quad\mathbb{P}^{i}[\widehat{\pi}(i)\geq\gamma\cdot\alpha]\geq 1-\delta,\ \mathcal{J}(\pi^{\beta,i})-\mathbb{E}^{i}[\mathcal{J}(\widehat{\pi})]\leq\xi,\ \forall i\geq 1.
(4)
By
Lemma
5
, we have that for each
i
>
1
i>1
,
ℙ
i
​
[
π
^
​
(
i
)
≥
γ
⋅
α
]
≤
2
⋅
ℙ
1
​
[
π
^
​
(
i
)
≥
γ
⋅
α
]
+
T
⋅
α
.
\displaystyle\mathbb{P}^{i}[\widehat{\pi}(i)\geq\gamma\cdot\alpha]\leq 2\cdot\mathbb{P}^{1}[\widehat{\pi}(i)\geq\gamma\cdot\alpha]+T\cdot\alpha.
Furthermore, on
ℳ
1
\mathcal{M}^{1}
we have
𝒥
​
(
π
β
,
1
)
−
𝔼
1
​
[
𝒥
​
(
π
^
)
]
=
𝔼
1
​
[
∑
i
>
1
π
^
​
(
i
)
]
\mathcal{J}(\pi^{\beta,1})-\mathbb{E}^{1}[\mathcal{J}(\widehat{\pi})]=\mathbb{E}^{1}[\sum_{i>1}\widehat{\pi}(i)]
.
Given this, we can upper bound
(
4
)
as
max
π
^
,
γ
⁡
γ
s.t.
ℙ
1
​
[
π
^
​
(
i
)
≥
γ
⋅
α
]
≥
1
2
⋅
(
1
−
δ
−
T
⋅
α
)
,
∀
i
>
1
,
𝔼
1
​
[
∑
i
>
1
π
^
​
(
i
)
]
≤
ξ
.
\displaystyle\max_{\widehat{\pi},\gamma}\gamma\quad\text{s.t.}\quad\mathbb{P}^{1}[\widehat{\pi}(i)\geq\gamma\cdot\alpha]\geq\frac{1}{2}\cdot(1-\delta-T\cdot\alpha),\forall i>1,\ \mathbb{E}^{1}[\sum_{i>1}\widehat{\pi}(i)]\leq\xi.
(5)
By Markov’s inequality, we have
ℙ
1
​
[
π
^
​
(
i
)
≥
γ
⋅
α
]
≤
𝔼
1
​
[
π
^
​
(
i
)
]
γ
⋅
α
.
\displaystyle\mathbb{P}^{1}[\widehat{\pi}(i)\geq\gamma\cdot\alpha]\leq\frac{\mathbb{E}^{1}[\widehat{\pi}(i)]}{\gamma\cdot\alpha}.
Furthermore, since we have assumed
δ
≤
1
/
4
\delta\leq 1/4
and
T
≤
1
4
​
α
T\leq\frac{1}{4\alpha}
, we have
1
2
⋅
(
1
−
δ
−
T
⋅
α
)
≥
1
4
\frac{1}{2}\cdot(1-\delta-T\cdot\alpha)\geq\frac{1}{4}
.
We can therefore bound
(
5
)
as
max
π
^
,
γ
⁡
γ
s.t.
𝔼
1
​
[
π
^
​
(
i
)
]
≥
1
4
⋅
γ
​
α
,
∀
i
>
1
,
𝔼
1
​
[
∑
i
>
1
π
^
​
(
i
)
]
≤
ξ
.
\displaystyle\max_{\widehat{\pi},\gamma}\gamma\quad\text{s.t.}\quad\mathbb{E}^{1}[\widehat{\pi}(i)]\geq\frac{1}{4}\cdot\gamma\alpha,\forall i>1,\ \mathbb{E}^{1}[\sum_{i>1}\widehat{\pi}(i)]\leq\xi.
(6)
However, we see then that we immediately have
γ
≤
ξ
4
​
(
A
−
1
)
​
α
.
\displaystyle\gamma\leq\frac{\xi}{4(A-1)\alpha}.
This proves the result as long as
A
>
1
A>1
.
∎
Appendix C
Posterior Demonstrator Policy for Gaussian Demonstrator
Let
P
(
⋅
∣
μ
)
P(\cdot\mid\mu)
denote the distribution
𝒩
​
(
μ
,
Σ
)
\mathcal{N}(\mu,\Sigma)
, where we assume
μ
\mu
is unknown and
Σ
\Sigma
is known. Assume that we have samples
𝔇
=
{
x
1
,
…
,
x
T
}
∼
P
(
⋅
∣
μ
⋆
)
\mathfrak{D}=\{x_{1},\ldots,x_{T}\}\sim P(\cdot\mid\mu^{\star})
.
Let
Q
prior
=
𝒩
​
(
0
,
Λ
0
)
Q_{\mathrm{prior}}=\mathcal{N}(0,\Lambda_{0})
denote the prior on
μ
\mu
.
Throughout this section we let
=
d
=^{d}
denote equality in distribution.
Lemma 6
.
Under
Q
prior
Q_{\mathrm{prior}}
, we have that the posterior
Q
post
Q_{\mathrm{post}}
on
μ
\mu
is:
Q
post
(
⋅
∣
𝔇
)
=
𝒩
(
Λ
post
Σ
−
1
⋅
∑
t
=
1
T
x
t
,
Λ
post
)
,
\displaystyle Q_{\mathrm{post}}(\cdot\mid\mathfrak{D})=\mathcal{N}\left(\Lambda_{\mathrm{post}}\Sigma^{-1}\cdot\sum_{t=1}^{T}x_{t},\Lambda_{\mathrm{post}}\right),
for
Λ
post
−
1
=
Λ
0
−
1
+
T
⋅
Σ
−
1
\Lambda_{\mathrm{post}}^{-1}=\Lambda_{0}^{-1}+T\cdot\Sigma^{-1}
.
Proof.
Dropping terms that do not depend on
μ
\mu
, we have
Q
post
​
(
μ
∣
𝔇
)
\displaystyle Q_{\mathrm{post}}(\mu\mid\mathfrak{D})
=
P
​
(
𝔇
∣
μ
)
​
Q
prior
​
(
μ
)
P
​
(
𝔇
)
\displaystyle=\frac{P(\mathfrak{D}\mid\mu)Q_{\mathrm{prior}}(\mu)}{P(\mathfrak{D})}
∝
exp
⁡
(
−
1
2
​
∑
t
=
1
T
(
x
t
−
μ
)
⊤
​
Σ
−
1
​
(
x
t
−
μ
)
)
⋅
exp
⁡
(
−
1
2
​
μ
⊤
​
Λ
0
​
μ
)
\displaystyle\propto\exp\left(-\frac{1}{2}\sum_{t=1}^{T}(x_{t}-\mu)^{\top}\Sigma^{-1}(x_{t}-\mu)\right)\cdot\exp\left(-\frac{1}{2}\mu^{\top}\Lambda_{0}\mu\right)
∝
exp
⁡
(
−
1
2
​
T
​
μ
⊤
​
Σ
−
1
​
μ
−
1
2
​
μ
⊤
​
Q
prior
−
1
​
μ
+
μ
⊤
​
Σ
−
1
⋅
∑
t
=
1
T
x
t
)
\displaystyle\propto\exp\left(-\frac{1}{2}T\mu^{\top}\Sigma^{-1}\mu-\frac{1}{2}\mu^{\top}Q_{\mathrm{prior}}^{-1}\mu+\mu^{\top}\Sigma^{-1}\cdot\sum_{t=1}^{T}x_{t}\right)
=
exp
⁡
(
−
1
2
​
(
μ
−
Λ
post
​
v
)
⊤
​
Λ
post
−
1
​
(
μ
−
Λ
post
​
v
)
+
1
2
​
v
⊤
​
Λ
post
​
v
)
\displaystyle=\exp\left(-\frac{1}{2}(\mu-\Lambda_{\mathrm{post}}v)^{\top}\Lambda_{\mathrm{post}}^{-1}(\mu-\Lambda_{\mathrm{post}}v)+\frac{1}{2}v^{\top}\Lambda_{\mathrm{post}}v\right)
for
Λ
post
−
1
=
Λ
0
−
1
+
T
⋅
Σ
−
1
\Lambda_{\mathrm{post}}^{-1}=\Lambda_{0}^{-1}+T\cdot\Sigma^{-1}
, and
v
=
Σ
−
1
⋅
∑
t
=
1
T
x
t
v=\Sigma^{-1}\cdot\sum_{t=1}^{T}x_{t}
.
∎
Lemma 7
(General version of
LABEL:prop:policy_post_opt
)
.
Let
μ
^
=
arg
​
min
μ
​
∑
t
=
1
T
(
μ
−
x
~
t
)
⊤
​
Σ
−
1
​
(
μ
−
x
~
t
)
+
(
μ
−
μ
~
)
⊤
​
Λ
0
−
1
​
(
μ
−
μ
~
)
,
\displaystyle\widehat{\mu}=\operatorname*{arg\,min}_{\mu}\sum_{t=1}^{T}(\mu-\widetilde{x}_{t})^{\top}\Sigma^{-1}(\mu-\widetilde{x}_{t})+(\mu-\widetilde{\mu})^{\top}\Lambda_{0}^{-1}(\mu-\widetilde{\mu}),
for
x
~
t
=
x
t
+
w
t
\widetilde{x}_{t}=x_{t}+w_{t}
,
w
t
∼
𝒩
​
(
0
,
Σ
)
w_{t}\sim\mathcal{N}(0,\Sigma)
, and
μ
~
∼
Q
prior
\widetilde{\mu}\sim Q_{\mathrm{prior}}
.
Then
μ
^
=
d
Q
post
(
⋅
∣
𝔇
)
\widehat{\mu}=^{d}Q_{\mathrm{post}}(\cdot\mid\mathfrak{D})
.
Proof.
By computing the gradient of the objective, setting it equal to 0, and solving for
μ
\mu
, we see that
μ
^
\displaystyle\widehat{\mu}
=
(
Λ
0
−
1
+
T
​
Σ
−
1
)
−
1
⋅
(
Σ
−
1
⋅
∑
t
=
1
T
x
~
t
+
Λ
0
−
1
​
μ
~
)
\displaystyle=(\Lambda_{0}^{-1}+T\Sigma^{-1})^{-1}\cdot\left(\Sigma^{-1}\cdot\sum_{t=1}^{T}\widetilde{x}_{t}+\Lambda_{0}^{-1}\widetilde{\mu}\right)
=
(
Λ
0
−
1
+
T
​
Σ
−
1
)
−
1
⋅
Σ
−
1
⋅
∑
t
=
1
T
x
t
+
(
Λ
0
−
1
+
T
​
Σ
−
1
)
−
1
⋅
(
Σ
−
1
⋅
∑
t
=
1
T
w
t
+
Λ
0
−
1
​
μ
~
)
.
\displaystyle=(\Lambda_{0}^{-1}+T\Sigma^{-1})^{-1}\cdot\Sigma^{-1}\cdot\sum_{t=1}^{T}x_{t}+(\Lambda_{0}^{-1}+T\Sigma^{-1})^{-1}\cdot\left(\Sigma^{-1}\cdot\sum_{t=1}^{T}w_{t}+\Lambda_{0}^{-1}\widetilde{\mu}\right).
Note that the first term in the above is deterministic conditioned on
𝔇
\mathfrak{D}
, and the second term is mean 0 and has covariance
(
Λ
0
−
1
+
T
​
Σ
−
1
)
−
1
(\Lambda_{0}^{-1}+T\Sigma^{-1})^{-1}
. We see then that the mean and covariance of
μ
^
\widehat{\mu}
match the mean the covariance of
Q
post
(
⋅
∣
𝔇
)
Q_{\mathrm{post}}(\cdot\mid\mathfrak{D})
given in
Lemma
6
, which proves the result.
∎
Appendix D
Additional Experimental Details
For all experiments we instantiate
PostBc
directly as suggested in
LABEL:alg:posterior_variance
and
LABEL:alg:posterior_bc
. We describe additional details on this instantiation next.
In all experiments, we parameterize
f
ℓ
f_{\ell}
in
LABEL:alg:posterior_variance
as an MLP (perhaps on top of a ResNet or other feature encoder, as described below). For the
Robomimic
experiments we let
f
ℓ
f_{\ell}
parameterize a Gaussian distribution and seek to model the actions in the dataset with a Gaussian, for other settings we simply have
f
ℓ
f_{\ell}
predict the actions directly (i.e. predicting a deterministic estimate of the actions rather than a distribution).
Note that simply training
f
ℓ
f_{\ell}
to predict the actions directly, rather than setting
f
ℓ
f_{\ell}
to a generative model that seeks to model the entire action distribution, is consistent with
LABEL:prop:policy_post_opt
—we aim to estimate a
sample
from the posterior distribution, for which it suffices to just fit a deterministic quantity, rather than fitting the entire
distribution
as generative modeling typically aims to do.
Furthermore, fitting a simple predictor on the actions directly usually requires fewer training iterations than fitting, for example, a diffusion model to the entire distribution, so this also reduces the computation required to fit the ensemble.
We found in practice that using bootstrap sampling to generate the datasets
𝔇
ℓ
\mathfrak{D}_{\ell}
in
LABEL:alg:posterior_variance
performs better than adding noise to the dataset as
LABEL:prop:policy_post_opt
suggests. We use both trajectory-level or state-action-level bootstrapping. For trajectory-level bootstrapping we generate
𝔇
ℓ
\mathfrak{D}_{\ell}
as in
Algorithm
1
.
1:
input:
demonstration dataset
𝔇
\mathfrak{D}
2:
𝔇
ℓ
←
∅
\mathfrak{D}_{\ell}\leftarrow\emptyset
3:
for
t
=
1
,
2
,
…
,
t=1,2,\ldots,
number of trajectories in
𝔇
\mathfrak{D}
do
4:
Sample trajectory
τ
∼
unif
​
(
𝔇
)
\tau\sim\mathrm{unif}(\mathfrak{D})
5:
𝔇
ℓ
←
𝔇
ℓ
∪
{
τ
}
\mathfrak{D}_{\ell}\leftarrow\mathfrak{D}_{\ell}\cup\{\tau\}
6:
return
𝔇
ℓ
\mathfrak{D}_{\ell}
Algorithm 1
Trajectory-Level Bootstrap Sampling
For state-action-level bootstrapping we generate
𝔇
ℓ
\mathfrak{D}_{\ell}
as in
Algorithm
2
.
1:
input:
demonstration dataset
𝔇
\mathfrak{D}
2:
𝔇
ℓ
←
∅
\mathfrak{D}_{\ell}\leftarrow\emptyset
3:
for
t
=
1
,
2
,
…
,
|
𝔇
|
t=1,2,\ldots,|\mathfrak{D}|
do
4:
Sample state-action pair
(
s
,
a
)
∼
unif
​
(
𝔇
)
(s,a)\sim\mathrm{unif}(\mathfrak{D})
5:
𝔇
ℓ
←
𝔇
ℓ
∪
{
(
s
,
a
)
}
\mathfrak{D}_{\ell}\leftarrow\mathfrak{D}_{\ell}\cup\{(s,a)\}
6:
return
𝔇
ℓ
\mathfrak{D}_{\ell}
Algorithm 2
Trajectory-Level Bootstrap Sampling
In all experiments we parameterize our final policy with a diffusion model. Given this,
LABEL:alg:posterior_bc
is trained on the standard diffusion loss.
Further details on each experiment are given below.
To leave room for RL improvement (i.e. to ensure performance is not saturated by the pretrained policy) we limit the number of demos per task in the pretraining dataset, for both the
Robomimic
and
Libero
experiments (see below for the precise number of trajectories used in pretraining).
D.1
Robomimic Experiments
We instantiate
π
^
θ
\widehat{\pi}^{\theta}
with a diffusion policy that uses an MLP architecture.
For
f
ℓ
f_{\ell}
, we train an MLP to simply predict the action directly in
𝔇
i
\mathfrak{D}_{i}
(i.e. we do not use a diffusion model for
f
ℓ
f_{\ell}
), but use the same architecture and dimensions for
f
ℓ
f_{\ell}
as the diffusion policies. We used trajectory-level bootstrapped sampling (
Algorithm
1
) to compute the ensemble. In all cases we pretrain on the Multi-Human
Robomimic
datasets, and in cases where we use less than the full dataset, we randomly select trajectories from the dataset to train on, using the same trajectories for each approach.
For each RL finetuning method, we sweep over the same hyperparameters for each pretrained policy method (i.e. BC,
σ
\sigma
-
Bc
,
PostBc
), and include results for the best one. For
σ
\sigma
-
Bc
, we swept over values of
σ
\sigma
and included results for the best-performing one.
For all experiments results are averaged over 5 seeds (we pretrain 5 policies for each approach and run RL finetuning on each of them once, for a total of 5 RL finetuning runs per pretraining method, finetuning method, and task).
For each evaluation, we roll out the policy 200 times.
For
Dppo
we utilize the default hyperparameters as stated in
Ren
et al.
(
2024
)
, and utilize DDPM sampling.
For
ValueDICE
, we use the officially published codebase, and the default hyperparameters provided there. We found that the
Iql
training on the data produced by
ValueDICE
could be somewhat unstable, and so to improve stability, for
Lift
, added LayerNorm to the
Iql
critic.
For
Dsrl
, we utilize a -1/0 success reward, and otherwise utilize a 0/1 success reward, using Robomimic’s built-in success detector to determine the reward.
We provide hyperparameters for the individual experiments below.
Table 1
:
Common
Dsrl
hyperparameters for all experiments.
Hyperparameter
Value
Learning rate
0.0003
0.0003
Batch size
256
256
Activation
Tanh
Target entropy
0
Target update rate (
τ
\tau
)
0.005
0.005
Number of actor and critic layers
3
3
Number of critics
2
2
Number of environments
4
4
Table 2
:
Dsrl
hyperparameters for
Robomimic
experiments.
Hyperparameter
Lift
Can
Square
Hidden size
2048
2048
2048
2048
2048
2048
Gradient steps per update
20
20
(
PostBc
,
Bc
),
10
10
(
σ
\sigma
-
Bc
)
20
20
(
PostBc
,
Bc
),
10
10
(
σ
\sigma
-
Bc
)
20
20
(
PostBc
,
Bc
),
10
10
(
σ
\sigma
-
Bc
)
Noise critic update steps
20
20
10
10
10
10
Discount factor
0.99
0.99
0.99
0.99
0.999
0.999
Action magnitude
1.5
1.5
1.5
1.5
1.5
1.5
Initial steps
24000
24000
24000
24000
32000
32000
Table 3
:
Hyperparameters for pretrained policies for
Robomimic
Dsrl
experiments.
Hyperparameter
Lift
Can
Square
Dataset size (number trajectories)
5
5
10
10
30
30
Action chunk size
4
4
4
4
4
4
train denoising steps
100
100
100
100
100
100
inference denoising steps
8
8
8
8
8
8
Hidden size
512
512
1024
1024
1024
1024
Hidden layers
3
3
3
3
3
3
Training epochs
3000
3000
3000
3000
3000
3000
Ensemble size (
PostBc
)
100
100
100
100
100
100
Ensemble training epochs (
PostBc
)
10000
10000
6000
6000
3000
3000
Posterior noise weight
α
\alpha
(
PostBc
)
1
1
0.5
0.5
1
1
Uniform noise
σ
\sigma
(
σ
\sigma
-
Bc
)
0.1
0.1
0.1
0.1
0.05
0.05
Table 4
:
Best-of-
N
N
hyperparameters for
Robomimic
experiments.
Hyperparameter
Lift
Can
Square
Total gradient steps
2000000
2000000
2000000
2000000
2000000
2000000
Iql
τ
\tau
(1000 rollouts)
0.5 (
Bc
,
PostBc
), 0.7 (
σ
\sigma
-
Bc
), 0.9 (
DICE
)
0.7
0.7
Iql
τ
\tau
(2000 rollouts)
0.5 (
Bc
), 0.7 (
σ
\sigma
-
Bc
,
PostBc
), 0.9 (
DICE
)
0.7
0.7
Discount factor
0.999
0.999
0.999
0.999
0.999
0.999
Table 5
:
Hyperparameters for pretrained policies for
Robomimic
Best-of-
N
N
experiments.
Hyperparameter
Lift
Can
Square
Dataset size (number trajectories)
20
20
300
300
300
300
Action chunk size
1
1
1
1
1
1
Train denoising steps
100
100
100
100
100
100
Hidden size
512
512
1024
1024
1024
1024
Hidden layers
3
3
3
3
3
3
Training epochs
3000
3000
3000
3000
3000
3000
Ensemble size (
PostBc
)
100
100
(1000 rollouts),
10
10
(2000 rollouts)
10
10
10
10
Ensemble training epochs (
PostBc
)
3000
3000
500
500
500
500
Posterior noise weight
α
\alpha
(
PostBc
)
1
1
(1000 rollouts),
2
2
(2000 rollouts)
1
1
1
1
(1000 rollouts),
2
2
(2000 rollouts)
Uniform noise
σ
\sigma
(
σ
\sigma
-
Bc
)
0.1
0.1
0.025
0.025
0.025
0.025
Table 6
:
Hyperparameters for pretrained policies for
Robomimic
Dppo
experiments.
Hyperparameter
Lift
Can
Square
Dataset size (number trajectories)
5
5
10
10
30
30
Action chunk size
4
4
4
4
4
4
train denoising steps
100
100
100
100
100
100
Hidden size
512
512
1024
1024
1024
1024
Hidden layers
3
3
3
3
3
3
Training epochs
3000
3000
3000
3000
3000
3000
Ensemble size (
PostBc
)
100
100
100
100
10
10
Ensemble training epochs (
PostBc
)
3000
3000
6000
6000
3000
3000
Posterior noise weight
α
\alpha
(
PostBc
)
0.5
0.5
0.25
0.25
1
1
Uniform noise
σ
\sigma
(
σ
\sigma
-
Bc
)
0.1
0.1
0.05
0.05
0.05
0.05
D.2
Libero Experiments
For Libero, we utilize the transformer architecture from
Dasari
et al.
(
2024
)
for
π
^
θ
\widehat{\pi}^{\theta}
.
For
PostBc
we use state-action bootstrap sampling (
Algorithm
1
) to generate
𝔇
ℓ
\mathfrak{D}_{\ell}
.
For
f
ℓ
f_{\ell}
, we utilize the same ResNet and tokenizer as
π
^
θ
\widehat{\pi}^{\theta}
, but simply utilize a 3-layer MLP head on top of it—trained to predict the actions directly—rather than a full diffusion transformer.
For the Best-of-
N
N
experiments,
PostBc
utilizes a diagonal posterior covariance estimate (that is, instead of computing the full covariance matrix as prescribed by
LABEL:alg:posterior_variance
, we compute the covariance dimension-wise, and construct a diagonal covariance matrix from this), while for the
Dsrl
runs it is trained with the full matrix posterior covariance estimate.
We train on
Libero 90
data from the first 3 scenes of
Libero 90
—
KITCHEN SCENE 1, KITCHEN SCENE 2
, and
KITCHEN SCENE 3
—and use 25 trajectories from each task in each scene. For task conditioning, we conditioning
π
^
θ
\widehat{\pi}^{\theta}
on the BERT language embedding
(Devlin
et al.
,
2019
)
of the corresponding text given for that task in the Libero dataset.
For each RL finetuning method, we sweep over the same hyperparameters for each pretrained policy method (i.e. BC,
σ
\sigma
-
Bc
,
PostBc
), and include results for the best one. We utilize the
Dsrl-Sac
variant of
Dsrl
from
Wagenmaker
et al.
(
2025
)
. For
σ
\sigma
-
Bc
, we swept over values of
σ
\sigma
and included results for the best-performing one.
The
Dsrl
experiments are averaged over 3 different pretraining runs per method, and one
Dsrl
run per pretrained run. The Best-of-
N
N
experiments are averaged over 2 different pretraining runs per method.
For each evaluation, we roll out the policy 100 times. In all cases, we utilize a -1/0 success reward, using Libero’s built-in success detector to determine the reward.
We provide hyperparameters for the individual experiments below.
Table 7
:
Dsrl
hyperparameters for all Libero experiments.
Hyperparameter
Value
Learning rate
0.0003
0.0003
Batch size
256
256
Activation
Tanh
Target entropy
0
Target update rate (
τ
\tau
)
0.005
0.005
Number of actor and critic layers
3
3
Layer size
1024
1024
Number of critics
2
2
Number of environments
1
1
Gradient steps per update
20
20
Discount factor
0.99
0.99
Action magnitude
1.5
1.5
Initial episode rollouts
20
20
Table 8
:
Best-of-
N
N
hyperparameters for all Libero experiments.
Hyperparameter
Value
Iql
learning rate
0.0003
0.0003
Iql
batch size
256
256
Iql
β
\beta
3
3
Activation
Tanh
Target update rate
0.005
0.005
Q
Q
and
V
V
number of layers
2
2
Q
Q
and
V
V
layer size
256
256
Number of critics
2
2
N
N
(Best-of-
N
N
samples)
32
32
Iql
gradient steps
50000
50000
Iql
τ
\tau
0.9
0.9
Discount factor
0.99
0.99
Table 9
:
Hyperparameters for DiT diffusion policy in Libero experiments.
Hyperparameter
Value
Batch size
150
150
Learning rate
0.0003
0.0003
Training steps
50000
50000
LR scheduler
cosine
Warmup steps
2000
2000
Action chunk size
4
4
Train denoising steps
100
100
Inference denoising steps
8
8
Image encoder
ResNet-18
Hidden size
256
256
Number of Heads
8
8
Number of Layers
4
4
Feedforward dimension
512
512
Token dimension
256
256
Ensemble size (
PostBc
)
5
5
Ensemble training steps (
PostBc
)
25000
25000
Ensemble layer size
512
512
Ensemble number of layers
3
3
Posterior noise weight (
PostBc
)
2
2
(
Dsrl
run),
4
4
(Best-of-
N
N
run)
Uniform noice
σ
\sigma
(
σ
\sigma
-
Bc
)
0.05
0.05
D.3
WidowX Experiments
Figure 2
:
Setup for WidowX “
Put corn in pot
” task.
Figure 3
:
Setup for WidowX “
Pick up banana
” task.
For the WidowX experiment, we collect 10 demonstrations on the “
Put corn in pot
” task. For the diffusion policy, we utilize a U-Net architecture with ResNet image encoder. For the
PostBc
ensemble predictors, we utilize a ResNet image encoder with MLP regression head, trained to directly predict the action in the dataset.
For both
Bc
and
PostBc
, we pretrain the policy on the 10 demonstrations, then roll out the pretrained policy 100 times on each task, manually resetting the scene each time and classifying each trajectory as success or failure. We utilize a 0/1 reward (every step is given a reward of 0 unless it succeeds, when it is given a reward of 1). We then train an
Iql
Q
Q
-function on the rollout data and, at test time, roll out the pretrained policy, sampling
N
N
actions at each step, and choosing the action with the maximum
Q
Q
-value.
For
Iql
, we utilize an MLP-based architecture, and to process the images, we utilize image features from a ResNet encoder pretrained on the Bridge v2 dataset
(Walke
et al.
,
2023
)
. For both
Bc
and
PostBc
, we try different values of
N
N
and different number of
Iql
training steps, and report the results for the best-performing values for each approach.
All hyperparameters for the diffusion policy are given in
Table
10
, and for
Iql
in
Table
11
.
Table 10
:
Hyperparameters for pretrained policies for WidowX experiments.
Hyperparameter
Both WidowX tasks
Action chunk size
1
1
Train denoising steps
100
100
Inference denoising steps
16
16
Image encoder
ResNet-18
U-Net channel size
[
256
,
512
,
1024
]
[256,512,1024]
U-Net kernel size
5
5
Training epochs
800
800
Ensemble predictor hidden size
512
512
Ensemble predictor hidden layers
3
3
Ensemble size (
PostBc
)
10
10
Ensemble training epochs (
PostBc
)
300
300
Posterior noise weight
α
\alpha
(
PostBc
)
1
1
Table 11
:
Best-of-
N
N
hyperparameters for WidowX experiments.
Hyperparameter
Put corn in pot
Pick up banana
Iql
learning rate
0.0003
0.0003
0.0003
0.0003
Iql
batch size
256
256
256
256
Iql
β
\beta
3
3
3
3
Activation
Tanh
Tanh
Target update rate
0.005
0.005
0.005
0.005
Q
Q
and
V
V
number of layers
2
2
2
2
Q
Q
and
V
V
layer size
256
256
256
256
Number of critics
2
2
2
2
N
N
(Best-of-
N
N
samples)
4
4
16
16
Iql
gradient steps
400000
400000
(
Bc
),
700000
700000
(
PostBc
)
100000
100000
Iql
τ
\tau
0.7
0.7
0.7
0.7
Discount factor
0.97
0.97
0.97
0.97
D.4
Additional Ablations
For all ablation experiments, other than the hyperparameter we vary, we utilize the hyperparameters given in
Section
D.1
.
In
Figure
4
we provide an additional ablation on the dataset size for
Robomimic Square
, and in
Figure
5
provide additional qualitative results on
Libero
.
Figure 4
:
Comparison of
Dsrl
finetuning performance combined with different BC pretraining approaches on
Robomimic Square
, varying the number of trajectories in the dataset the policies are pretrained on. As can be seen, the finetuning performance of policies pretrained with
PostBc
is largely unaffected by the size of the pretraining dataset, while BC and
σ
\sigma
-
Bc
are both very sensitive to dataset size. For large enough datasets (50 trajectories), BC and
σ
\sigma
-
Bc
perform as well as
PostBc
. This is to be expected—if we train on enough data, our uncertainty will be low, so
PostBc
will essentially reduce to BC. These results illustrate that
PostBc
gracefully interpolates between settings where BC overfits to small amounts of data, hurting its finetuning performance, and settings where BC is sufficient for effective finetuning.
Figure 5
:
Additional density heatmaps of pretrained policies on tasks 6-21 from
Libero 90
. See
Table
12
for task commands.
Table 12
:
Task descriptions for Libero tasks in
Kitchen Scene 1-3
.
Task ID
Task description
Task 6
Open the bottom drawer of the cabinet
Task 7
Open the top drawer of the cabinet
Task 8
Open the top drawer of the cabinet and put the bowl in it
Task 9
Put the black bowl on the plate
Task 10
Put the black bowl on top of the cabinet
Task 11
Open the top drawer of the cabinet
Task 12
Put the black bowl at the back on the plate
Task 13
Put the black bowl at the front on the plate
Task 14
Put the middle black bowl on the plate
Task 15
Put the middle black bowl on top of the cabinet
Task 16
Stack the black bowl at the front on the black bowl in the middle
Task 17
Stack the middle black bowl on the back black bowl
Task 18
Put the frying pan on the stove
Task 19
Put the moka pot on the stove
Task 20
Turn on the stove
Task 21
Turn on the stove and put the frying pan on it