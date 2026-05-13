---
title: Learning Bipedal Walking on a Quadruped Robot via Adversarial Motion Priors
id: learning-bipedal-walking-on-a-quadruped-robot-via-adversarial-motion-priors
tags:
- legged-rl-budgets
- amp
- quadruped
- bipedal
- isaac-gym
- training-budget
created: '2026-05-06T07:30:47.178760Z'
updated: '2026-05-06T07:34:55.336754Z'
source: https://arxiv.org/html/2407.02282v1
source_domain: arxiv.org
fetched_at: '2026-05-06T07:30:47.178760Z'
fetch_provider: builtin
status: draft
type: note
tier: institutional
content_type: paper
deprecated: false
summary: 'Peng et al. (arXiv 2407.02282, 2024, Leeds/UCL). Adapts AMP+teacher-student
  framework to bipedal locomotion using only two rear legs of a quadruped (Unitree
  A1). Generates reference data using TOWR trajectory optimization (2.4s clips of
  walking and running gaits). AMP discriminator state: 31-dimensional (joint positions,
  point velocities, base velocity, angular velocity, base height). Gradient penalty
  coefficient alpha_qp=10. KEY TRAINING BUDGET (verbatim from Section III-B-3): ''500
  parallel agents on different types of terrains with increasing difficulties using
  the Isaac Gym simulator in overall 26000 iterations and cost 15.88 hours in total...
  All trainings were performed on a single NVIDIA RTX 4070 GPU.'' Episode length:
  max 1000 steps, control frequency 50 Hz. Total steps: 500 agents x 26000 iterations
  x (1000 step episodes / unroll varies) — exact step count not stated but approximately
  500 x 26000 x 24 steps/iteration = ~312M total env steps (inferred). Evaluation
  on 6 terrain types at 4 speeds showing 79-93% success rate. Domain randomization:
  mass, friction, Kp/Kd gains.'
---

Learning Bipedal Walking on a Quadruped Robot via Adversarial Motion Priors
Learning Bipedal Walking on a Quadruped Robot via Adversarial Motion Priors
Tianhu Peng
1
, Lingfan Bao
1
, Joseph Humphreys
1
, Andromachi Maria Delfaki
2
,
Dimitrios Kanoulas
2
and Chengxu Zhou
2*
This work was supported by the Royal Society [grant number RG\R2\232409] and the UKRI Future Leaders Fellowship [grant number MR/V025333/1]. Please refer to the video for an overview of our framework and results at
https://youtu.be/JYD1RlrQRWM
.For the purpose of open access, the authors have applied a Creative Commons Attribution (CC BY) licence to any Author Accepted Manuscript version arising from this submission.
1
School of Mechanical Engineering, University of Leeds, UK.
2
Department of Computer Science, University College London, UK.
*
Corresponding author,
chengxu.zhou@ucl.ac.uk
Abstract
Previous studies have successfully demonstrated agile and robust locomotion in challenging terrains for quadrupedal robots. However, the bipedal locomotion mode for quadruped robots remains unverified. This paper explores the adaptation of a learning framework originally designed for quadrupedal robots to operate blind locomotion in biped mode. We leverage a framework that incorporates Adversarial Motion Priors with a teacher-student policy to enable imitation of a reference trajectory and navigation on tough terrain. Our work involves transferring and evaluating a similar learning framework on a quadruped robot in biped mode, aiming to achieve stable walking on both flat and complicated terrains. Our simulation results demonstrate that the trained policy enables the quadruped robot to navigate both flat and challenging terrains, including stairs and uneven surfaces.
Index Terms:
Legged Robots, Bipedal Locomotion, Deep Reinforcement Learning, Adversarial Motion Priors
I
Introduction
Legged robots exhibit superior terrain adaptability compared to their wheeled and tracked counterparts. Although quadrupedal robots are known for their stability and agility, bipedal robots offer greater flexibility by freeing the upper body for complex tasks. This flexibility suggests the potential for quadrupedal robots to walk in a bipedal gait, using the rear legs for walking and the front legs for manipulation.
The primary challenge in adapting quadruped robots for bipedal locomotion stems from their mechanical design constraints. First, unlike typical bipedal robots that have firm, flat feet, quadruped robots often feature soft, point-contact feet that inherently lack stability. Second, the rear legs of quadruped robots are not specifically designed for bipedal walking, their limited range of motion and underactuation contribute to unnatural and unstable bipedal gaits. This design mismatch explains why quadruped robots struggle with bipedal walking modes. This leads to high requirements for locomotion controllers during bipedal modes.
To achieve bipedal walking for quadruped robots, there are primarily two approaches: the model-based method and the learning-based method
[
1
]
. Model-based methods are based on highly accurate mathematical models, which have proven to be effective in executing highly dynamic motions in both quadruped and bipedal robots. However, these methods lack robustness and generalization in unseen scenarios, largely due to the difficulty of accurately modeling ground interactions and contact dynamics. In contrast learning-based methods, reinforcement learning (RL), provides a more adaptable solution by enabling the exploration of the robots’ full dynamics and interactions with the environment, thus offering greater flexibility in controlling complex locomotive behaviors. Early research in RL on legged robots primarily utilized unrealistic models within physical simulators
[
2
,
3
,
4
]
. In transitioning to a practical bipedal robot and learning natural and robust gaits, previous studies have primarily designed a reference-free learning framework by designing periodic composition reward
[
5
]
or mimicking predefined references
[
6
,
7
,
8
]
.
Reference-free methods explore various gait patterns efficiently, while reference-based methods leverage prior knowledge to accelerate learning, resulting in efficient policy exploration and robust locomotion skills. These methods incorporate expert information and predefined reference trajectories from motion capture data or trajectory optimization (TO). Generative Adversarial Imitation Learning
[
2
]
and Adversarial motion priors (AMP)
[
8
]
predict state transitions and evaluate the similarity between reference and agent data, promoting stable gait maintenance. AMP was implemented with a study of human reference behavior in biped robots
[
9
,
10
]
, combining it with periodic rewards to promote stable gait maintenance.
Figure 1
:
Overview of the teacher-student learning framework. (a) The teacher policy, which leverages privileged data
S
t
p
subscript
superscript
𝑆
𝑝
𝑡
S^{p}_{t}
italic_S start_POSTSUPERSCRIPT italic_p end_POSTSUPERSCRIPT start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT
,terrain information
o
t
e
subscript
superscript
𝑜
𝑒
𝑡
o^{e}_{t}
italic_o start_POSTSUPERSCRIPT italic_e end_POSTSUPERSCRIPT start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT
and prospective data
o
t
p
subscript
superscript
𝑜
𝑝
𝑡
o^{p}_{t}
italic_o start_POSTSUPERSCRIPT italic_p end_POSTSUPERSCRIPT start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT
through RL, aims to maximize a total reward
r
t
subscript
𝑟
𝑡
r_{t}
italic_r start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT
comprising command task reward
r
t
g
superscript
subscript
𝑟
𝑡
𝑔
r_{t}^{g}
italic_r start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_g end_POSTSUPERSCRIPT
, style reward
r
t
g
superscript
subscript
𝑟
𝑡
𝑔
r_{t}^{g}
italic_r start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_g end_POSTSUPERSCRIPT
based on AMP, and regulation reward
r
t
g
superscript
subscript
𝑟
𝑡
𝑔
r_{t}^{g}
italic_r start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_g end_POSTSUPERSCRIPT
for ensuring safety and smooth motion. (b) The student policy, trained via supervised learning, seeks to imitate the teacher’s actions
a
t
t
⁢
e
⁢
a
⁢
c
⁢
h
⁢
e
⁢
r
subscript
superscript
𝑎
𝑡
𝑒
𝑎
𝑐
ℎ
𝑒
𝑟
𝑡
a^{teacher}_{t}
italic_a start_POSTSUPERSCRIPT italic_t italic_e italic_a italic_c italic_h italic_e italic_r end_POSTSUPERSCRIPT start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT
and reconstruct the teacher’s latent states
l
t
t
⁢
e
⁢
a
⁢
c
⁢
h
⁢
e
⁢
r
subscript
superscript
𝑙
𝑡
𝑒
𝑎
𝑐
ℎ
𝑒
𝑟
𝑡
l^{teacher}_{t}
italic_l start_POSTSUPERSCRIPT italic_t italic_e italic_a italic_c italic_h italic_e italic_r end_POSTSUPERSCRIPT start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT
from historical and prospective observations
H
t
:
[
O
t
−
N
,
O
t
−
N
−
1
,
…
,
O
t
−
1
]
:
subscript
𝐻
𝑡
subscript
𝑂
𝑡
𝑁
subscript
𝑂
𝑡
𝑁
1
…
subscript
𝑂
𝑡
1
H_{t}:[O_{t-N},O_{t-N-1},...,O_{t-1}]
italic_H start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT : [ italic_O start_POSTSUBSCRIPT italic_t - italic_N end_POSTSUBSCRIPT , italic_O start_POSTSUBSCRIPT italic_t - italic_N - 1 end_POSTSUBSCRIPT , … , italic_O start_POSTSUBSCRIPT italic_t - 1 end_POSTSUBSCRIPT ]
To generate appropriate predefined references, several methods are utilized. Motion capture technology is commonly employed to produce reference data for various types of legged robots
[
10
,
11
,
12
]
. This technology captures comprehensive kinematic data from real-world scenarios, enabling versatile data collection that is not confined to a specific robot model. However, adapting this data to different robotic platforms often requires a re-targeting process, which increases both complexity and manual labor. On the other hand, TO in reduced-order
[
13
]
and full dynamics models
[
7
]
, has been employed. This approach reduces complexity and eliminates the need for further re-targeting, making it a more efficient solution for generating references. Additionally, compared to full dynamics models, reduced-order models can decrease the computational resources required for optimization and offer greater generalization across all quadruped robots. Regarding predesigned references from the optimization method, various models are utilized. Besides HZD-based full dynamics reference
[
7
]
, reduced-order dynamics such as Single Rigid Body Dynamics (SRBD)
[
14
]
are also utilized in the training procedure. Based on the reduced-order model, task-space learning focuses on the foot setpoints and based velocity
[
15
,
16
]
.
Another significant challenge exists in bridging the sim-to-real gap. To extend the robustness of locomotion and overcome the sim-to-real gap, frameworks using the privileged learning paradigm have been introduced
[
17
,
18
]
. By combining the strengths of AMP in reference-based learning and privileged learning, there is potential to enable quadrupedal robots to adopt bipedal walking modes. Similar frameworks have been introduced
[
11
,
9
]
, but none have been validated on bipedal robots or quadruped robots for bipedal mode.
Our objective is to train a policy that enables a quadrupedal robot to achieve bipedal locomotion using only its two rear legs, thus freeing its front legs for more complex tasks. This capability aims to enhance the robot’s versatility and functionality in various practical applications. This paper presents a novel framework that enables quadrupedal robots to achieve robust and agile blind bipedal locomotion on flat terrain. We adopt a teacher-student policy framework, where privileged information that the robot cannot directly access is encoded. The student policy uses historical observation information to infer this privileged information, thereby enhancing robustness. Additionally, we integrate the AMP training framework to learn and imitate the style behaviors of reference data generated through TO based on a SRBD model. Different from previous work
[
19
]
with assistant devices, this comprehensive training framework equips the policy to support agile bipedal motions in quadrupedal robots.
In summary, the primary contribution of this paper is to develop a novel framework (shown in Fig.
1
) that allows quadrupedal robots to perform robust and agile bipedal blind locomotion on flat terrain using only their rear legs. This bipedal mode frees the front legs for more complex tasks, significantly enhancing the robot’s versatility and functionality. Besides, We evaluate our model on the A1 quadrupedal robot using a biped gait model in the Isaac Gym simulation environment, demonstrating agile and robust movements.
II
Methodology
II-A
Reinforcement Learning on Legged Robots
The task of learning legged locomotion poses significant challenges due to the complex environment and limitations in sensor data. To address this, a partially observable Markov decision process (POMDP) framework was adopted, denoted as
(
s
t
,
a
t
,
P
,
r
t
,
p
0
,
γ
)
subscript
𝑠
𝑡
subscript
𝑎
𝑡
𝑃
subscript
𝑟
𝑡
subscript
𝑝
0
𝛾
(s_{t},a_{t},P,r_{t},p_{0},\gamma)
( italic_s start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT , italic_a start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT , italic_P , italic_r start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT , italic_p start_POSTSUBSCRIPT 0 end_POSTSUBSCRIPT , italic_γ )
, where
s
t
subscript
𝑠
𝑡
s_{t}
italic_s start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT
represents the state at time step
t
𝑡
t
italic_t
,
a
t
subscript
𝑎
𝑡
a_{t}
italic_a start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT
is the action taken by the agent,
P
⁢
(
s
t
+
1
|
s
t
,
a
t
)
𝑃
conditional
subscript
𝑠
𝑡
1
subscript
𝑠
𝑡
subscript
𝑎
𝑡
P(s_{t+1}|s_{t},a_{t})
italic_P ( italic_s start_POSTSUBSCRIPT italic_t + 1 end_POSTSUBSCRIPT | italic_s start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT , italic_a start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT )
describes the system dynamics, predicting the next state based on the current state and action, ,
r
t
⁢
(
s
t
,
a
t
,
s
t
+
1
)
subscript
𝑟
𝑡
subscript
𝑠
𝑡
subscript
𝑎
𝑡
subscript
𝑠
𝑡
1
r_{t}(s_{t},a_{t},s_{t+1})
italic_r start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT ( italic_s start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT , italic_a start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT , italic_s start_POSTSUBSCRIPT italic_t + 1 end_POSTSUBSCRIPT )
is the reward function, quantifying the immediate benefit of taking action
a
t
subscript
𝑎
𝑡
a_{t}
italic_a start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT
in state
s
t
subscript
𝑠
𝑡
s_{t}
italic_s start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT
leading to
s
t
+
1
subscript
𝑠
𝑡
1
s_{t+1}
italic_s start_POSTSUBSCRIPT italic_t + 1 end_POSTSUBSCRIPT
,
p
0
subscript
𝑝
0
p_{0}
italic_p start_POSTSUBSCRIPT 0 end_POSTSUBSCRIPT
denotes the initial state distribution, and
γ
t
superscript
𝛾
𝑡
\gamma^{t}
italic_γ start_POSTSUPERSCRIPT italic_t end_POSTSUPERSCRIPT
is the discount factor, determining the importance of future rewards in the decision-making process.
The objective of RL in this context is to identify an optimal policy
π
θ
subscript
𝜋
𝜃
\pi_{\theta}
italic_π start_POSTSUBSCRIPT italic_θ end_POSTSUBSCRIPT
parameterized by
θ
𝜃
\theta
italic_θ
, that maximizes the expected discounted return over future trajectories. This is formalized by the objective function
J
⁢
(
θ
)
𝐽
𝜃
J(\theta)
italic_J ( italic_θ )
:
J
⁢
(
θ
)
=
𝔼
π
θ
⁢
[
∑
t
=
0
∞
γ
t
⁢
r
t
]
𝐽
𝜃
subscript
𝔼
subscript
𝜋
𝜃
delimited-[]
subscript
superscript
𝑡
0
superscript
𝛾
𝑡
subscript
𝑟
𝑡
\displaystyle J(\theta)=\mathbb{E}_{\pi_{\theta}}[\sum^{\infty}_{t=0}\gamma^{t%
}r_{t}]
italic_J ( italic_θ ) = blackboard_E start_POSTSUBSCRIPT italic_π start_POSTSUBSCRIPT italic_θ end_POSTSUBSCRIPT end_POSTSUBSCRIPT [ ∑ start_POSTSUPERSCRIPT ∞ end_POSTSUPERSCRIPT start_POSTSUBSCRIPT italic_t = 0 end_POSTSUBSCRIPT italic_γ start_POSTSUPERSCRIPT italic_t end_POSTSUPERSCRIPT italic_r start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT ]
(1)
In the state space, the state
s
t
t
⁢
e
⁢
a
⁢
c
⁢
h
⁢
e
⁢
r
superscript
subscript
𝑠
𝑡
𝑡
𝑒
𝑎
𝑐
ℎ
𝑒
𝑟
s_{t}^{teacher}
italic_s start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_t italic_e italic_a italic_c italic_h italic_e italic_r end_POSTSUPERSCRIPT
includes proprioceptive observation
o
t
p
∈
ℝ
48
superscript
subscript
𝑜
𝑡
𝑝
superscript
ℝ
48
o_{t}^{p}\in\mathbb{R}^{48}
italic_o start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_p end_POSTSUPERSCRIPT ∈ blackboard_R start_POSTSUPERSCRIPT 48 end_POSTSUPERSCRIPT
, privileged state
s
t
p
∈
ℝ
45
superscript
subscript
𝑠
𝑡
𝑝
superscript
ℝ
45
s_{t}^{p}\in\mathbb{R}^{45}
italic_s start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_p end_POSTSUPERSCRIPT ∈ blackboard_R start_POSTSUPERSCRIPT 45 end_POSTSUPERSCRIPT
and terrain information
o
t
e
∈
ℝ
187
subscript
superscript
𝑜
𝑒
𝑡
superscript
ℝ
187
o^{e}_{t}\in\mathbb{R}^{187}
italic_o start_POSTSUPERSCRIPT italic_e end_POSTSUPERSCRIPT start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT ∈ blackboard_R start_POSTSUPERSCRIPT 187 end_POSTSUPERSCRIPT
. The proprioceptive observation
o
t
p
∈
ℝ
48
superscript
subscript
𝑜
𝑡
𝑝
superscript
ℝ
48
o_{t}^{p}\in\mathbb{R}^{48}
italic_o start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_p end_POSTSUPERSCRIPT ∈ blackboard_R start_POSTSUPERSCRIPT 48 end_POSTSUPERSCRIPT
encompasses critical data such as the orientation of the gravity vector, base linear and angular velocity in the robot’s frame, joint positions and velocities, the previous action
a
t
−
1
∈
ℝ
12
subscript
𝑎
𝑡
1
superscript
ℝ
12
a_{t-1}\in\mathbb{R}^{12}
italic_a start_POSTSUBSCRIPT italic_t - 1 end_POSTSUBSCRIPT ∈ blackboard_R start_POSTSUPERSCRIPT 12 end_POSTSUPERSCRIPT
executed by the current policy. The privileged state
s
t
p
superscript
subscript
𝑠
𝑡
𝑝
s_{t}^{p}
italic_s start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_p end_POSTSUPERSCRIPT
contains the information include ground friction coefficients, ground restitution coefficients, contact forces, external forces within positions on the robot, and collision state.
The privileged state
s
t
p
∈
ℝ
45
superscript
subscript
𝑠
𝑡
𝑝
superscript
ℝ
45
s_{t}^{p}\in\mathbb{R}^{45}
italic_s start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_p end_POSTSUPERSCRIPT ∈ blackboard_R start_POSTSUPERSCRIPT 45 end_POSTSUPERSCRIPT
includes additional key details that the physical robot cannot directly access in the real-world environment. This information encompasses ground friction and restitution coefficients, contact and external forces at specific robot positions, collision state information. The terrain information
o
t
e
∈
ℝ
187
subscript
superscript
𝑜
𝑒
𝑡
superscript
ℝ
187
o^{e}_{t}\in\mathbb{R}^{187}
italic_o start_POSTSUPERSCRIPT italic_e end_POSTSUPERSCRIPT start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT ∈ blackboard_R start_POSTSUPERSCRIPT 187 end_POSTSUPERSCRIPT
contains the 187 height measurement sampled from grid around robot base to the ground.
In contrast, the student policy state utilizes a sequence history of proprioceptive observations
H
t
:
[
O
t
−
N
,
O
t
−
N
−
1
,
…
,
O
t
]
:
subscript
𝐻
𝑡
subscript
𝑂
𝑡
𝑁
subscript
𝑂
𝑡
𝑁
1
…
subscript
𝑂
𝑡
H_{t}:[O_{t-N},O_{t-N-1},...,O_{t}]
italic_H start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT : [ italic_O start_POSTSUBSCRIPT italic_t - italic_N end_POSTSUBSCRIPT , italic_O start_POSTSUBSCRIPT italic_t - italic_N - 1 end_POSTSUBSCRIPT , … , italic_O start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT ]
to approximate the privileged information. By learning from this historical data, the student policy aims to imitate the inaccessible privileged state, enhancing its decision-making capabilities in the absence of direct access to certain environmental variables.
Regarding the action space, the policy action
a
t
subscript
𝑎
𝑡
a_{t}
italic_a start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT
is a 12 dimensional target joint position offset added to the time-invariant nominal joint position. This specification guides the joint PD controller, utilizing fixed gains to compute torque commands effectively for motor position control.
II-B
Adversarial Motion Priors and Rewards Design
The AMP framework utilizes adversarial learning to train two neural networks—a generator and a discriminator—in a competitive setup. The generator produces motion predictions for the robot, while the discriminator evaluates their quality and realism. Style rewards are used to measure the similarity between the demonstrator’s behavior and the robot’s, with higher similarity yielding more rewards.
Using the reference dataset
D
𝐷
D
italic_D
, the AMP-based style reward function encourages the robot to replicate the same gait style. According to
[
20
]
, a neural network-based discriminator
D
φ
subscript
𝐷
𝜑
D_{\varphi}
italic_D start_POSTSUBSCRIPT italic_φ end_POSTSUBSCRIPT
predicts whether a state transition
(
S
t
,
S
t
+
1
)
subscript
𝑆
𝑡
subscript
𝑆
𝑡
1
(S_{t},S_{t+1})
( italic_S start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT , italic_S start_POSTSUBSCRIPT italic_t + 1 end_POSTSUBSCRIPT )
is from the dataset
D
𝐷
D
italic_D
or generated by the agent
A
𝐴
A
italic_A
. Each state
S
t
AMP
∈
ℝ
31
superscript
subscript
𝑆
𝑡
AMP
superscript
ℝ
31
S_{t}^{\text{AMP}}\in\mathbb{R}^{31}
italic_S start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT start_POSTSUPERSCRIPT AMP end_POSTSUPERSCRIPT ∈ blackboard_R start_POSTSUPERSCRIPT 31 end_POSTSUPERSCRIPT
includes joint positions, point velocities, base linear velocity, base angular velocity, and base height relative to the terrain. To avoid mode collapse, the dataset
D
𝐷
D
italic_D
contains only trot gait motion clips.
The discriminator’s training objective includes a gradient penalty to enforce smoothness and is defined as:
arg
⁡
min
φ
subscript
𝜑
\displaystyle\arg\min_{\varphi}
roman_arg roman_min start_POSTSUBSCRIPT italic_φ end_POSTSUBSCRIPT
𝔼
(
s
t
,
s
t
+
1
)
∼
D
⁢
[
(
D
φ
⁢
(
s
t
,
s
t
+
1
)
−
1
)
2
]
subscript
𝔼
similar-to
subscript
𝑠
𝑡
subscript
𝑠
𝑡
1
𝐷
delimited-[]
superscript
subscript
𝐷
𝜑
subscript
𝑠
𝑡
subscript
𝑠
𝑡
1
1
2
\displaystyle\mathbb{E}_{(s_{t},s_{t+1})\sim D}[(D_{\varphi}(s_{t},s_{t+1})-1)%
^{2}]
blackboard_E start_POSTSUBSCRIPT ( italic_s start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT , italic_s start_POSTSUBSCRIPT italic_t + 1 end_POSTSUBSCRIPT ) ∼ italic_D end_POSTSUBSCRIPT [ ( italic_D start_POSTSUBSCRIPT italic_φ end_POSTSUBSCRIPT ( italic_s start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT , italic_s start_POSTSUBSCRIPT italic_t + 1 end_POSTSUBSCRIPT ) - 1 ) start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT ]
(2)
+
𝔼
(
s
t
,
s
t
+
1
)
∼
A
⁢
[
(
D
φ
⁢
(
s
t
,
s
t
+
1
)
−
1
)
2
]
subscript
𝔼
similar-to
subscript
𝑠
𝑡
subscript
𝑠
𝑡
1
𝐴
delimited-[]
superscript
subscript
𝐷
𝜑
subscript
𝑠
𝑡
subscript
𝑠
𝑡
1
1
2
\displaystyle+\mathbb{E}_{(s_{t},s_{t+1})\sim A}[(D_{\varphi}(s_{t},s_{t+1})-1%
)^{2}]
+ blackboard_E start_POSTSUBSCRIPT ( italic_s start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT , italic_s start_POSTSUBSCRIPT italic_t + 1 end_POSTSUBSCRIPT ) ∼ italic_A end_POSTSUBSCRIPT [ ( italic_D start_POSTSUBSCRIPT italic_φ end_POSTSUBSCRIPT ( italic_s start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT , italic_s start_POSTSUBSCRIPT italic_t + 1 end_POSTSUBSCRIPT ) - 1 ) start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT ]
+
α
q
⁢
p
2
⁢
𝔼
(
s
t
,
s
t
+
1
)
∼
D
⁢
[
‖
∇
φ
D
φ
⁢
(
s
t
,
s
t
+
1
)
‖
2
2
]
,
superscript
𝛼
𝑞
𝑝
2
subscript
𝔼
similar-to
subscript
𝑠
𝑡
subscript
𝑠
𝑡
1
𝐷
delimited-[]
superscript
subscript
norm
subscript
∇
𝜑
subscript
𝐷
𝜑
subscript
𝑠
𝑡
subscript
𝑠
𝑡
1
2
2
\displaystyle+\frac{\alpha^{qp}}{2}\mathbb{E}_{(s_{t},s_{t+1})\sim D}[\|\nabla%
_{\varphi}D_{\varphi}(s_{t},s_{t+1})\|_{2}^{2}],
+ divide start_ARG italic_α start_POSTSUPERSCRIPT italic_q italic_p end_POSTSUPERSCRIPT end_ARG start_ARG 2 end_ARG blackboard_E start_POSTSUBSCRIPT ( italic_s start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT , italic_s start_POSTSUBSCRIPT italic_t + 1 end_POSTSUBSCRIPT ) ∼ italic_D end_POSTSUBSCRIPT [ ∥ ∇ start_POSTSUBSCRIPT italic_φ end_POSTSUBSCRIPT italic_D start_POSTSUBSCRIPT italic_φ end_POSTSUBSCRIPT ( italic_s start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT , italic_s start_POSTSUBSCRIPT italic_t + 1 end_POSTSUBSCRIPT ) ∥ start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT ] ,
where
α
q
⁢
p
superscript
𝛼
𝑞
𝑝
\alpha^{qp}
italic_α start_POSTSUPERSCRIPT italic_q italic_p end_POSTSUPERSCRIPT
is a manually specified coefficient (
α
q
⁢
p
=
10
superscript
𝛼
𝑞
𝑝
10
\alpha^{qp}=10
italic_α start_POSTSUPERSCRIPT italic_q italic_p end_POSTSUPERSCRIPT = 10
).
The style reward is defined by:
r
t
s
⁢
[
(
s
t
,
s
t
+
1
)
∼
A
]
=
max
⁡
[
0
,
1
−
0.25
⁢
(
d
t
s
⁢
c
⁢
o
⁢
r
⁢
e
−
1
)
2
]
,
superscript
subscript
𝑟
𝑡
𝑠
delimited-[]
similar-to
subscript
𝑠
𝑡
subscript
𝑠
𝑡
1
𝐴
0
1
0.25
superscript
subscript
superscript
𝑑
𝑠
𝑐
𝑜
𝑟
𝑒
𝑡
1
2
\displaystyle r_{t}^{s}[(s_{t},s_{t+1})\sim A]=\mathbb{\max}[0,1-0.25(d^{score%
}_{t}-1)^{2}],
italic_r start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_s end_POSTSUPERSCRIPT [ ( italic_s start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT , italic_s start_POSTSUBSCRIPT italic_t + 1 end_POSTSUBSCRIPT ) ∼ italic_A ] = roman_max [ 0 , 1 - 0.25 ( italic_d start_POSTSUPERSCRIPT italic_s italic_c italic_o italic_r italic_e end_POSTSUPERSCRIPT start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT - 1 ) start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT ] ,
(3)
where
d
t
s
⁢
c
⁢
o
⁢
r
⁢
e
=
D
φ
⁢
(
s
t
,
s
t
+
1
)
subscript
superscript
𝑑
𝑠
𝑐
𝑜
𝑟
𝑒
𝑡
subscript
𝐷
𝜑
subscript
𝑠
𝑡
subscript
𝑠
𝑡
1
d^{score}_{t}=D_{\varphi}(s_{t},s_{t+1})
italic_d start_POSTSUPERSCRIPT italic_s italic_c italic_o italic_r italic_e end_POSTSUPERSCRIPT start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT = italic_D start_POSTSUBSCRIPT italic_φ end_POSTSUBSCRIPT ( italic_s start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT , italic_s start_POSTSUBSCRIPT italic_t + 1 end_POSTSUBSCRIPT )
and is scaled to the range
[
0
,
1
]
0
1
[0,1]
[ 0 , 1 ]
.
The overall reward function is:
r
t
=
r
t
g
+
r
t
s
+
r
t
l
,
subscript
𝑟
𝑡
superscript
subscript
𝑟
𝑡
𝑔
superscript
subscript
𝑟
𝑡
𝑠
superscript
subscript
𝑟
𝑡
𝑙
\displaystyle r_{t}=r_{t}^{g}+r_{t}^{s}+r_{t}^{l},
italic_r start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT = italic_r start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_g end_POSTSUPERSCRIPT + italic_r start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_s end_POSTSUPERSCRIPT + italic_r start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_l end_POSTSUPERSCRIPT ,
(4)
where
r
t
g
superscript
subscript
𝑟
𝑡
𝑔
r_{t}^{g}
italic_r start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_g end_POSTSUPERSCRIPT
is the task reward,
r
t
s
superscript
subscript
𝑟
𝑡
𝑠
r_{t}^{s}
italic_r start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_s end_POSTSUPERSCRIPT
is the style reward, and
r
t
l
superscript
subscript
𝑟
𝑡
𝑙
r_{t}^{l}
italic_r start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_l end_POSTSUPERSCRIPT
is the regularization reward. Task rewards typically include tracking base linear and angular velocities:
r
t
g
=
ω
v
⁢
exp
⁡
(
−
|
v
^
t
x
⁢
y
−
v
t
x
⁢
y
|
)
+
ω
ω
⁢
exp
⁡
(
−
|
ω
^
t
z
−
ω
t
z
|
)
,
superscript
subscript
𝑟
𝑡
𝑔
superscript
𝜔
𝑣
superscript
subscript
^
𝑣
𝑡
𝑥
𝑦
superscript
subscript
𝑣
𝑡
𝑥
𝑦
superscript
𝜔
𝜔
superscript
subscript
^
𝜔
𝑡
𝑧
superscript
subscript
𝜔
𝑡
𝑧
r_{t}^{g}=\omega^{v}\exp\left(-|\hat{v}_{t}^{xy}-v_{t}^{xy}|\right)+\omega^{%
\omega}\exp\left(-|\hat{\omega}_{t}^{z}-\omega_{t}^{z}|\right),
italic_r start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_g end_POSTSUPERSCRIPT = italic_ω start_POSTSUPERSCRIPT italic_v end_POSTSUPERSCRIPT roman_exp ( - | over^ start_ARG italic_v end_ARG start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_x italic_y end_POSTSUPERSCRIPT - italic_v start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_x italic_y end_POSTSUPERSCRIPT | ) + italic_ω start_POSTSUPERSCRIPT italic_ω end_POSTSUPERSCRIPT roman_exp ( - | over^ start_ARG italic_ω end_ARG start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_z end_POSTSUPERSCRIPT - italic_ω start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_z end_POSTSUPERSCRIPT | ) ,
(5)
where
ω
v
superscript
𝜔
𝑣
{\omega}^{v}
italic_ω start_POSTSUPERSCRIPT italic_v end_POSTSUPERSCRIPT
and
ω
ω
superscript
𝜔
𝜔
{\omega}^{\omega}
italic_ω start_POSTSUPERSCRIPT italic_ω end_POSTSUPERSCRIPT
are coefficients, and
v
^
t
x
⁢
y
superscript
subscript
^
𝑣
𝑡
𝑥
𝑦
\hat{v}_{t}^{xy}
over^ start_ARG italic_v end_ARG start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_x italic_y end_POSTSUPERSCRIPT
and
ω
^
t
z
superscript
subscript
^
𝜔
𝑡
𝑧
\hat{\omega}_{t}^{z}
over^ start_ARG italic_ω end_ARG start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_z end_POSTSUPERSCRIPT
are the velocity commands.
Regularization rewards promote safe, smooth motion and minimize energy costs, enhancing the adaptability and efficiency of learned behaviors for real-world applications. This component contributes to the robustness and effectiveness of the overall motion.
II-C
Reference Generation
In our research, we refine the imitation and learning of reference data for precise bipedal locomotion. Using a single TO formulation from previous work
[
13
]
, we generate walking and running gaits for the A1 biped robot, focusing on its back legs’ trajectories.
We streamline this process with TOWR (Trajectory Optimization for Walking Robots)
[
21
]
, which eliminates the need for manual tuning. TOWR generates dynamically feasible, energy-efficient motions by optimizing smooth and stable trajectories. To improve imitation fidelity, we integrate inverse kinematics into TOWR, producing joint space data that closely mimics reference trajectories.
Our generated trajectories encompass various locomotion patterns, including forward walking and two distinct running gaits, each lasting 2.4 seconds. Utilizing TO for motion dataset generation offers several advantages. Firstly, it enables precise matching of the state space between the simulated agent and demonstrator, leveraging kinematic dynamics models to refine trajectory suitability. Moreover, this approach circumvents complexities associated with other motion re-targeting techniques, ensuring a more seamless and accurate replication of desired motions.
III
Framework and Training
III-A
Learning Framework
III-A
1
Teacher Policy Architecture
The teacher policy
π
θ
teacher
superscript
subscript
𝜋
𝜃
teacher
\pi_{\theta}^{\text{teacher}}
italic_π start_POSTSUBSCRIPT italic_θ end_POSTSUBSCRIPT start_POSTSUPERSCRIPT teacher end_POSTSUPERSCRIPT
is trained using Proximal Policy Optimization
[
22
]
with the total reward
r
t
subscript
𝑟
𝑡
r_{t}
italic_r start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT
as specified in Section
II-B
. During training, the teacher performs a rollout in the environment to generate a state transition
(
s
t
AMP
,
s
t
+
1
AMP
)
superscript
subscript
𝑠
𝑡
AMP
superscript
subscript
𝑠
𝑡
1
AMP
(s_{t}^{\text{AMP}},s_{t+1}^{\text{AMP}})
( italic_s start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT start_POSTSUPERSCRIPT AMP end_POSTSUPERSCRIPT , italic_s start_POSTSUBSCRIPT italic_t + 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT AMP end_POSTSUPERSCRIPT )
. This state transition is then fed into the discriminator described in Section
II-B
.
The teacher policy consists of three Multilayer Perceptron (MLP) networks. Two of these MLPs encode low-dimensional latent representations:
l
t
e
∈
ℝ
16
superscript
subscript
𝑙
𝑡
𝑒
superscript
ℝ
16
l_{t}^{e}\in\mathbb{R}^{16}
italic_l start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_e end_POSTSUPERSCRIPT ∈ blackboard_R start_POSTSUPERSCRIPT 16 end_POSTSUPERSCRIPT
for terrain data and
l
t
p
∈
ℝ
8
superscript
subscript
𝑙
𝑡
𝑝
superscript
ℝ
8
l_{t}^{p}\in\mathbb{R}^{8}
italic_l start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_p end_POSTSUPERSCRIPT ∈ blackboard_R start_POSTSUPERSCRIPT 8 end_POSTSUPERSCRIPT
for privileged data. Using two separate MLPs for encoding helps mitigate information loss that often occurs during the compression process, thereby preserving crucial and necessary information.
The preservation of essential data significantly aids the student policy in reconstructing the latent representations, facilitating a more efficient and accurate learning process. The third MLP acts as a low-level network, utilizing the proprioceptive observation state along with the two encoded latent representations to generate the teacher’s action in the environment.
The learning framework is shown in Fig.
1
.
III-A
2
Student Policy Architecture
The student policy is designed to emulate the teacher policy, replicating actions without relying on privileged state and terrain information. Throughout the student training process, a supervised approach is employed, minimizing two key losses: imitation loss and reconstruction loss. The imitation loss ensures that the student policy closely mimic the action
a
t
t
⁢
e
⁢
a
⁢
c
⁢
h
⁢
e
⁢
r
∈
ℝ
12
superscript
subscript
𝑎
𝑡
𝑡
𝑒
𝑎
𝑐
ℎ
𝑒
𝑟
superscript
ℝ
12
a_{t}^{teacher}\in\mathbb{R}^{12}
italic_a start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_t italic_e italic_a italic_c italic_h italic_e italic_r end_POSTSUPERSCRIPT ∈ blackboard_R start_POSTSUPERSCRIPT 12 end_POSTSUPERSCRIPT
dictated by the teacher’s policy. Simultaneously, the reconstruction loss encourages the memory encoder within the student’s policy to faithfully reproduce the latent representation
l
t
t
⁢
e
⁢
a
⁢
c
⁢
h
⁢
e
⁢
r
∈
ℝ
24
superscript
subscript
𝑙
𝑡
𝑡
𝑒
𝑎
𝑐
ℎ
𝑒
𝑟
superscript
ℝ
24
l_{t}^{teacher}\in\mathbb{R}^{24}
italic_l start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_t italic_e italic_a italic_c italic_h italic_e italic_r end_POSTSUPERSCRIPT ∈ blackboard_R start_POSTSUPERSCRIPT 24 end_POSTSUPERSCRIPT
consists of the terrain latent
l
o
t
∈
ℝ
16
superscript
subscript
𝑙
𝑜
𝑡
superscript
ℝ
16
l_{o}^{t}\in\mathbb{R}^{16}
italic_l start_POSTSUBSCRIPT italic_o end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_t end_POSTSUPERSCRIPT ∈ blackboard_R start_POSTSUPERSCRIPT 16 end_POSTSUPERSCRIPT
and privileged latent
l
o
t
∈
ℝ
8
superscript
subscript
𝑙
𝑜
𝑡
superscript
ℝ
8
l_{o}^{t}\in\mathbb{R}^{8}
italic_l start_POSTSUBSCRIPT italic_o end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_t end_POSTSUPERSCRIPT ∈ blackboard_R start_POSTSUPERSCRIPT 8 end_POSTSUPERSCRIPT
employed by the teacher.
The overarching architecture comprises a memory encoder and a low-level MLP
[
23
]
that maintains an identical structure to the teacher’s low-level network. Memory encoders are implemented by stacking a sequence of 45 historical observations information
H
t
:
[
O
t
−
N
,
O
t
−
N
−
1
,
…
,
O
t
−
1
]
:
subscript
𝐻
𝑡
subscript
𝑂
𝑡
𝑁
subscript
𝑂
𝑡
𝑁
1
…
subscript
𝑂
𝑡
1
H_{t}:[O_{t-N},O_{t-N-1},...,O_{t-1}]
italic_H start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT : [ italic_O start_POSTSUBSCRIPT italic_t - italic_N end_POSTSUBSCRIPT , italic_O start_POSTSUBSCRIPT italic_t - italic_N - 1 end_POSTSUBSCRIPT , … , italic_O start_POSTSUBSCRIPT italic_t - 1 end_POSTSUBSCRIPT ]
into the input of an MLP.
III-B
Training and Implementation Details
III-B
1
Termination
The robot’s locomotion training is governed by specific termination conditions designed to ensure safety and effectiveness. Episodes are terminated upon detecting collisions involving the trunk, upper limbs, thighs, and calves, prioritizing the study’s focus on bipedal locomotion with only two-foot ground contact.
III-B
2
Domain Randomization
To facilitate the transfer of learned behaviors from simulation to the real world, domain randomization has been employed. This approach involves randomizing various parameters crucial for robot locomotion, such as terrain friction, base mass, joint PD controller gains, ground friction, restitution, and perturbations to the robot’s base velocity. During training, sampled velocity vectors are added to the robot’s current base velocity at random intervals. The specific randomization variables and their corresponding uniform distribution ranges are detailed in Table
I
, enabling robust policy adaptation and testing in diverse real-world environments.
TABLE I:
Randomized Simulation Parameters
Parameters
Range
Unit
Link Mass
[0.8, 1.2]
kg
Payload Mass
[0, 3]
kg
Payload Range
[-3]
-
Ground Friction
[0.05, 2.75]
-
Ground Restitution
[0.0, 1.0]
-
Joint
K
p
subscript
𝐾
𝑝
K_{p}
italic_K start_POSTSUBSCRIPT italic_p end_POSTSUBSCRIPT
[0.8, 1.2]
×
\times
×
20
-
Joint
K
d
subscript
𝐾
𝑑
K_{d}
italic_K start_POSTSUBSCRIPT italic_d end_POSTSUBSCRIPT
[0.8, 1.2]
×
\times
×
0.5
-
Figure 2
:
Terrains in Isaac Gym Simulations.
Figure 3
:
Base linear velocity in the x direction and base angular velocity in yaw for the robot on both uniform and discrete obstacle terrains.
III-B
3
Simulation Setup
We trained 500 parallel agents on different types of terrains with increasing difficulties using the Isaac Gym simulator
[
24
]
in overall 26000 iterations and cost 15.88 hours in total. Fig
2
showed the simulation in different terrain type. Each
RL episode lasts for a maximum of 1000 steps, and terminates early if it reaches the termination criteria. The control frequency of the policy is 50 Hz in the simulation. All training’s were performed on a single NVIDIA RTX 4070 GPU.
In A1 locomotion scenarios, actions are represented by
a
t
∈
R
12
subscript
𝑎
𝑡
superscript
𝑅
12
a_{t}\in R^{12}
italic_a start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT ∈ italic_R start_POSTSUPERSCRIPT 12 end_POSTSUPERSCRIPT
a 12-dimensional vector specifying the desired positional adjustments for each actuated joint as dictated by the Proportional-Derivative (PD) controller.
IV
Results
In our experiments within the Isaac Gym simulation environment, we evaluated the performance of the updated policy for a quadrupedal robot in biped locomotion. The results, depicted in Table
II
, show varying success rates and tracking accuracy across different terrains and speeds.
TABLE II
:
Comparison of tracking accuracy and success rate on different terrain and speed
Terrain Types
Uniform
Wave
Stepping Stones
Sloped
Stairs
Obstacles
0.5 m/s
Acc (%)
82.76
80.84
80.06
81.37
77.67
80.88
Succ (%)
92.3
91.76
91.6
84.21
90.09
89.55
1.0 m/s
Acc (%)
81.29
79.64
77.73
80.01
71.57
73.51
Succ (%)
90
91.59
90.01
82.14
86.14
86.59
1.5 m/s
Acc (%)
76.60
77.19
73.46
78.43
64.57
71.47
Succ (%)
86.96
85.06
88.73
79.5
84.91
84.55
2.0 m/s
Acc (%)
63.73
66.28
64.69
64.68
53.38
61.52
Succ (%)
79.55
84.22
78.55
76.23
81.53
78.1
The robot generally performs well at lower speeds, with higher success rates on less challenging terrains. However, success rates and tracking accuracy decrease as speed and terrain difficulty increase, notably on sloped terrains and obstacles due to mechanical design limitations.
The uniform terrain and discrete obstacle terrain were specifically chosen for detailed analysis as they are highly representative of uneven terrains. The uniform terrain has uneven features, while the discrete obstacle terrain combines elements of both flat and stepped obstacles, akin to stairs, providing a comprehensive challenge for evaluating the robot’s locomotion policy.
Figure 4
:
Robot’s feet contact forces on both uniform and discrete obstacle terrains.
Figure 5
:
Snapshots illustrating the response of a robot subjected to a 100N push force (indicated by a red arrow) applied along the x-axis over a duration of 0.1 seconds. The sequence shows the robot’s movement and stabilization process.
Fig.
3
presents the base linear velocity in the x direction and the base angular velocity in yaw for the robot on both uniform and discrete obstacle terrains. During the push test, noticeable changes in these velocities occur, reflecting the robot’s dynamic response to the external force. The robot’s base linear velocity tracking shows acceptable performance, with the measured velocities closely following the commanded values. The base angular velocity tracking, while not as accurate as the linear velocity, still demonstrates the robot’s ability to follow the commanded trajectory to a reasonable extent. The large spike observed around the 2-second mark is attributed to the external push force. Despite this disturbance, the robot quickly regains stability, indicating robust recovery capabilities of the locomotion policy. This ability to reject disturbances is crucial for maintaining stable locomotion in unpredictable environments.
The contact force data, shown in Fig.
4
, provides additional insights into the interaction between the robot’s feet and the terrain. The vertical ground reaction forces
F
z
subscript
𝐹
𝑧
F_{z}
italic_F start_POSTSUBSCRIPT italic_z end_POSTSUBSCRIPT
were measured for both the rear-left (RL) and rear-right (RR) legs on discrete obstacles and uniform terrains. The data reveals asymmetry in the contact forces, indicating that the learned policy does not exhibit a perfectly symmetrical gait.To address this issue and enhance the robot’s performance, future work could incorporate a symmetrical reward function or a reward based on the orientation of base tracking into the learning algorithm.
The push test results and performance analysis provide additional insights into the robot’s capabilities and resilience. As illustrated in Fig.
5
, the robot’s response to a 100N push applied along the x-axis over a duration of 0.1 seconds is captured, showcasing the policy’s effectiveness in handling external disturbances. Despite the significant push, the robot manages to stabilize itself quickly, demonstrating the robustness and resilience of the developed locomotion policy.
The results indicate that the legged robot’s locomotion policy exhibits satisfactory performance in both base linear velocity tracking and recovery from external disturbances. The ability to maintain stability and follow commanded trajectories on different terrains, as well as the efficient recovery from disturbances, underscores the robustness of the developed policy.
Overall, the experimental outcomes validate the effectiveness of our legged locomotion policy in achieving stable, efficient, and resilient movement across various terrains, even under external disturbances. These findings contribute valuable knowledge towards the development of energy-efficient and robust legged robots capable of operating in diverse and challenging environments.
V
Conclusion and Future work
In this paper, we proposed a novel framework for learning robust, agile, and natural bipedal locomotion skills for quadruped robots in simulation. Utilizing a teacher-student learning framework with privileged and terrain information, we enhanced the robustness of the learned policy and helped bridge the sim-to-real gap. By integrating adversarial motion imitation, the learned gait mimics the style and behavior of a TO reference gait. Our results demonstrate high-performance blind locomotion in a quadruped robot in biped mode.
Overall, our findings highlight the potential of imitation learning and TO in achieving agile and robust locomotion across diverse robotic platforms. Future work will focus on developing more robust biped motion capabilities on uneven terrain, transferring these capabilities to physical robots, and refining the transition from quadrupedal to biped mode to enhance legged robots’ versatility.
References
[1]
L. Bao, J. Humphreys, T. Peng, and C. Zhou, “Deep reinforcement learning for bipedal locomotion: A brief survey,” 2024.
[2]
J. Ho and S. Ermon, “Generative adversarial imitation learning,” in
International Conference on Neural Information Processing Systems
, 2016, pp. 4572–4580.
[3]
X. Peng, G. Berseth, K. Yin, and M. Panne, “DeepLoco: dynamic locomotion skills using hierarchical deep reinforcement learning,”
ACM Transactions on Graphics
, vol. 36, pp. 1–13, 2017.
[4]
Z. Xie, H. Ling, N. Kim, and M. Panne, “ALLSTEPS: Curriculum-driven learning of stepping stone skills,”
Computer Graphics Forum
, vol. 39, pp. 213–224, 2020.
[5]
J. Siekmann, Y. Godse, A. Fern, and J. Hurst, “Sim-to-real learning of all common bipedal gaits via periodic reward composition,” in
IEEE International Conference on Robotics and Automation
, 2021, pp. 7309–7315.
[6]
Z. Xie, P. Clary, J. Dao, P. Morais, J. Hurst, and M. van de Panne, “Learning locomotion skills for cassie: Iterative design and sim-to-real,” in
Conference on Robot Learning
, 2020, pp. 317–329.
[7]
Z. Li, X. Cheng, X. B. Peng, P. Abbeel, S. Levine, G. Berseth, and K. Sreenath, “Reinforcement learning for robust parameterized locomotion control of bipedal robots,” in
IEEE International Conference on Robotics and Automation
, 2021, pp. 2811–2817.
[8]
Q. Zhang, P. Cui, D. Yan, J. Sun, Y. Duan, A. Zhang, and R. Xu, “Whole-body humanoid robot locomotion with human reference,”
arXiv preprint arXiv:2402.18294
, 2024.
[9]
J. Wu, G. Xin, C. Qi, and Y. Xue, “Learning robust and agile legged locomotion using adversarial motion priors,”
IEEE Robotics and Automation Letters
, vol. 8, no. 8, pp. 4975–4982, 2023.
[10]
A. Escontrela, X. B. Peng, W. Yu, T. Zhang, A. Iscen, K. Goldberg, and P. Abbeel, “Adversarial motion priors make good substitutes for complex reward functions,” in
IEEE/RSJ International Conference on Intelligent Robots and Systems
, 2022, pp. 25–32.
[11]
Y. Wang, Z. Jiang, and J. Chen, “Learning robust, agile, natural legged locomotion skills in the wild,” in
RoboLetics: Workshop on Robot Learning in Athletics@ CoRL
, 2023.
[12]
Q. Zhang, P. Cui, D. Yan, J. Sun, Y. Duan, A. Zhang, and R. Xu, “Whole-body humanoid robot locomotion with human reference,”
arXiv preprint arXiv:2402.18294
, 2024.
[13]
A. Winkler, C. D. Bellicoso, M. Hutter, and J. Buchli, “Gait and trajectory optimization for legged systems through phase-based end-effector parameterization,”
IEEE Robotics and Automation Letters
, vol. 3, no. 3, pp. 1560–1567, 2018.
[14]
F. Yu, R. Batke, J. Dao, J. Hurst, K. Green, and A. Fern, “Dynamic bipedal turning through sim-to-real reinforcement learning,” in
IEEE-RAS International Conference on Humanoid Robots
, 2022, pp. 903–910.
[15]
H. Duan, J. Dao, K. Green, T. Apgar, A. Fern, and J. Hurst, “Learning task space actions for bipedal locomotion,” in
IEEE International Conference on Robotics and Automation
, 2021, pp. 1276–1282.
[16]
G. A. Castillo, B. Weng, S. Yang, W. Zhang, and A. Hereid, “Template model inspired task space learning for robust bipedal locomotion,” in
IEEE/RSJ International Conference on Intelligent Robots and Systems
, 2023, pp. 8582–8589.
[17]
J. Lee, J. Hwangbo, L. Wellhausen, V. Koltun, and M. Hutter, “Learning quadrupedal locomotion over challenging terrain,”
Science robotics
, vol. 5, no. 47, p. eabc5986, 2020.
[18]
A. Kumar, Z. Fu, D. Pathak, and J. Malik, “Rma: Rapid motor adaptation for legged robots,”
arXiv preprint arXiv:2107.04034
, 2021.
[19]
C. Yu and A. Rosendo, “Multi-modal legged locomotion framework with automated residual reinforcement learning,”
IEEE Robotics and Automation Letters
, vol. 7, no. 4, pp. 10 312–10 319, 2022.
[20]
X. B. Peng, Z. Ma, P. Abbeel, S. Levine, and A. Kanazawa, “AMP: Adversarial motion priors for stylized physics-based character control,”
ACM Transactions on Graphics
, vol. 40, no. 4, pp. 1–20, 2021.
[21]
A. W. Winkler, “TOWR–an open-source trajectory optimizer for legged robots in c,” 2018, [Online]. Available:
https://github.com/ethz-adrl/towr
.
[22]
J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, “Proximal policy optimization algorithms,”
arXiv preprint arXiv:1707.06347
, 2017.
[23]
G. B. Margolis, G. Yang, K. Paigwar, T. Chen, and P. Agrawal, “Rapid locomotion via reinforcement learning,”
arXiv preprint arXiv:2205.02824
, 2022.
[24]
N. Rudin, D. Hoeller, P. Reist, and M. Hutter, “Learning to walk in minutes using massively parallel deep reinforcement learning,” in
Conference on Robot Learning
.   PMLR, 2022, pp. 91–100.