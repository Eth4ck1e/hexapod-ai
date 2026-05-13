---
title: Learning to Walk and Fly with Adversarial Motion Priors
id: learning-to-walk-and-fly-with-adversarial-motion-priors
tags:
- legged-rl-budgets
- amp
- humanoid
- multi-modal
- isaac-gym
- training-budget
- long-source
created: '2026-05-06T07:30:56.427905Z'
updated: '2026-05-06T07:35:19.411928Z'
source: https://arxiv.org/html/2309.12784v2
source_domain: arxiv.org
fetched_at: '2026-05-06T07:30:56.427905Z'
fetch_provider: builtin
status: draft
type: note
tier: institutional
content_type: paper
deprecated: false
summary: 'L''Erario et al. (arXiv 2309.12784, 2023, IIT/Zurich). AMP applied to aerial
  humanoid robot (iRonCub, 44kg, 23 joints, 4 jet engines) to learn smooth walk-to-fly
  transitions without explicit mode switching. AMP-trained policy selects locomotion
  mode based on environment. Motion datasets: (i) human walking mocap retargeted via
  IK from CMU dataset; (ii) flying motions from trajectory optimization. Discriminator
  fed consecutive observation pairs. KEY TRAINING BUDGET (verbatim from Section IV-B):
  ''4096 agents using PPO, controlled at 60 Hz. The training requires ~2 hours on
  an NVIDIA Quadro RTX 6000.'' PPO parameters from Table I: mini-batch size 32768,
  discount 0.99, learning rate 5e-5, clip 0.2, entropy coefficient 0.0, KL threshold
  0.008. Total env steps not explicitly stated but with 4096 agents and ~2 hour wall-clock,
  at typical Isaac Gym rates this is in the range of hundreds of millions. No BC pretrain
  stage mentioned. No DR finetune budget breakdown. >10,500 words — flagged for analyst
  delegation.'
---

Learning to Walk and Fly with Adversarial Motion Priors
Learning to Walk and Fly with Adversarial Motion Priors
Giuseppe L’Erario
1,3
, Drew Hanover
2
, Ángel Romero
2
, Yunlong Song
2
,
Gabriele Nava
1
, Paolo Maria Viceconte
1
, Daniele Pucci
1,3
, Davide Scaramuzza
2
1
Artificial and Mechanical Intelligence, Istituto Italiano di Tecnologia, Genoa, Italy
2
Robotics and Perception Group, University of Zurich, Switzerland.
3
University of Manchester, Manchester, UK.
Abstract
Robot multimodal locomotion encompasses the ability to transition between walking and flying, representing a significant challenge in robotics. This work presents an approach that enables automatic smooth transitions between legged and aerial locomotion. Leveraging the concept of Adversarial Motion Priors, our method allows the robot to imitate motion datasets and accomplish the desired task without the need for complex reward functions. The robot learns walking patterns from human-like gaits and aerial locomotion patterns from motions obtained using trajectory optimization.
Through this process, the robot adapts the locomotion scheme based on environmental feedback using reinforcement learning, with the spontaneous emergence of mode-switching behavior.
The results highlight the potential for achieving multimodal locomotion in aerial humanoid robotics through automatic control of walking and flying modes, paving the way for applications in diverse domains such as search and rescue, surveillance, and exploration missions.
This research contributes to advancing the capabilities of aerial humanoid robots in terms of versatile locomotion in various environments.
Video
:
https://youtu.be/qvE-XFtcfjU
.
I
Introduction
The transition between locomotion styles is a phenomenon that many species across the animal kingdom exhibit.
Not many robots, however, can combine aerial and terrestrial locomotion, thus leaving the problem of traversing
different environments still partially open. The broader research community has begun to study what it takes to enable multimodal locomotion for various bio-inspired robotic systems
[
1
]
.
Countless examples have demonstrated systems with bespoke forms of locomotion: bat-inspired robots that can fly and walk
[
2
]
,
propeller-driven bipedal robots
[
3
]
, morphing rovers
[
4
]
, rolling quadrupeds
[
5
]
, underwater robots
[
6
,
7
]
and even flying humanoid robots
[
8
,
9
]
.
However, little work exists on understanding how and when to transition between the locomotion forms.
For example, when dealing with an aerial-humanoid robot, what is the proper time to transition between modes such as walking or flying when considering a high-level locomotion task?
This goal can be accomplished by breaking out the controller into submodules, where each module handles an individual locomotion task, as proposed by
[
3
]
.
Nevertheless, the transition between locomotion forms is a significant challenge and remains an open question.
This paper moves forward with a learning-based method that enables
aerial humanoid robots to exhibit multimodal locomotion capabilities.
A standard approach to address the locomotion problem is to decompose the system into two layers
[
10
]
:
1) A trajectory optimization (TO) layer, whose role is to provide a set of feasible trajectories
[
11
,
12
]
;
2) Whole-body control, which stabilizes the trajectories produced by the TO layer
[
13
]
.
Figure 1:
iRonCub, the aerial humanoid robot, on a complex terrain.
The TO layer lies at the core of the locomotion architecture, and it is responsible for generating a set of feasible trajectories that the robot can track. It is formulated as an optimization problem that minimizes a cost function – encoding the desired robot behavior – subject to constraints – characterizing the system dynamics, also called
model
.
Typical approaches for legged systems as humanoid robots rely on
simplified
models such as the linear inverted pendulum (LIP)
[
14
]
, or the divergent component of motion (DCM)
[
15
]
.
While optimization methods using these simplified models are fast, they cannot exploit the robot’s complete structure. In contrast, approaches leveraging the robot’s
full
dynamics can utilize its inherent system structure, albeit they are computationally expensive.
Reduced
models as the
centroidal momentum dynamics
are a compromise between the two extremes and reduce the problem’s complexity
[
16
,
17
,
18
,
19
]
.
Nonetheless, TO methods remain computationally intensive and demand prior knowledge of the environment. On the other side of the spectrum, model-free techniques such as Reinforcement Learning (RL) can learn a policy that maps the state of the system to action without the need for a model
[
20
]
. These tools enable a new set of control strategies for the robotics realm
[
21
,
22
,
23
,
24
,
25
]
. The main challenge in RL for robot control lies in the limited ability of trained agents to exhibit natural behavior in specific tasks, often resulting in inefficient solutions that exploit simulator inaccuracies.
To address this limitation, researchers have focused on how nature’s ingenuity has shaped the movements of animals, such as motor skills exhibited by humans, which they embed into simulated through data-driven approaches
[
26
,
27
,
28
,
29
,
30
]
.
These techniques recently found their way into robotics, where they are used to train robots to imitate animal-like motions
[
31
,
32
,
33
]
.
One such method, called Adversarial motion priors (AMP)
[
27
]
, takes cues from Generative Adversarial Imitation Learning (GAIL) to train an agent to replicate the “style” embedded in a reference dataset
[
34
]
.
This paper progresses in this direction by proposing a method that utilizes AMP and RL to study the spontaneous emergence of automatic and smooth locomotion transitions in aerial humanoid robots.
AMP allows the robot to learn and mimic human-like motions when walking - enhancing the naturalness of the gait - and imitating flying motion obtained via trajectory optimization.
The output of the proposed approach is a locomotion pattern that switches between human-like and flying locomotion based on the task and the environment.
We train locomotion policies for iRonCub, a jet-powered aerial humanoid robot with terrestrial and aerial locomotion capabilities.
We learn to mimic the motion styles provided within the motion datasets while achieving a high-level waypoint tracking task.
The transition between flying and walking is managed by including an energy proxy term in the reward function, encouraging the robot to walk when the ground is within reach and fly otherwise. The method is tested in both the cases of
ideal
thrust propulsion and
specific
jet-powered actuation modeled from real-world data.
We performed experiments within the Nvidia Isaac Gym environment
[
35
,
36
]
, forcing the agent to traverse complex terrains and rough courses.
To our knowledge, this is the first demonstration of an aerial humanoid robot exhibiting smooth transitions between walking and flying without explicitly tracking a trajectory provided by trajectory optimization or using a state machine to switch between locomotion modes.
Instead, our agent learns how to fly, walk, and navigate without explicitly being told how or what locomotion to use.
The paper is organized as follows. Sec
II
introduces notation and recalls some critical ideas about floating-base systems. Sec
III
presents the whole approach by recalling the AMP method and introducing the system modeling. Sec
IV
validates the technique on iRonCub, a flying humanoid robot. Sec
V
concludes the paper with remarks and future work.
II
Background
II-A
Floating-base modeling
A robot can be modeled as a multi-body system composed of
n
+
1
𝑛
1
n+1
italic_n + 1
rigid bodies – called links – connected by
n
𝑛
n
italic_n
joints with one degree of freedom. Using of the
floating base
formalism
[
37
]
, the robot’s configuration is defined by the tuple
q
=
(
p
ℬ
ℐ
,
R
ℬ
ℐ
,
θ
)
∈
ℚ
𝑞
superscript
subscript
𝑝
ℬ
ℐ
superscript
subscript
𝑅
ℬ
ℐ
𝜃
ℚ
q=({}^{\mathcal{I}}p_{\mathcal{B}},{}^{\mathcal{I}}R_{\mathcal{B}},\theta)\in%
\mathbb{Q}
italic_q = ( start_FLOATSUPERSCRIPT caligraphic_I end_FLOATSUPERSCRIPT italic_p start_POSTSUBSCRIPT caligraphic_B end_POSTSUBSCRIPT , start_FLOATSUPERSCRIPT caligraphic_I end_FLOATSUPERSCRIPT italic_R start_POSTSUBSCRIPT caligraphic_B end_POSTSUBSCRIPT , italic_θ ) ∈ blackboard_Q
, where
p
ℬ
ℐ
∈
ℝ
3
superscript
subscript
𝑝
ℬ
ℐ
superscript
ℝ
3
{}^{\mathcal{I}}p_{\mathcal{B}}\in\mathbb{R}^{3}
start_FLOATSUPERSCRIPT caligraphic_I end_FLOATSUPERSCRIPT italic_p start_POSTSUBSCRIPT caligraphic_B end_POSTSUBSCRIPT ∈ blackboard_R start_POSTSUPERSCRIPT 3 end_POSTSUPERSCRIPT
and
R
ℬ
ℐ
∈
S
⁢
O
⁢
(
3
)
superscript
subscript
𝑅
ℬ
ℐ
𝑆
𝑂
3
{}^{\mathcal{I}}R_{\mathcal{B}}\in SO(3)
start_FLOATSUPERSCRIPT caligraphic_I end_FLOATSUPERSCRIPT italic_R start_POSTSUBSCRIPT caligraphic_B end_POSTSUBSCRIPT ∈ italic_S italic_O ( 3 )
are the position and orientation of the robot base frame
ℬ
ℬ
\mathcal{B}
caligraphic_B
w.r.t. the inertial frame
ℐ
ℐ
\mathcal{I}
caligraphic_I
, and
θ
∈
ℝ
n
𝜃
superscript
ℝ
𝑛
\theta\in\mathbb{R}^{n}
italic_θ ∈ blackboard_R start_POSTSUPERSCRIPT italic_n end_POSTSUPERSCRIPT
are the joint positions.
The element
ν
=
(
p
˙
ℬ
ℐ
,
ω
ℬ
ℐ
,
θ
˙
)
𝜈
superscript
subscript
˙
𝑝
ℬ
ℐ
superscript
subscript
𝜔
ℬ
ℐ
˙
𝜃
\nu=({}^{\mathcal{I}}\dot{p}_{\mathcal{B}},{}^{\mathcal{I}}\omega_{\mathcal{B}%
},\dot{\theta})
italic_ν = ( start_FLOATSUPERSCRIPT caligraphic_I end_FLOATSUPERSCRIPT over˙ start_ARG italic_p end_ARG start_POSTSUBSCRIPT caligraphic_B end_POSTSUBSCRIPT , start_FLOATSUPERSCRIPT caligraphic_I end_FLOATSUPERSCRIPT italic_ω start_POSTSUBSCRIPT caligraphic_B end_POSTSUBSCRIPT , over˙ start_ARG italic_θ end_ARG )
is the robot’s velocity, where
θ
˙
˙
𝜃
\dot{\theta}
over˙ start_ARG italic_θ end_ARG
are the joint velocities and
(
p
˙
ℬ
ℐ
,
ω
ℬ
ℐ
)
superscript
subscript
˙
𝑝
ℬ
ℐ
superscript
subscript
𝜔
ℬ
ℐ
({}^{\mathcal{I}}\dot{p}_{\mathcal{B}},{}^{\mathcal{I}}\omega_{\mathcal{B}})
( start_FLOATSUPERSCRIPT caligraphic_I end_FLOATSUPERSCRIPT over˙ start_ARG italic_p end_ARG start_POSTSUBSCRIPT caligraphic_B end_POSTSUBSCRIPT , start_FLOATSUPERSCRIPT caligraphic_I end_FLOATSUPERSCRIPT italic_ω start_POSTSUBSCRIPT caligraphic_B end_POSTSUBSCRIPT )
are the linear and angular velocity of the base frame
ℬ
ℬ
\mathcal{B}
caligraphic_B
w.r.t.
ℐ
ℐ
\mathcal{I}
caligraphic_I
such that
R
ℬ
ℐ
˙
=
S
⁢
(
ω
ℬ
ℐ
)
⁢
R
ℬ
ℐ
˙
superscript
subscript
𝑅
ℬ
ℐ
𝑆
superscript
subscript
𝜔
ℬ
ℐ
superscript
subscript
𝑅
ℬ
ℐ
\dot{{}^{\mathcal{I}}R_{\mathcal{B}}}=S({}^{\mathcal{I}}\omega_{\mathcal{B}}){%
}^{\mathcal{I}}R_{\mathcal{B}}
over˙ start_ARG start_FLOATSUPERSCRIPT caligraphic_I end_FLOATSUPERSCRIPT italic_R start_POSTSUBSCRIPT caligraphic_B end_POSTSUBSCRIPT end_ARG = italic_S ( start_FLOATSUPERSCRIPT caligraphic_I end_FLOATSUPERSCRIPT italic_ω start_POSTSUBSCRIPT caligraphic_B end_POSTSUBSCRIPT ) start_FLOATSUPERSCRIPT caligraphic_I end_FLOATSUPERSCRIPT italic_R start_POSTSUBSCRIPT caligraphic_B end_POSTSUBSCRIPT
is satisfied.
By applying the Euler-Poincaré formalism, the equation of motion of a system exchanging
n
b
subscript
𝑛
𝑏
n_{b}
italic_n start_POSTSUBSCRIPT italic_b end_POSTSUBSCRIPT
wrenches with the environment results in
M
⁢
(
q
)
⁢
ν
˙
+
C
⁢
(
q
,
ν
)
⁢
ν
+
G
⁢
(
q
)
=
[
0
6
×
1
τ
]
+
∑
i
=
1
n
b
J
i
⊤
⁢
f
i
,
𝑀
𝑞
˙
𝜈
𝐶
𝑞
𝜈
𝜈
𝐺
𝑞
matrix
subscript
0
6
1
𝜏
superscript
subscript
𝑖
1
subscript
𝑛
𝑏
subscript
superscript
𝐽
top
𝑖
subscript
f
𝑖
M(q)\dot{\nu}+C(q,\nu)\nu+G(q)=\begin{bmatrix}0_{6\times 1}\\
\tau\end{bmatrix}+\sum_{i=1}^{n_{b}}J^{\top}_{i}\mathrm{f}_{i},
italic_M ( italic_q ) over˙ start_ARG italic_ν end_ARG + italic_C ( italic_q , italic_ν ) italic_ν + italic_G ( italic_q ) = [ start_ARG start_ROW start_CELL 0 start_POSTSUBSCRIPT 6 × 1 end_POSTSUBSCRIPT end_CELL end_ROW start_ROW start_CELL italic_τ end_CELL end_ROW end_ARG ] + ∑ start_POSTSUBSCRIPT italic_i = 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_n start_POSTSUBSCRIPT italic_b end_POSTSUBSCRIPT end_POSTSUPERSCRIPT italic_J start_POSTSUPERSCRIPT ⊤ end_POSTSUPERSCRIPT start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT roman_f start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ,
(1)
where
M
,
C
∈
ℝ
(
n
+
6
)
×
(
n
+
6
)
𝑀
𝐶
superscript
ℝ
𝑛
6
𝑛
6
M,C\in\mathbb{R}^{(n+6)\times(n+6)}
italic_M , italic_C ∈ blackboard_R start_POSTSUPERSCRIPT ( italic_n + 6 ) × ( italic_n + 6 ) end_POSTSUPERSCRIPT
are the mass and Coriolis matrix,
G
∈
ℝ
n
+
6
𝐺
superscript
ℝ
𝑛
6
G\in\mathbb{R}^{n+6}
italic_G ∈ blackboard_R start_POSTSUPERSCRIPT italic_n + 6 end_POSTSUPERSCRIPT
is the gravity vector,
τ
∈
ℝ
n
𝜏
superscript
ℝ
𝑛
\tau\in\mathbb{R}^{n}
italic_τ ∈ blackboard_R start_POSTSUPERSCRIPT italic_n end_POSTSUPERSCRIPT
are the internal actuation torques,
f
i
subscript
f
𝑖
\mathrm{f}_{i}
roman_f start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT
is the
i
𝑖
i
italic_i
-th of the
n
b
subscript
𝑛
𝑏
n_{b}
italic_n start_POSTSUBSCRIPT italic_b end_POSTSUBSCRIPT
external wrench applied on the origin of the frame
𝒞
i
subscript
𝒞
𝑖
\mathcal{C}_{i}
caligraphic_C start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT
and
J
i
∈
ℝ
6
×
(
n
+
6
)
subscript
𝐽
𝑖
superscript
ℝ
6
𝑛
6
J_{i}\in\mathbb{R}^{6\times(n+6)}
italic_J start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ∈ blackboard_R start_POSTSUPERSCRIPT 6 × ( italic_n + 6 ) end_POSTSUPERSCRIPT
is the jacobian mapping the system velocity
ν
𝜈
\nu
italic_ν
to the velocity
(
p
˙
𝒞
i
ℐ
,
ω
𝒞
i
ℐ
)
superscript
subscript
˙
𝑝
subscript
𝒞
𝑖
ℐ
superscript
subscript
𝜔
subscript
𝒞
𝑖
ℐ
({}^{\mathcal{I}}\dot{p}_{\mathcal{C}_{i}},{}^{\mathcal{I}}\omega_{\mathcal{C}%
_{i}})
( start_FLOATSUPERSCRIPT caligraphic_I end_FLOATSUPERSCRIPT over˙ start_ARG italic_p end_ARG start_POSTSUBSCRIPT caligraphic_C start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT end_POSTSUBSCRIPT , start_FLOATSUPERSCRIPT caligraphic_I end_FLOATSUPERSCRIPT italic_ω start_POSTSUBSCRIPT caligraphic_C start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT end_POSTSUBSCRIPT )
of the frame
𝒞
i
subscript
𝒞
𝑖
\mathcal{C}_{i}
caligraphic_C start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT
.
These quantities are useful to understand the choice of the state and action space of the robot.
II-B
Reinforcement Learning
We model the problem as a Markov Decision Process (MDP) defined by a tuple
(
𝒮
,
𝒜
,
𝒫
,
ℛ
,
γ
)
𝒮
𝒜
𝒫
ℛ
𝛾
(\mathcal{S},\mathcal{A},\mathcal{P},\mathcal{R},\gamma)
( caligraphic_S , caligraphic_A , caligraphic_P , caligraphic_R , italic_γ )
[
20
]
. The state space
𝒮
𝒮
\mathcal{S}
caligraphic_S
is the set of all possible states of the system and the environment, while the action space
𝒜
𝒜
\mathcal{A}
caligraphic_A
is the set of all possible actions that the agent can execute. The transition probability
𝒫
:
𝒮
×
𝒜
×
𝒮
→
𝖯𝗋
⁢
[
𝒮
]
:
𝒫
→
𝒮
𝒜
𝒮
𝖯𝗋
delimited-[]
𝒮
\mathcal{P}:\mathcal{S}\times\mathcal{A}\times\mathcal{S}\rightarrow\mathsf{Pr%
}[\mathcal{S}]
caligraphic_P : caligraphic_S × caligraphic_A × caligraphic_S → sansserif_Pr [ caligraphic_S ]
defines the probability of transitioning from a state
s
t
subscript
𝑠
𝑡
s_{t}
italic_s start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT
to a state
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
given an action
a
t
subscript
𝑎
𝑡
a_{t}
italic_a start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT
.
The discount factor
γ
∈
[
0
,
1
)
𝛾
0
1
\gamma\in[0,1)
italic_γ ∈ [ 0 , 1 )
trades off
long-term against short-term rewards. In the context of Deep RL, the agent is modeled as a neural network policy
π
σ
⁢
(
a
t
|
s
t
)
subscript
𝜋
𝜎
conditional
subscript
𝑎
𝑡
subscript
𝑠
𝑡
\pi_{\sigma}(a_{t}|s_{t})
italic_π start_POSTSUBSCRIPT italic_σ end_POSTSUBSCRIPT ( italic_a start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT | italic_s start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT )
that maps a state
s
t
subscript
𝑠
𝑡
s_{t}
italic_s start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT
to an action
a
t
subscript
𝑎
𝑡
a_{t}
italic_a start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT
, with
σ
𝜎
\sigma
italic_σ
the parameters of the neural network. The goal of DRL is to find the optimal set of parameters
σ
𝜎
\sigma
italic_σ
that maximize the expected discounted return
J
⁢
(
σ
)
=
𝔼
π
σ
⁢
[
∑
t
=
0
T
−
1
γ
t
⁢
r
t
]
,
𝐽
𝜎
subscript
𝔼
subscript
𝜋
𝜎
delimited-[]
superscript
subscript
𝑡
0
𝑇
1
superscript
𝛾
𝑡
subscript
𝑟
𝑡
J(\sigma)=\mathbb{E}_{\pi_{\sigma}}\left[\sum_{t=0}^{T-1}\gamma^{t}r_{t}\right],
italic_J ( italic_σ ) = blackboard_E start_POSTSUBSCRIPT italic_π start_POSTSUBSCRIPT italic_σ end_POSTSUBSCRIPT end_POSTSUBSCRIPT [ ∑ start_POSTSUBSCRIPT italic_t = 0 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_T - 1 end_POSTSUPERSCRIPT italic_γ start_POSTSUPERSCRIPT italic_t end_POSTSUPERSCRIPT italic_r start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT ] ,
(2)
where
ℛ
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
ℛ
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
\mathcal{R}(s_{t},a_{t},s_{t+1})
caligraphic_R ( italic_s start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT , italic_a start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT , italic_s start_POSTSUBSCRIPT italic_t + 1 end_POSTSUBSCRIPT )
represents the reward
r
t
subscript
𝑟
𝑡
r_{t}
italic_r start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT
received when the agent executes an action
a
t
subscript
𝑎
𝑡
a_{t}
italic_a start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT
in
s
t
subscript
𝑠
𝑡
s_{t}
italic_s start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT
and transitions to
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
.
III
Method
Our work aims to train a policy
π
𝜋
\pi
italic_π
capable of walking and flying, automatically selecting the best locomotion pattern to accomplish a task without needing a high-level planner. The agent should be able to imitate the locomotion patterns extracted from a set of motion datasets
𝒟
𝒟
\mathcal{D}
caligraphic_D
, in which each sample represents a snapshot capturing the system’s kinematics at a specific instant.
III-A
Adversarial Motion Priors
Figure 2:
The discriminator learns to distinguish between samples from the dataset and samples produced by the agent. The policy
π
σ
subscript
𝜋
𝜎
\pi_{\sigma}
italic_π start_POSTSUBSCRIPT italic_σ end_POSTSUBSCRIPT
is trained to imitate the dataset’s motion and accomplish a task simultaneously by maximizing the total reward
r
⁢
(
t
)
𝑟
𝑡
r(t)
italic_r ( italic_t )
that expresses the quality of the motion and the task accomplishment.
The agent is trained to imitate the motion of the dataset and to accomplish a task at the same time. This requirement translates into a reward function composed of two terms, as proposed in AMP
[
27
]
:
r
t
=
w
G
⁢
r
t
G
+
w
S
⁢
r
t
S
.
subscript
𝑟
𝑡
subscript
𝑤
G
superscript
subscript
𝑟
𝑡
G
subscript
𝑤
S
superscript
subscript
𝑟
𝑡
S
r_{t}=w_{\text{G}}{}^{\text{G}}r_{t}+w_{\text{S}}{}^{\text{S}}r_{t}.
italic_r start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT = italic_w start_POSTSUBSCRIPT G end_POSTSUBSCRIPT start_FLOATSUPERSCRIPT G end_FLOATSUPERSCRIPT italic_r start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT + italic_w start_POSTSUBSCRIPT S end_POSTSUBSCRIPT start_FLOATSUPERSCRIPT S end_FLOATSUPERSCRIPT italic_r start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT .
(3)
The
task reward
r
G
superscript
𝑟
G
{}^{\text{G}}r
start_FLOATSUPERSCRIPT G end_FLOATSUPERSCRIPT italic_r
specifies the goal the agent should accomplish, e.g., reach a target or have a desired base velocity. The
style reward
r
S
superscript
𝑟
S
{}^{\text{S}}r
start_FLOATSUPERSCRIPT S end_FLOATSUPERSCRIPT italic_r
enforces the policy to produce locomotion patterns resembling the motion of the dataset. The total reward encourages the agent to imitate a desired set of motion patterns while choosing the one that best accomplishes the desired goal. Given the environment, no high-level planner is required to select a specific motion, and the switching behavior emerges naturally. An overview of the system is given in Fig.
2
.
III-A
1
Style-reward
Generating data that resemble existing information is a characteristic of Generative Adversarial Networks (GANs)
[
38
]
. GANs consist of two competing neural networks: a generator that creates new data that resembles the training set and a discriminator that tries to distinguish between real and generated data, trained in a game-like fashion.
Similarly, we can use a discriminator
D
𝐷
D
italic_D
to distinguish between the samples of the dataset and those produced by the agent
π
σ
subscript
𝜋
𝜎
\pi_{\sigma}
italic_π start_POSTSUBSCRIPT italic_σ end_POSTSUBSCRIPT
that, in turn, acts as a generator aiming at creating realistic movements.
In practice, the discriminator
D
Φ
:
ℝ
k
×
ℝ
k
→
ℝ
:
subscript
𝐷
Φ
→
superscript
ℝ
𝑘
superscript
ℝ
𝑘
ℝ
D_{\Phi}:\mathbb{R}^{k}\times\mathbb{R}^{k}\rightarrow\mathbb{R}
italic_D start_POSTSUBSCRIPT roman_Φ end_POSTSUBSCRIPT : blackboard_R start_POSTSUPERSCRIPT italic_k end_POSTSUPERSCRIPT × blackboard_R start_POSTSUPERSCRIPT italic_k end_POSTSUPERSCRIPT → blackboard_R
is a neural network with parameters
Φ
Φ
\Phi
roman_Φ
that maps a couple of consecutive samples
χ
t
,
χ
t
+
1
subscript
𝜒
𝑡
subscript
𝜒
𝑡
1
\chi_{t},\chi_{t+1}
italic_χ start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT , italic_χ start_POSTSUBSCRIPT italic_t + 1 end_POSTSUBSCRIPT
of dimension
k
𝑘
k
italic_k
to a scalar value and aims at differentiating motions from the dataset
𝒟
𝒟
\mathcal{D}
caligraphic_D
from movements produced by the RL agent following the policy
π
σ
subscript
𝜋
𝜎
\pi_{\sigma}
italic_π start_POSTSUBSCRIPT italic_σ end_POSTSUBSCRIPT
.
The objective proposed in
[
27
,
28
]
to train the discriminator is
arg
⁢
min
Φ
−
𝔼
(
χ
t
,
χ
t
+
1
)
∼
𝒟
⁢
[
log
⁡
(
D
Φ
⁢
(
χ
t
,
χ
t
+
1
)
)
]
−
𝔼
(
χ
t
,
χ
t
+
1
)
∼
π
σ
⁢
[
log
⁡
(
1
−
D
Φ
⁢
(
χ
t
,
χ
t
+
1
)
)
]
+
w
gp
⁢
𝔼
(
χ
t
,
χ
t
+
1
)
∼
𝒟
⁢
[
‖
∇
Φ
D
Φ
⁢
(
χ
t
,
χ
t
+
1
)
‖
2
]
,
subscript
arg
min
Φ
subscript
𝔼
similar-to
subscript
𝜒
𝑡
subscript
𝜒
𝑡
1
𝒟
delimited-[]
subscript
𝐷
Φ
subscript
𝜒
𝑡
subscript
𝜒
𝑡
1
subscript
𝔼
similar-to
subscript
𝜒
𝑡
subscript
𝜒
𝑡
1
subscript
𝜋
𝜎
delimited-[]
1
subscript
𝐷
Φ
subscript
𝜒
𝑡
subscript
𝜒
𝑡
1
subscript
𝑤
gp
subscript
𝔼
similar-to
subscript
𝜒
𝑡
subscript
𝜒
𝑡
1
𝒟
delimited-[]
superscript
norm
subscript
∇
Φ
subscript
𝐷
Φ
subscript
𝜒
𝑡
subscript
𝜒
𝑡
1
2
\begin{split}\operatorname*{arg\,min}_{\Phi}\ &-\mathbb{E}_{(\chi_{t},\chi_{t+%
1})\sim\mathcal{D}}[\log(D_{\Phi}(\chi_{t},\chi_{t+1}))]\\
&-\mathbb{E}_{(\chi_{t},\chi_{t+1})\sim\mathcal{\pi_{\sigma}}}[\log(1-D_{\Phi}%
(\chi_{t},\chi_{t+1}))]\\
&+w_{\text{gp}}\mathbb{E}_{(\chi_{t},\chi_{t+1})\sim\mathcal{D}}\big{[}||%
\nabla_{\Phi}D_{\Phi}(\chi_{t},\chi_{t+1})||^{2}],\end{split}
start_ROW start_CELL start_OPERATOR roman_arg roman_min end_OPERATOR start_POSTSUBSCRIPT roman_Φ end_POSTSUBSCRIPT end_CELL start_CELL - blackboard_E start_POSTSUBSCRIPT ( italic_χ start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT , italic_χ start_POSTSUBSCRIPT italic_t + 1 end_POSTSUBSCRIPT ) ∼ caligraphic_D end_POSTSUBSCRIPT [ roman_log ( italic_D start_POSTSUBSCRIPT roman_Φ end_POSTSUBSCRIPT ( italic_χ start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT , italic_χ start_POSTSUBSCRIPT italic_t + 1 end_POSTSUBSCRIPT ) ) ] end_CELL end_ROW start_ROW start_CELL end_CELL start_CELL - blackboard_E start_POSTSUBSCRIPT ( italic_χ start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT , italic_χ start_POSTSUBSCRIPT italic_t + 1 end_POSTSUBSCRIPT ) ∼ italic_π start_POSTSUBSCRIPT italic_σ end_POSTSUBSCRIPT end_POSTSUBSCRIPT [ roman_log ( 1 - italic_D start_POSTSUBSCRIPT roman_Φ end_POSTSUBSCRIPT ( italic_χ start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT , italic_χ start_POSTSUBSCRIPT italic_t + 1 end_POSTSUBSCRIPT ) ) ] end_CELL end_ROW start_ROW start_CELL end_CELL start_CELL + italic_w start_POSTSUBSCRIPT gp end_POSTSUBSCRIPT blackboard_E start_POSTSUBSCRIPT ( italic_χ start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT , italic_χ start_POSTSUBSCRIPT italic_t + 1 end_POSTSUBSCRIPT ) ∼ caligraphic_D end_POSTSUBSCRIPT [ | | ∇ start_POSTSUBSCRIPT roman_Φ end_POSTSUBSCRIPT italic_D start_POSTSUBSCRIPT roman_Φ end_POSTSUBSCRIPT ( italic_χ start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT , italic_χ start_POSTSUBSCRIPT italic_t + 1 end_POSTSUBSCRIPT ) | | start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT ] , end_CELL end_ROW
(4)
where the first two terms
force the discriminator to output
D
Φ
⁢
(
χ
t
,
χ
t
+
1
)
=
1
subscript
𝐷
Φ
subscript
𝜒
𝑡
subscript
𝜒
𝑡
1
1
D_{\Phi}(\chi_{t},\chi_{t+1})=1
italic_D start_POSTSUBSCRIPT roman_Φ end_POSTSUBSCRIPT ( italic_χ start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT , italic_χ start_POSTSUBSCRIPT italic_t + 1 end_POSTSUBSCRIPT ) = 1
when the discriminator is fed with dataset transitions and
D
Φ
⁢
(
χ
t
,
χ
t
+
1
)
=
0
subscript
𝐷
Φ
subscript
𝜒
𝑡
subscript
𝜒
𝑡
1
0
D_{\Phi}(\chi_{t},\chi_{t+1})=0
italic_D start_POSTSUBSCRIPT roman_Φ end_POSTSUBSCRIPT ( italic_χ start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT , italic_χ start_POSTSUBSCRIPT italic_t + 1 end_POSTSUBSCRIPT ) = 0
when it is fed with samples produced by the agent. The final term is a gradient penalty that penalizes nonzero gradients on dataset transition, resulting in improved stability and quality of the training
[
39
]
.
When fed with samples produced by the agent, the discriminator serves as a critic that evaluates the quality of the motion produced by the policy. The output of
D
⁢
(
χ
t
,
χ
t
+
1
)
𝐷
subscript
𝜒
𝑡
subscript
𝜒
𝑡
1
D(\chi_{t},\chi_{t+1})
italic_D ( italic_χ start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT , italic_χ start_POSTSUBSCRIPT italic_t + 1 end_POSTSUBSCRIPT )
is hence used to compute the style reward
[
27
,
28
]
r
t
S
=
−
log
⁡
(
1
−
1
1
+
e
−
D
Φ
⁢
(
χ
t
,
χ
t
+
1
)
)
,
superscript
subscript
𝑟
𝑡
S
1
1
1
superscript
𝑒
subscript
𝐷
Φ
subscript
𝜒
𝑡
subscript
𝜒
𝑡
1
{}^{\text{S}}r_{t}=-\log{\Big{(}1-\dfrac{1}{1+e^{-D_{\Phi}(\chi_{t},\chi_{t+1}%
)}}\Big{)}},
start_FLOATSUPERSCRIPT S end_FLOATSUPERSCRIPT italic_r start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT = - roman_log ( 1 - divide start_ARG 1 end_ARG start_ARG 1 + italic_e start_POSTSUPERSCRIPT - italic_D start_POSTSUBSCRIPT roman_Φ end_POSTSUBSCRIPT ( italic_χ start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT , italic_χ start_POSTSUBSCRIPT italic_t + 1 end_POSTSUBSCRIPT ) end_POSTSUPERSCRIPT end_ARG ) ,
(5)
in which the negative logarithm is used to map the output of the discriminator to a reward that is high when the agent produces a transition similar to the dataset.
III-A
2
Task-reward
We want the robot to reach a target and follow a route made of checkpoints. Ideally, we could reward the agent when it hits the target. This approach makes associating a reward with an individual action complex since it introduces reward sparsity. A common practice is to use proxy rewards that guide the agent to the true objective and provide continuous feedback to the agent. We use three rewards to accomplish the task.
At each time instant, the agent can be associated with a checkpoint. First, we want to minimize the distance of the robot base to the target at each time instant
t
𝑡
t
italic_t
. This requirement translates into the reward:
r
t
c
=
exp
⁡
(
−
c
1
⁢
‖
x
d
−
p
t
‖
2
)
,
superscript
subscript
𝑟
𝑡
𝑐
subscript
𝑐
1
superscript
norm
subscript
𝑥
𝑑
subscript
𝑝
𝑡
2
{}^{c}r_{t}=\exp(-c_{1}\|x_{d}-p_{t}\|^{2}),
start_FLOATSUPERSCRIPT italic_c end_FLOATSUPERSCRIPT italic_r start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT = roman_exp ( - italic_c start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT ∥ italic_x start_POSTSUBSCRIPT italic_d end_POSTSUBSCRIPT - italic_p start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT ∥ start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT ) ,
(6)
where
x
d
∈
ℝ
3
subscript
𝑥
𝑑
superscript
ℝ
3
x_{d}\in\mathbb{R}^{3}
italic_x start_POSTSUBSCRIPT italic_d end_POSTSUBSCRIPT ∈ blackboard_R start_POSTSUPERSCRIPT 3 end_POSTSUPERSCRIPT
is the target position,
p
t
∈
ℝ
3
subscript
𝑝
𝑡
superscript
ℝ
3
p_{t}\in\mathbb{R}^{3}
italic_p start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT ∈ blackboard_R start_POSTSUPERSCRIPT 3 end_POSTSUPERSCRIPT
is the position of the robot base, and
c
1
subscript
𝑐
1
c_{1}
italic_c start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT
a hyperparameter.
Second, we want the robot to approach the checkpoint with a desired velocity
v
d
∈
ℝ
subscript
𝑣
𝑑
ℝ
v_{d}\in\mathbb{R}
italic_v start_POSTSUBSCRIPT italic_d end_POSTSUBSCRIPT ∈ blackboard_R
. This reward is written as:
r
t
v
=
exp
(
−
c
2
∥
v
d
−
g
⁢
(
p
t
)
−
g
⁢
(
p
t
−
1
)
Δ
⁢
t
∥
2
)
,
{}^{v}r_{t}=\exp\left(-c_{2}\left\lVert v_{d}-\frac{g(p_{t})-g(p_{t-1})}{%
\Delta t}\right\lVert^{2}\right),
start_FLOATSUPERSCRIPT italic_v end_FLOATSUPERSCRIPT italic_r start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT = roman_exp ( - italic_c start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT ∥ italic_v start_POSTSUBSCRIPT italic_d end_POSTSUBSCRIPT - divide start_ARG italic_g ( italic_p start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT ) - italic_g ( italic_p start_POSTSUBSCRIPT italic_t - 1 end_POSTSUBSCRIPT ) end_ARG start_ARG roman_Δ italic_t end_ARG ∥ start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT ) ,
(7)
where
g
⁢
(
p
)
=
‖
p
−
x
d
‖
𝑔
𝑝
norm
𝑝
subscript
𝑥
𝑑
g(p)=\|p-x_{d}\|
italic_g ( italic_p ) = ∥ italic_p - italic_x start_POSTSUBSCRIPT italic_d end_POSTSUBSCRIPT ∥
is a function that returns the scalar distance between the robot base and the target,
Δ
⁢
t
Δ
𝑡
\Delta t
roman_Δ italic_t
is the time step, and
c
2
subscript
𝑐
2
c_{2}
italic_c start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT
is a hyperparameter. This term encourages the agent to travel at a desired velocity
v
d
subscript
𝑣
𝑑
v_{d}
italic_v start_POSTSUBSCRIPT italic_d end_POSTSUBSCRIPT
on a line connecting the robot base to the target.
The last reward demands the agent to face the checkpoint, i.e., the robot base
x
𝑥
x
italic_x
-axis should point to the projection of the target in the horizontal plane:
r
t
f
=
min
⁡
(
0
,
f
⁢
(
p
t
)
x
⁢
y
⋅
i
x
⁢
y
ℐ
)
,
superscript
subscript
𝑟
𝑡
𝑓
0
⋅
𝑓
subscript
subscript
𝑝
𝑡
𝑥
𝑦
superscript
subscript
𝑖
𝑥
𝑦
ℐ
{}^{f}r_{t}=\min(0,f(p_{t})_{xy}\cdot{}^{\mathcal{I}}i_{xy}),
start_FLOATSUPERSCRIPT italic_f end_FLOATSUPERSCRIPT italic_r start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT = roman_min ( 0 , italic_f ( italic_p start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT ) start_POSTSUBSCRIPT italic_x italic_y end_POSTSUBSCRIPT ⋅ start_FLOATSUPERSCRIPT caligraphic_I end_FLOATSUPERSCRIPT italic_i start_POSTSUBSCRIPT italic_x italic_y end_POSTSUBSCRIPT ) ,
(8)
where
f
⁢
(
p
)
=
x
d
−
p
‖
x
d
−
p
‖
𝑓
𝑝
subscript
𝑥
𝑑
𝑝
norm
subscript
𝑥
𝑑
𝑝
f(p)=\frac{x_{d}-p}{\|x_{d}-p\|}
italic_f ( italic_p ) = divide start_ARG italic_x start_POSTSUBSCRIPT italic_d end_POSTSUBSCRIPT - italic_p end_ARG start_ARG ∥ italic_x start_POSTSUBSCRIPT italic_d end_POSTSUBSCRIPT - italic_p ∥ end_ARG
is a function that returns the unit vector pointing from the robot base to the target, the subscript
x
⁢
y
𝑥
𝑦
xy
italic_x italic_y
extracts the horizontal components, and the unit vector
i
ℐ
superscript
𝑖
ℐ
{}^{\mathcal{I}}i
start_FLOATSUPERSCRIPT caligraphic_I end_FLOATSUPERSCRIPT italic_i
is the robot
x
𝑥
x
italic_x
-axis in the inertial frame.
The reward components (
6
), (
7
), and (
8
) fully describe the task and become equal to
1
1
1
1
when the robot reaches the target.
Additionally, we want to minimize thrust usage. The thrust penalty can be considered a proxy term that minimizes propulsion expenditure and encourages the agent to use legged locomotion when possible. The thrust penalty is written as
r
t
T
=
−
∥
T
t
T
max
∥
2
,
{}^{\text{T}}r_{t}=-\left\lVert\frac{T_{t}}{T_{\text{max}}}\right\lVert^{2},
start_FLOATSUPERSCRIPT T end_FLOATSUPERSCRIPT italic_r start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT = - ∥ divide start_ARG italic_T start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT end_ARG start_ARG italic_T start_POSTSUBSCRIPT max end_POSTSUBSCRIPT end_ARG ∥ start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT ,
(9)
where
T
max
subscript
𝑇
max
T_{\text{max}}
italic_T start_POSTSUBSCRIPT max end_POSTSUBSCRIPT
is the maximum thrust each jet can exert while
T
t
∈
ℝ
m
subscript
𝑇
𝑡
superscript
ℝ
𝑚
T_{t}\in\mathbb{R}^{m}
italic_T start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT ∈ blackboard_R start_POSTSUPERSCRIPT italic_m end_POSTSUPERSCRIPT
is the thrust vector acting on the system.
Finally, the total task reward is written as
r
t
G
=
w
c
⁢
r
t
c
+
w
v
⁢
r
t
v
+
w
f
⁢
r
f
⁢
t
+
w
T
⁢
r
t
T
,
superscript
subscript
𝑟
𝑡
G
subscript
𝑤
𝑐
superscript
subscript
𝑟
𝑡
𝑐
subscript
𝑤
𝑣
superscript
subscript
𝑟
𝑡
𝑣
subscript
𝑤
𝑓
superscript
𝑟
𝑓
𝑡
subscript
𝑤
T
superscript
subscript
𝑟
𝑡
T
{}^{\text{G}}r_{t}=w_{c}{}^{c}r_{t}+w_{v}{}^{v}r_{t}+w_{f}{}^{f}rt+w_{\text{T}%
}{}^{\text{T}}r_{t},
start_FLOATSUPERSCRIPT G end_FLOATSUPERSCRIPT italic_r start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT = italic_w start_POSTSUBSCRIPT italic_c end_POSTSUBSCRIPT start_FLOATSUPERSCRIPT italic_c end_FLOATSUPERSCRIPT italic_r start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT + italic_w start_POSTSUBSCRIPT italic_v end_POSTSUBSCRIPT start_FLOATSUPERSCRIPT italic_v end_FLOATSUPERSCRIPT italic_r start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT + italic_w start_POSTSUBSCRIPT italic_f end_POSTSUBSCRIPT start_FLOATSUPERSCRIPT italic_f end_FLOATSUPERSCRIPT italic_r italic_t + italic_w start_POSTSUBSCRIPT T end_POSTSUBSCRIPT start_FLOATSUPERSCRIPT T end_FLOATSUPERSCRIPT italic_r start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT ,
(10)
where
w
c
subscript
𝑤
𝑐
w_{c}
italic_w start_POSTSUBSCRIPT italic_c end_POSTSUBSCRIPT
,
w
v
subscript
𝑤
𝑣
w_{v}
italic_w start_POSTSUBSCRIPT italic_v end_POSTSUBSCRIPT
,
w
f
subscript
𝑤
𝑓
w_{f}
italic_w start_POSTSUBSCRIPT italic_f end_POSTSUBSCRIPT
, and
w
T
subscript
𝑤
T
w_{\text{T}}
italic_w start_POSTSUBSCRIPT T end_POSTSUBSCRIPT
are hyperparameters that weight the contribution of each reward.
III-B
Model representation
Sec
II-A
recalls how the quantities describing a multi-body system affect the robot’s evolution. We build the action and the observation spaces accordingly, considering also the environment and task information.
III-B
1
Observation Space
The observation space consists of two components: one relates to the robot, and one distillates information about the task and the environment. We define the robot observation vector as
o
robot
=
[
ρ
⊤
v
ℬ
⊤
ℬ
ω
ℬ
⊤
ℬ
θ
⊤
θ
˙
⊤
p
EE
⊤
T
⊤
]
⊤
,
subscript
𝑜
robot
superscript
matrix
superscript
𝜌
top
superscript
superscript
subscript
𝑣
ℬ
top
ℬ
superscript
superscript
subscript
𝜔
ℬ
top
ℬ
superscript
𝜃
top
superscript
˙
𝜃
top
superscript
subscript
𝑝
EE
top
superscript
𝑇
top
top
o_{\text{robot}}=\begin{bmatrix}\rho^{\top}&{}^{\mathcal{B}}v_{\mathcal{B}}^{%
\top}&{}^{\mathcal{B}}\omega_{\mathcal{B}}^{\top}&\theta^{\top}&\dot{\theta}^{%
\top}&p_{\text{EE}}^{\top}&T^{\top}\end{bmatrix}^{\top},
italic_o start_POSTSUBSCRIPT robot end_POSTSUBSCRIPT = [ start_ARG start_ROW start_CELL italic_ρ start_POSTSUPERSCRIPT ⊤ end_POSTSUPERSCRIPT end_CELL start_CELL start_FLOATSUPERSCRIPT caligraphic_B end_FLOATSUPERSCRIPT italic_v start_POSTSUBSCRIPT caligraphic_B end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ⊤ end_POSTSUPERSCRIPT end_CELL start_CELL start_FLOATSUPERSCRIPT caligraphic_B end_FLOATSUPERSCRIPT italic_ω start_POSTSUBSCRIPT caligraphic_B end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ⊤ end_POSTSUPERSCRIPT end_CELL start_CELL italic_θ start_POSTSUPERSCRIPT ⊤ end_POSTSUPERSCRIPT end_CELL start_CELL over˙ start_ARG italic_θ end_ARG start_POSTSUPERSCRIPT ⊤ end_POSTSUPERSCRIPT end_CELL start_CELL italic_p start_POSTSUBSCRIPT EE end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ⊤ end_POSTSUPERSCRIPT end_CELL start_CELL italic_T start_POSTSUPERSCRIPT ⊤ end_POSTSUPERSCRIPT end_CELL end_ROW end_ARG ] start_POSTSUPERSCRIPT ⊤ end_POSTSUPERSCRIPT ,
(11)
where
ρ
𝜌
\rho
italic_ρ
is the quaternion representation of the rotation of the base
R
B
ℐ
superscript
subscript
𝑅
𝐵
ℐ
{}^{\mathcal{I}}R_{B}
start_FLOATSUPERSCRIPT caligraphic_I end_FLOATSUPERSCRIPT italic_R start_POSTSUBSCRIPT italic_B end_POSTSUBSCRIPT
,
v
ℬ
ℬ
superscript
subscript
𝑣
ℬ
ℬ
{}^{\mathcal{B}}v_{\mathcal{B}}
start_FLOATSUPERSCRIPT caligraphic_B end_FLOATSUPERSCRIPT italic_v start_POSTSUBSCRIPT caligraphic_B end_POSTSUBSCRIPT
and
ω
ℬ
ℬ
superscript
subscript
𝜔
ℬ
ℬ
{}^{\mathcal{B}}\omega_{\mathcal{B}}
start_FLOATSUPERSCRIPT caligraphic_B end_FLOATSUPERSCRIPT italic_ω start_POSTSUBSCRIPT caligraphic_B end_POSTSUBSCRIPT
are the base velocity in the base frame,
θ
𝜃
\theta
italic_θ
and
θ
˙
˙
𝜃
\dot{\theta}
over˙ start_ARG italic_θ end_ARG
are the joints position and velocity,
p
EE
subscript
𝑝
EE
p_{\text{EE}}
italic_p start_POSTSUBSCRIPT EE end_POSTSUBSCRIPT
is the vector containing the relative position of the end effectors, namely hands and feet, and
T
=
[
T
1
,
…
,
T
m
]
𝑇
subscript
𝑇
1
…
subscript
𝑇
𝑚
T=[T_{1},\dots,T_{m}]
italic_T = [ italic_T start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT , … , italic_T start_POSTSUBSCRIPT italic_m end_POSTSUBSCRIPT ]
is the thrust acting on the system as specified in Eq. (
16
).
The environment observation gives information about the robot’s interaction with the environment. We use an elevation map simulating a scan of the terrain around the robot
o
env
=
[
h
11
h
12
…
h
0
⁢
L
h
21
h
22
…
h
2
⁢
L
⋮
⋮
⋱
⋮
h
V
⁢
1
h
V
⁢
2
…
h
VL
]
.
subscript
𝑜
env
matrix
subscript
ℎ
11
subscript
ℎ
12
…
subscript
ℎ
0
L
subscript
ℎ
21
subscript
ℎ
22
…
subscript
ℎ
2
L
⋮
⋮
⋱
⋮
subscript
ℎ
V
1
subscript
ℎ
V
2
…
subscript
ℎ
VL
{o}_{\text{env}}=\begin{bmatrix}h_{11}&h_{12}&\dots&h_{0\text{L}}\\
h_{21}&h_{22}&\dots&h_{2\text{L}}\\
\vdots&\vdots&\ddots&\vdots\\
h_{\text{V}1}&h_{\text{V}2}&\dots&h_{\text{VL}}\end{bmatrix}.
italic_o start_POSTSUBSCRIPT env end_POSTSUBSCRIPT = [ start_ARG start_ROW start_CELL italic_h start_POSTSUBSCRIPT 11 end_POSTSUBSCRIPT end_CELL start_CELL italic_h start_POSTSUBSCRIPT 12 end_POSTSUBSCRIPT end_CELL start_CELL … end_CELL start_CELL italic_h start_POSTSUBSCRIPT 0 L end_POSTSUBSCRIPT end_CELL end_ROW start_ROW start_CELL italic_h start_POSTSUBSCRIPT 21 end_POSTSUBSCRIPT end_CELL start_CELL italic_h start_POSTSUBSCRIPT 22 end_POSTSUBSCRIPT end_CELL start_CELL … end_CELL start_CELL italic_h start_POSTSUBSCRIPT 2 L end_POSTSUBSCRIPT end_CELL end_ROW start_ROW start_CELL ⋮ end_CELL start_CELL ⋮ end_CELL start_CELL ⋱ end_CELL start_CELL ⋮ end_CELL end_ROW start_ROW start_CELL italic_h start_POSTSUBSCRIPT V 1 end_POSTSUBSCRIPT end_CELL start_CELL italic_h start_POSTSUBSCRIPT V 2 end_POSTSUBSCRIPT end_CELL start_CELL … end_CELL start_CELL italic_h start_POSTSUBSCRIPT VL end_POSTSUBSCRIPT end_CELL end_ROW end_ARG ] .
(12)
The elevation map is centered at the robot base and moves with the robot. The grid is discretized into
V
×
L
V
L
\text{V}\times\text{L}
V × L
cells.
h
i
⁢
j
∈
ℝ
subscript
ℎ
𝑖
𝑗
ℝ
h_{ij}\in\mathbb{R}
italic_h start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT ∈ blackboard_R
is the height of the
i
𝑖
i
italic_i
-th row and
j
𝑗
j
italic_j
-th column of the grid map with respect to the robot base.
The task observation vector gives the agent information about the goal. We use the position of the target
x
d
subscript
𝑥
𝑑
x_{d}
italic_x start_POSTSUBSCRIPT italic_d end_POSTSUBSCRIPT
with respect to the robot base
p
𝑝
p
italic_p
o
task
=
p
d
ℬ
=
R
ℬ
⊤
ℐ
⁢
(
x
d
−
p
)
.
subscript
𝑜
task
superscript
subscript
𝑝
𝑑
ℬ
superscript
superscript
subscript
𝑅
ℬ
top
ℐ
subscript
𝑥
𝑑
𝑝
o_{\text{task}}={}^{\mathcal{B}}p_{d}={}^{\mathcal{I}}R_{\mathcal{B}}^{\top}(x%
_{d}-p).
italic_o start_POSTSUBSCRIPT task end_POSTSUBSCRIPT = start_FLOATSUPERSCRIPT caligraphic_B end_FLOATSUPERSCRIPT italic_p start_POSTSUBSCRIPT italic_d end_POSTSUBSCRIPT = start_FLOATSUPERSCRIPT caligraphic_I end_FLOATSUPERSCRIPT italic_R start_POSTSUBSCRIPT caligraphic_B end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ⊤ end_POSTSUPERSCRIPT ( italic_x start_POSTSUBSCRIPT italic_d end_POSTSUBSCRIPT - italic_p ) .
(13)
The observation space is the concatenation of the robot, environment, and task observations
o
=
[
o
robot
⊤
o
¯
env
⊤
o
task
⊤
]
⊤
,
𝑜
superscript
matrix
superscript
subscript
𝑜
robot
top
superscript
subscript
¯
𝑜
env
top
superscript
subscript
𝑜
task
top
top
o=\begin{bmatrix}o_{\text{robot}}^{\top}&\bar{o}_{\text{env}}^{\top}&o_{\text{%
task}}^{\top}\end{bmatrix}^{\top},
italic_o = [ start_ARG start_ROW start_CELL italic_o start_POSTSUBSCRIPT robot end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ⊤ end_POSTSUPERSCRIPT end_CELL start_CELL over¯ start_ARG italic_o end_ARG start_POSTSUBSCRIPT env end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ⊤ end_POSTSUPERSCRIPT end_CELL start_CELL italic_o start_POSTSUBSCRIPT task end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ⊤ end_POSTSUPERSCRIPT end_CELL end_ROW end_ARG ] start_POSTSUPERSCRIPT ⊤ end_POSTSUPERSCRIPT ,
(14)
where
o
¯
env
subscript
¯
𝑜
env
\bar{o}_{\text{env}}
over¯ start_ARG italic_o end_ARG start_POSTSUBSCRIPT env end_POSTSUBSCRIPT
is the flattened environment observation matrix.
III-B
2
Action Space
The action space
𝒜
𝒜
\mathcal{A}
caligraphic_A
is composed of the desired joint positions of the robot
θ
d
∈
ℝ
n
subscript
𝜃
𝑑
superscript
ℝ
𝑛
\theta_{d}\in\mathbb{R}^{n}
italic_θ start_POSTSUBSCRIPT italic_d end_POSTSUBSCRIPT ∈ blackboard_R start_POSTSUPERSCRIPT italic_n end_POSTSUPERSCRIPT
and the desired thrust dynamics input
u
=
[
u
1
,
…
,
u
m
]
⊤
∈
ℝ
m
𝑢
superscript
subscript
𝑢
1
…
subscript
𝑢
𝑚
top
superscript
ℝ
𝑚
u=[u_{1},\dots,u_{m}]^{\top}\in\mathbb{R}^{m}
italic_u = [ italic_u start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT , … , italic_u start_POSTSUBSCRIPT italic_m end_POSTSUBSCRIPT ] start_POSTSUPERSCRIPT ⊤ end_POSTSUPERSCRIPT ∈ blackboard_R start_POSTSUPERSCRIPT italic_m end_POSTSUPERSCRIPT
a
=
[
θ
d
⊤
u
⊤
]
⊤
,
𝑎
superscript
matrix
superscript
subscript
𝜃
𝑑
top
superscript
𝑢
top
top
a=\begin{bmatrix}\theta_{d}^{\top}&u^{\top}\end{bmatrix}^{\top},
italic_a = [ start_ARG start_ROW start_CELL italic_θ start_POSTSUBSCRIPT italic_d end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ⊤ end_POSTSUPERSCRIPT end_CELL start_CELL italic_u start_POSTSUPERSCRIPT ⊤ end_POSTSUPERSCRIPT end_CELL end_ROW end_ARG ] start_POSTSUPERSCRIPT ⊤ end_POSTSUPERSCRIPT ,
(15)
where
n
𝑛
n
italic_n
is the number of joints of the robot and
m
𝑚
m
italic_m
is the number of links on which the thrust acts. The desired joint positions
θ
d
subscript
𝜃
𝑑
\theta_{d}
italic_θ start_POSTSUBSCRIPT italic_d end_POSTSUBSCRIPT
are fed to a PD controller. The resulting thrust intensities
T
𝑇
T
italic_T
are computed as the increment of the thrust intensity with respect to the previous time step
k
−
1
𝑘
1
k-1
italic_k - 1
as
T
=
T
⁢
[
k
−
1
]
+
g
⁢
(
T
⁢
[
k
−
1
]
,
u
)
⁢
Δ
⁢
t
𝑇
𝑇
delimited-[]
𝑘
1
𝑔
𝑇
delimited-[]
𝑘
1
𝑢
Δ
𝑡
{T=T[k-1]+g(T[k-1],u)\Delta t}
italic_T = italic_T [ italic_k - 1 ] + italic_g ( italic_T [ italic_k - 1 ] , italic_u ) roman_Δ italic_t
and act on the specified links, generating a pure force
f
i
=
R
i
ℐ
⁢
(
q
)
⁢
[
0
0
T
i
]
,
∀
i
∈
[
1
,
m
]
,
formulae-sequence
subscript
𝑓
𝑖
superscript
subscript
𝑅
𝑖
ℐ
𝑞
matrix
0
0
subscript
𝑇
𝑖
for-all
𝑖
1
𝑚
f_{i}={}^{\mathcal{I}}R_{i}(q)\begin{bmatrix}0\\
0\\
T_{i}\end{bmatrix},\forall i\in[1,m],
italic_f start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT = start_FLOATSUPERSCRIPT caligraphic_I end_FLOATSUPERSCRIPT italic_R start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ( italic_q ) [ start_ARG start_ROW start_CELL 0 end_CELL end_ROW start_ROW start_CELL 0 end_CELL end_ROW start_ROW start_CELL italic_T start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT end_CELL end_ROW end_ARG ] , ∀ italic_i ∈ [ 1 , italic_m ] ,
(16)
where
R
i
ℐ
⁢
(
q
)
superscript
subscript
𝑅
𝑖
ℐ
𝑞
{}^{\mathcal{I}}R_{i}(q)
start_FLOATSUPERSCRIPT caligraphic_I end_FLOATSUPERSCRIPT italic_R start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ( italic_q )
represents the orientation of the frame
i
𝑖
i
italic_i
on which the force
f
i
subscript
𝑓
𝑖
f_{i}
italic_f start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT
is applied and
Δ
⁢
t
Δ
𝑡
\Delta t
roman_Δ italic_t
is the time step between two consecutive actions.
g
⁢
(
T
⁢
[
k
−
1
]
,
u
)
𝑔
𝑇
delimited-[]
𝑘
1
𝑢
g(T[k-1],u)
italic_g ( italic_T [ italic_k - 1 ] , italic_u )
is a function that computes the thrust dynamics given the previous thrust intensities
T
⁢
[
k
−
1
]
𝑇
delimited-[]
𝑘
1
T[k-1]
italic_T [ italic_k - 1 ]
and the input
u
𝑢
u
italic_u
, which in the
ideal
case is considered to be
g
⁢
(
T
⁢
[
k
−
1
]
,
u
)
=
T
˙
𝑔
𝑇
delimited-[]
𝑘
1
𝑢
˙
𝑇
g(T[k-1],u)=\dot{T}
italic_g ( italic_T [ italic_k - 1 ] , italic_u ) = over˙ start_ARG italic_T end_ARG
, i.e., thrust intensity rate-of-change, while in the
specific
use case of the jet-powered actuation, it is a function identified from real-world data, see Sec.
IV-G
for more details.
III-C
Discriminator Observation Space
The discriminator
D
𝐷
D
italic_D
gives feedback about the quality of the motion produced by the agent. Selecting a meaningful set of features to feed into the discriminator is crucial since it should be able to capture the robot’s motion. We choose the discriminator observation space to be the same as the robot observation vector
χ
=
o
robot
𝜒
subscript
𝑜
robot
\chi=o_{\text{robot}}
italic_χ = italic_o start_POSTSUBSCRIPT robot end_POSTSUBSCRIPT
. The discriminator is fed with couples of consecutive observations
χ
t
,
χ
t
+
1
subscript
𝜒
𝑡
subscript
𝜒
𝑡
1
\chi_{t},\chi_{t+1}
italic_χ start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT , italic_χ start_POSTSUBSCRIPT italic_t + 1 end_POSTSUBSCRIPT
and trained with batches of samples from the motion dataset
𝒟
𝒟
\mathcal{D}
caligraphic_D
and samples produced by the policy
π
σ
subscript
𝜋
𝜎
\pi_{\sigma}
italic_π start_POSTSUBSCRIPT italic_σ end_POSTSUBSCRIPT
.
IV
Results
In this section, we present the simulated results of the proposed approach.
We designed our experiments to answer the following questions: (i) Can smooth multimodal locomotion be achieved with an AMP-based method? (ii) What type of information does the policy need to guide the emergence of the desired locomotion pattern? (iii) Can the same result be obtained using a classic RL approach?
Our testbench is iRonCub, a flying humanoid robot that expresses a degree of terrestrial and aerial locomotion. iRonCub has 23 joints, weights
44
⁢
kg
44
kg
44~{}$\mathrm{k}\mathrm{g}$
44 roman_kg
, and is equipped with four jet engines,
two fixed on the chest and two moved by the arms, which in the specific use case are the commercial JetCat P250 engines
[
40
]
, capable of exerting
T
m
⁢
a
⁢
x
=
250
⁢
N
subscript
𝑇
𝑚
𝑎
𝑥
250
N
T_{max}=250~{}$\mathrm{N}$
italic_T start_POSTSUBSCRIPT italic_m italic_a italic_x end_POSTSUBSCRIPT = 250 roman_N
of thrust each.
The training consists of two main tasks. The first task is a multimodal locomotion scenario on flat ground, in which the robot has to catch several waypoints located on the ground and in the air. The second scenario consists of a world containing different terrains, in which the robot has to understand which locomotion modality is more suited. While the scenarios above are tested in the
ideal
thrust case, we also validate the approach in the
specific
use case of jet-powered actuation, in which the model of the jet engines is identified from real-world data with a neural network and embedded in the simulation environment, see Sec.
IV-G
.
IV-A
Motion Priors
(a)
(b)
(c)
Figure 3:
Snapshots of the flying motion obtained using TO.
(a)
(b)
(c)
Figure 4:
Snapshots of the walking motion obtained using inverse kinematics from the CMU human walking dataset.
The used motion data comes from two sources. (i) Walking motions are produced from recorded mocap clip
[
41
,
26
]
and retargeted to the iRonCub model using inverse kinematics implemented using
iDynTree
[
42
]
(Fig.
4
). (ii) Trajectory optimization for aerial humanoid robots produces flying motions containing the full state
[
43
]
. These datasets consist of motions described in terms of the system’s kinematics, i.e., joint positions, base poses, and thrust intensities (Fig.
4
)
In the case of walking motions, the thrust is set to zero.
If the original demonstrator is different from the agent, the motion might not be feasible. In this case, the agent will produce a motion similar to the original one but feasible for the agent.
TABLE I:
PPO parameters.
Parameter
Value
Discount rate
γ
𝛾
\gamma
italic_γ
0.99
Learning rate
5e-5
GAE parameter
λ
𝜆
\lambda
italic_λ
0.95
Entropy coefficient
0.0
Clip parameter
0.2
Mini-batch size
32768
Critic loss coefficient
5
KL threshold
0.008
Number of actors
4096
IV-B
Experimental Setup
The training environment is developed using the IsaacGym simulator, which allows massive parallel training. We trained 4096 agents using PPO, controlled at
60
⁢
hz
60
hz
60~{}$\mathrm{h}\mathrm{z}$
60 roman_hz
. The training requires
∼
2
similar-to
absent
2
\sim 2
∼ 2
hours on an NVIDIA Quadro RTX 6000. The training parameters are shown in Tab.
I
. Fig.
5
shows the learning curves of average reward and episode duration over ten training runs for the test presented in Sec.
IV-D
. The curves show a small variance, demonstrating the stability of the training. The reward coefficients are shown in Tab.
II
.
Figure 5:
Training curves over 10 runs on the scenario from Sec.
IV-D
.
IV-C
Traversing flat ground
The first scenario involves the robot walking, take-off, flying, and landing on flat ground at a desired velocity
v
d
subscript
𝑣
𝑑
v_{d}
italic_v start_POSTSUBSCRIPT italic_d end_POSTSUBSCRIPT
of
0.8
m
/
s
times
0.8
m
s
0.8\text{\,}\mathrm{m}\mathrm{/}\mathrm{s}
start_ARG 0.8 end_ARG start_ARG times end_ARG start_ARG roman_m / roman_s end_ARG
. The robot must reach several waypoints set in an interval between
0.7
m
times
0.7
m
0.7\text{\,}\mathrm{m}
start_ARG 0.7 end_ARG start_ARG times end_ARG start_ARG roman_m end_ARG
and
2.0
m
times
2.0
m
2.0\text{\,}\mathrm{m}
start_ARG 2.0 end_ARG start_ARG times end_ARG start_ARG roman_m end_ARG
. Once a waypoint is hit, a new one appears forward, and the robot moves toward it, as shown in Fig.
6
. In this case, the height map
is reduced to a single value, i.e., the height of the robot base w.r.t. the terrain. The test shows that the robot effectively learns terrestrial and aerial locomotion and can choose when to switch between them. Each episode is terminated when the distance of the robot base from the ground is less than
0.4
⁢
m
0.4
m
0.4~{}$\mathrm{m}$
0.4 roman_m
or the maximum number of steps is reached.
Figure 6:
Walk-to-fly maneuver. The robot lands and walks to catch a waypoint on the ground level, takes off, and flies to hit the aerial target.
Figure 7:
The robot walks over any reachable terrain. The height map enables
terrain-aware
multimodal locomotion.
IV-D
Terrain-aware locomotion
In the second scenario, the robot deals with diverse terrains, i.e., flat and rough terrain, stepping stones, and pits at a desired velocity
v
d
subscript
𝑣
𝑑
v_{d}
italic_v start_POSTSUBSCRIPT italic_d end_POSTSUBSCRIPT
of
0.8
m
/
s
times
0.8
m
s
0.8\text{\,}\mathrm{m}\mathrm{/}\mathrm{s}
start_ARG 0.8 end_ARG start_ARG times end_ARG start_ARG roman_m / roman_s end_ARG
. As in the previous scenario, the route comprises waypoints the robot needs to catch. The policy is fed with a height map of
7
×
9
7
9
7\times 9
7 × 9
cells, where each cell is
0.3
m
times
0.3
m
0.3\text{\,}\mathrm{m}
start_ARG 0.3 end_ARG start_ARG times end_ARG start_ARG roman_m end_ARG
wide.
Using the height map as observation, the robot learns to use small portions of reachable terrain to step on, walk when possible, and fly when the ground is not reachable, as shown in Fig.
7
.
The termination strategy is the same as the previous case: when the base height is smaller than
0.4
⁢
m
0.4
m
0.4~{}$\mathrm{m}$
0.4 roman_m
, the episode is over.
Tuning the thrust penalization is not trivial. If the penalty is too high, the robot prefers to stay at the edge of the terrain rather than fly and reach the following reachable terrain since using the thrust leads to a higher penalty than the reward. If the penalty is too low, the robot prefers to fly rather than walk when the terrain is sparser. The penalty should be high enough to make the robot prefer walking over flying but not too high to make the robot choose to walk when flying is needed.
TABLE II:
Reward coefficients.
Reward term
Value
Target reward weight
w
c
subscript
𝑤
𝑐
w_{c}
italic_w start_POSTSUBSCRIPT italic_c end_POSTSUBSCRIPT
0.1
Velocity reward weight
w
v
subscript
𝑤
𝑣
w_{v}
italic_w start_POSTSUBSCRIPT italic_v end_POSTSUBSCRIPT
0.7
Facing reward weight
w
f
subscript
𝑤
𝑓
w_{f}
italic_w start_POSTSUBSCRIPT italic_f end_POSTSUBSCRIPT
0.2
Thrust penalty weight
w
T
subscript
𝑤
T
w_{\text{T}}
italic_w start_POSTSUBSCRIPT T end_POSTSUBSCRIPT
0.11
Target reward hyperparameter
c
1
subscript
𝑐
1
c_{1}
italic_c start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT
0.5
Velocity reward hyperparameter
c
2
subscript
𝑐
2
c_{2}
italic_c start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT
0.5
IV-E
Ablation Studies
We perform ablation studies to show the influence of the motion priors. We trained the policies using the same parameters but without the motion priors or part of them. We keep the thrust penalization constant and use the scenario described in Sec.
IV-C
as a testbench. Fig.
8
shows episode duration over the maximum duration, reward
r
𝑟
r
italic_r
collected over a maximum observed reward
r
¯
¯
𝑟
\bar{r}
over¯ start_ARG italic_r end_ARG
, and thrust usage as
1
−
T
/
T
m
⁢
a
⁢
x
1
𝑇
subscript
𝑇
𝑚
𝑎
𝑥
1-T/T_{max}
1 - italic_T / italic_T start_POSTSUBSCRIPT italic_m italic_a italic_x end_POSTSUBSCRIPT
.
Figure 8:
The radar chart shows the average quantities in percentage. The policy trained with only walk motion prior shows less usage of the thrust but collects less reward. The other three policies collect higher rewards, with the policy trained with both motion priors – in yellow – being efficient in thrust usage and covering a larger area.
IV-E
1
No Motion Prior
The policy trained without any motion prior does not show any particular behavior. The reward collected is similar to the policy trained with both motion priors, but the thrust usage is higher.
IV-E
2
Only Walking Motion Prior
The policy trained without aerial motion priors learns how to walk, but it does not show any flying behavior, preventing the robot from reaching the waypoints in the air and collecting rewards. The usage of the thrust is lower than the other policies.
IV-E
3
Only Flying Motion Prior
The policy trained without terrestrial motion priors does not learn to walk and fly to reach all the waypoints, collecting a reward similar to the other policies. Despite the thrust usage being higher than the policy trained with both motion priors, it is lower than the policy trained without any motion prior
IV-E
1
, suggesting that the aerial motion prior embeds an optimized thrust usage coming from the trajectory optimization solution.
IV-E
4
Walking + Flying Motion Prior
The policy trained with both motion priors learns how to walk and fly. The collected reward is similar to other policies, but the thrust usage is lower, indicating that switching between walking and flying is also more efficient in propulsion expenditure.
Figure 9:
The plot shows the identification results obtained on the JetCat P250. The orange line is the thrust measured from a load cell in a custom test bench for jet data collection. The red line is the thrust obtained by integrating the identified LSTM model.
IV-F
Comparison with Trajectory Optimization
The trajectory optimization method proposed in
[
43
]
is formulated as a multiple-shooting optimal control problem using the centroidal dynamics over an optimized horizon of 100 nodes that requires minimization of thrust usage. The contact sequence is computed using a complementarity condition formulation, and the problem is solved using the IPOPT solver
[
44
]
.
The TO method is able to produce a transition from legged to aerial locomotion but requires a complex cost function and a long computation time preventing replanning capabilities. A trajectory optimization approach would need the concatenation of several offline trajectories computed for each possible scenario, stabilized afterward by an online controller, while the proposed method can adapt to the environment online. Conversely, the policy can learn how to switch between walking and flying without the need for a complex reward function while inferring the terrain information from the elevation map, which is infeasible with the compared trajectory optimization method or classical control method, needing a separate module that deals with the terrain information. Table
IV
compares the two approaches.
IV-G
Use case: Jet-powered actuation
This section presents the results and considerations in the use case of jet-powered actuation. The jet engines are modeled using a Long Short-Term Memory (LSTM) neural network
[
45
]
, which is trained to predict the thrust given the input and the state of the jet engine. The model is trained using the data collected from the JetCat P250, a commercial jet engine capable of exerting a thrust of
T
m
⁢
a
⁢
x
=
250
⁢
N
subscript
𝑇
𝑚
𝑎
𝑥
250
N
T_{max}=250~{}$\mathrm{N}$
italic_T start_POSTSUBSCRIPT italic_m italic_a italic_x end_POSTSUBSCRIPT = 250 roman_N
[
40
]
. The data comprises thrust measured with a load cell at different throttle inputs in an ad-hoc designed test bench
[
46
]
.
The resulting discrete jet dynamics model is
T
=
T
⁢
[
k
−
1
]
+
g
⁢
(
T
⁢
[
k
−
1
]
,
u
)
⁢
Δ
⁢
t
,
𝑇
𝑇
delimited-[]
𝑘
1
𝑔
𝑇
delimited-[]
𝑘
1
𝑢
Δ
𝑡
T=T[k-1]+g(T[k-1],u)\Delta t,
italic_T = italic_T [ italic_k - 1 ] + italic_g ( italic_T [ italic_k - 1 ] , italic_u ) roman_Δ italic_t ,
(17)
where
T
𝑇
T
italic_T
and
T
⁢
[
k
−
1
]
𝑇
delimited-[]
𝑘
1
T[k-1]
italic_T [ italic_k - 1 ]
are the actual and the previous thrust,
u
𝑢
u
italic_u
is the throttle input,
g
𝑔
g
italic_g
is the LSTM model and
Δ
⁢
t
Δ
𝑡
\Delta t
roman_Δ italic_t
is the time step.
The model is validated by comparing the thrust obtained by integrating the model against the thrust measured from the load cell, as shown in Fig.
9
, with performance in Table
III
.
TABLE III:
Identification errors.
Error
Value
Mean Absolute Error
4.995
N
N
\mathrm{N}
roman_N
Root Mean Squared Error
8.061
N
N
\mathrm{N}
roman_N
The model is then embedded in the simulation environment, and the policy is trained as in scenarios described in Sec.
IV-D
. The policy outputs the throttle input
u
𝑢
u
italic_u
, which is then passed to the LSTM model (
17
) along with the previous thrust
T
⁢
[
k
−
1
]
𝑇
delimited-[]
𝑘
1
T[k-1]
italic_T [ italic_k - 1 ]
to obtain the thrust
T
𝑇
T
italic_T
.
The response dynamics of the jet propulsion system exhibit a comparatively slower response rate when compared to the ideal thrust case: attaining the take-off thrust requires a longer time.
For this reason, the policy is trained by reducing the thrust penalization to
1
⁢
e
−
8
1
𝑒
8
1e-8
1 italic_e - 8
. Furthermore, the minimum throttle input is set to
15
%
percent
15
15\%
15 %
, which leads to a minimum thrust of
∼
40
⁢
N
similar-to
absent
40
N
\sim 40~{}$\mathrm{N}$
∼ 40 roman_N
. The policy can learn how to transition between walking and flying, although the transition is slower than the ideal thrust case, e.g., when traversing the stepping stones, the robot employs the aerial locomotion mode.
TABLE IV:
Comparison between the policy and the TO method.
Feature
Method
Our
TO
Computation time
≃
2
similar-to-or-equals
absent
2
~{}\simeq 2
≃ 2
hours
≃
30
similar-to-or-equals
absent
30
\simeq 30
≃ 30
min
Length
Episode length
7 seconds
Cost function
4 terms
12 terms
Terrain-aware
✓
✗
Online
✓
✗
Automatic Transition
✓
✓
V
Conclusions
Our paper presents a method that enables aerial humanoid robots to seamlessly transition between walking and flying modes.
The proposed strategy leverages the concept of Adversarial Motion Priors to learn a natural gait pattern from human-like gaits and an efficient aerial locomotion pattern from motions obtained using trajectory optimization.
The robot can traverse complex terrains, switching automatically between both locomotion forms, without an explicit constraint on the form of navigation. Although our method has been tested in simulation environments and necessitates further investigations of its reliability in real-world domains, this result marks progress towards the application of aerial humanoid robots in various fields, such as search and rescue and monitoring missions, where diverse and demanding scenarios are encountered.
Looking forward, the integration of more motion priors might broaden the potential of aerial humanoid robotics in future challenges.
References
[1]
R. Lock, S. Burgess, and R. Vaidyanathan, “Multi-modal locomotion: from animal
to application,”
Bioinspiration & biomimetics
, vol. 9, no. 1, 2013.
[2]
L. Daler, S. Mintchev, C. Stefanini, and D. Floreano, “A bioinspired
multi-modal flying and walking robot,”
Bioinspiration & biomimetics
,
vol. 10, no. 1, 2015.
[3]
K. Kim, P. Spieler, E.-S. Lupu, A. Ramezani, and S.-J. Chung, “A bipedal
walking robot that can fly, slackline, and skateboard,”
Science
Robotics
, vol. 6, no. 59, 2021.
[4]
E. Sihite, A. Kalantari, R. Nemovi, A. Ramezani, and M. Gharib, “Multi-modal
mobility morphobot (m4) with appendage repurposing for locomotion plasticity
enhancement,”
Nature communications
, vol. 14, no. 1, 2023.
[5]
M. Bjelonic, C. D. Bellicoso, Y. de Viragh, D. Sako, F. D. Tresoldi,
F. Jenelten, and M. Hutter, “Keep rollin’—whole-body motion control and
planning for wheeled quadrupedal robots,”
IEEE Robotics and Automation
Letters
, vol. 4, no. 2, 2019.
[6]
Q. Yu and N. Gravish, “Multimodal locomotion in a soft robot through
hierarchical actuation,”
Soft Robotics
, 2023.
[7]
J. Yu, M. Wang, W. Wang, M. Tan, and J. Zhang, “Design and control of a
fish-inspired multimodal swimming robot,” in
2011 IEEE International
Conference on Robotics and Automation
.   IEEE, 2011.
[8]
D. Pucci, S. Traversaro, and F. Nori, “Momentum control of an underactuated
flying humanoid robot,”
IEEE Robotics and Automation Letters
,
vol. 3, no. 1, 2017.
[9]
G. Nava, L. Fiorio, S. Traversaro, and D. Pucci, “Position and Attitude
Control of an Underactuated Flying Humanoid Robot,” in
2018 IEEE-RAS
18th International Conference on Humanoid Robots (Humanoids)
, 2018.
[10]
G. Romualdi, S. Dafarra, Y. Hu, P. Ramadoss, F. J. A. Chavez, S. Traversaro,
and D. Pucci, “A benchmarking of dcm-based architectures for position,
velocity and torque-controlled humanoid robots,”
International Journal
of Humanoid Robotics
, vol. 17, no. 01, 2020.
[11]
M. Kelly, “An introduction to trajectory optimization: How to do your own
direct collocation,”
SIAM Review
, vol. 59, no. 4, 2017. [Online].
Available:
https://epubs.siam.org/page/terms
[12]
S. Dafarra, G. Romualdi, and D. Pucci, “Dynamic complementarity conditions and
whole-body trajectory optimization for humanoid robot locomotion,”
IEEE Transactions on Robotics
, vol. 38, no. 6, 2022.
[13]
S. Kuindersma, R. Deits, M. Fallon, A. Valenzuela, H. Dai, F. Permenter,
T. Koolen, P. Marion, and R. Tedrake, “Optimization-based locomotion
planning, estimation, and control design for the atlas humanoid robot,”
Autonomous robots
, vol. 40, 2016.
[14]
S. Kajita, F. Kanehiro, K. Kaneko, K. Fujiwara, K. Harada, K. Yokoi, and
H. Hirukawa, “Biped walking pattern generation by using preview control of
zero-moment point,” in
2003 IEEE International Conference on Robotics
and Automation
, vol. 2.   IEEE, 2003.
[15]
J. Englsberger, C. Ott, and A. Albu-Schäffer, “Three-dimensional bipedal
walking control using divergent component of motion,” in
2013 IEEE/RSJ
International Conference on Intelligent Robots and Systems
.   IEEE, 2013.
[16]
D. E. Orin, A. Goswami, and S.-H. Lee, “Centroidal dynamics of a humanoid
robot,”
Autonomous robots
, vol. 35, no. 2.
[17]
H. Dai, A. Valenzuela, and R. Tedrake, “Whole-body motion planning with
centroidal dynamics and full kinematics,” in
2014 IEEE-RAS
International Conference on Humanoid Robots
, 2014.
[18]
B. Ponton, A. Herzog, A. Del Prete, S. Schaal, and L. Righetti, “On Time
Optimization of Centroidal Momentum Dynamics,” in
Proceedings - IEEE
International Conference on Robotics and Automation
.   Institute of Electrical and Electronics Engineers Inc., 9
2018. [Online]. Available:
https://git-amd.tuebingen.mpg.de/bponton/timeoptimization.https://arxiv.org/abs/1709.09265v3
[19]
G. Romualdi, S. Dafarra, G. L’Erario, I. Sorrentino, S. Traversaro, and
D. Pucci, “Online non-linear centroidal mpc for humanoid robot locomotion
with step adjustment,” in
2022 International Conference on Robotics
and Automation (ICRA)
, 2022.
[20]
R. S. Sutton and A. G. Barto,
Reinforcement learning: An
introduction
.   MIT press, 2018.
[21]
V. Tsounis, M. Alge, J. Lee, F. Farshidian, and M. Hutter, “Deepgait: Planning
and control of quadrupedal gaits using deep reinforcement learning,”
IEEE Robotics and Automation Letters
, vol. 5, no. 2, 2020.
[22]
Y. Song, A. Romero, M. Mueller, V. Koltun, and D. Scaramuzza, “Reaching the
limit in autonomous racing: Optimal control versus reinforcement learning,”
Science Robotics
, 2023.
[23]
D. Rodriguez and S. Behnke, “Deepwalk: Omnidirectional bipedal gait by deep
reinforcement learning,” in
2021 IEEE international conference on
robotics and automation (ICRA)
.   IEEE,
2021.
[24]
Z. Li, X. Cheng, X. B. Peng, P. Abbeel, S. Levine, G. Berseth, and K. Sreenath,
“Reinforcement learning for robust parameterized locomotion control of
bipedal robots,” in
2021 IEEE International Conference on Robotics and
Automation (ICRA)
.   IEEE, 2021.
[25]
E. Kaufmann, L. Bauersfeld, A. Loquiercio, M. Müller, V. Koltun, and
D. Scaramuzza, “Champion-level drone-racing with deep reinforcement
learning,”
Nature
, 2023.
[26]
X. B. Peng, P. Abbeel, S. Levine, and M. Van de Panne, “Deepmimic:
Example-guided deep reinforcement learning of physics-based character
skills,”
ACM Transactions On Graphics (TOG)
, vol. 37, no. 4, 2018.
[27]
X. B. Peng, Z. Ma, P. Abbeel, S. Levine, and A. Kanazawa, “Amp: Adversarial
motion priors for stylized physics-based character control,”
ACM
Transactions on Graphics (TOG)
, vol. 40, no. 4, 2021.
[28]
X. B. Peng, Y. Guo, L. Halper, S. Levine, and S. Fidler, “Ase: Large-scale
reusable adversarial skill embeddings for physically simulated characters,”
ACM Transactions On Graphics (TOG)
, vol. 41, no. 4, 2022.
[29]
S. Starke, Y. Zhao, T. Komura, and K. Zaman, “Local motion phases for learning
multi-contact character movements,”
ACM Transactions on Graphics
(TOG)
, vol. 39, no. 4, 2020.
[30]
S. Starke, I. Mason, and T. Komura, “Deepphase: Periodic autoencoders for
learning motion phase manifolds,”
ACM Transactions on Graphics (TOG)
,
vol. 41, no. 4, 2022.
[31]
A. Escontrela, X. B. Peng, W. Yu, T. Zhang, A. Iscen, K. Goldberg, and
P. Abbeel, “Adversarial motion priors make good substitutes for complex
reward functions,” in
2022 IEEE/RSJ International Conference on
Intelligent Robots and Systems (IROS)
.   IEEE, 2022.
[32]
X. B. Peng, E. Coumans, T. Zhang, T.-W. E. Lee, J. Tan, and S. Levine,
“Learning agile robotic locomotion skills by imitating animals,” in
Robotics: Science and Systems
, 07 2020.
[33]
P. M. Viceconte, R. Camoriano, G. Romualdi, D. Ferigo, S. Dafarra,
S. Traversaro, G. Oriolo, L. Rosasco, and D. Pucci, “Adherent: Learning
human-like trajectory generators for whole-body control of humanoid robots,”
IEEE Robotics and Automation Letters
, vol. 7, no. 2, 2022.
[34]
J. Ho and S. Ermon, “Generative adversarial imitation learning,”
Advances in neural information processing systems
, vol. 29, 2016.
[35]
V. Makoviychuk,
et al.
, “Isaac gym: High performance gpu-based physics
simulation for robot learning,” 2021.
[36]
N. Rudin, D. Hoeller, P. Reist, and M. Hutter, “Learning to walk in minutes
using massively parallel deep reinforcement learning,” in
Conference
on Robot Learning
.   PMLR, 2022.
[37]
S. Traversaro, “Modelling, estimation and identification of humanoid robots
dynamics,” Ph.D. dissertation, Italian Institute of Technology Genoa, Italy,
2017.
[38]
I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair,
A. Courville, and Y. Bengio, “Generative adversarial nets,”
Advances
in neural information processing systems
, vol. 27, 2014.
[39]
L. Mescheder, A. Geiger, and S. Nowozin, “Which training methods for gans do
actually converge?” in
International conference on machine
learning
.   PMLR, 2018.
[40]
“JetCatP250.” [Online]. Available:
www.jetcat.de/en/productdetails/produkte/jetcat/produkte/Professionell/p250%20pro%20s
[41]
C. M. University, “Cmu graphics lab motion capture database,”
http://mocap.cs.cmu.edu/
, accessed: 2023-06-07.
[42]
F. Nori, S. Traversaro, J. Eljaik, F. Romano, A. Del Prete, and D. Pucci,
“icub whole-body control through force regulation on rigid noncoplanar
contacts,”
Frontiers in Robotics and AI
, vol. 2, no. 6, 2015.
[Online]. Available:
http://www.frontiersin.org/humanoid˙robotics/10.3389/frobt.2015.00006/abstract
[43]
G. L’Erario, G. Nava, G. Romualdi, F. Bergonti, V. Razza, S. Dafarra, and
D. Pucci, “Whole-body trajectory optimization for robot multimodal
locomotion,” in
2022 IEEE-RAS 21st International Conference on
Humanoid Robots (Humanoids)
, 2022.
[44]
A. Wächter and L. T. Biegler, “On the implementation of an interior-point
filter line-search algorithm for large-scale nonlinear programming,”
Mathematical programming
, vol. 106, no. 1, 2006.
[45]
S. Hochreiter and J. Schmidhuber, “Long short-term memory,”
Neural
computation
, vol. 9, no. 8, 1997.
[46]
G. L’Erario, L. Fiorio, G. Nava, F. Bergonti, H. A. O. Mohamed, E. Benenati,
S. Traversaro, and D. Pucci, “Modeling, identification and control of model
jet engines for jet powered robotics,”
IEEE Robotics and Automation
Letters
, vol. 5, no. 2, 2020.