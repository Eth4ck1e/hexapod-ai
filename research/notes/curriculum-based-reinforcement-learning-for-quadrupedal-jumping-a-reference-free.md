---
title: 'Curriculum-Based Reinforcement Learning for Quadrupedal Jumping: A Reference-free
  Design'
id: curriculum-based-reinforcement-learning-for-quadrupedal-jumping-a-reference-free
tags:
- legged-rl-budgets
- quadruped
- curriculum
- ppo
- training-budget
- sim-to-real
- jumping
created: '2026-05-06T07:31:02.755786Z'
updated: '2026-05-06T07:35:47.904182Z'
source: https://arxiv.org/html/2401.16337v2
source_domain: arxiv.org
fetched_at: '2026-05-06T07:31:02.755786Z'
fetch_provider: builtin
status: draft
type: note
tier: institutional
content_type: paper
deprecated: false
summary: 'Oxford/TU Delft 2024 paper demonstrating reference-free quadrupedal jumping
  via a 3-stage PPO curriculum on the Unitree Go1. Stage I (jumping in place, vertical):
  3k iterations; Stage II (long-distance forward/diagonal jump): 10k iterations; Stage
  III (jump with obstacles): 10k iterations. All stages use 4096 parallel agents and
  24 environment steps per agent per update. Wall-clock times on a single RTX 3090
  GPU: Stage I ~1.4 hours, Stage II ~4.1 hours, Stage III ~4.8 hours (~10.3 hours
  total). Total environment steps: Stage I = 3,000 × 4096 × 24 ≈ 295M; Stage II =
  10,000 × 4096 × 24 ≈ 983M; Stage III = 10,000 × 4096 × 24 ≈ 983M; Grand total ≈
  2.26 billion steps. Policy operates at 50 Hz; simulation at 200 Hz. Architecture:
  shared MLP [256, 128, 64] with ELU activations; history of last 20 steps in observations.
  No BC pretraining stage — reference-free from scratch. Key result: 90 cm forward
  jump, zero-shot transfer to outdoor grass not seen in training. Curriculum progression
  is automatic (success-triggered difficulty increase within stages) and manual (task-level
  stage transition).'
---

Curriculum-Based Reinforcement Learning for Quadrupedal Jumping: A Reference-free Design
HTML conversions
sometimes display errors
due to content that did not convert correctly from the source. This paper uses the following packages that are not yet supported by the HTML conversion tool. Feedback on these issues are not necessary; they are known and are being worked on.
failed: picinpar
failed: pbox
Authors: achieve the best HTML results from your LaTeX submissions by following these
best practices
.
License: CC BY 4.0
arXiv:2401.16337v2 [cs.RO] 04 Mar 2024
Curriculum-Based Reinforcement Learning for Quadrupedal Jumping: A Reference-free Design
Vassil Atanassov*, Jiatao Ding*, Jens Kober, Ioannis Havoutis, Cosimo Della Santina
Vassil Atanassov and Ioannis Havoutis are with the Oxford Robotics Institute, Department of Engineering Science, University of Oxford, U.K (emails: {vassilatanassov, ioannis}@robots.ox.ac.uk). Jiatao Ding, Jens Kober and Cosimo Della Santina are with the Department of Cognitive Robotics, Delft University of Technology, Building 34, Mekelweg 2, 2628CD, Delft, The Netherlands (e-mails: {J.Ding-2, C.DellaSantina, j.kober}@tudelft.nl). Cosimo Della Santina is also with the Institute of Robotics and Mechatronics, German Aerospace Center (DLR), 82234 Wessling, Germany (e-mail: cosimodellasantina@gmail.com).

* Vassil Atanassov and Jiatao Ding are the corresponding authors.
Abstract
Deep reinforcement learning (DRL) has emerged as a promising solution to mastering explosive and versatile quadrupedal jumping skills. However, current DRL-based frameworks usually rely on pre-existing reference trajectories obtained by capturing animal motions or transferring experience from existing controllers. This work aims to prove that learning dynamic jumping is possible without relying on imitating a reference trajectory by leveraging a curriculum design. Starting from a vertical in-place jump, we generalize the learned policy to forward and diagonal jumps and, finally, we learn to jump across obstacles.
Conditioned on the desired landing location, orientation, and obstacle dimensions, the proposed approach yields a wide range of omnidirectional jumping motions in real-world experiments. Particularly we achieve a 90cm forward jump, exceeding all previous records for similar robots reported in the existing literature. Additionally, the robot can reliably execute continuous jumping on soft grassy grounds, which is especially remarkable as such conditions were not included in the training stage.
Note:
A supplementary video can be found on:
https://www.youtube.com/watch?v=nRaMCrwU5X8
. The code associated with this work can be found on:
https://github.com/Vassil17/Curriculum-Quadruped-Jumping-DRL
.
I
Introduction
Through millions of years of evolution, legged animals have adapted to loco-mote in highly complex and discontinuous environments that widely exist in nature. Goats, for example, are capable of scaling nearly vertical mountainsides and jumping across chasms several times their body length. While many works have tackled dynamic locomotion recently
[
1
,
2
,
3
]
, achieving such complex controlled behaviour is still an open challenge.
Quadrupedal jumping has traditionally been investigated through model-based control, where an accurate model of the dynamical system is needed to generate optimal control inputs
[
4
,
5
,
6
,
7
]
. In addition, these methods rely on various heuristics necessary to render the approach feasible, which limit the search space and result in conservative performance.
In contrast to model-based optimisation, model-free reinforcement learning (RL) has emerged as an effective alternative that does not require expert knowledge for control engineering and tedious gain tuning. Especially, deep RL (DRL) has shown impressive generalisation and robustness capabilities in executing locomotion tasks
[
1
,
8
,
9
,
10
,
2
]
. For quadrupedal jumping, a series of correct actions need to be taken for the robot to succeed. Paired with an inherently sparse reward structure (the robot has either jumped or not), it is exceptionally hard for the robot to learn, as most of its trials will fail. Current RL approaches tackle this by directly transferring skills from demonstrations
[
11
,
12
]
or optimal controllers
[
13
,
14
,
15
]
. However, balancing the degree to which the agent should imitate the demonstration and generalise to new tasks is challenging and remains an open question.
Figure 1:
The Go1 robot jumps across grassland (top), jumps down onto grassland (middle) and jumps across a gap onto a lower box (bottom).
In this work, we push robots to learn to jump on their own by combining curriculum learning with DRL, eliminating the reliability of pre-computed motion references. By conditioning the policy on the desired landing location and orientation, our approach produces versatile jumping motions with just one single policy. Furthermore, by incorporating partial knowledge of the obstacles surrounding it, the robot learns different manoeuvres adapted to complex real-world scenarios.
The main contributions are summarised as follows:
•
We propose a curriculum-based DRL framework, which is capable of learning jumping motions without requiring motion capture data or a reference trajectory.
•
We generalise across a wide range of jumps with a single policy for both indoor and outdoor environments. With our method, the real robot can jump 90cm forward, which, to the best of our knowledge, is the longest distance achieved on quadrupeds of a similar size. It has been demonstrated that continuous jumping across grassland and robust jumping across uneven terrains can be achieved in a zero-shot manner.
•
We incorporate partial environmental information into the learning stage, which allows the robot to jump over more complex terrains.
In Section
II
, we introduce the existing RL-based jumping controllers. In Section
III
and Section
IV
, we separately present the curriculum design and DRL formulation. After extensively evaluating our method in Section
V
, we discuss our approach and directions for future work in Section
VI
.
II
Related work
II-A
Reinforcement learning for quadrupedal jumping
DRL is a promising solution for accomplishing jumping tasks by offloading the computational complexity to offline training. One approach to learning quadrupedal jumping is by learning from demonstrations, such as from trajectories generated through optimal control
[
13
,
14
]
, or hand-tuned reference motions
[
11
,
12
]
. To address the challenges associated with the selection of relevant states to mimic and manage conflicting objectives, generative adversarial imitation learning (GAIL) has recently been widely adopted
[
16
,
17
,
18
]
, even when dealing with partially incomplete demonstrations
[
11
]
. In
[
19
]
transfer learning is used to learn policies capable of diverse agile motions from a database of existing RL and model-based controllers.
However, most imitation-based methods have so far shown a limited generalisation capability beyond the imitation domain. Furthermore, many of the aforementioned works rely on learning a separate policy for each unique type of motion, rather than a common task- or goal-conditioned policy.
To reduce the dependency on a motion prior,
[
20
]
use a variational auto-encoder (VAE) to encapsulate motion capture data into a latent space and then combine it with a Bayesian diversity search to discover viable take-off states.
[
21
]
trained a high-level motion planning module to produce desired centre of mass (CoM) trajectories for small hops, conditioned on visual inputs and then tracked by a model-based controller. In
[
13
]
, deviations to reference trajectories generated by a non-linear optimal trajectory
[
5
]
are learned, providing better generalisation to out-of-training domains. Similarly,
[
22
]
learn action residuals to a model-based controller to achieve continuous jumping. Another work focusing on continuous hopping
[
23
]
uses a learned centroidal policy to output desired centre of mass trajectories, which are tracked by a quadratic-programming-based(QP) ground reaction force (GRF) controller. Rudin et al.
[
24
]
show cat-like jumping in low gravity by using a more complex reward function, without imitating motion clips. However, this approach has not yet been verified on Earth-like gravitational conditions. Recently, Vezzi et al.
[
25
]
proposed learning to jump by combining a first-stage evolution strategy with a second-stage DRL. Compared to
[
25
]
, our approach is less complex by using proximal policy optimization (PPO)
[
26
]
for all curriculum stages, and is capable of executing jumps conditioned on the desired jumping length and orientation.
II-B
Curriculum learning in dynamic quadrupedal locomotion
Curriculum learning (CL) is a training framework which progressively provides more challenging data or tasks as the policy improves. As the name suggests, the idea behind the approach borrows from human education, where complex tasks are taught by breaking them into simpler parts.
In legged locomotion, CL has seen wide use, mainly in terms of terrain adaptation. Xie et al.
[
27
]
show how an adaptive curriculum can be used to learn stepping stone skills much more efficiently than other methods like uniform sampling. Similarly, other automatic curriculum learning methods have been proposed to vary environmental parameters based on the performance of the agents
[
10
]
, rather than using a manually specified curriculum. On the rewards side, Hwangbo et al.
[
1
]
employ a curriculum which scales down certain rewards at the start. This design allows the policy to first learn how to locomote and only afterwards to be polished to satisfy the additional constraints and limits of the problem. In
[
28
]
,parkour locomotion skills are learned through a well-designed terrain curriculum with a single policy, which is then distilled to a exteroception-conditioned policy. Similar parkour skills are acquired in
[
29
]
, but the method requires separate policies for each skill, as well as a perception and navigation network, which greatly increases the computational complexity. Barkour
[
30
]
uses a similar approach, but distils the specialist controllers into a single generalist transformer policy.
To learn dynamic parkour skills,
[
31
]
adopt a two-stage curriculum, transitioning from soft to hard dynamic constraints in the second stage.
Recently,
[
12
]
used multi-stage training to learn imitation-based vertical jumping, and then transferred that knowledge to forward jumping. While similar to our approach, however, there are a couple of significant differences - we do not require any reference trajectories, and we learn a single policy for versatile tasks.
III
Curriculum design
Defining and constraining the behaviour of jumping across specific distances is challenging as it combines two distinct behaviours: that of "jumping" and that of reaching a desired spatial point. Furthermore, an easily learnable local optimum exists, where the robot could simply walk (or crawl) toward the target point without actually jumping. To avoid converging to such undesired behaviour we use curriculum learning to decompose the problem into several simpler sub-tasks.
Figure 2:
The curricula: jumping in place (left), long-distance jump (middle) and long-distance jump with obstacles (right). The latter two vary the jump distance/orientation and obstacle height, respectively.
In our approach, we adopt two types of curriculum - on a local difficulty level and on a task level, as can be seen in Fig.
2
. The former involves progressively (and automatically) making the environment more complex as the agent succeeds. In particular, upon successful jumps, we increase the range of desired jumping distances and obstacle heights that we sample from. The task-level curriculum is, on the other hand, manually selected and consists of training the agent for a certain number of steps at a given task. After mastering the easier jumping skill, the policy is loaded onto the next task, which might be defined differently and contains a new set of rewards.
In the remainder , we describe each of these task-level and difficulty curricula in the progressive order of training.
III-A
Stage I: Jumping in place
Vertical jumping without traversing a certain horizontal distance, i.e. jumping in place, is the basic component of agile jumping.
However, the lack of reference results in a learning problem with sparse rewards, given that the agent needs to first learn certain behaviours (e.g., squatting down and then pushing hard against the ground to take off) before it can reach the reward-rich states, i.e. being high in the air. As the robot does not experience these jumping-specific rewards initially, it is prone to converging to a local optimum, such as standing in place, where small rewards are collected safely.
To avoid getting stuck in this local optimum behaviour, we adopt a modified form of the reference state initialisation (RSI) technique
[
32
]
. In imitation learning, RSI initialises the agent at random points of the reference trajectory, allowing the agent to explore such reward-rich states before it has learned the actions necessary to reach them. As we do not use a reference trajectory, we instead modify RSI to sample a random height and upward velocity from a predefined range.
III-B
Stage II: Long-distance jump
Once the robot has converged to a jumping-in-place behaviour, we further train it to perform precise forward and diagonal jumps. The first part of the command vector
𝐠
∈
ℛ
13
𝐠
superscript
ℛ
13
\mathbf{g}\in\mathcal{R}^{13}
bold_g ∈ caligraphic_R start_POSTSUPERSCRIPT 13 end_POSTSUPERSCRIPT
(see Fig.
4
) in the observations specifies the desired landing point and orientation to create a goal-conditioned policy. Similarly to the
jumping in place
sub-task, we also adopt a curriculum-style sampling for desired landing points, where successful agents are progressed to more difficult environments where the desired jumping distance and landing yaw are sampled from a greater range.
III-C
Stage III: Long-distance jump across obstacles
Finally, we introduce obstacles in the environment. Without loss of generality, we choose three classes of obstacles, including thin barrier-like objects, box-shaped obstacles and slopes. Depending on the desired landing pose, the obstacle location and the type, the agent needs to either jump onto or over it. While it is possible to learn a general behaviour that can accomplish this without any exteroception, such behaviour will be conservative, sub-optimal and potentially much less robust. With this in mind, we incorporate information about the distance to the centre of the obstacles and its general dimensions (length, width, and height). In the real world, we manually specify these parameters
1
1
1
A separate module that estimates obstacle dimensions could be utilised. One future work would be linking exteroceptive sensors to the policy and removing the parameterisation of the world around the robot.
.
Similarly to the previous stage, we start with obstacles of smaller height. Then, successful robots progress towards more challenging terrains, whereas failing ones are demoted to easier environments. To ensure that the robot remembers the previously learned behaviour we also randomly send a certain percentage of robots to jump on flat ground, as in Stage II.
Figure 3:
Control diagram of the system. The observations
𝐨
t
subscript
𝐨
𝑡
\mathbf{o}_{t}
bold_o start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT
include user command (in green) and a history of system states (in yellow). The policy is parameterised by a neural network (shown in blue). The output actions
𝐚
t
+
1
subscript
𝐚
𝑡
1
\mathbf{a}_{t+1}
bold_a start_POSTSUBSCRIPT italic_t + 1 end_POSTSUBSCRIPT
are added to the nominal joint angles
𝐪
nom
superscript
𝐪
nom
\mathbf{q}^{\mathrm{nom}}
bold_q start_POSTSUPERSCRIPT roman_nom end_POSTSUPERSCRIPT
. The desired joint angles are then tracked via a PD controller which computes torque commands.
IV
DRL formulation
This section details the DLR formulation, as illustrated in Fig.
3
. First, preliminaries are introduced. Then, we define the key components of goal-conditioned RL, including observations, actions and reward functions. Finally, we introduce our domain randomisation scheme to mitigate the sim2real gap.
IV-A
Preliminaries
RL infers a policy
π
⁢
(
a
t
|
s
t
)
𝜋
conditional
subscript
𝑎
𝑡
subscript
𝑠
𝑡
\pi(a_{t}|s_{t})
italic_π ( italic_a start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT | italic_s start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT )
of how to act by constantly interacting with the environment. The RL problem is typically formulated as a Markov decision process (MDP), where at each step the agent interacts with the environment by taking an
action
𝐚
t
∈
𝒜
subscript
𝐚
𝑡
𝒜
\textbf{a}_{t}\in\mathcal{A}
a start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT ∈ caligraphic_A
. Subsequently, it receives the new states of the environment
𝐬
t
+
1
∈
𝒪
subscript
𝐬
𝑡
1
𝒪
\textbf{s}_{t+1}\in\mathcal{O}
s start_POSTSUBSCRIPT italic_t + 1 end_POSTSUBSCRIPT ∈ caligraphic_O
in the form of
observation
, and the associated
reward
ℛ
t
subscript
ℛ
𝑡
\mathcal{R}_{t}
caligraphic_R start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT
that it has earned. Based on the observed state
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
and its policy
π
⁢
(
a
t
+
1
|
s
t
+
1
)
𝜋
conditional
subscript
𝑎
𝑡
1
subscript
𝑠
𝑡
1
\pi(a_{t+1}|s_{t+1})
italic_π ( italic_a start_POSTSUBSCRIPT italic_t + 1 end_POSTSUBSCRIPT | italic_s start_POSTSUBSCRIPT italic_t + 1 end_POSTSUBSCRIPT )
the agent can then choose a new action
a
t
+
1
subscript
𝑎
𝑡
1
a_{t+1}
italic_a start_POSTSUBSCRIPT italic_t + 1 end_POSTSUBSCRIPT
. In this way, the RL algorithm optimises behaviours that yield high rewards.
In goal-conditioned RL, the action policy can also be conditioned on specific goals, i.e.
π
⁢
(
a
t
|
s
t
,
g
)
𝜋
conditional
subscript
𝑎
𝑡
subscript
𝑠
𝑡
𝑔
\pi(a_{t}|s_{t},g)
italic_π ( italic_a start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT | italic_s start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT , italic_g )
. Such a policy can be used to produce diverse behaviours depending on the specific command
g
𝑔
g
italic_g
, enabling the learning of multiple distinct behaviours under a single policy.
In this work, we formulate the following objective: finding a policy
π
⁢
(
a
|
s
,
g
)
𝜋
conditional
𝑎
𝑠
𝑔
\pi(a|s,g)
italic_π ( italic_a | italic_s , italic_g )
which maximises the cumulative sum of rewards earned over the task duration. As often immediate rewards are more valuable than rewards in the distant future, a discount factor
γ
∈
(
0
,
1
]
𝛾
0
1
\gamma\in(0,1]
italic_γ ∈ ( 0 , 1 ]
is commonly used. Mathematically, the full objective of maximising the sum of discounted rewards
J
𝐽
J
italic_J
, known as the return, can be written as:
arg
⁡
max
π
J
⁢
(
π
)
=
𝔼
τ
∼
p
π
⁢
(
τ
)
⁢
[
∑
t
=
0
T
γ
t
⁢
R
t
|
s
0
=
s
]
,
subscript
𝜋
𝐽
𝜋
subscript
𝔼
similar-to
𝜏
superscript
𝑝
𝜋
𝜏
delimited-[]
conditional
superscript
subscript
𝑡
0
𝑇
superscript
𝛾
𝑡
subscript
𝑅
𝑡
subscript
𝑠
0
𝑠
\mathop{\arg\max\limits_{\mathbf{\pi}}}\quad J(\pi)=\mathbb{E}_{\tau\sim p^{%
\pi}(\tau)}\left[\sum_{t=0}^{T}\gamma^{t}R_{t}|s_{0}=s\right],
start_BIGOP roman_arg roman_max start_POSTSUBSCRIPT italic_π end_POSTSUBSCRIPT end_BIGOP italic_J ( italic_π ) = blackboard_E start_POSTSUBSCRIPT italic_τ ∼ italic_p start_POSTSUPERSCRIPT italic_π end_POSTSUPERSCRIPT ( italic_τ ) end_POSTSUBSCRIPT [ ∑ start_POSTSUBSCRIPT italic_t = 0 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_T end_POSTSUPERSCRIPT italic_γ start_POSTSUPERSCRIPT italic_t end_POSTSUPERSCRIPT italic_R start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT | italic_s start_POSTSUBSCRIPT 0 end_POSTSUBSCRIPT = italic_s ] ,
(1)
where
R
t
subscript
𝑅
𝑡
R_{t}
italic_R start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT
is the immediate reward at time
t
𝑡
t
italic_t
and
s
0
subscript
𝑠
0
s_{0}
italic_s start_POSTSUBSCRIPT 0 end_POSTSUBSCRIPT
is the initial state. The expectation of the return is taken over a trajectory
τ
𝜏
\tau
italic_τ
sampled by following the policy.
IV-B
Observation and action space
Figure 4:
The definition of observations. The command
𝐠
𝐠
\mathbf{g}
bold_g
and jump toggle
j
are provided by the user, while the remaining observations are either directly read from the sensors, or estimated using sensory data.
Observation space:
Using a memory of previous observations and actions allows the agent to implicitly reason about its own dynamics and the interaction with the environment
[
1
,
10
]
. Here,
we use a concatenated history of the last
N
steps as input to the policy
2
2
2
In practice, we found that using the last 20 steps is sufficient for the task while also being fast for training.
. As illustrated in Fig.
4
, the observation space consists of the historical base linear velocity
𝐯
∈
ℝ
3
×
N
𝐯
superscript
ℝ
3
𝑁
\mathbf{v}\in\mathbb{R}^{3\times N}
bold_v ∈ blackboard_R start_POSTSUPERSCRIPT 3 × italic_N end_POSTSUPERSCRIPT
, base angular velocity
𝝎
∈
ℝ
3
×
N
𝝎
superscript
ℝ
3
𝑁
\bm{\omega}\in\mathbb{R}^{3\times N}
bold_italic_ω ∈ blackboard_R start_POSTSUPERSCRIPT 3 × italic_N end_POSTSUPERSCRIPT
(both in the base frame), joint position
𝐪
∈
ℝ
12
×
N
𝐪
superscript
ℝ
12
𝑁
\mathbf{q}\in\mathbb{R}^{12\times N}
bold_q ∈ blackboard_R start_POSTSUPERSCRIPT 12 × italic_N end_POSTSUPERSCRIPT
, joint velocity
𝐪
˙
∈
ℝ
12
×
N
˙
𝐪
superscript
ℝ
12
𝑁
\mathbf{\dot{q}}\in\mathbb{R}^{12\times N}
over˙ start_ARG bold_q end_ARG ∈ blackboard_R start_POSTSUPERSCRIPT 12 × italic_N end_POSTSUPERSCRIPT
, previous actions
𝐚
t
−
1
∈
ℝ
12
×
N
subscript
𝐚
𝑡
1
superscript
ℝ
12
𝑁
\mathbf{a}_{t-1}\in\mathbb{R}^{12\times N}
bold_a start_POSTSUBSCRIPT italic_t - 1 end_POSTSUBSCRIPT ∈ blackboard_R start_POSTSUPERSCRIPT 12 × italic_N end_POSTSUPERSCRIPT
, the base orientation (as a quaternion)
𝐪
¯
∈
ℝ
4
×
N
¯
𝐪
superscript
ℝ
4
𝑁
\mathbf{\bar{q}}\in\mathbb{R}^{4\times N}
over¯ start_ARG bold_q end_ARG ∈ blackboard_R start_POSTSUPERSCRIPT 4 × italic_N end_POSTSUPERSCRIPT
and the foot contact states
𝐜
∈
ℝ
4
×
N
𝐜
superscript
ℝ
4
𝑁
\mathbf{c}\in\mathbb{R}^{4\times N}
bold_c ∈ blackboard_R start_POSTSUPERSCRIPT 4 × italic_N end_POSTSUPERSCRIPT
.
Note that our policy is also conditioned on the command
𝐠
∈
ℝ
13
𝐠
superscript
ℝ
13
\mathbf{g}\in\mathbb{R}^{13}
bold_g ∈ blackboard_R start_POSTSUPERSCRIPT 13 end_POSTSUPERSCRIPT
and jump toggle
j
∈
{
0
,
1
}
𝑗
0
1
j\in\{0,1\}
italic_j ∈ { 0 , 1 }
, see the green block in Fig.
4
. As illustrated in Fig.
5
, the command
𝐠
∈
[
Δ
⁢
𝐩
des
,
Δ
⁢
𝐪
¯
des
,
𝐩
obs
,
𝐝𝐢𝐦
obs
]
𝐠
Δ
subscript
𝐩
des
Δ
subscript
¯
𝐪
des
subscript
𝐩
obs
subscript
𝐝𝐢𝐦
obs
\mathbf{g}\in[\Delta\mathbf{p}_{\mathrm{des}},\Delta\mathbf{\bar{q}}_{\mathrm{%
des}},\mathbf{p}_{\mathrm{{obs}}},\mathbf{dim}_{\mathrm{obs}}]
bold_g ∈ [ roman_Δ bold_p start_POSTSUBSCRIPT roman_des end_POSTSUBSCRIPT , roman_Δ over¯ start_ARG bold_q end_ARG start_POSTSUBSCRIPT roman_des end_POSTSUBSCRIPT , bold_p start_POSTSUBSCRIPT roman_obs end_POSTSUBSCRIPT , bold_dim start_POSTSUBSCRIPT roman_obs end_POSTSUBSCRIPT ]
contains the desired landing position (
Δ
⁢
𝐩
des
∈
ℝ
3
Δ
subscript
𝐩
des
superscript
ℝ
3
\Delta\mathbf{p}_{\mathrm{des}}\in\mathbb{R}^{3}
roman_Δ bold_p start_POSTSUBSCRIPT roman_des end_POSTSUBSCRIPT ∈ blackboard_R start_POSTSUPERSCRIPT 3 end_POSTSUPERSCRIPT
), desired landing orientation (
Δ
⁢
𝐪
¯
des
∈
ℝ
4
Δ
subscript
¯
𝐪
des
superscript
ℝ
4
\Delta\mathbf{\bar{q}}_{\mathrm{des}}\in\mathbb{R}^{4}
roman_Δ over¯ start_ARG bold_q end_ARG start_POSTSUBSCRIPT roman_des end_POSTSUBSCRIPT ∈ blackboard_R start_POSTSUPERSCRIPT 4 end_POSTSUPERSCRIPT
), the centre of the obstacle (
𝐩
obs
∈
ℝ
3
subscript
𝐩
obs
superscript
ℝ
3
\mathbf{p}_{\mathrm{obs}}\in\mathbb{R}^{3}
bold_p start_POSTSUBSCRIPT roman_obs end_POSTSUBSCRIPT ∈ blackboard_R start_POSTSUPERSCRIPT 3 end_POSTSUPERSCRIPT
if present), and its dimensions (
𝐝𝐢𝐦
obs
∈
ℝ
3
subscript
𝐝𝐢𝐦
obs
superscript
ℝ
3
\mathbf{dim}_{\mathrm{obs}}\in\mathbb{R}^{3}
bold_dim start_POSTSUBSCRIPT roman_obs end_POSTSUBSCRIPT ∈ blackboard_R start_POSTSUPERSCRIPT 3 end_POSTSUPERSCRIPT
including height, width, and length)
3
3
3
In the training process, we sample the landing pose and obtain the obstacle parameters from the simulator. In the real world, the command vector is specified by the user.
. Due to the lack of long-term memory in the feed-forward neural network, we use the jump toggle
j
𝑗
j
italic_j
to indicate whether the robot has already jumped, similar to
[
32
]
. However, in our case, the jump toggle also serves as a control switch, where the robot remains standing until its value is changed.
Figure 5:
The command vector
𝐠
𝐠
\mathbf{g}
bold_g
for a forward jump onto an obstacle. In the first two training stages (
π
I
subscript
𝜋
𝐼
\pi_{I}
italic_π start_POSTSUBSCRIPT italic_I end_POSTSUBSCRIPT
and
π
I
⁢
I
subscript
𝜋
𝐼
𝐼
\pi_{II}
italic_π start_POSTSUBSCRIPT italic_I italic_I end_POSTSUBSCRIPT
), where no obstacles are considered, the information of the obstacle is set to zero.
Action space:
Our policy generates the twelve actuated joint angles (
𝐪
des
∈
ℝ
12
superscript
𝐪
des
superscript
ℝ
12
\mathbf{q}^{\mathrm{des}}\in\mathbb{R}^{12}
bold_q start_POSTSUPERSCRIPT roman_des end_POSTSUPERSCRIPT ∈ blackboard_R start_POSTSUPERSCRIPT 12 end_POSTSUPERSCRIPT
) for jumping control. Particularly, we learn the deviations from the nominal joint positions
𝐪
nom
∈
ℝ
12
superscript
𝐪
nom
superscript
ℝ
12
\mathbf{q}^{\mathrm{nom}}\in\mathbb{R}^{12}
bold_q start_POSTSUPERSCRIPT roman_nom end_POSTSUPERSCRIPT ∈ blackboard_R start_POSTSUPERSCRIPT 12 end_POSTSUPERSCRIPT
. To smooth the output actions, we used an exponential moving average (EMA) low-pass filter with a cut-off frequency of 5 Hz. The filtered actions are then scaled and added to
𝐪
nom
superscript
𝐪
nom
\mathbf{q}^{\mathrm{nom}}
bold_q start_POSTSUPERSCRIPT roman_nom end_POSTSUPERSCRIPT
to generate
𝐪
des
superscript
𝐪
des
\mathbf{q}^{\mathrm{des}}
bold_q start_POSTSUPERSCRIPT roman_des end_POSTSUPERSCRIPT
for the motor servos, i.e.
𝐪
des
=
𝐚
+
𝐪
nom
superscript
𝐪
des
𝐚
superscript
𝐪
nom
\mathbf{q}^{\mathrm{des}}=\mathbf{a}+\mathbf{q}^{\mathrm{nom}}
bold_q start_POSTSUPERSCRIPT roman_des end_POSTSUPERSCRIPT = bold_a + bold_q start_POSTSUPERSCRIPT roman_nom end_POSTSUPERSCRIPT
. A PD feedback controller then produces the desired torque at a higher frequency, as shown in Fig.
3
. To guarantee safety, we clip
𝐪
des
superscript
𝐪
des
\mathbf{q}^{\mathrm{des}}
bold_q start_POSTSUPERSCRIPT roman_des end_POSTSUPERSCRIPT
within the feasibility range when the real joint angles approach the limits.
IV-C
Rewards
TABLE I:
Rewards definition. The light orange colour indicates task-based rewards, while the light purple shade describes regularisation rewards.
w
×
subscript
𝑤
w_{\times}
italic_w start_POSTSUBSCRIPT × end_POSTSUBSCRIPT
is the weight,
σ
×
subscript
𝜎
\sigma_{\times}
italic_σ start_POSTSUBSCRIPT × end_POSTSUBSCRIPT
is a scaling factor for the exponential kernel,
e
⁡
(
⋅
)
e
⋅
\operatorname{e}(\cdot)
roman_e ( ⋅ )
and
log
⁡
(
⋅
)
log
⋅
\operatorname{log}(\cdot)
roman_log ( ⋅ )
separately denote the exponent and logarithm operation.
Name
Type
Stance
Flight
Landing
Landing position
Single
0
0
w
𝐩
⁢
(
e
⁢
(
−
∑
‖
𝐩
land
−
𝐩
des
‖
2
)
/
σ
p
,
land
)
subscript
𝑤
𝐩
𝑒
superscript
norm
subscript
𝐩
land
subscript
𝐩
des
2
subscript
𝜎
𝑝
land
w_{\mathbf{p}}(e(-\sum||\mathbf{p}_{\mathrm{land}}-\mathbf{p}_{\mathrm{des}}||%
^{2})/\sigma_{p\mathrm{,land}})
italic_w start_POSTSUBSCRIPT bold_p end_POSTSUBSCRIPT ( italic_e ( - ∑ | | bold_p start_POSTSUBSCRIPT roman_land end_POSTSUBSCRIPT - bold_p start_POSTSUBSCRIPT roman_des end_POSTSUBSCRIPT | | start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT ) / italic_σ start_POSTSUBSCRIPT italic_p , roman_land end_POSTSUBSCRIPT )
Landing orientation
Single
0
0
w
ori
(
e
(
−
|
|
log
(
𝐪
¯
land
−
1
*
𝐪
¯
des
|
|
2
)
/
σ
ori
,
land
)
w_{\mathrm{ori}}(\operatorname{e}(-||\operatorname{log}(\mathbf{\bar{q}}_{%
\mathrm{land}}^{-1}*\mathbf{\bar{q}}_{\mathrm{des}}||^{2})/\sigma_{\mathrm{ori%
,land}})
italic_w start_POSTSUBSCRIPT roman_ori end_POSTSUBSCRIPT ( roman_e ( - | | roman_log ( over¯ start_ARG bold_q end_ARG start_POSTSUBSCRIPT roman_land end_POSTSUBSCRIPT start_POSTSUPERSCRIPT - 1 end_POSTSUPERSCRIPT * over¯ start_ARG bold_q end_ARG start_POSTSUBSCRIPT roman_des end_POSTSUBSCRIPT | | start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT ) / italic_σ start_POSTSUBSCRIPT roman_ori , roman_land end_POSTSUBSCRIPT )
Max height
Single
0
0
w
h
(
e
(
|
|
h
max
−
0.9
|
|
2
)
/
σ
p
z
,
max
)
)
w_{h}(\operatorname{e}(||h_{\mathrm{max}}-0.9||^{2})/\sigma_{p_{z}\mathrm{,max%
}}))
italic_w start_POSTSUBSCRIPT italic_h end_POSTSUBSCRIPT ( roman_e ( | | italic_h start_POSTSUBSCRIPT roman_max end_POSTSUBSCRIPT - 0.9 | | start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT ) / italic_σ start_POSTSUBSCRIPT italic_p start_POSTSUBSCRIPT italic_z end_POSTSUBSCRIPT , roman_max end_POSTSUBSCRIPT ) )
Jumping
Single
0
0
w
jump
subscript
𝑤
jump
w_{\mathrm{jump}}
italic_w start_POSTSUBSCRIPT roman_jump end_POSTSUBSCRIPT
Base Position
Continuous
w
p
z
,
st
⁢
(
e
⁡
(
−
‖
p
z
−
0.20
‖
2
/
σ
p
z
,
st
)
)
)
w_{p_{z},\mathrm{st}}(\operatorname{e}(-||p_{z}-0.20||^{2}/\sigma_{p_{z},%
\mathrm{st)}}))
italic_w start_POSTSUBSCRIPT italic_p start_POSTSUBSCRIPT italic_z end_POSTSUBSCRIPT , roman_st end_POSTSUBSCRIPT ( roman_e ( - | | italic_p start_POSTSUBSCRIPT italic_z end_POSTSUBSCRIPT - 0.20 | | start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT / italic_σ start_POSTSUBSCRIPT italic_p start_POSTSUBSCRIPT italic_z end_POSTSUBSCRIPT , roman_st ) end_POSTSUBSCRIPT ) )
w
p
z
,
fl
⁢
(
e
⁡
(
−
‖
p
z
−
0.7
‖
2
/
σ
p
z
,
fl
)
)
subscript
𝑤
subscript
𝑝
𝑧
fl
e
superscript
norm
subscript
𝑝
𝑧
0.7
2
subscript
𝜎
subscript
𝑝
𝑧
fl
w_{p_{z},\mathrm{fl}}(\operatorname{e}(-||p_{z}-0.7||^{2}/\sigma_{p_{z}\mathrm%
{,fl}}))
italic_w start_POSTSUBSCRIPT italic_p start_POSTSUBSCRIPT italic_z end_POSTSUBSCRIPT , roman_fl end_POSTSUBSCRIPT ( roman_e ( - | | italic_p start_POSTSUBSCRIPT italic_z end_POSTSUBSCRIPT - 0.7 | | start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT / italic_σ start_POSTSUBSCRIPT italic_p start_POSTSUBSCRIPT italic_z end_POSTSUBSCRIPT , roman_fl end_POSTSUBSCRIPT ) )
w
𝐩
,
l
⁢
(
e
⁡
(
−
∑
‖
𝐩
−
𝐩
des
‖
2
/
σ
p
,
l
)
)
subscript
𝑤
𝐩
l
e
superscript
norm
𝐩
subscript
𝐩
des
2
subscript
𝜎
𝑝
l
w_{\mathbf{p},\mathrm{l}}(\operatorname{e}(-\sum||\mathbf{p}-\mathbf{p}_{%
\mathrm{des}}||^{2}/\sigma_{p\mathrm{,l}}))
italic_w start_POSTSUBSCRIPT bold_p , roman_l end_POSTSUBSCRIPT ( roman_e ( - ∑ | | bold_p - bold_p start_POSTSUBSCRIPT roman_des end_POSTSUBSCRIPT | | start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT / italic_σ start_POSTSUBSCRIPT italic_p , roman_l end_POSTSUBSCRIPT ) )
Orientation Tracking
Continuous
w
ori
,
st
(
e
(
−
|
|
log
(
𝐪
¯
base
−
1
*
𝐪
¯
des
|
|
2
/
σ
ori
,
st
)
)
w_{\mathrm{ori,st}}(\operatorname{e}(-||\log(\mathbf{\bar{q}}_{\mathrm{base}}^%
{-1}*\mathbf{\bar{q}}_{\mathrm{des}}||^{2}/\sigma_{\mathrm{ori,st}}))
italic_w start_POSTSUBSCRIPT roman_ori , roman_st end_POSTSUBSCRIPT ( roman_e ( - | | roman_log ( over¯ start_ARG bold_q end_ARG start_POSTSUBSCRIPT roman_base end_POSTSUBSCRIPT start_POSTSUPERSCRIPT - 1 end_POSTSUPERSCRIPT * over¯ start_ARG bold_q end_ARG start_POSTSUBSCRIPT roman_des end_POSTSUBSCRIPT | | start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT / italic_σ start_POSTSUBSCRIPT roman_ori , roman_st end_POSTSUBSCRIPT ) )
0
w
ori
,
l
⁢
(
e
⁡
(
−
‖
log
⁡
(
𝐪
¯
base
−
1
*
𝐪
¯
des
)
‖
2
/
σ
ori
,
l
)
)
subscript
𝑤
ori
l
e
superscript
norm
superscript
subscript
¯
𝐪
base
1
subscript
¯
𝐪
des
2
subscript
𝜎
ori
l
w_{\mathrm{ori,l}}(\operatorname{e}(-||\log(\mathbf{\bar{q}}_{\mathrm{base}}^{%
-1}*\mathbf{\bar{q}}_{\mathrm{des}})||^{2}/\sigma_{\mathrm{ori,l}}))
italic_w start_POSTSUBSCRIPT roman_ori , roman_l end_POSTSUBSCRIPT ( roman_e ( - | | roman_log ( over¯ start_ARG bold_q end_ARG start_POSTSUBSCRIPT roman_base end_POSTSUBSCRIPT start_POSTSUPERSCRIPT - 1 end_POSTSUPERSCRIPT * over¯ start_ARG bold_q end_ARG start_POSTSUBSCRIPT roman_des end_POSTSUBSCRIPT ) | | start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT / italic_σ start_POSTSUBSCRIPT roman_ori , roman_l end_POSTSUBSCRIPT ) )
Base linear velocity
Continuous
0
w
𝐯
x
,
y
⁢
(
−
e
⁡
(
∑
‖
𝐯
x
,
y
−
𝐯
des
‖
2
/
σ
v
)
)
subscript
𝑤
subscript
𝐯
𝑥
𝑦
e
superscript
norm
subscript
𝐯
𝑥
𝑦
subscript
𝐯
des
2
subscript
𝜎
𝑣
w_{\mathbf{v}_{x,y}}(-\operatorname{e}(\sum||\mathbf{v}_{x,y}-\mathbf{v}_{%
\mathrm{des}}||^{2}/\sigma_{v}))
italic_w start_POSTSUBSCRIPT bold_v start_POSTSUBSCRIPT italic_x , italic_y end_POSTSUBSCRIPT end_POSTSUBSCRIPT ( - roman_e ( ∑ | | bold_v start_POSTSUBSCRIPT italic_x , italic_y end_POSTSUBSCRIPT - bold_v start_POSTSUBSCRIPT roman_des end_POSTSUBSCRIPT | | start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT / italic_σ start_POSTSUBSCRIPT italic_v end_POSTSUBSCRIPT ) )
0
Base angular velocity
Continuous
0
w
𝝎
⁢
(
e
⁡
(
−
∑
‖
𝝎
−
𝝎
des
‖
2
/
σ
ω
)
)
subscript
𝑤
𝝎
e
superscript
norm
𝝎
subscript
𝝎
des
2
subscript
𝜎
𝜔
w_{\bm{\omega}}(\operatorname{e}(-\sum||\bm{\omega}-\bm{\omega}_{\mathrm{des}}%
||^{2}/\sigma_{\omega}))
italic_w start_POSTSUBSCRIPT bold_italic_ω end_POSTSUBSCRIPT ( roman_e ( - ∑ | | bold_italic_ω - bold_italic_ω start_POSTSUBSCRIPT roman_des end_POSTSUBSCRIPT | | start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT / italic_σ start_POSTSUBSCRIPT italic_ω end_POSTSUBSCRIPT ) )
0.1
⁢
w
𝝎
⁢
(
e
⁡
(
−
∑
‖
𝝎
‖
2
/
σ
ω
)
)
0.1
subscript
𝑤
𝝎
e
superscript
norm
𝝎
2
subscript
𝜎
𝜔
0.1w_{\bm{\omega}}(\operatorname{e}(-\sum||\bm{\omega}||^{2}/\sigma_{\omega}))
0.1 italic_w start_POSTSUBSCRIPT bold_italic_ω end_POSTSUBSCRIPT ( roman_e ( - ∑ | | bold_italic_ω | | start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT / italic_σ start_POSTSUBSCRIPT italic_ω end_POSTSUBSCRIPT ) )
Feet clearance
Continuous
0
w
feet
⁢
(
‖
p
feet
−
p
feet
0
+
[
0.0
,
0.0
,
−
0.15
]
‖
2
)
subscript
𝑤
feet
superscript
norm
subscript
𝑝
feet
superscript
subscript
𝑝
feet
0
0.0
0.0
0.15
2
w_{\mathrm{feet}}(||p_{\mathrm{feet}}-p_{\mathrm{feet}}^{0}+[0.0,0.0,-0.15]||^%
{2})
italic_w start_POSTSUBSCRIPT roman_feet end_POSTSUBSCRIPT ( | | italic_p start_POSTSUBSCRIPT roman_feet end_POSTSUBSCRIPT - italic_p start_POSTSUBSCRIPT roman_feet end_POSTSUBSCRIPT start_POSTSUPERSCRIPT 0 end_POSTSUPERSCRIPT + [ 0.0 , 0.0 , - 0.15 ] | | start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT )
0
Symmetry
Continuous
w
sym
⁢
(
∑
joint
|
𝐪
left
−
𝐪
right
|
2
)
subscript
𝑤
sym
subscript
joint
superscript
subscript
𝐪
left
subscript
𝐪
right
2
w_{\mathrm{sym}}(\sum_{\mathrm{joint}}|\mathbf{q}_{\mathrm{left}}-\mathbf{q}_{%
\mathrm{right}}|^{2})
italic_w start_POSTSUBSCRIPT roman_sym end_POSTSUBSCRIPT ( ∑ start_POSTSUBSCRIPT roman_joint end_POSTSUBSCRIPT | bold_q start_POSTSUBSCRIPT roman_left end_POSTSUBSCRIPT - bold_q start_POSTSUBSCRIPT roman_right end_POSTSUBSCRIPT | start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT )
Nominal pose
Continuous
w
𝐪
(
e
(
−
∑
joint
|
|
𝐪
j
−
𝐪
j
,
nom
|
|
2
/
σ
q
)
w_{\mathbf{q}}(\operatorname{e}(-\sum_{\mathrm{joint}}||\mathbf{q}_{j}-\mathbf%
{q}_{j,\mathrm{nom}}||^{2}/\sigma_{q})
italic_w start_POSTSUBSCRIPT bold_q end_POSTSUBSCRIPT ( roman_e ( - ∑ start_POSTSUBSCRIPT roman_joint end_POSTSUBSCRIPT | | bold_q start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT - bold_q start_POSTSUBSCRIPT italic_j , roman_nom end_POSTSUBSCRIPT | | start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT / italic_σ start_POSTSUBSCRIPT italic_q end_POSTSUBSCRIPT )
0.1
w
𝐪
(
e
(
−
∑
joint
|
|
𝐪
j
−
𝐪
j
,
nom
|
|
2
/
σ
q
)
w_{\mathbf{q}}(\operatorname{e}(-\sum_{\mathrm{joint}}||\mathbf{q}_{j}-\mathbf%
{q}_{j,\mathrm{nom}}||^{2}/\sigma_{q})
italic_w start_POSTSUBSCRIPT bold_q end_POSTSUBSCRIPT ( roman_e ( - ∑ start_POSTSUBSCRIPT roman_joint end_POSTSUBSCRIPT | | bold_q start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT - bold_q start_POSTSUBSCRIPT italic_j , roman_nom end_POSTSUBSCRIPT | | start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT / italic_σ start_POSTSUBSCRIPT italic_q end_POSTSUBSCRIPT )
w
𝐪
(
e
(
−
∑
joint
|
|
𝐪
j
−
𝐪
j
,
nom
|
|
2
/
σ
q
)
w_{\mathbf{q}}(\operatorname{e}(-\sum_{\mathrm{joint}}||\mathbf{q}_{j}-\mathbf%
{q}_{j,\mathrm{nom}}||^{2}/\sigma_{q})
italic_w start_POSTSUBSCRIPT bold_q end_POSTSUBSCRIPT ( roman_e ( - ∑ start_POSTSUBSCRIPT roman_joint end_POSTSUBSCRIPT | | bold_q start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT - bold_q start_POSTSUBSCRIPT italic_j , roman_nom end_POSTSUBSCRIPT | | start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT / italic_σ start_POSTSUBSCRIPT italic_q end_POSTSUBSCRIPT )
Energy
Continuous
w
energy
⁢
(
𝝉
T
⁢
𝐪
˙
)
subscript
𝑤
energy
superscript
𝝉
𝑇
˙
𝐪
w_{\mathrm{energy}}(\bm{\tau}^{T}\mathbf{\dot{q}})
italic_w start_POSTSUBSCRIPT roman_energy end_POSTSUBSCRIPT ( bold_italic_τ start_POSTSUPERSCRIPT italic_T end_POSTSUPERSCRIPT over˙ start_ARG bold_q end_ARG )
Base acceleration
Continuous
w
acc
⁢
|
𝐯
˙
|
2
subscript
𝑤
acc
superscript
˙
𝐯
2
w_{\mathrm{acc}}|\mathbf{\dot{v}}|^{2}
italic_w start_POSTSUBSCRIPT roman_acc end_POSTSUBSCRIPT | over˙ start_ARG bold_v end_ARG | start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT
Contact change
Continuous
w
c
⁢
∑
feet
(
c
foot
⁢
(
t
)
−
c
foot
⁢
(
t
−
1
)
)
subscript
𝑤
𝑐
subscript
feet
subscript
𝑐
foot
𝑡
subscript
𝑐
foot
𝑡
1
w_{c}\sum_{\mathrm{feet}}(c_{\mathrm{foot}}(t)-c_{\mathrm{foot}}(t-1))
italic_w start_POSTSUBSCRIPT italic_c end_POSTSUBSCRIPT ∑ start_POSTSUBSCRIPT roman_feet end_POSTSUBSCRIPT ( italic_c start_POSTSUBSCRIPT roman_foot end_POSTSUBSCRIPT ( italic_t ) - italic_c start_POSTSUBSCRIPT roman_foot end_POSTSUBSCRIPT ( italic_t - 1 ) )
Maintain Contact
Continuous
w
contact
⁢
∑
feet
c
foot
⁢
(
t
)
subscript
𝑤
contact
subscript
feet
subscript
𝑐
foot
𝑡
w_{\mathrm{contact}}\sum_{\mathrm{feet}}c_{\mathrm{foot}}(t)
italic_w start_POSTSUBSCRIPT roman_contact end_POSTSUBSCRIPT ∑ start_POSTSUBSCRIPT roman_feet end_POSTSUBSCRIPT italic_c start_POSTSUBSCRIPT roman_foot end_POSTSUBSCRIPT ( italic_t )
0
0
Contact forces
Continuous
w
F
c
⁢
∑
i
=
0
n
f
|
F
i
−
F
¯
|
subscript
𝑤
subscript
𝐹
𝑐
superscript
subscript
𝑖
0
subscript
𝑛
f
subscript
𝐹
𝑖
¯
𝐹
w_{F_{c}}\sum_{i=0}^{n_{\mathrm{f}}}|F_{i}-\bar{F}|
italic_w start_POSTSUBSCRIPT italic_F start_POSTSUBSCRIPT italic_c end_POSTSUBSCRIPT end_POSTSUBSCRIPT ∑ start_POSTSUBSCRIPT italic_i = 0 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_n start_POSTSUBSCRIPT roman_f end_POSTSUBSCRIPT end_POSTSUPERSCRIPT | italic_F start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT - over¯ start_ARG italic_F end_ARG |
Action rate
Continuous
w
a
⁢
∑
joint
|
𝐚
⁢
(
t
)
−
𝐚
⁢
(
t
−
1
)
|
2
subscript
𝑤
𝑎
subscript
joint
superscript
𝐚
𝑡
𝐚
𝑡
1
2
w_{a}\sum_{\mathrm{joint}}|\mathbf{a}(t)-\mathbf{a}(t-1)|^{2}
italic_w start_POSTSUBSCRIPT italic_a end_POSTSUBSCRIPT ∑ start_POSTSUBSCRIPT roman_joint end_POSTSUBSCRIPT | bold_a ( italic_t ) - bold_a ( italic_t - 1 ) | start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT
Joint acceleration
Continuous
w
q
¨
⁢
∑
joint
|
𝐪
¨
j
|
2
subscript
𝑤
¨
𝑞
subscript
joint
superscript
subscript
¨
𝐪
𝑗
2
w_{\ddot{q}}\sum_{\mathrm{joint}}|\mathbf{\ddot{q}}_{j}|^{2}
italic_w start_POSTSUBSCRIPT over¨ start_ARG italic_q end_ARG end_POSTSUBSCRIPT ∑ start_POSTSUBSCRIPT roman_joint end_POSTSUBSCRIPT | over¨ start_ARG bold_q end_ARG start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT | start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT
Joint limits
Continuous
w
q
l
⁢
i
⁢
m
⁢
∑
joint
|
𝐪
j
−
𝐪
j
,
l
⁢
i
⁢
m
|
2
subscript
𝑤
subscript
𝑞
𝑙
𝑖
𝑚
subscript
joint
superscript
subscript
𝐪
𝑗
subscript
𝐪
𝑗
𝑙
𝑖
𝑚
2
w_{q_{lim}}\sum_{\mathrm{joint}}|\mathbf{q}_{j}-\mathbf{q}_{j,lim}|^{2}
italic_w start_POSTSUBSCRIPT italic_q start_POSTSUBSCRIPT italic_l italic_i italic_m end_POSTSUBSCRIPT end_POSTSUBSCRIPT ∑ start_POSTSUBSCRIPT roman_joint end_POSTSUBSCRIPT | bold_q start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT - bold_q start_POSTSUBSCRIPT italic_j , italic_l italic_i italic_m end_POSTSUBSCRIPT | start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT
Ideally, we expect the agent to accomplish the task while maximising the rewards it receives. However, poor choice of reward scaling could lead the agent to converge to the local minima, e.g., standing behaviour without jumping, where only certain penalties like energy cost and joint acceleration are minimised.
To avoid this, instead of naively summing them, we multiply the positive component of the reward by the exponent of the squared negative component, i.e.
r
total
=
r
+
⁢
e
⁡
(
−
‖
r
−
‖
2
/
σ
)
subscript
𝑟
total
superscript
𝑟
e
superscript
norm
superscript
𝑟
2
𝜎
r_{\mathrm{total}}=r^{+}\operatorname{e}(-||r^{-}||^{2}/\sigma)
italic_r start_POSTSUBSCRIPT roman_total end_POSTSUBSCRIPT = italic_r start_POSTSUPERSCRIPT + end_POSTSUPERSCRIPT roman_e ( - | | italic_r start_POSTSUPERSCRIPT - end_POSTSUPERSCRIPT | | start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT / italic_σ )
4
4
4
For conciseness, notation
e
⁡
(
−
‖
x
‖
2
/
σ
)
e
superscript
norm
𝑥
2
𝜎
\operatorname{e}(-||x||^{2}/\sigma)
roman_e ( - | | italic_x | | start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT / italic_σ )
is used to represent passing the squared error
‖
x
‖
2
superscript
norm
𝑥
2
||x||^{2}
| | italic_x | | start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT
through an exponential kernel of the form
exp
⁡
(
−
‖
x
‖
2
σ
)
exp
superscript
norm
𝑥
2
𝜎
\operatorname{exp}(\frac{-||x||^{2}}{\sigma})
roman_exp ( divide start_ARG - | | italic_x | | start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT end_ARG start_ARG italic_σ end_ARG )
. This ensures the reward is positive and scales it between 0 and 1.
. This allows the agent to always receive a strictly positive reward, scaled down by the amount of penalties, which improves the learning stability.
As listed in Table
I
, three phases are used to describe when each reward is given. In particular, ‘stance’ indicates that the robot has been given a command to jump but is still on the ground. Then, ‘flight’ is triggered when the robot is in mid-air and has no contact with the ground. Finally, the ‘landing’ begins upon landing and lasts until the end of the episode. In each phase, task-based rewards (in orange) and regularisation rewards (in violet) are considered. On the other hand, the rewards items can be divided into
Single
type and
Continuous
type, where the former is given once per episode (typically at the end), and the latter is given once per each simulation step that satisfies the conditions.
Task rewards:
First, sparse rewards are introduced to encourage the general behaviour for accomplishing the desired jumping task, including those of detecting contact (‘landing’) after several steps of no contact (‘flight’), the maximum height the agent reached, and whether it has landed at the desired position with the desired orientation. These rewards are only given once at the end of the episode, marked by ‘Single’ in Table
I
. In addition, continuous task-related objectives are also defined to simplify the exploration, including
•
Tracking the desired linear velocity (
𝐯
x
,
y
,
des
b
subscript
superscript
𝐯
𝑏
𝑥
𝑦
des
\mathbf{v}^{b}_{x,y,\mathrm{des}}
bold_v start_POSTSUPERSCRIPT italic_b end_POSTSUPERSCRIPT start_POSTSUBSCRIPT italic_x , italic_y , roman_des end_POSTSUBSCRIPT
) and yaw angular velocity while in flight, and tracking zero angular velocity after landing.
•
Squatting down to a height of 0.2m while on the ground and tracking a certain height in the air.
•
Maintaining a constant base position and tracking the desired orientation after landing.
Notably, in order to ensure enough clearance when jumping forward and over obstacles, we introduce a foot clearance reward that tracks the nominal foot position (i.e. at the nominal joint angles
q
nom
superscript
𝑞
nom
q^{\mathrm{nom}}
italic_q start_POSTSUPERSCRIPT roman_nom end_POSTSUPERSCRIPT
) on the xy-plane, and simultaneously, minimises the z-distance between each foot and the centre of mass. This objective encourages the robot to tuck its legs in close to its body while in the air.
Regularisation rewards:
As we do not imprint any reference motions onto the agent, auxiliary regularisation rewards are needed to achieve smooth, feasible and safe behaviour. Specifically, we penalise the action rate, together with any violations of predefined soft limits for the joint position. Besides, the instantaneous energy power, computed as the dot product between actuator torque and joint velocity, is penalised for generating an energy-efficient motion. Considering that various quadrupedal jumps seen in nature exhibit high left- and right-side symmetry, we drive the robot towards maintaining this symmetry with an additional reward. Finally, we noticed that the robot often stomped its feet rapidly during the squat-down stage in the training process. To eliminate this unnecessary behaviour, we add a small reward for maintaining contact in the first few steps of the episode, as well as a penalty on frequent contact state changes.
Termination:
We terminate each episode when the following events occur:
•
Collision between body links and the environment.
•
Base height lower than 0.12 m.
•
Orientation error larger than 3.0 rad.
•
Landing position error bigger than 0.15 m.
IV-D
Domain randomisation
TABLE II:
Randomised variables and their ranges.
Name
Randomisation range
Ground friction
[0.01, 3.0]
Ground restitution
[0.0, 0.4]
Additional payload
[-1.0, 3.0]
kg
kilogram
\mathrm{kg}
roman_kg
Link mass factor
[0.7,1.3] x
Centre of mass displacement
[-0.1, 0.1]
m
meter
\mathrm{m}
roman_m
Episodic Latency
[0.0, 40.0]
ms
millisecond
\mathrm{ms}
roman_ms
Extra per-step latency
[-5.0, 5.0]
ms
millisecond
\mathrm{ms}
roman_ms
Motor Strength factor
[0.9, 1.1] x
Joint offsets
[-0.02, 0.02]
rad
radian
\mathrm{rad}
roman_rad
PD Gains factor
[0.9, 1.1] x
Joint friction
[0.0, 0.04]
Joint damping
[0.0, 0.01]
N
m
s
rad
−
1
times
newton
meter
second
radian
1
\mathrm{N}\text{\,}\mathrm{m}\text{\,}\mathrm{s}\text{\,}{\mathrm{rad}}^{-1}
start_ARG roman_N end_ARG start_ARG times end_ARG start_ARG roman_m end_ARG start_ARG times end_ARG start_ARG roman_s end_ARG start_ARG times end_ARG start_ARG power start_ARG roman_rad end_ARG start_ARG - 1 end_ARG end_ARG
To bridge the gap between simulation and real-world scenarios, we implement zero-shot domain randomisation. The ground friction, restitution, and link mass are sampled at random at the start of every episode. In addition, we add a random offset to the joint encoder values, randomise proportional and derivative gains of the PD controller and randomise the strength of the motors for every episode. The range of each randomized variable is listed in Table
II
.
For hardware control, unmodelled communication delays and latencies strongly weaken the performance of learning-based policies. To tackle this issue, at the beginning of each episode, we sample a latency value from the range of
l
∈
[
0
,
50
]
𝑙
0
50
l\in[0,50]
italic_l ∈ [ 0 , 50 ]
ms. Then, at each step, we add a small random value to reflect the effect of stochastic communication delays.
V
Experimental validation
In this section, we first validate the policy trained on the first two curriculum stages (i.e. policy
π
I
⁢
I
subscript
𝜋
𝐼
𝐼
\pi_{II}
italic_π start_POSTSUBSCRIPT italic_I italic_I end_POSTSUBSCRIPT
, shown in Fig.
2
), through various experiments - forward and diagonal jumps, continuous jumps, and robust jumping in the presence of environmental disturbances and uneven terrains. Then, we validate the policy after the final training stage (policy
π
I
⁢
I
⁢
I
subscript
𝜋
𝐼
𝐼
𝐼
\pi_{III}
italic_π start_POSTSUBSCRIPT italic_I italic_I italic_I end_POSTSUBSCRIPT
) when jumping onto and over obstacles.
V-A
Training setup
The
implementation is based on the open-source Gym environment provided by ETH Zurich
[
9
]
. Particularly, we use 4096 agents and 24 environmental steps per agent per update step. For the vertical jump, we train for 3k iterations, while for the forward jump without and with obstacles we train for 10k steps each. The actor policy and the critic are parameterised by a shared MLP with 3 hidden layers of dimensions
[
256
,
128
,
64
]
256
128
64
[256,128,64]
[ 256 , 128 , 64 ]
, with exponential linear unit (ELU) activations after each layer. Using a single RTX 3090 GPU, the three highly parallelised training stages took approximately 1.4 hours, 4.1 hours and 4.8 hours, respectively.
The policy operates at a frequency of 50 Hz, and the simulation runs at 200 Hz.
We performed all of the experiments on the Unitree Go1. During the hardware validations, we use a constant joint friction value of 0.04, joint damping of 0.01 Nm s rad
−
1
1
{}^{-1}
start_FLOATSUPERSCRIPT - 1 end_FLOATSUPERSCRIPT
and a constant latency of 30 ms.
V-B
Versatile jumping on flat ground
V-B
1
Forward jumping
Figure 6:
Real world (top) and simulation (bottom) execution of a forward jump. The yellow marker indicates the desired 60cm jumping distance.
(a)
Joint angles, velocities and torques for the front right (FR) leg during the 60cm forward jump. The flight phase for the hardware experiment is indicated by the yellow-shaded region.
(b)
Base angular and linear velocity during the 60cm forward jump. The flight phase for the hardware test is indicated in light yellow.
Figure 7:
Hardware and simulation quantitative results for the 60cm forward jump.
First, we evaluate the policy on a variety of forward jumps. Fig.
6
compares hardware and simulation motions of a 60cm forward jump while Fig.
7
presents the quantitative results. As can be seen, the real-world behaviour closely matches the simulated prediction. One noticeable deviation is in the peak torques at take-off - where the measured torques deviate from both the desired torques (computed by the PD control law using the desired joint angles) and the simulation torques. Besides, larger joint angles for the hip and thigh are measured upon landing in real-world tests, likely due to poor impact modelling in the simulation. Finally, the Euler angles show a slight variation between simulation and hardware. We believe that this mismatch is mainly due to the motor modelling inaccuracies, coupled with the weight of the additional mass on top of the robot, shifting its centre of mass. Despite these state deviations, the jumping distance is well-tracked, and the base velocity matches the expected behaviour, showing a good sim2real adaptation.
Figure 8:
Hardware results for a 90cm forward jump (top) and a 50cm
×
\times
×
30cm diagonal jump with desired yaw of
30
⁢
°
30
°
30\degree
30 °
(bottom).
We then tested the maximum distance it could jump across. Fig.
8
(top) illustrates a 90 cm forward jump, with the target landing point shown by the yellow marker.
Despite slipping on the soft pads as it lands, the robot recovers quickly, demonstrating its robustness against uncertainties
5
5
5
It is worth mentioning that we reward the position of the base upon landing, rather than the feet. As a result, in the trial, the base cleared the 90cm distance, but the rear left foot landed a bit behind.
. To the best of our knowledge, this is the largest jumping distance achieved by robots of similar size and similar actuators (see Table
III
).
TABLE III:
Maximal jump length comparison with state-of-the-art.
Method
[
19
]
[
21
]
[
30
]
[
22
]
[
23
]
[
28
]
Ours
Jump length [m]
0.2
0.26
0.5
0.6
0.7
0.8
0.9
V-B
2
Diagonal jumping
Fig.
8
(bottom) shows a diagonal jump of 50 cm x 30 cm with a desired yaw of
30
⁢
°
30
°
30\degree
30 °
. Both the landing position and yaw are tracked accurately.
Figure 9:
Tracking performance as a function of the desired X- and Y-axis jumping distances, with the error (in cm) shown by the colour gradient (left); and the tracking performance in terms of overall desired vs actual jumping distance (right). The environments that have been terminated (due to any non-foot collisions) are shown in red, and the black
45
⁢
°
45
°
45\degree
45 °
dashed line indicates the ideal tracking performance. Data is gathered from 8000 trials across the whole jumping range
x
∈
[
0
,
1.2
]
,
y
∈
[
−
0.3
,
0.3
]
formulae-sequence
𝑥
0
1.2
𝑦
0.3
0.3
x\in[0,1.2],y\in[-0.3,0.3]
italic_x ∈ [ 0 , 1.2 ] , italic_y ∈ [ - 0.3 , 0.3 ]
. Only 112 robots have been terminated, leading to a success rate of 98.6%.
Furthermore, we evaluated the policy across the whole jumping range in simulation, of which the success rate and tracking metrics are presented in Fig.
9
. As can be seen from the left plot, the tracking error is lowest for narrow jumps of forward distance up to 50 cm. As both the longitudinal and lateral distances increase, so does the final landing error. Interestingly, the majority of failed environments asymmetrically occur in the lower right corner of the plot. The right plot in Fig.
9
shows the same data but grouped by total desired distance vs actual achieved distance. We found that the data closely follow the
45
⁢
°
45
°
45\degree
45 °
line (i.e. ideal performance) for the smaller jumps with the gradient slowly decreasing after 50 cm.
V-C
Jumping onto/across rough terrain
We here evaluate how well the policy performs in the presence of environmental disturbances, despite not being trained on uneven or rough ground. In this section, we ran several experiments, including jumping with obstacles surrounding the robot, blindly jumping from and onto a box, and jumping from asphalt onto a soft grassy terrain.
As shown by the top two time-lapses in Fig.
10
, the policy enables robust jumping onto both soft and stiff objects that could (and did) slip under the feet of the robot. The third row demonstrates that the robot could jump from hard asphalt onto soft grass, despite training on flat ground only.
Figure 10:
Several experiments showcasing the robustness of the policy
π
I
⁢
I
subscript
𝜋
𝐼
𝐼
\pi_{II}
italic_π start_POSTSUBSCRIPT italic_I italic_I end_POSTSUBSCRIPT
to variations in the terrain: jumping across discrete hard and soft objects (rows 1 and 2), asphalt-to-grass jump (row 3), nine consecutive jump on grass (row 4).
Next, we tested the policy on a continuous jumping task, where a new command of a 40cm forward jump is given following each jump without resetting the robot states. As seen in the fourth row of Fig.
10
, the policy is robust enough to execute a jump from a variety of different initial states. Despite the fact that the soft ground causes some hip angle deviation upon landing, the robot was able to execute at least nine consecutive jumps.
V-D
Forward jumping with obstacles
Figure 11:
Jumping over a 5cm tall, 5cm wide obstacle (top row) and jumping onto a 10cm tall box (bottom).
To further demonstrate the versatility, we tested forward jumping with obstacles, using policy
π
I
⁢
I
⁢
I
subscript
𝜋
𝐼
𝐼
𝐼
\pi_{III}
italic_π start_POSTSUBSCRIPT italic_I italic_I italic_I end_POSTSUBSCRIPT
. To be brief, only two scenarios are presented here, including jumping over a 5 cm tall thin obstacle and landing on a 10 cm box. In the first task, the robot had to jump across 80 cm to avoid collision. As seen in the top row of Fig.
11
, the robot succeeded in jumping over the barrier and landed successfully. In the second case, the robot needed to leap over 70 cm while maintaining a large height. As a result, better performance was observed, considering that a shorter forward distance enabled the robot to achieve a larger height throughout the flight.
V-E
Ablation study
To better understand the effect of our curriculum, we compared our approach to several baselines in Fig.
12
:
•
No RSI
: Training Stage I without RSI, i.e. no height and upward velocity initialisation,
•
No curriculum
: Directly training Stage II without pre-training Stage I, but with RSI height and velocity initialisation. For fairness, we train this baseline for an additional 3k steps,
•
No curriculum and no RSI
: Same as above, but without any RSI.
(a)
(b)
Figure 12:
Mean reward throughout training for the (a) Stage I: Jumping in place, and (b) Stage II: Long-distance jump tasks.
As can be seen from Fig.
12a
, the RSI is required for learning the jumping-in-place task. Without it, the agent converges to a local optimum and fails to complete the task. Despite the overall high reward, it can be seen in Fig.
12b
that directly training the long-distance jump also results in an early convergence to a standing behaviour, which highlights the need for our curriculum strategy.
VI
Discussion and conclusion
In this work, we present a curriculum-based end-to-end deep reinforcement learning approach, capable of learning a variety of precise short- and long-distance jumps, while also reaching the desired yaw upon landing. Unlike many existing methods, we have achieved this through a single policy, without the need for reference trajectories and additional imitation rewards.
Furthermore, through domain randomisation, we successfully deployed the policy onto the real system and closely matched the expected behaviour from the simulation. The system was robust to the noisy sensor data, especially the foot contact sensors and the velocity state estimates. The jumps exhibited high accuracy, both in simulation and on the hardware, in terms of tracking the desired landing position and orientation. Furthermore, our policy achieved a 90 cm forward jump on the Unitree Go1 robot, a distance greater than those reported by other model- and learning-based controllers. We demonstrated additional outdoor tests, where the robot successfully performed nine consecutive jumps on soft grass, without previously encountering such environments in its training. In addition, we showed that simulating obstacles throughout training and conditioning the policy on their properties can enhance the mobility of the robot, allowing it to safely leap over or land on objects of up to 10cm.
When executing a long-distance jump, real animals exhibit a four-legged contact phase, followed by an upward pitch and pure rear-leg contact at take-off. During landing a mirrored behaviour is observed - the body is pitching downwards and contact is first gained with the front legs. Previous model-based control works
[
4
,
5
]
have manually incorporated this contact schedule into their optimisers. It would be interesting to investigate how such behaviour can be learned through DRL without supplying a reference trajectory, and validate its benefits compared to the style of jumping exhibited here.
References
[1]
Jemin Hwangbo et al.
“Learning agile and dynamic motor skills for legged robots”
In
Science Robotics
4.26
, 2019, pp. eaau5872
[2]
Takahiro Miki et al.
“Learning robust perceptive locomotion for quadrupedal robots in the wild”
In
Science Robotics
7.62
, 2022, pp. eabk2822
[3]
Ananye Agarwal, Ashish Kumar, Jitendra Malik and Deepak Pathak
“Legged Locomotion in Challenging Terrains using Egocentric Vision” arXiv:2211.07638 [cs, eess]
arXiv, 2022
DOI:
10.48550/arXiv.2211.07638
[4]
Quan Nguyen et al.
“Optimized Jumping on the MIT Cheetah 3 Robot”
In
International Conference on Robotics and Automation
, 2019, pp. 7448–7454
[5]
Chuong Nguyen and Quan Nguyen
“Contact-timing and trajectory optimization for 3d jumping on quadruped robots”
In
IEEE/RSJ International Conference on Intelligent Robots and Systems
, 2022, pp. 11994–11999
[6]
Marko Bjelonic et al.
“Offline motion libraries and online MPC for advanced mobility skills”
In
The International Journal of Robotics Research
41.9-10
SAGE Publications Sage UK: London, England, 2022, pp. 903–924
[7]
Jiatao Ding et al.
“Robust Jumping with an Articulated Soft Quadruped via Trajectory Optimization and Iterative Learning”
In
IEEE Robotics and Automation Letters
9.1
IEEE, 2023, pp. 255–262
[8]
Ashish Kumar, Zipeng Fu, Deepak Pathak and Jitendra Malik
“RMA: Rapid Motor Adaptation for Legged Robots” arXiv:2107.04034 [cs]
arXiv, 2021
DOI:
10.48550/arXiv.2107.04034
[9]
Nikita Rudin, David Hoeller, Philipp Reist and Marco Hutter
“Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning”
In
Conference on Robot Learning
, 2022, pp. 91–100
[10]
Joonho Lee et al.
“Learning quadrupedal locomotion over challenging terrain”
In
Science Robotics
5.47
, 2020, pp. eabc5986
[11]
Chenhao Li et al.
“Learning Agile Skills via Adversarial Imitation of Rough Partial Demonstrations”
In
Conference on Robot Learning
205
, 2022, pp. 342–352
[12]
Zhongyu Li et al.
“Robust and Versatile Bipedal Jumping Control through Multi-Task Reinforcement Learning”
In
CoRR
abs/2302.09450
, 2023
[13]
Guillaume Bellegarda, Chuong Nguyen and Quan Nguyen
“Robust Quadruped Jumping via Deep Reinforcement Learning”
In
CoRR
abs/2011.07089
, 2023
[14]
Yuni Fuchioka, Zhaoming Xie and Michiel Panne
“OPT-Mimic: Imitation of Optimized Trajectories for Dynamic Quadruped Behaviors”
In
International Conference on Robotics and Automation
, 2023, pp. 5092–5098
[15]
Fangzhou Yu et al.
“Dynamic Bipedal Maneuvers through Sim-to-Real Reinforcement Learning”
In
CoRR
abs/2207.07835
, 2022
[16]
Xue Bin Peng et al.
“Amp: Adversarial motion priors for stylized physics-based character control”
In
ACM Transactions on Graphics (ToG)
40.4
ACM New York, NY, USA, 2021, pp. 1–20
[17]
Alejandro Escontrela et al.
“Adversarial Motion Priors Make Good Substitutes for Complex Reward Functions”
arXiv, 2022
arXiv:
2203.15103
[18]
Eric Vollenweider et al.
“Advanced Skills through Multiple Adversarial Motion Priors in Reinforcement Learning”
arXiv, 2022
arXiv:
2203.14912
[19]
Laura M. Smith et al.
“Learning and Adapting Agile Locomotion Skills by Transferring Experience”
In
Robotics: Science and Systems XIX
, 2023
[20]
Zhiqi Yin, Zeshi Yang, Michiel Panne and KangKang Yin
“Discovering diverse athletic jumping strategies”
In
ACM Trans. Graph.
40.4
, 2021, pp. 91:1–91:17
DOI:
10.1145/3450626.3459817
[21]
Gabriel B. Margolis et al.
“Learning to Jump from Pixels”
In
Conference on Robot Learning
164
PMLR, 2021, pp. 1025–1034
[22]
Yuxiang Yang et al.
“Continuous Versatile Jumping Using Learned Action Residuals”
In
Annual Learning for Dynamics and Control Conference
, 2023, pp. 770–782
[23]
Yuxiang Yang et al.
“CAJun: Continuous Adaptive Jumping using a Learned Centroidal Controller”
arXiv, 2023
arXiv:
2306.09557
[24]
Nikita Rudin, Hendrik Kolvenbach, Vassilios Tsounis and Marco Hutter
“Cat-like Jumping and Landing of Legged Robots in Low-gravity Using Deep Reinforcement Learning”
In
IEEE Transactions on Robotics
38.1
, 2022, pp. 317–328
[25]
Francecso Vezzi et al.
“Two-Stage Learning of Highly Dynamic Motions with Rigid and Articulated Soft Quadrupeds”
arXiv, 2023
arXiv:
2309.09682
[26]
John Schulman et al.
“Proximal policy optimization algorithms”
In
arXiv preprint arXiv:1707.06347
, 2017
[27]
Zhaoming Xie, Hung Yu Ling, Nam Hee Kim and Michiel Panne
“ALLSTEPS: Curriculum-driven Learning of Stepping Stone Skills”
In
Computer Graphics Forum
39.8
, 2020, pp. 213–224
[28]
Xuxin Cheng, Kexin Shi, Ananye Agarwal and Deepak Pathak
“Extreme Parkour with Legged Robots”
arXiv, 2023
arXiv:
2309.14341
[29]
David Hoeller, Nikita Rudin, Dhionis Sako and Marco Hutter
“ANYmal Parkour: Learning Agile Navigation for Quadrupedal Robots”
arXiv, 2023
arXiv:
2306.14874
[30]
Ken Caluwaerts et al.
“Barkour: Benchmarking Animal-level Agility with Quadruped Robots”, 2023
[31]
Ziwen Zhuang et al.
“Robot Parkour Learning”
arXiv, 2023
arXiv:
2309.05665
[32]
Xue Bin Peng, Pieter Abbeel, Sergey Levine and Michiel Panne
“DeepMimic: example-guided deep reinforcement learning of physics-based character skills”
In
ACM Transactions on Graphics
37.4
, 2018, pp. 143:1–143:14