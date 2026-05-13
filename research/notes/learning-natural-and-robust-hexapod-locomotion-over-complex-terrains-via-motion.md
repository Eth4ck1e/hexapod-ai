---
title: Learning Natural and Robust Hexapod Locomotion over Complex Terrains via Motion
  Priors based on Deep Reinforcement Learning
id: learning-natural-and-robust-hexapod-locomotion-over-complex-terrains-via-motion
tags:
- legged-rl-budgets
- hexapod
- amp
- ppo
- sim-to-real
- training-budget
created: '2026-05-06T07:30:20.271886Z'
updated: '2026-05-06T07:33:20.983462Z'
source: https://arxiv.org/html/2511.03167
source_domain: arxiv.org
fetched_at: '2026-05-06T07:30:20.271886Z'
fetch_provider: builtin
status: draft
type: note
tier: institutional
content_type: paper
deprecated: false
summary: 'Liu et al. (SJTU, Nov 2025): PPO+AMP hexapod locomotion controller on a
  real PhantomX-class hexapod. AMP discriminator trained on trajectory-optimization
  priors (8.6 s of tripod gait data). Asymmetric Actor-Critic with terrain encoder
  (87-dim privileged state) + state estimator for sim-to-real. Training setup: 4096
  parallel robots in IsaacGym, 1000 steps/episode (20 s at 50 Hz), 50,000 episodes
  total, ~35 hours on single RTX 3090Ti GPU. Total env steps = 4096 x 1000 x 50000
  = 204.8B implied steps, but episodes use early termination so effective step count
  is lower. Reward: task tracking + AMP style score + penalty terms (z-vel, roll/pitch,
  torques, collisions, joint limits). Domain randomization includes joint stiffness
  [0.8-1.2]x, link mass [0.9-1.1]x, foot friction [0.1-2.5], payload [0-5 kg]. Successfully
  transfers zero-shot to real robot. Outperforms RMA, Concurrent, Baseline, MPC on
  velocity perturbation tolerance (ours +/-0.803 m/s vs RMA +/-0.738 vs Baseline +/-0.201).
  Paper explicitly does NOT decompose training into BC/RL/DR stages — single end-to-end
  PPO+AMP run.'
---

Learning Natural and Robust Hexapod Locomotion over Complex Terrains via Motion Priors based on Deep Reinforcement Learning
Learning Natural and Robust Hexapod Locomotion over Complex Terrains via Motion Priors based on Deep Reinforcement Learning
Xin Liu, Jinze Wu, Yinghui Li, Chenkun Qi
∗
, Yufei Xue, Feng Gao
This work was supported by the National Key Research and Development Plan (2021YFF0307900). All authors are with School of Mechanical Engineering, Shanghai Jiao Tong University, Shanghai, China. Email:
chenkqi@sjtu.edu.cn
∗
Corresponding Author
Abstract
Multi-legged robots offer enhanced stability to navigate complex terrains with their multiple legs interacting with the environment. However, how to effectively coordinate the multiple legs in a larger action exploration space to generate natural and robust movements is a key issue. In this paper, we introduce a motion prior-based approach, successfully applying deep reinforcement learning algorithms to a real hexapod robot. We generate a dataset of optimized motion priors, and train an adversarial discriminator based on the priors to guide the hexapod robot to learn natural gaits. The learned policy is then successfully transferred to a real hexapod robot, and demonstrate natural gait patterns and remarkable robustness without visual information in complex terrains. This is the first time that a reinforcement learning controller has been used to achieve complex terrain walking on a real hexapod robot.
I
INTRODUCTION
Hexapod robots, like their natural counterparts, are known for superior terrain adaptability and stability, with their quasi-static gait requiring minimal muscle output
[
Zhong7876812
]
. As a result, they have gained significant interest and application. However, developing a controller for natural gait and robust motion on complex terrains remains a challenge.
Previous research on hexapod robots often relies on static gaits, like crawling, which limits their ability to navigate challenging terrains quickly and reliably. Locomotion controllers for bipedal and quadrupedal robots focus on two main types: model-based, which rely on simplified environmental and robot dynamics, and model-free, which use data-driven approaches without explicit modeling.
Model-free deep reinforcement learning (DRL) algorithms have proven more robust in complex environments compared to model-based methods, leading to increased use of DRL in legged locomotion control
[
schilling2020decentralized
,
hwangbo2019learning
,
lee2020learning
,
wjz
,
ji2022concurrent
]
. Despite this, no DRL algorithm has yet been applied effectively to real hexapod robots for natural and robust locomotion in complex terrains. The challenge lies in the increased complexity due to more legs, making it harder for robots to generate natural gaits. This work presents a DRL-based method for hexapod motion control, incorporating motion priors, to enable robust, natural locomotion in challenging environments.
Figure 1:
The hexapod robot showcases its ability to achieve natural and robust locomotion across diverse terrains.
The main contributions are listed as follows:
1.
We produce motion data for the hexapod robot on flat terrain using trajectory optimization (TO). Subsequently, we trained a motion discriminator to assist the hexapod robot in achieving a natural and robust locomotion in challenging terrains.
2.
We propose an asymmetric DRL framework based on adversarial discriminator for training a motion controller and deploy it on a real hexapod robot to achieve blind locomotion in challenging terrains.
II
RELATED WORK
II-A
Locomotion Control Algorithms for Legged Robots
Researchers have studied legged robot motion control to enable adaptation to complex terrains. Model-based methods, such as model predictive control (MPC)
[
meduri2023biconmp
]
and whole body control (WBC)
[
mit2019wbc
]
, require simplification and modeling of robot dynamics and the environment. However, these approaches struggle with unstructured or unknown terrains, which can lead to optimization failures.
An alternative approach incorporates biological concepts like central pattern generators (CPG) into control methods to reduce task complexity
[
schilling2020decentralized
,
lele2020learning
]
. However, adjusting CPG parameters online in changing environments is challenging, often compromising stability, especially in dynamic or unknown conditions.
Recently, data-driven algorithms, particularly reinforcement learning (RL), have gained popularity for controlling bipedal and quadrupedal robots
[
hwangbo2019learning
,
lee2020learning
,
miki2022learning
,
Liu2024Skill
]
. These methods rely on proprioceptive sensors, like joint encoders and the IMU, offering a more robust solution for unstructured environments. However, robots with more legs face increased difficulty in learning natural and stable gaits due to a larger exploration space, making convergence harder and reward function design more complex.
II-B
Reinforcement Learning for Locomotion
RL controllers have proven effective for legged robots, especially quadrupeds, enhancing their motion capabilities and adaptability to complex terrains
[
hwangbo2019learning
,
lee2020learning
,
wjz
,
ji2022concurrent
]
.
[
hwangbo2019learning
]
used an actuator network to model actuator dynamics and ensure smooth transition from simulation to reality. Building on this,
[
lee2020learning
]
improved ANYmal’s robustness by training it on various terrains. Some RL controllers for bipedal robots, like Cassie, adjust reference motions from a pre-defined model-based controller
[
xie2018feedback
,
xie2020learning
]
, speeding up training but limiting motion flexibility and exploration.
Hexapod robots, with more points of contact with the ground, offer better stability and interaction with the terrain, allowing greater perception of terrain complexity. However, most RL research on hexapods focuses on crawling gaits
[
schilling2020decentralized
,
lele2020learning
]
, limiting agility and speed in complex environments.
[
azayev2020blind
]
proposed a scalable two-level framework for blind hexapod locomotion in complex environments using RL, training expert policies on discrete terrain distributions. However, this method has only been tested in simulations. Currently, no RL framework exists for real hexapod robots to learn natural, robust gaits for challenging terrains using only proprioception.
II-C
Motion Imitation Learning
Designing complex reward functions is laborious, especially for hexapod robots exploring higher-dimensional spaces. Achieving a natural, robust gait via meticulously crafted reward functions is challenging. Imitation learning offers an alternative: by imitating real animal motion or manually crafted animation data, learning can converge faster and achieve higher-quality performance
[
peng2018deepmimic
,
peng2020learning
]
. However, while this approach effectively replicates individual motion clips, it struggles to handle multiple reference motions with a single phase variable.
Adversarial Motion Priors (AMP)
[
peng2021amp
]
address this issue using a GAIL framework
[
ho2016generative
]
that builds an adversarial discriminator. The discriminator discerns whether state transition pairs
(
s
t
,
s
t
+
1
)
\left(s_{t},s_{t+1}\right)
come from prior data or the learned policy, guiding the agent toward the motion characteristics of the prior data.
This approach allows simulated agents to perform complex tasks while adopting motion styles from large, unstructured motion datasets
[
wjz
,
escontrela2022adversarial
,
vollenweider2022advanced
]
.
In this work, we employ a more general motion imitation approach based on adversarial imitation learning and construct an asymmetric reinforcement learning network. This enables it to be trained using privileged information in simulation, relying solely on proprioceptive sensors for zero-shot generalization to the real hexapod robot without the need for fine-tuning. This allows our hexapod robot to exhibit similar behavior to a raw motion dataset on flat terrain without motion clips and to adapt to challenging terrains.
III
LEARNING FROM MOTION PRIORS
Figure 2:
The asymmetric Actor-Critic reinforcement learning framework. We formulate three types of rewards to facilitate tripod gait styles. The style-specific reward is given by the discriminator of adversarial motion priors. During deployment, the desired joint position calculated by summing the policy output with the default joint position is sent to the CSP controller to calculate the torque.
We consider a discrete-time dynamic model. At each time step
t
t
, the state is
𝒙
t
\boldsymbol{x}_{t}
. An action
𝒂
t
\boldsymbol{a}_{t}
is taken according to the policy, leading to the next state
𝒙
t
+
1
\boldsymbol{x}_{t+1}
with probability
P
​
(
𝒙
t
+
1
∣
𝒙
t
,
𝒂
t
)
P\left({{\boldsymbol{x}_{t+1}}\mid{\boldsymbol{x}_{t}},{\boldsymbol{a}_{t}}}\right)
and yielding a reward
r
t
r_{t}
. The goal of RL is to learn a policy parameterized by
θ
\theta
, denoted
π
θ
{\pi_{\theta}}
, that maximizes the discounted cumulative return:
J
​
(
θ
)
=
𝔼
π
θ
​
(
∑
t
=
0
∞
γ
t
​
r
t
)
J\left(\theta\right)=\mathbb{E}_{\pi_{\theta}}\left({\textstyle\sum_{t=0}^{\infty}}\gamma^{t}r_{t}\right)
.
Our controller does not use exteroreception, so the robot cannot obtain terrain data from cameras or radars. Consequently, the problem is modeled as a partially observable markov decision process (POMDP). We employ an asymmetric Actor-Critic framework
[
asymmetricAC
]
to train the controller: the Critic has full access to the state (including terrain and privileged robot data), while the Actor can only access partial observations from proprioceptive sensors.
Observation and Action Space:
As shown in Fig.
2
, the Actor and Critic receive different inputs, reflecting their asymmetric roles. The Critic’s input includes comprehensive state observations for evaluating the Actor’s actions: proprioceptive data
𝒐
t
p
∈
ℝ
42
\boldsymbol{o}_{t}^{p}\in{\mathbb{R}^{42}}
, the previous action
𝒂
t
−
1
∈
ℝ
18
\boldsymbol{a}_{t-1}\in{\mathbb{R}^{18}}
, the target base velocity
𝒗
t
des
=
(
v
x
,
v
y
,
ω
z
)
∈
ℝ
3
\boldsymbol{v}_{t}^{\rm des}=\left(v_{x},v_{y},\omega_{z}\right)\in{\mathbb{R}^{3}}
, privileged state data
𝒔
t
p
∈
ℝ
42
\boldsymbol{s}_{t}^{p}\in{\mathbb{R}^{42}}
, and terrain elevation scanning points
𝒊
t
e
∈
ℝ
187
\boldsymbol{i}_{t}^{e}\in{\mathbb{R}^{187}}
. Proprioceptive data consists of the robot’s angular velocities
𝝎
t
∈
ℝ
3
\boldsymbol{\omega}_{t}\in{\mathbb{R}^{3}}
, gravity vector projection
𝒆
g
∈
ℝ
3
\boldsymbol{e}_{g}\in{\mathbb{R}^{3}}
, joint positions
𝜽
t
∈
ℝ
18
\boldsymbol{\theta}_{t}\in{\mathbb{R}^{18}}
, and joint velocities
𝜽
˙
t
∈
ℝ
18
\boldsymbol{\dot{\theta}}_{t}\in{\mathbb{R}^{18}}
. Privileged state data includes base velocity
𝒗
t
∈
ℝ
3
\boldsymbol{v}_{t}\in{\mathbb{R}^{3}}
, base height
h
b
∈
ℝ
{h}_{b}\in{\mathbb{R}}
, ground friction
f
n
∈
ℝ
{f}_{n}\in{\mathbb{R}}
, foot contact forces
𝒇
c
∈
ℝ
18
\boldsymbol{f}_{c}\in{\mathbb{R}^{18}}
, external perturbation and its direction
𝒇
p
∈
ℝ
6
\boldsymbol{f}_{p}\in{\mathbb{R}^{6}}
, and collision states of the trunk, thighs, and calves
𝕀
c
∈
ℝ
13
{\mathbb{I}}_{c}\in{\mathbb{R}^{13}}
, which are less directly measurable. Terrain information is collected from multiple surrounding points, indicating vertical displacement from the robot’s base.
By contrast, the Actor’s input is limited to proprioceptive data, the previous action, and the target base velocity. The policy action
𝒂
t
\boldsymbol{a}_{t}
is an 18-dimensional vector specifying a target joint position offset. This offset is added to the nominal joint position
𝒒
0
\boldsymbol{q}_{0}
, which remains constant, to determine the desired motor position
𝒒
d
\boldsymbol{q}_{d}
. The following low-level joint CSP control law then computes torques:
𝝉
=
𝑲
p
​
2
​
(
𝑲
p
​
1
​
(
𝒒
d
−
𝒒
)
−
𝒒
˙
)
\boldsymbol{\tau}=\boldsymbol{K}_{p2}\left(\boldsymbol{K}_{p1}\left(\boldsymbol{q}_{d}-\boldsymbol{q}\right)-\boldsymbol{\dot{q}}\right)
.
Reward Design:
Designing reward functions for hexapod robots can be challenging and requires expert tuning. When using rewards from quadrupeds, hexapods typically fail to develop the tripod gait. To address this, we design a reward with three components: a task tracking reward
r
t
g
r^{g}_{t}
, a penalty
r
t
l
r^{l}_{t}
, and a tripod-style reward
r
t
s
r^{s}_{t}
. Their sum forms the total reward
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
r_{t}=r^{g}_{t}+r^{s}_{t}+r^{l}_{t}
.
The task reward emphasizes accurate tracking of linear and angular velocities. The penalty promotes motion stability, smoothness, and safety. Specifically, penalties are applied to the body’s vertical velocity and roll/pitch angular velocities to maintain stability. Excessive joint torque and acceleration are penalized to reduce motor stress and conserve energy. The rate of action change is penalized for smooth motion. Joint torque and velocity exceeding thresholds are penalized to prevent hardware overload. Collisions and contact forces are penalized to encourage minimal collisions and prevent excessive body damage. The tripod-style reward, based on adversarial motion priors, encourages adopting a tripod gait on diverse terrains (see Section
III-B
for details). Table
I
provides the specific reward functions and their scales.
TABLE I:
Reward terms for task tracking, style, and penalty.
Term
Annotation
Equation
Task
r
g
r^{g}
Linear velocity
1
∗
exp
⁡
(
‖
𝐯
t
,
x
​
y
−
𝐯
t
,
x
​
y
des
‖
2
/
0.15
)
1*\exp\left(\|\mathbf{v}_{t,xy}-\mathbf{v}_{t,xy}^{\rm des}\|_{2}\big/0.15\right)
Angular velocity
0.5
∗
exp
⁡
(
‖
ω
t
,
z
−
ω
t
,
z
des
‖
2
/
0.15
)
0.5*\exp\left(\|{\omega}_{t,z}-{\omega}_{t,z}^{\rm des}\|_{2}\big/0.15\right)
Style
r
s
r^{s}
D Score
1
∗
max
⁡
[
0
,
1
−
0.25
​
(
d
t
score
−
1
)
2
]
1*\max\left[0,1-0.25\left(d_{t}^{\rm score}-1\right)^{2}\right]
Penalty
r
l
r^{l}
Linear velocity
−
1
∗
v
t
,
z
2
-1*{v}_{t,z}^{2}
Angular velocity
−
0.08
∗
‖
𝝎
t
,
x
​
y
‖
2
-0.08*\|\boldsymbol{\omega}_{t,xy}\|_{2}
Joint torque
−
2
​
e
−
6
∗
‖
𝝉
‖
2
-2e^{-6}*\|\boldsymbol{\tau}\|_{2}
Joint acceleration
−
1.5
​
e
−
7
∗
‖
𝐪
¨
‖
2
-1.5e^{-7}*\|\mathbf{\ddot{q}}\|_{2}
Action rate
−
0.01
∗
‖
𝐚
t
−
𝐚
t
−
1
‖
2
-0.01*\|\mathbf{a}_{t}-\mathbf{a}_{t-1}\|_{2}
Collisions
−
0.05
∗
n
c
​
o
​
l
​
l
​
i
​
s
​
i
​
o
​
n
-0.05*n_{collision}
Joint torque limits
−
0.05
∗
‖
max
⁡
(
|
𝝉
t
|
−
𝝉
l
​
i
​
m
​
i
​
t
,
0
)
‖
2
-0.05*\|\max\left(\left|\boldsymbol{\tau}_{t}\right|-\boldsymbol{\tau}^{limit},0\right)\|_{2}
Joint velocity limits
−
0.5
∗
‖
max
⁡
(
|
𝒒
˙
t
|
−
𝒒
˙
l
​
i
​
m
​
i
​
t
,
0
)
‖
2
-0.5*\|\max\left(\left|\boldsymbol{\dot{q}}_{t}\right|-\boldsymbol{\dot{q}}^{limit},0\right)\|_{2}
Contact force
−
0.1
∗
‖
max
⁡
(
|
𝐟
t
|
−
𝐟
l
​
i
​
m
​
i
​
t
,
0
)
‖
2
-0.1*\|\max\left(\left|\mathbf{f}_{t}\right|-\mathbf{f}^{limit},0\right)\|_{2}
We randomize dynamic parameters for both robots and environments to reflect differences between real and simulated conditions. This enhances policy robustness and smooth transfer from simulation to the real world. Details of the parameter randomization are listed in Table
II
.
TABLE II:
The range of the randomized parameters.
Parameters
Range
Unit
Joint Stiffness
[0.8, 1.2]
×
\times
100
-
Joint Damping
[0.8, 1.2]
×
\times
2
-
Joint Position
[0.6, 1.4]
×
\times
nominal value
rad
Link Mass
[0.9, 1.1]
×
\times
nominal value
Kg
Payload Mass
[0, 5]
Kg
Payload Position
[-0.15, 0.15] relative to base position
m
Foot Friction
[0.1, 2.5]
-
Motor Strength
[0.8, 1.2]
-
III-A
Motion Priors Generation
The tripod gait is common in hexapod arthropods and is crucial for challenging terrain. To equip our hexapod robot with a high-quality tripod gait, we generate a motion dataset
𝒟
\mathcal{D}
on flat ground using TO (see Fig.
2
), which is the most cost-effective way to obtain prior motion data. The resulting trajectories last 8.6 seconds and cover forward, backward, lateral, steering, and combined motions, each maintaining a consistent gait cycle. This ensures the motion data fully corresponds to both the simulated robot and the demonstrator, avoiding extra retargeting
[
peng2018deepmimic
]
.
Each state
𝒔
t
A
​
M
​
P
∈
ℝ
61
\boldsymbol{s}_{t}^{AMP}\in{\mathbb{R}^{61}}
includes joint positions, joint velocity, base linear and angular velocity, base height relative to the terrain, and foot heights in the base frame. State transitions drawn from
𝒟
\mathcal{D}
serve as real samples for discriminator training.
III-B
Tripod Style Reward Based on Motion Priors
The style-specific reward promotes a tripod gait similar to the
𝒟
\mathcal{D}
while leaving the robot free to traverse challenging terrain (i.e., it does not force strict imitation). Tripod mode, common in hexapods, ensures the center of gravity remains within the triangular support domain, balancing stability and flexibility. Following
[
peng2021amp
]
, we train a discriminator
D
φ
D_{\varphi}
with parameters
φ
\varphi
to classify whether each state transition
T
s
=
(
𝒔
t
,
𝒔
t
+
1
)
{T}_{s}=(\boldsymbol{s}_{t},\boldsymbol{s}_{t+1})
is from the prior dataset or generated by the robot’s policy. If the discriminator detects a difference, it assigns a lower reward, indicating the robot has yet to learn the tripod style. As training progresses, the robot’s transitions become indistinguishable from the prior data, resulting in a higher reward.
The discriminator’s objective is:
arg
⁡
min
φ
𝔼
T
s
∼
𝒟
​
[
(
D
φ
​
(
T
s
)
−
1
)
2
]
+
𝔼
T
s
∼
π
​
[
(
D
φ
​
(
T
s
)
+
1
)
2
]
+
α
g
​
p
2
[
∥
∇
φ
D
φ
(
T
s
)
∥
2
]
T
s
∼
𝒟
,
\begin{split}\mathop{\arg\min}\limits_{\varphi}&\mathbb{E}_{{T}_{s}\sim\mathcal{D}}\left[\left(D_{\varphi}({T}_{s})-1\right)^{2}\right]+\mathbb{E}_{{T}_{s}\sim\pi}\left[\left(D_{\varphi}({T}_{s})+1\right)^{2}\right]\\
&+\frac{{{\alpha^{gp}}}}{2}{{}_{{T}_{s}\sim{\cal D}}}\left[\left\|{{\nabla_{\varphi}}{D_{\varphi}}({T}_{s})}\right\|_{2}\right],\end{split}
(1)
where the first two terms use a least square GAN formulation to minimize the Pearson divergence between transitions from
π
\pi
and
𝒟
\mathcal{D}
. To stabilize training, a gradient penalty is introduced in the second term
[
peng2021amp
]
, controlled by
α
g
​
p
\alpha^{gp}
. The tripod style reward is then defined as:
r
t
s
​
[
T
s
∼
π
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
​
(
D
φ
​
(
T
s
)
−
1
)
2
]
,
r_{t}^{s}\left[{T}_{s}\sim\pi\right]=\max\left[0,1-0.25\left({D_{\varphi}}({T}_{s})-1\right)^{2}\right],
(2)
and is scaled to the range
[
0
,
1
]
\left[0,1\right]
.
IV
NETWORK DESIGN AND TRAINING
IV-A
Network Architecture
We establish an asymmetric Actor-Critic RL framework: the Critic network receives privileged data and terrain details via two encoders to evaluate the current policy’s actions, while the Actor network relies solely on observable measurements (velocity commands, previous actions, and proprioceptive observations) for deployment. We encode terrain information
𝒊
t
e
\boldsymbol{i}_{t}^{e}
into a 16-dimensional latent variable
𝒍
t
e
\boldsymbol{l}_{t}^{e}
using a terrain encoder
g
e
g_{e}
, and encode privileged data
𝒔
t
p
\boldsymbol{s}_{t}^{p}
into an 8-dimensional latent variable
𝒍
t
p
\boldsymbol{l}_{t}^{p}
using a privileged encoder
g
p
g_{p}
. A three-layer Critic MLP then processes these latent representations and the observable data to produce target values
V
t
V_{t}
for advantage estimation.
Because it is difficult to obtain accurate linear velocity on real robots, we introduced a state estimator within the Actor network that computes linear velocity from the last five proprioceptive observations
𝒐
t
−
N
+
1
p
,
…
,
𝒐
t
−
1
p
,
𝒐
t
p
,
(
N
=
5
)
\boldsymbol{o}_{t-N+1}^{p},...,\boldsymbol{o}_{t-1}^{p},\boldsymbol{o}_{t}^{p},(N=5)
. We also designed a short-term memory encoder to compress these past observations into a latent variable
𝒉
t
\boldsymbol{h}_{t}
, allowing the robot to infer terrain characteristics from its history. The observable variables, estimated velocity, and the latent representation of past states are then passed to a low-level MLP, which produces the policy action
𝒂
t
\boldsymbol{a}_{t}
.
The discriminator
D
φ
D_{\varphi}
is a simpler network with two hidden layers and a linear output. Further details can be found in Table
III
.
TABLE III:
Network architecture for RL training framework.
Module
Inputs
Hidden Layers
Outputs
Estimator (MLP)
O
t
−
4
p
,
…
,
O
t
p
{O}_{t-4}^{p},...,{O}_{t}^{p}
[64, 32]
v
^
t
\hat{v}_{t}
Memory (MLP)
O
t
−
4
p
,
…
,
O
t
p
{O}_{t-4}^{p},...,{O}_{t}^{p}
[512, 256, 128]
h
t
h_{t}
Low-Level (MLP)
c
​
m
​
d
,
a
t
−
1
,
o
t
p
,
v
^
t
,
h
t
cmd,a_{t-1},o_{t}^{p},\hat{v}_{t},h_{t}
[256, 128, 64]
a
t
a_{t}
g
p
g_{p}
(MLP)
s
t
p
s_{t}^{p}
[64, 32]
l
t
p
l_{t}^{p}
g
e
g_{e}
(MLP)
i
t
e
i_{t}^{e}
[256, 128]
l
t
e
l_{t}^{e}
Critic (MLP)
c
​
m
​
d
,
a
t
−
1
,
o
t
p
,
l
t
p
,
l
t
e
cmd,a_{t-1},o_{t}^{p},l_{t}^{p},l_{t}^{e}
[512, 256, 128]
V
t
V_{t}
D
φ
D_{\varphi}
(MLP)
s
t
A
​
M
​
P
,
s
t
+
1
A
​
M
​
P
s_{t}^{AMP},s_{t+1}^{AMP}
[1024, 512]
d
t
score
d_{t}^{\rm score}
IV-B
Training
We train the policy using Proximal Policy Optimization (PPO)
[
schulman2017proximal
]
with privileged state and terrain data. At the start of each episode, the robot receives random velocity commands
𝒗
t
des
\boldsymbol{v}_{t}^{\rm des}
,
representing longitudinal, lateral, and yaw velocities. Following the terrain curriculum
[
wjz
]
, the yaw velocity is provided directly for efficient tracking. The policy network estimates the robot’s linear velocity
𝒗
^
t
\hat{\boldsymbol{v}}_{t}
through supervised learning using privileged information (see Fig.
2
).
We update the discriminator and policy networks concurrently. Specifically, we randomly extract state transition pairs
𝑻
s
p
=
(
𝒔
t
p
,
𝒔
t
+
1
p
)
\boldsymbol{T}_{s}^{p}=\left(\boldsymbol{s}_{t}^{p},\boldsymbol{s}_{t+1}^{p}\right)
from prior data, while the policy generates its own pairs
𝑻
s
π
=
(
𝒔
t
π
,
𝒔
t
+
1
π
)
\boldsymbol{T}_{s}^{\pi}=\left(\boldsymbol{s}_{t}^{\pi},\boldsymbol{s}_{t+1}^{\pi}\right)
.
The discriminator
D
φ
D_{\varphi}
evaluates these pairs and outputs
D
φ
​
(
T
s
)
{D_{\varphi}}({T}_{s})
, which is used to compute the
r
t
s
r^{s}_{t}
. The policy learns the prior motion style by generating actions that deceive the discriminator, which is updated simultaneously to better distinguish between the prior data and the agent’s behavior.
V
SIMULATIONS AND EXPERIMENTS
Simulation:
We created the terrains in the IsaacGym and trained 4096 robots simultaneously
[
rudin2022learning
]
. Each episode involved 1000 steps over 20 seconds, with early termination if the condition was met. The policy ran at a control frequency of
50
Hz
50\text{\,}\mathrm{Hz}
. We conducted 50,000 episodes, and the training took about 35 hours on a NVIDIA RTX 3090Ti GPU.
Hardware:
Our hexapod robot has a symmetrical design with six legs: right front (RF), right middle (RM), right rear (RR), left front (LF), left middle (LM), and left rear (LR). Each leg has three degrees of freedom, including the hip, thigh, and shank joints. To prevent leg collisions and increase the support area, the middle legs are extended 13.7 cm outward compared to the front and rear legs. The robot weighs 25.5 kg and stands 30 cm tall.
V-A
Ablation Study for the Design of the Reward Terms
To ascertain the necessity of each type of reward term, we trained three policies considering different combinations of the rewards, including
r
t
g
+
r
t
s
r^{g}_{t}+r^{s}_{t}
,
r
t
g
+
r
t
l
r^{g}_{t}+r^{l}_{t}
, and
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
r^{g}_{t}+r^{s}_{t}+r^{l}_{t}
.
We first analyzed the locomotion behavior of the three policies on flat terrain. Fig.
3
compares the tracking performance of the policies on flat ground with velocity commands in simulation. Fig.
3
(a), (b), and (c) show the performance of the policies on sinusoidal velocity commands in the x and y directions and yaw angular velocity. The results show that the policy guided by
r
t
g
+
r
t
l
r^{g}_{t}+r^{l}_{t}
exhibits significant jitter and deviation in velocity tracking, leading to unnatural behavior, as seen in Fig.
3
(h).
Fig.
3
(d), (e), and (f) compare the stability of the policies in the z-direction linear velocity and roll and pitch angles. The severe deviation of the curve guided by
r
t
g
+
r
t
l
r^{g}_{t}+r^{l}_{t}
shows that without the style reward, the policy fails to suppress movement in unexpected directions. This suggests that the style reward
r
t
s
r^{s}_{t}
helps the policy learn behaviors that better capture the reference tripod gait.
Figure 3:
Comparison of three policies in terms of ability to track sinusoidal velocity commands in the simulation. (a)-(c) Base velocity tracking in x, y, yaw directions. (d)-(f) Base velocity deviations in z-axis, and orientation deviations along the x, y axes. (g) Locomotion guided by
r
t
g
+
r
t
s
r^{g}_{t}+r^{s}_{t}
. (h) Locomotion guided by
r
t
g
+
r
t
l
r^{g}_{t}+r^{l}_{t}
. (i) Locomotion guided by
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
r^{g}_{t}+r^{s}_{t}+r^{l}_{t}
.
Next, we compared the traversability of the three policies across various terrains. In Fig.
4
, the vertical axis shows the terrain difficulty, and the horizontal axis represents iterations. The results show that the policies guided by
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
r^{g}_{t}+r^{s}_{t}+r^{l}_{t}
and
r
t
g
+
r
t
s
r^{g}_{t}+r^{s}_{t}
enable the robot to navigate more difficult terrains faster and reach higher levels. Specifically, for challenging terrains like stairs, the policies with
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
r^{g}_{t}+r^{s}_{t}+r^{l}_{t}
and
r
t
g
+
r
t
s
r^{g}_{t}+r^{s}_{t}
perform better than the
r
t
g
+
r
t
l
r^{g}_{t}+r^{l}_{t}
policy, as seen in Fig.
4
and
4
. This suggests that relying solely on task rewards and penalties may lead to abnormal behavior, limiting traversal of complex terrain. The style reward helps the robot learn more natural behaviors and explore its motion capabilities. Additionally, the policy with
r
t
g
+
r
t
s
r^{g}_{t}+r^{s}_{t}
performs better in the early stages, but
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
r^{g}_{t}+r^{s}_{t}+r^{l}_{t}
helps the robot navigate more difficult terrain in later stages, as shown in Fig.
4
.
Figure 4:
Comparison of policies in terms of ability to travel different terrains.
V-B
Robustness Experiments
We also trained and compared other advanced RL-based controllers and OCS2-based MPC controllers. We assessed each controller’s performance on flat terrain and its ability to navigate challenging terrains. These controllers are all blind locomotion controllers that rely on proprioception, including:
1.
Baseline
[
rudin2022learning
]
:
A policy trained without privileged access to information about the environment.
2.
Concurrent
[
ji2022concurrent
]
:
A policy is trained without terrain information as input for the actor network, and concurrently trained with an estimator network that estimates the body state.
3.
RMA
[
kumar2021rma
]
:
A policy trained using a teacher-student framework without any expert priors.
4.
MPC
[
FARSHIDIAN7989016
]
:
An MPC controller based on OCS2 fine-tunes the leg lift height, body height, and gait.
We used 5 random seeds and the same low-level network.
To test robustness, we applied random disturbances to the robot on flat ground in simulation. Specifically, we applied velocity perturbations along the three coordinate axes, ranging from small to large magnitudes. These perturbations affected the robot’s center of mass position at one-second intervals until a termination condition was met. Table
IV
shows the maximum velocity disturbances each controller could handle without causing the robot to fall. Results show the robot is least disturbance‑tolerant along the Y‑axis; the table below gives its Y‑axis tolerance range.
TABLE IV:
The controllers’ velocity disturbance tolerance range.
Controllers
Disturbances [Min, Max] (m/s)
Ours
[
-0.803
,
0.803
]
RMA
[-0.738, 0.738]
Concurrent
[-0.463, 0.463]
Baseline
[-0.201, 0.201]
MPC
[-0.112, 0.112]
The results showed controllers could regain stability after disturbances. Exceeding the threshold caused a loss of control, highlighting the different robustness among the controllers. Notably, our method demonstrated superior robustness, handling larger disturbances better than the others.
V-C
Indoor and Outdoor Experiments
As shown in Fig.
1
, we tested the robot on stairs ranging from 3 cm to 20 cm in height and on slopes with gradients from 5° to 30°. The robot moved at 0.3 m/s for 10 s. Success was defined as completing the tasks—ascending/descending stairs or traversing slopes—without falling. We conducted 10 tests for each controller and calculated the success rate.
As shown in Fig.
5
(a)-(c), our controller successfully navigated all terrains. RMA can access terrain information during teacher policy training, allowing some adaptation to terrains. However, its fixed low-level network updates limit adaptability to more complex terrains. The asymmetric Actor-Critic method addresses this by continuously updating the low-level network. Additionally, RL controllers trained with Baseline, Concurrent, or the MPC controller struggled to adapt to complex terrains without terrain information.
Figure 5:
Success rates of different controllers in different terrains
In the outdoor test, we navigated the robot at 0.5 m/s across a flower bed with a 15 cm step and over approximately 46 m of uneven grassland, as shown in Fig.
1
. Success was defined as crossing the flower bed without falling. We conducted 10 tests for each controller and calculated the success rate. As shown in Fig.
5
, our controller consistently outperformed the others. This demonstrates its ability to adapt to soft, uneven grass terrain, not encountered in simulation, with the memory encoding network’s terrain inference helping the controller adjust to complex terrain.
VI
CONCLUSIONS
In this paper, we propose a novel approach that combines motion priors with reinforcement learning (RL) algorithms. An RL controller is trained with an adversarial discriminator using these motion priors. This method enables the hexapod robot to perform natural and robust blind locomotion in complex terrains. Simulations and experiments show that the learned policy transfers successfully to the real robot, demonstrating natural gaits and strong robustness without visual input in challenging environments.