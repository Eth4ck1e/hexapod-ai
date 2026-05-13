---
title: 'Towards bridging the gap: Systematic sim-to-real transfer for diverse legged
  robots'
id: towards-bridging-the-gap-systematic-sim-to-real-transfer-for-diverse-legged-robo
tags:
- legged-rl-budgets
- quadruped
- sim-to-real
- ppo
- training-budget
- energy-efficiency
- multi-platform
created: '2026-05-06T07:31:01.370242Z'
updated: '2026-05-06T07:35:46.725925Z'
source: https://arxiv.org/html/2509.06342v1
source_domain: arxiv.org
fetched_at: '2026-05-06T07:31:01.369242Z'
fetch_provider: builtin
status: draft
type: note
tier: institutional
content_type: paper
deprecated: false
summary: 'ETH Zurich Robotic Systems Lab 2025 paper presenting PACE (Physics-grounded
  Actuator Calibration and Energy-efficient locomotion), a systematic sim-to-real
  pipeline for quadrupedal locomotion across 13+ platforms (ANYmal, Tytan, Minimal,
  NAO, Aibo, GR-1, etc.). Core method: bottom-up actuator parameter identification
  using CMA-ES optimization on 4096 parallel simulated environments from ~20 in-air
  data trajectories per robot (~20 seconds encoder data each), without torque sensing.
  PPO training hyperparameters (Table 7): 30,000 iterations, 4096 parallel environments,
  24 steps per environment per update, 5 mini-batches, 5 learning epochs per iteration.
  Total environment steps per training run = 30,000 × 4096 × 24 = ~2.95 billion steps.
  No dynamics randomization used (PACE identification substitutes for it). No wall-clock
  training time reported for policy training. Hardware: single NVIDIA GeForce RTX
  3080 for PACE parameter optimization (CMA-ES). Isaac Gym used for identification.
  Result: 32% reduction in Cost of Transport for ANYmal (CoT = 1.27 vs 1.86 for prior
  SOTA). Reward function uses only 4 terms, replacing complex multi-term hand-tuned
  rewards. No BC pretraining stage — trains policy from scratch using identified parameters.'
---

Towards bridging the gap: Systematic sim-to-real transfer for diverse legged robots
\corrauth
Filip Bjelonic, filipb@leggedrobotics.com
Towards bridging the gap: Systematic sim-to-real transfer for diverse legged robots
Filip Bjelonic
1
1
affiliationmark:
Fabian Tischhauser
1
1
affiliationmark:
and Marco Hutter
1
1
affiliationmark:
1
1
affiliationmark:
Robotic Systems Lab, ETH Zurich, Zurich, Switzerland
Abstract
Legged robots must achieve both robust locomotion and energy efficiency to be practical in real-world environments. Yet controllers trained in simulation often fail to transfer reliably, and most existing approaches neglect actuator-specific energy losses or depend on complex, hand-tuned reward formulations.
We propose a framework that integrates sim-to-real reinforcement learning with a physics-grounded energy model for permanent magnet synchronous motors. The framework requires a minimal parameter set to capture the simulation–reality gap and employs a compact four-term reward with a first-principle-based energetic loss formulation that balances electrical and mechanical dissipation.
We evaluate and validate the approach through a bottom-up dynamic parameter identification study, spanning actuators, full-robot
in-air
trajectories and
on-ground
locomotion. The framework is tested on three primary platforms and deployed on ten additional robots, demonstrating reliable policy transfer without randomization of dynamic parameters. Our method improves the energetic efficiency over state-of-the-art methods, achieving a
\qty
32 reduction in the full Cost of Transport of
\robot
ANYmal (
1.27
1.27
).
All code, models, and datasets will be released.
keywords:
Legged robots, Quadrupedal locomotion, Reinforcement learning, Simulation-to-real transfer, Reality gap, Energy efficiency
(a)
Real
(b)
Sim (ours)
(c)
Delta phase portrait
Figure 1
:
Comparison of real and simulated robot trajectories under a 0.1 –
2.0
Hz
2.0\text{\,}\mathrm{Hz}
chirp input. (a) Picture of the real robot with an overlaid trace showing its motion sequence from experiment. (b) Equivalent view in simulation using our modeling approach. In both (a) and (b), the robot’s base is suspended so that its legs swing freely. (c) Phase portrait of the “delta” joint state—i.e., at each timestep, the real joint position minus the simulated joint position (x-axis) and the real joint velocity minus the simulated joint velocity (y-axis)—plotted separately for three modeling approaches: no model (light gray), state-of-the-art actuator network (blue), and our proposed method (orange). Points nearer to (0, 0) indicate that the simulated trajectory more closely matches the real trajectory.
Legged robots promise versatile mobility in environments that are inaccessible to wheeled or tracked platforms. However, achieving robust and efficient locomotion remains a central challenge: Controllers trained in simulation only show the same performance on the real system if the system and actuator dynamics are accurately modeled. Modeling errors, in particular in the actuator dynamics, where electrical and mechanical losses are often not properly modeled, lead to major inefficiencies and hence directly affect robot endurance and payload capacity.
Prior work has addressed the simulation-to-reality gap through extensive domain randomization, residual modeling, or full physics identification. While these methods have demonstrated impressive sim-to-real transfer, they often require specialized sensors, exhaustive parameter identification, or iterative expert tuning of heuristic models and reward functions. As a result, current RL controller pipelines often rely on complex, high-dimensional reward functions. As modeling errors increase the sim-to-real gap, environment design often resorts to repeatedly adjusting rewards based on observed discrepancies in the real system—an ad-hoc process that can be repeated many times without clear structure, and which further disconnects rewards from physical notions such as efficiency.
This work introduces a framework that addresses these challenges by combining a sim-to-real RL pipeline with systematic actuator modeling. We perform a bottom-up analysis: first characterizing single actuator models, then estimating dynamic parameters on the full-system level, and finally validating locomotion control across multiple platforms.
A focus of our framework is the integration of a physics-grounded energy model for
\acp
pmsm, which enables training of locomotion policies that are both transferable and energy-efficient.
The main contributions of this work are:
1.
Sim-to-real pipeline:
An open-source modeling pipeline that integrates actuator and system models into RL training (see Section
2
).
2.
Bottom-up performance analysis:
Multi-level evaluation from single actuators to full-robot locomotion, including comparisons with state-of-the-art black-box approaches (see Section
3.2
).
3.
Cross-platform validation:
Deployment on three primary platforms (
\robot
ANYmal,
\robot
Tytan,
\robot
Minimal) and over ten additional systems (see Section
3.3
).
4.
Energetic assessment:
Quantitative analysis of electrical and mechanical efficiency, demonstrating improvements over previous methods (see Section
3.3.3
).
Additional contributions include:
•
A joint position control strategy with position saturation for hardware protection during locomotion (see Section
2.3.2
).
•
Empirical evidence that only four reward terms suffice for effective locomotion training (see Section
2.3.3
and
4.2
).
•
A physics-grounded energetic reward that balances electrical and mechanical losses to minimize overall consumption (see Section
2.3.3
).
All code, models, and raw data underlying the results, including those for most figures and tables, will be released.
1
Related Work
Our manuscript focuses on state-of-the-art methods for rigid robots actuated by
\acfp
pmsm, which are the dominant choice in modern legged robots due to their efficiency and controllability (see Figure
2
for a few prominent examples
hutter2017anymal
;
unitree2025motorsdk
;
aractingi2023controlling
;
katz2018low
;
liu2024diablo
). For clarity, we note that while “
\acf
bldc” is often used in industry as a generic label for brushless motors, it formally refers to machines with trapezoidal back-EMF driven by six-step commutation, whereas most legged robots employ sinusoidal
\acp
pmsm with field-oriented control
derammelaere2016quantitative
;
gamazo2010position
.
1.1
Modeling
Modeling approaches for sim-to-real transfer can be broadly categorized by the trade-off between physical prior knowledge and the amount of real-world data required. At one end, simplified physics models and domain randomization require little real-world data but assume accurate rigid-body dynamics. Residual physics approaches combine moderate priors with learned corrections, needing additional data to capture unmodeled effects. At the other end, full dynamics models minimize assumptions but demand extensive real-world interaction to learn complete system dynamics. Both model-based and purely data-driven approaches are represented across this spectrum.
Low-data approaches.
These methods rely mostly on rigid-body dynamics from the robot description. Cassie SysID combined with domain randomization
xie2020learning
, hand-calibrated parameters with randomized variations
li2024learning
, and classical dynamics randomization for reinforcement learning
bellegarda2022robust
or imitation learning
peng2020learning
fall into this category. Large-scale randomization has been shown effective for dexterous manipulation (DexTreme
handa2023dextreme
). Variants include active domain randomization
mehta2020active
, probabilistic approaches such as BayesSim
ramos2019bayessim
, adversarial domain randomization
shi2024rethinking
, and methods like DROPO
tiboni2023dropo
.
Moderate-data approaches.
Residual models combine physics priors with learned corrections, often focusing on actuator or body-level dynamics. Neural augmentations of simulators capture uncertainties
ajay2018augmenting
, while adversarial techniques such as SimGAN identify simulation parameters
jiang2021simgan
. Neural-Augmented Simulation (NAS) introduces recurrent residual dynamics
golemo2018sim
.
Joint-centric methods include actuator networks that leverage joint torque sensors
hwangbo2019learning
, unsupervised actuator models without torque sensing
fey2025bridging
, drive-limit estimations on
\robot
Spot
miller2025high
, and delta-action models through real-world rollouts (ASAP
he2025asap
).
Body-centric methods aim to identify physical parameters at the system level, for instance via sampling-based active exploration (SPI-Active
sobanbabu2025sampling
) or learning discrepancies from human demonstrations (DROID
tsai2021droid
).
Hybrid strategies combine low-level SysID with residual dynamics models, such as aerodynamic compensation for the floating robot BALLU
sontakke2023residual
.
High-data approaches.
Full dynamics-based methods attempt to learn the complete robot dynamics with little or no reliance on physical priors. DayDreamer
wu2023daydreamer
learns world models from scratch, updating them through real-world rollouts while replaying the best policy. Offline world models similarly combine simulation-initialized dynamics with real-world fine-tuning
li2025offline
. Other approaches estimate grounded forward and inverse dynamics transformations
hanna2021grounded
to enhance simulator accuracy.
Online adaptation.
To cope with discrepancies that remain during deployment, online adaptation mechanisms are employed. Strategies include online fine-tuning
smith2022legged
, meta-learning for rapid adaptation
song2020rapidly
, and student–teacher schemes for online parameter identification
lee2020learning
.
For comprehensive surveys on sim-to-real, see Muratore et al.
muratore2022robot
or Ju et al.
ju2022transferring
.
While prior work has successfully employed domain randomization
xie2020learning
;
handa2023dextreme
or residual dynamics
hwangbo2019learning
;
fey2025bridging
, these approaches often require specialized sensors or exhaustive parameter searches. Our method instead introduces a minimal set of parameters that compactly capture the simulation–reality gap, placing it within the class of moderate-data approaches. We further show that actuator drive dynamics are largely linear (
q
^
→
q
\hat{q}\rightarrow q
), which enables fast optimization of a concise and physically interpretable parameter set that transfers across platforms.
1.2
Control
Control of legged robots in complex terrain has been approached with both Model Predictive Control (MPC) and
\ac
rl
grandia2023perceptive
;
miki2022learning
;
xue2024full
;
rudin2025parkour
. Current state-of-the-art emphasizes
\ac
rl, often based on
\ac
ppo
schulman2017proximal
and its constrained derivatives, such as IPO
liu2020ipo
or TRPO-based formulations for locomotion
kim2024not
.
Partial observability.
Real-world control is inherently a partially observable Markov decision process (POMDP). Compared to fully observable MDPs, learning under partial observability is more challenging, as policies must cope with state aliasing and incomplete information
kaelbling1998planning
. In practice, standard RL methods often exhibit degraded performance or unstable training when policies are restricted to partial observations
hausknecht2015deep
;
pinto2017asymmetric
. Several strategies have been proposed to mitigate these challenges, including large-scale parallelized RL training
rudin2022learning
, teacher–student distillation
lee2020learning
, and asymmetric actor–critic architectures where the critic has privileged state access
pinto2017asymmetric
.
Reward design.
Reward shaping is a central challenge, as locomotion policies often involve more than ten hand-crafted reward terms
lee2020learning
;
miki2022learning
;
ji2022concurrent
;
shin2023actuator
, making tuning difficult and requiring expert heuristics. Constraint-based formulations can reduce the dimensionality of reward terms
kim2024not
, but in practice, they shift part of the tuning complexity to the choice and scaling of constraints. Multi-morphology training similarly offloads tuning to central pattern generators
shafiee2024manyquadrupeds
. Other approaches attempt to automate reward design with large language models (Eureka
ma2023eureka
) or remove explicit rewards altogether by optimizing intrinsic objectives, such as DIAYN
eysenbach2018diversity
.
Most recent controllers
rudin2022learning
;
miki2022learning
achieve agility, relying on high-dimensional and ad hoc reward shaping. We instead formulate rewards directly from actuator energy losses, yielding a compact and physically meaningful objective. Our formulation only requires four reward terms, which reduces tuning complexity.
1.3
Energy-efficient locomotion
Energy-efficient control is critical for autonomous legged robots.
Total losses consist of controller-independent terms such as sensing, computation, and inverter switching, and motion- respectively control-policy-dependent losses, which can be optimized. The literature on modeling motion-dependent energy losses can be grouped into pseudo-approximations, low-fidelity models, and high-fidelity models.
Pseudo-approximations.
Simplified proxies such as squared torque are frequently used to approximate energy costs. However, many works omit mechanical power consumption altogether
lee2020learning
;
shin2023actuator
;
miki2022learning
;
ji2022concurrent
;
kim2024not
, or introduce objectives with weak correlation to actual energy, e.g., penalizing changes in power
shafiee2024manyquadrupeds
.
Low-fidelity models.
These methods typically split energy into mechanical power and Joule heating
yang2022fast
. Scaling between these terms is often hand-tuned or derived from motor characteristics
wensing2017proprioceptive
;
fadini2021computational
, enabling training of energy-aware locomotion controllers
aractingi2023controlling
;
roux2025constrained
.
High-fidelity models.
Advanced approaches numerically compute individual loss components, including copper losses (
P
Cu
P_{\mathrm{Cu}}
), iron losses (
P
FE
P_{\mathrm{FE}}
), permanent magnet losses (
P
PM
P_{\mathrm{PM}}
), and mechanical dissipation
ferrari2022flux
. Recent data-driven work has employed accurate power measurement to train neural networks that predict instantaneous power consumption in legged robots
valsecchi2024accurate
.
Existing energy models range from simplified proxies such as torque-squared penalties
lee2020learning
;
kim2024not
to computationally intensive numerical loss calculations
ferrari2022flux
. Building on prior actuator-aware formulations
wensing2017proprioceptive
;
fadini2021computational
, we propose a loss model tailored to
\acp
pmsm that captures the dominant sources of energy dissipation while remaining tractable within modern reinforcement learning simulators. This balance enables training of locomotion policies that are explicitly energy-efficient.
1.4
Notation
We adopt the following notation conventions throughout the paper:
•
Scalars are written normal
(
x
)
(x)
, vectors in bold
(
𝒙
)
(\boldsymbol{x})
.
•
Target values are denoted with a hat
(
x
^
)
(\hat{x})
.
•
All vector norms are Euclidean,
∥
𝐱
∥
=
∥
𝐱
∥
2
\lVert\mathbf{x}\rVert=\lVert\mathbf{x}\rVert_{2}
.
•
The term
motor
refers to the actuator input side, and
joint
refers to the output side.
•
Reduced inertia with respect to the joint is indicated with a tilde
(
I
~
)
(\tilde{I})
.
•
Robot legs are indexed LF (left front), RF (right front), LH (left hind), RH (right hind).
(a)
\robot
ANYmal D
(b)
\robot
B2
(c)
\robot
Solo
(d)
\robot
Mini-Cheetah
(e)
\robot
DIABLO
Figure 2
:
Representative legged robots with explicitly documented use of
\ac
pmsm for actuation.
2
Method
Figure 3
:
Overview of the proposed
\ac
pace pipeline with policy training. (i) Collection of real in-air data on a fixed-base setup (top left). (ii) Evolutionary parameter fitting of joint dynamics to align simulated and measured trajectories (top right). (iii) Blind policy training in simulation with zero-shot deployment on hardware (bottom).
Our pipeline to bridge the reality gap comprises three stages, illustrated in Figure
3
.
(i)
Data collection on the real robot
(Section
2.1
), using joint impedance (PD) control to execute and record trajectories.
(ii)
Simulator alignment via evolutionary parameter identification
(Section
2.2
), where we fit a compact set of parameters so that simulated in-air trajectories match the recorded ones.
(iii)
Policy learning and deployment
(Section
2.3
), where we train a blind locomotion controller and deploy it zero-shot on hardware.
We refer to the overall approach as
\acf
pace.
2.1
Data collection
Data collection is conducted with a fixed base to avoid motion cross-coupling between different limbs.
We excite all joints simultaneously using chirp signals of
\qtyrange
2060 per sequence, sweeping from a minimum to a maximum frequency. Signals and measurements are time-synchronized and logged at high sampling rates (typically
\qtyrange
40010000). We found the following conditions to work best.
Fixed base (no base motion)
During simulation replay we rigidly fix the base in air.
For robots with symmetry planes (e.g., two for
\robot
ANYmal/
\robot
Minimal, one for
\robot
Tytan), we cancel net base wrenches by commanding symmetric joint trajectories.
No contacts (legs free in air).
We avoid all contacts (including inter-leg) so the identification is not confounded by unmeasured external forces. Moreover, in stance the base inertia dominates the effective joint dynamics; collecting in-air data isolates leg/drive dynamics. A simplified model in Appendix
B
shows that in agile stance the effective base inertia exceeds leg/drive inertias by 1–2 orders of magnitude (
\robot
ANYmal example).
Excitation bandwidth.
Ideally, trajectories cover up to
f
policy
/
2
f_{\text{policy}}/2
(Nyquist of the control policy), since this is the highest possible frequency that the controller can excite in the system. Structural constraints may limit this (e.g.,
\qty
2 on
\robot
ANYmal;
\qty
10 on
\robot
Tytan/
\robot
Minimal). If structural or practical constraints prevent reaching this range, the excitation should at least cover twice the highest frequency expected in the locomotion controller’s motion (e.g.,
\qty
1 for our robots walking at
\qty
1
\per
).
Joint-level PD gains.
Gains
P
τ
,
D
τ
P_{\tau},D_{\tau}
set the closed-loop poles of the joint tracking, cf. the transfer function
H
q
​
(
s
)
=
e
−
s
​
T
d
​
P
τ
I
a
​
s
2
+
(
d
+
D
τ
)
​
s
+
P
τ
,
\displaystyle H_{q}(s)\;=\;e^{-sT_{d}}\,\frac{P_{\tau}}{I_{a}s^{2}+(d+D_{\tau})s+P_{\tau}},
(1)
where
I
a
I_{a}
is the (armature/effective) inertia,
d
d
viscous damping, and
T
d
T_{d}
a (lumped) delay. High gains push poles to higher frequencies, demanding higher excitation bandwidth (often infeasible). We therefore use small gains in simulation for both identification and policy training—so that the characteristic dynamics (poles) appear at lower frequencies.
2.2
Parameter identification
We align the simulator by fitting a small set of parameters that dominantly shape the joint-space dynamics: per-joint armature/inertia
𝐈
a
\mathbf{I}_{a}
, viscous damping
𝐝
\mathbf{d}
, Coulomb friction
𝝉
f
\boldsymbol{\tau}_{f}
, and joint bias
𝐪
~
b
\tilde{\mathbf{q}}_{b}
, plus a global command delay
T
d
T_{d}
. With
n
n
actuated joints, the parameter vector is
𝐩
=
[
𝐈
a
,
𝐝
,
𝝉
f
,
𝐪
~
b
,
T
d
]
​
\tran
∈
ℝ
4
​
n
+
1
.
\displaystyle\mathbf{p}\;=\;[\mathbf{I}_{a},\;\mathbf{d},\;\boldsymbol{\tau}_{f},\;\tilde{\mathbf{q}}_{b},\;T_{d}]\tran\in\mathbb{R}^{4n+1}.
(2)
We instantiate
N
=
4096
N=4096
parallel environments with the real-experiment base pose, each with parameters
𝐩
e
\mathbf{p}_{e}
. We replay the recorded joint targets at the simulation rate used later for
\ac
rl, and measure the simulated joint trajectories
𝐪
i
,
e
sim
\mathbf{q}_{i,e}^{\text{sim}}
. The identification objective for environment
e
e
is the time-averaged mean-squared joint-position error:
ℓ
e
=
1
k
​
∑
i
=
1
k
‖
𝐪
i
real
−
𝐪
i
,
e
sim
‖
2
,
\displaystyle\ell_{e}\;=\;\frac{1}{k}\sum_{i=1}^{k}\|\mathbf{q}_{i}^{\text{real}}-\mathbf{q}_{i,e}^{\text{sim}}\|^{2},
(3)
yielding a loss vector
𝓛
∈
ℝ
N
\boldsymbol{\mathcal{L}}\in\mathbb{R}^{N}
. We optimize
𝐩
∗
=
\argmin
𝐩
​
𝔼
​
[
ℓ
e
]
,
\displaystyle\mathbf{p}^{*}\;=\;\argmin_{\mathbf{p}}\,\mathbb{E}[\ell_{e}],
(4)
using
\ac
cmaes
nomura2024cmaes
over the population. Evolutionary search is robust to local minima
telikani2021evolutionary
;
hansen2016cma
;
hansen2001completely
and works well at this moderate dimensionality (typically
≈
49
\approx 49
parameters for our robots).
We found
\ac
cmaes reliable and sample-efficient in massively parallel GPU simulation; with fewer samples, Bayesian optimization is an alternative.
Single-joint dynamics model.
For reference, a single joint obeys
I
a
​
q
¨
+
d
​
q
˙
=
τ
i
+
τ
comp
+
τ
f
,
\displaystyle I_{a}\ddot{q}+d\,\dot{q}\;=\;\tau_{i}+\tau_{\text{comp}}+\tau_{f},
(5)
where
τ
f
\tau_{f}
models Coulomb friction and
τ
comp
\tau_{\text{comp}}
denotes firmware-level compensations (e.g., cogging compensations, plant inversion, friction observers). Assuming tight current control and a saturation nonlinearity on commanded torque (cf. Figure
13
), a practical closed-loop form is
I
a
​
q
¨
+
d
​
q
˙
=
sat
​
(
P
τ
​
(
q
^
−
q
+
q
~
b
)
−
D
τ
​
q
˙
+
τ
comp
)
+
τ
f
.
\displaystyle I_{a}\ddot{q}+d\,\dot{q}\;=\;\mathrm{sat}\!\big{(}P_{\tau}(\hat{q}-q+\tilde{q}_{b})-D_{\tau}\dot{q}+\tau_{\text{comp}}\big{)}+\tau_{f}.
(6)
Drive-level simplifications (e.g., load-independent damping) are common but imperfect; actual damping and torque–current maps are often state dependent. Hence, fitting parameters against full-robot in-air data (rather than isolated drives; see Section
4.1.2
) is critical.
Finally, note a non-uniqueness if PD gains are co-optimized with the dynamics. Any common scaling
u
c
u_{c}
of
{
I
a
,
d
,
P
τ
,
D
τ
}
\{I_{a},d,P_{\tau},D_{\tau}\}
preserves (
6
) and trajectories:
u
c
​
I
a
∗
​
q
¨
+
u
c
​
d
∗
​
q
˙
=
u
c
​
P
τ
∗
​
Δ
​
q
−
u
c
​
D
τ
∗
​
q
˙
.
\displaystyle u_{c}I_{a}^{*}\ddot{q}+u_{c}d^{*}\dot{q}\;=\;u_{c}P_{\tau}^{*}\Delta q-u_{c}D_{\tau}^{*}\dot{q}.
(7)
We therefore
do not
include PD gains in the identification and assume these to be known.
2.3
Learning environment
Having addressed simulator alignment through parameter identification, we now turn to the learning-based control stage. Because
\ac
pace identifies the dynamics end-to-end, we
do not
use dynamics randomization. We randomize the
task
(pushes, ground friction) and terrains (flat/rough, stairs, boxes, slopes). We otherwise follow standard practice
rudin2022learning
, emphasizing only differences below.
2.3.1
Observations
We model the problem as a POMDP. Accordingly, we use asymmetric
\ac
ppo
pinto2017asymmetric
: the policy receives proprioception
𝐨
prop
⊂
𝐬
\mathbf{o}^{\text{prop}}\subset\mathbf{s}
, while the critic observes privileged state
𝐬
\mathbf{s}
.
Policy (proprioception).
Base linear velocity
𝐯
B
\mathbf{v}_{B}
, base angular velocity
𝝎
B
\boldsymbol{\omega}_{B}
, gravity in base frame
𝐠
B
\mathbf{g}_{B}
(from a state estimator); user commands
(
v
B
x
,
v
B
y
,
ω
B
yaw
)
(v_{B}^{x},v_{B}^{y},\omega_{B}^{\text{yaw}})
; joint positions
𝐪
\mathbf{q}
and velocities
𝐪
˙
\dot{\mathbf{q}}
; previous action
𝐚
t
−
1
\mathbf{a}_{t-1}
. We add i.i.d. noise to all but commands and
𝐚
t
−
1
\mathbf{a}_{t-1}
. The observation has dimension
𝐨
prop
∈
ℝ
48
\mathbf{o}^{\text{prop}}\in\mathbb{R}^{48}
.
Critic (privileged).
Noise-free
𝐨
prop
⁣
∗
\mathbf{o}^{\text{prop}*}
plus base wrench
(
𝐅
B
,
𝝉
B
)
(\mathbf{F}_{B},\boldsymbol{\tau}_{B})
, ground friction, binary foot contacts, and a height scan centered at the base CoM spanning
2
m
2\text{\,}\mathrm{m}
×
\times
3
m
3\text{\,}\mathrm{m}
at
0.15
m
0.15\text{\,}\mathrm{m}
resolution. Thus
𝐨
priv
∈
ℝ
305
\mathbf{o}^{\text{priv}}\in\mathbb{R}^{305}
and
𝐬
=
[
𝐨
prop
⁣
∗
,
𝐨
priv
]
∈
ℝ
353
\mathbf{s}=[\mathbf{o}^{\text{prop}*},\,\mathbf{o}^{\text{priv}}]\in\mathbb{R}^{353}
.
2.3.2
Actions and hard-limit safe PD control
The policy outputs joint position offsets
𝐚
t
=
π
​
(
𝐨
t
prop
)
\mathbf{a}_{t}=\pi(\mathbf{o}^{\text{prop}}_{t})
relative to a default posture
𝐪
0
\mathbf{q}_{0}
, which are converted to torques and tracked by a current loop (cf. Figure
14
):
𝝉
t
=
P
τ
​
(
𝐚
t
+
𝐪
0
−
𝐪
)
−
D
τ
​
𝐪
˙
.
\displaystyle\boldsymbol{\tau}_{t}\;=\;P_{\tau}\big{(}\mathbf{a}_{t}+\mathbf{q}_{0}-\mathbf{q}\big{)}\;-\;D_{\tau}\dot{\mathbf{q}}.
(8)
Before sending targets
𝐪
^
t
=
𝐚
t
+
𝐪
0
\hat{\mathbf{q}}_{t}=\mathbf{a}_{t}+\mathbf{q}_{0}
to the drives we apply a
joint-limit saturation
to guarantee zero commanded torque toward a hard limit while preserving motion away from it.
Let
𝒬
feas
\mathcal{Q}_{\text{feas}}
be the feasible set and
𝒬
soft
⊂
𝒬
feas
\mathcal{Q}_{\text{soft}}\subset\mathcal{Q}_{\text{feas}}
a soft-limit band. For joint
j
j
with soft/hard bounds
q
j
soft
,
q
j
hard
q_{j}^{\text{soft}},q_{j}^{\text{hard}}
,
q
^
j
=
{
q
^
j
−
q
j
−
q
j
soft
q
j
hard
−
q
j
soft
​
(
q
^
j
−
q
j
hard
)
,
q
j
∈
𝒬
soft
,
q
^
j
∉
𝒬
feas
,
q
^
j
,
otherwise
.
\displaystyle\hat{q}_{j}\;=\;\begin{cases}\hat{q}_{j}-\tfrac{q_{j}-q_{j}^{\text{soft}}}{q_{j}^{\text{hard}}-q_{j}^{\text{soft}}}\big{(}\hat{q}_{j}-q_{j}^{\text{hard}}\big{)},&q_{j}\!\in\!\mathcal{Q}_{\text{soft}},~\hat{q}_{j}\!\notin\!\mathcal{Q}_{\text{feas}},\\[6.0pt]
\hat{q}_{j},&\text{otherwise}.\end{cases}
(9)
At
q
j
=
q
j
hard
q_{j}=q_{j}^{\text{hard}}
this yields
q
^
j
=
q
j
hard
\hat{q}_{j}=q_{j}^{\text{hard}}
so the PD term toward the limit vanishes. The same logic applies near the lower bound. This saturation runs in both simulation and on hardware.
2.3.3
Rewards
Thanks to fitted dynamics parameters, we use only four terms: velocity tracking and energy (physics-based), plus two structural penalties (collisions and foot touchdown velocity).
Velocity tracking.
Following
lee2020learning
, we reward proximity to commanded base velocities:
r
v
=
exp
⁡
(
−
‖
𝐯
^
B
,
x
​
y
−
𝐯
B
,
x
​
y
‖
2
2
σ
v
)
+
exp
⁡
(
−
(
ω
^
B
,
yaw
−
ω
B
,
yaw
)
2
σ
v
)
.
\displaystyle\boxed{r_{v}\;=\;\exp\!\Big{(}-\tfrac{\|\hat{\mathbf{v}}_{B,xy}-\mathbf{v}_{B,xy}\|_{2}^{2}}{\sigma_{v}}\Big{)}+\exp\!\Big{(}-\tfrac{(\hat{\omega}_{B,\text{yaw}}-\omega_{B,\text{yaw}})^{2}}{\sigma_{v}}\Big{)}}.
(10)
Energy.
We combine electrical dissipation
P
el
P_{\text{el}}
with mechanical power
P
mech
P_{\text{mech}}
(including regeneration) and gravitational potential power
P
pot
P_{\text{pot}}
:
P
total
=
P
el
+
P
mech
+
P
pot
.
\displaystyle P_{\text{total}}\;=\;P_{\text{el}}+P_{\text{mech}}+P_{\text{pot}}.
(11)
Assuming dominant
q
q
-axis current
i
q
i_{q}
and negligible
i
d
i_{d}
,
P
el
\displaystyle P_{\text{el}}
=
∑
j
=
1
n
R
j
​
i
q
,
j
2
=
∑
j
=
1
n
τ
j
2
​
R
j
r
j
2
​
k
i
,
j
2
,
\displaystyle=\sum_{j=1}^{n}R_{j}i_{q,j}^{2}\;=\;\sum_{j=1}^{n}\tau_{j}^{2}\,\frac{R_{j}}{r_{j}^{2}k_{i,j}^{2}},
(12)
P
mech
\displaystyle P_{\text{mech}}
=
{
𝝉
⊤
​
𝐪
˙
,
𝝉
⊤
​
𝐪
˙
>
0
,
k
regen
​
𝝉
⊤
​
𝐪
˙
,
𝝉
⊤
​
𝐪
˙
<
0
,
\displaystyle=\begin{cases}\boldsymbol{\tau}^{\top}\dot{\mathbf{q}},&\boldsymbol{\tau}^{\top}\dot{\mathbf{q}}>0,\\
k_{\text{regen}}\,\boldsymbol{\tau}^{\top}\dot{\mathbf{q}},&\boldsymbol{\tau}^{\top}\dot{\mathbf{q}}<0,\end{cases}
(13)
P
pot
\displaystyle P_{\text{pot}}
=
∑
b
=
1
B
m
b
​
g
​
v
b
,
z
,
\displaystyle=\sum_{b=1}^{B}m_{b}g\,v_{b,z},
(14)
where
r
j
r_{j}
is the gear ratio,
k
i
,
j
k_{i,j}
the motor constant,
R
j
R_{j}
the coil resistance,
g
g
gravity, and
v
b
,
z
v_{b,z}
the center-of-mass velocity along
−
𝐠
-\,\mathbf{g}
. Because many robotic systems are black-box, Figure
22
reports the characteristic Joule heating scale from (
12
) for our in-house developed robots, providing a rough estimate across different platforms.
Since joint speeds scale with base speed, dissipation grows roughly with
‖
𝐯
^
B
‖
2
\|\hat{\mathbf{v}}_{B}\|^{2}
. We thus apply a velocity-dependent normalization,
γ
v
=
1
‖
𝐯
^
B
‖
2
2
+
1
,
\displaystyle\gamma_{v}\;=\;\frac{1}{\|\hat{\mathbf{v}}_{B}\|_{2}^{2}+1},
(15)
and define
r
e
=
γ
v
​
P
total
.
\displaystyle\boxed{r_{e}\;=\;\gamma_{v}\,P_{\text{total}}}.
(16)
Foot-touchdown (FTD) penalty.
To discourage braking by impacts (gear wear, noise), we penalize the maximum foot speed in a short history window upon touchdown. With buffer length
n
ftd
=
3
n_{\text{ftd}}=3
and foot
j
j
,
v
j
,
ftd
\displaystyle v_{j,\text{ftd}}
=
{
max
k
∈
{
t
−
2
,
t
−
1
,
t
}
⁡
‖
𝐯
j
,
k
foot
‖
,
if touchdown at
​
t
,
0
,
otherwise
,
\displaystyle=\begin{cases}\max_{k\in\{t-2,t-1,t\}}\|\mathbf{v}^{\text{foot}}_{j,k}\|,&\text{if touchdown at }t,\\
0,&\text{otherwise},\end{cases}
(17)
r
ftd
\displaystyle r_{\text{ftd}}
=
∑
j
#
​
feet
v
j
,
ftd
.
\displaystyle=\sum_{j}^{\#\text{feet}}v_{j,\text{ftd}}.
(18)
Collision penalty.
We penalize joint-limit and thigh–environment collisions via an indicator:
r
c
=
{
1
,
if collision at
​
t
,
0
,
otherwise
.
\displaystyle\boxed{r_{c}\;=\;\begin{cases}1,&\text{if collision at }t,\\
0,&\text{otherwise}.\end{cases}}
(19)
Penalty scheduling.
Early in training, large penalties can stall gait discovery. We therefore schedule energy and FTD penalties with an exponential factor
k
decay
=
e
−
λ
​
t
k_{\text{decay}}=e^{-\lambda t}
and
κ
=
1
−
k
decay
\kappa=1-k_{\text{decay}}
:
r
=
c
v
​
r
v
+
c
c
​
r
c
+
κ
​
(
c
e
​
r
e
+
c
ftd
​
r
ftd
)
.
\displaystyle r\;=\;c_{v}r_{v}+c_{c}r_{c}+\kappa\,(c_{e}r_{e}+c_{\text{ftd}}r_{\text{ftd}}).
(20)
We typically choose a half-life of 500 iterations for
λ
\lambda
, but this is task dependent.
2.3.4
Entropy scheduling
Exploration is essential early but harms precision later. We anneal the entropy coefficient with a smooth tanh schedule:
ℰ
​
(
t
)
\displaystyle\mathcal{E}(t)
=
ℰ
∞
+
ϵ
​
(
ℰ
0
−
ℰ
∞
)
,
\displaystyle=\mathcal{E}_{\infty}+\epsilon\,(\mathcal{E}_{0}-\mathcal{E}_{\infty}),
(21)
ϵ
\displaystyle\epsilon
=
1
2
−
1
2
​
tanh
⁡
(
η
​
(
t
−
T
ℰ
)
)
,
\displaystyle=\tfrac{1}{2}-\tfrac{1}{2}\tanh\!\big{(}\eta\,(t-T_{\mathcal{E}})\big{)},
(22)
so that
ℰ
​
(
0
)
≈
ℰ
0
\mathcal{E}(0)\!\approx\!\mathcal{E}_{0}
transitions to
ℰ
∞
\mathcal{E}_{\infty}
near
T
ℰ
T_{\mathcal{E}}
.
2.4
Remarks
Drive-side filtering.
Joint velocities are often low-pass filtered (cutoffs as low as
25
25\text{\,}
–
50
Hz
50\text{\,}\mathrm{Hz}
). Although
\ac
pace uses positions in the loss, filtered
q
˙
\dot{q}
still affects commanded PD torques. If noticeable, include this filter in simulation and identify its parameters.
Units of PD gains.
Non-
SI
\mathrm{S}\mathrm{I}
units still allow identification but break generalization across gains and skew energy terms (damping/ohmic vs. mechanical). Use consistent
SI
\mathrm{S}\mathrm{I}
units.
Further assumptions (empirically satisfied in Figure
4
).
•
Correct kinematics (URDF/USD, frames).
•
High-bandwidth current control or an LTI-approximable drive (closed-loop behavior is fitted).
•
Mild temperature dependence during data collection.
•
Nonlinearities modest in the excited range (or absorbed by fitted terms).
•
Sufficient structural stiffness in the excitation band (Section
2.1
).
Finally, we stress that adding parameters indiscriminately can harm identifiability. The compact set
{
𝐈
a
,
𝐝
,
𝝉
f
,
𝐪
~
b
,
T
d
}
\{\mathbf{I}_{a},\mathbf{d},\boldsymbol{\tau}_{f},\tilde{\mathbf{q}}_{b},T_{d}\}
proved sufficient across platforms; co-optimizing PD gains leads to non-unique optima (Eq. (
7
)).
3
Experiments
This section introduces the robotic platforms and the experimental protocols. We evaluate
\ac
pace in two stages.
First, controlled in-air validation and evaluation experiments (Section
3.2
) establish methodological soundness at different levels of system complexity, ranging from a single drive to the fully suspended robot.
Second, on-ground locomotion experiments (Section
3.3
) apply the approach in realistic deployment scenarios, assessing performance and energetic efficiency.
We proceed bottom–up:
(In–air: i) single–actuator analysis on
\robot
Tytan (full access to firmware, electronics, and mechanics, Section
3.2.1
);
(In–air: ii) full–robot identification and validation, including a comparison on
\robot
ANYmal against a zero–model baseline and a state-of-the-art actuator network
hwangbo2019learning
(Section
3.2.2
);
(On–ground: iii) full–robot locomotion experiments (Sections
3.3.1
3.3.2
) under the same conditions as (In–air: ii) and
(On–ground: iv) long-duration energetic evaluations (Section
3.3.3
).
Each experimental stage has a distinct objective.
The single–drive study validates high-bandwidth motor torque tracking and mechanical identification under fully known conditions.
The
in-air
full–robot experiments test whether the approach scales to system level, generalizes across PD gains and trajectories, and allow benchmarking against a zero–model baseline and a learned actuator network
hwangbo2019learning
.
Finally, the
on–ground
locomotion trials demonstrate the practical applicability of
\ac
pace, evaluating tracking accuracy, energetic efficiency, and long–duration performance.
3.1
Robots
We perform analysis on three quadrupeds—
\robot
Tytan,
\robot
ANYmal, and
\robot
Minimal—chosen to span actuation types, scales, and transparency (open vs. closed source).
All three share the same leg topology: Hip Abduction–Adduction (HAA), Hip Flexion–Extension (HFE), and Knee Flexion–Extension (KFE), with point feet (rubber end caps).
All joints are
\ac
pmsm driven.
Key characteristics are summarized in Table
1
with
“?” marks indicate confidential specifications we are not permitted to disclose.
Figure 4
:
Primary robotic platforms evaluated in this study. Top row (left to right):
\robot
ANYmal,
\robot
Tytan, and
\robot
Minimal. Additional systems include Sony’s
\robot
Aibo, Softbank’s
\robot
NAO,
\robot
ALMA,
\robot
Spacehopper,
\robot
LEVA,
\robot
Magnecko v2 and v1, and Fourier’s
\robot
GR-1. All robots are depicted in operation using the proposed
\ac
pace parameter fitting.
\robot
Tytan
—a custom platform developed at ETH Zurich and based on
\robot
Barry
valsecchi2023barry
—uses pseudo-direct drives at HAA/HFE (fixed ratio
r
=
5.6
r\!=\!5.6
) and a variable-ratio ball-screw lever at KFE (
r
∈
[
0.8
,
9
]
r\in[0.8,9]
). It employs a low inertia leg design by moving the knee-motor towards its hip.
\robot
ANYmal
hutter2016anymal
employs series-elastic harmonic drives with high, fixed ratios. It serves as a closed-source testbed demonstrating applicability when only limited low-level access is available. Our method does
not
rely on joint-torque sensors, in contrast to the actuator network baseline.
\robot
Minimal
is a small, largely 3D-printed quadruped. All joints share a variable-ratio lever mechanism driven by T-Motor units and a lead-screw transmission. In total
\qty
76 of its total mass is situated in the base with only
\qty
6 at each leg.
Beyond these three primary platforms,
\ac
pace has also been deployed on a diverse set of robots, including
\robot
Aibo
watanabe2025learning
,
\robot
NAO,
\robot
ALMA
bellicoso2019alma
;
ma2025learning
,
\robot
Spacehopper
spiridonov2024spacehopper
,
\robot
LEVA
arnold2025leva
,
\robot
Magnecko v1
leuthard2024magnecko
/v2,
\robot
GR-1
he2025attention
, the
\robot
Allegro Hand, and the
\robot
Ability Hand (see Fig.
4
), as well as additional unpublished systems not shown here.
Table 1
:
Main robot characteristics.
Tytan
ANYmal
Minimal
Hips
Knee
Weight [kg]
52.3
52.3
52.8
52.8
4.2
4.2
Shoulder height [m]
0.62
0.62
0.55
0.55
0.25
0.25
Shoulder width [m]
0.22
0.22
0.20
0.20
0.16
0.16
Shoulder depth [m]
0.73
0.73
0.75
0.75
0.39
0.39
Thigh length [m]
0.40
0.40
0.30
0.30
0.15
0.15
Shank length [m]
0.37
0.37
0.38
0.38
0.16
0.16
Regen. coeff.
k
regen
k_{\text{regen}}
[-]
0.3
0.3
0.0
0.0
0.3
0.3
Bus voltage
u
u
[V]
48
48
48
48
18
18
Gear ratio
r
r
[-]
5.6
5.6
0.8
0.8
-
?
7.2
7.2
-
9
9
16
16
Max. joint torque [Nm]
140
140
28
28
-
89
89
2.9
2.9
-
315
315
6.4
6.4
Max. joint speed [rad/s]
16.8
16.8
3
3
-
8.5
8.5
45
45
-
36
36
99.4
99.4
Max. motor torque [Nm]
25
25
35
35
?
0.4
0.4
Max. motor speed [rad/s]
94
94
29
29
?
716
716
Motor constant
k
i
k_{i}
[Nm/A]
0.59
0.59
1.25
1.25
?
0.0252
0.0252
Coil resistance
R
R
[
Ω
\Omega
]
1.04
1.04
1.71
1.71
?
0.194
0.194
3.1.1
Data collection
On each system, we record multi-joint chirps with varying amplitudes. Typical sequence duration is
\qtyrange
2040 with
f
0
=
\qty
​
0.1
f_{0}=\qty{0.1}{}
and a maximum frequency of
\qty
10 to avoid excessive actuator stress. Structural constraints limit the achievable bandwidth on suspended setups (full
\robot
Tytan:
\qty
8;
\robot
ANYmal:
\qty
2; cf. Section
2.1
).
All full-robot logs (and low-level control) run at
\qty
400. We capture time-synchronized data via shared-memory logging using SignalLogger
anybotics_signal_logger
.
For additional validation on
\robot
Minimal, we also collect random joint steps at
\qty
2 update (every
\qty
0.5); we
do not
use this trajectory on larger machines to avoid transmission wear.
Unless noted otherwise, all
\ac
pace optimizations and simulations run on a single NVIDIA GeForce RTX 3080.
3.2
In-air evaluation and validation
(a)
Single-drive experimental setup for joint-level characterization.
(b)
Identified versus target inertia as a function of lever arm radius, with drive compensations disabled.
(c)
Full-robot setup on
\robot
Tytan for multi-joint data collection.
(d)
Trajectory replay on the LF HFE joint of
\robot
Tytan, comparing real and simulated joint positions.
Figure 5
:
Experimental setups and representative evaluation across abstraction levels.
The following experiments validate
\ac
pace under controlled, contact-free conditions.
The single-drive setup isolates mechanical effects and directly reveals the physical meaning of the identified inertia
I
a
I_{a}
and damping
d
d
.
The in-air full-robot experiments then test whether the approach scales when all actuators, electronics, and dynamics interact simultaneously.
Throughout, we concentrate on effective inertia
I
a
I_{a}
(mechanically verifiable); damping and friction are identified but not dissected further, as detailed tribology would require dedicated equipment.
3.2.1
Single drive
A single hip actuator is rigidly mounted to an aluminum frame (Figure
5(a)
). The motor shaft is vertical (gravity effects negligible). An aluminum interface attaches a discrete mass at known radii using a locking pin, enabling controlled changes of output inertia. The power supply is four
\qty
12 car-batteries in series (
\qty
48), peak
\qty
10.8
\kilo
; a
\qty
100 fuse limits to
\qty
4.8
\kilo
. The drive accepts up to
\qty
32 peak current. This setup emulates a perfect voltage source.
The drive’s output-reduced inertia comprises the rotor, sun-gear assembly, planet gears, and output shaft. Using d’Alembert’s principle, the combined reduced output inertia is
I
~
output
\displaystyle\tilde{I}_{\text{output}}
=
I
~
rotor
+
I
~
sun
+
I
~
planet
+
I
~
shaft
\displaystyle=\tilde{I}_{\text{rotor}}+\tilde{I}_{\text{sun}}+\tilde{I}_{\text{planet}}+\tilde{I}_{\text{shaft}}
=
(
1.14
+
0.511
+
0.00203
+
0.00146
)
×
10
−
2
​
kg
m
2
\displaystyle=(1.14+0.511+0.00203+0.00146)\times 10^{-2}\;$\mathrm{kg}\text{\,}{\mathrm{m}}^{2}$
≈
1.65
×
10
−
2
kg
m
2
,
\displaystyle\approx$1.65\text{\times}{10}^{-2}\text{\,}\mathrm{kg}\text{\,}{\mathrm{m}}^{2}$,
(23)
showing rotor and sun-assembly dominance; planets and output-shaft are
∼
10
3
\sim\!10^{3}
smaller.
The interface inertia is
I
~
interface
=
\qty
​
8.67
​
e
−
3
​
\squared
\tilde{I}_{\text{interface}}=\qty{8.67e-3}{\squared}
. The attachable mass has
m
=
\qty
​
503
m=\qty{503}{}
and own inertia about its CoM of
\qty
1.8e-4
\squared
. By varying the mounting radius of this mass, the joint inertia can be adjusted in the range
\qtyrange
​
1.656.27
​
e
−
2
​
\squared
\qtyrange{1.65}{6.27e-2}{\squared}
.
Current loop validation.
The goal of this experiment is to verify the motor drives’ ability to precisely track commanded currents
i
q
i_{q}
, and thereby commanded motor torques
τ
m
\tau_{m}
, over a wide frequency range.
With the interface removed (free spin), we command a current chirp (
\qty
1–
\qty
1250, amplitude
\qty
2,
\qty
25) and record measured currents at
\qty
10
\kilo
. This experiment does not serve parameter identification, but rather confirms high-bandwidth motor torque authority. Joint-level torque fidelity is addressed end-to-end by
\ac
pace.
Mechanical loop identification.
For single-drive identification, we fit three parameters: armature inertia
I
a
I_{a}
, viscous damping
d
d
, and friction
τ
f
\tau_{f}
. Joint bias is irrelevant (no gravity coupling), and the control/communication delay
T
d
T_{d}
is negligible (microseconds). We simulate in Isaac Gym a single-link, single-joint model (link inertia set to zero so
I
a
I_{a}
absorbs total inertia).
We collect chirps from
\qtyrange
0.110 at
\qty
2.5
\kilo
logging.
In total
30
30
experiments are conducted:
15
15
with nominal firmware feed-forward compensations enabled (cogging/friction; cf. Figure
14
) and
15
15
with compensations disabled. Each set spans five load cases (No load; Interface only; Interface+mass at
\qty
15.9
\centi
,
\qty
21.6
\centi
,
\qty
27.3
\centi
) and three PD configurations:
•
P
τ
=
\qty
​
60
​
\per
P_{\tau}=\qty{60}{\per}
,
D
τ
=
\qty
​
2
​
\per
D_{\tau}=\qty{2}{\per}
(locomotion default),
•
P
τ
=
\qty
​
145
​
\per
P_{\tau}=\qty{145}{\per}
,
D
τ
=
\qty
​
5
​
\per
D_{\tau}=\qty{5}{\per}
,
•
P
τ
=
\qty
​
250
​
\per
P_{\tau}=\qty{250}{\per}
,
D
τ
=
\qty
​
10
​
\per
D_{\tau}=\qty{10}{\per}
.
Each experiment is fitted independently (three parameters), totaling
∼
\qty
​
5
\sim\!\qty{5}{}
across all runs.
3.2.2
Full robot
At the system level we identify all parameters, including joint position biases and a global delay, with the base stationary and legs moving in air. We collect identification and validation sets on
\robot
Tytan,
\robot
ANYmal, and
\robot
Minimal.
For
\robot
Tytan, identification uses sinusoidal joint targets at
P
τ
=
\qty
​
60
​
\per
P_{\tau}\!=\!\qty{60}{\per}
,
D
τ
=
\qty
​
2
​
\per
D_{\tau}\!=\!\qty{2}{\per}
; validation repeats with
P
τ
=
\qty
​
145
​
\per
P_{\tau}\!=\!\qty{145}{\per}
,
D
τ
=
\qty
​
5
​
\per
D_{\tau}\!=\!\qty{5}{\per}
.
For
\robot
Minimal, we additionally validate on random joint steps (
\qty
0.5 dwell).
On
\robot
ANYmal, low-level gains are fixed; we therefore use the vendor defaults (matching the actuator-network training setup). The actuator-network baseline is the current ANYbotics model (LSTM-augmented, trained on a larger dataset) rather than the original 2019 version
hwangbo2019learning
.
3.3
On-ground locomotion analysis
Having validated
\ac
pace under controlled conditions, we now assess its performance in locomotion scenarios under ground contact.
These experiments evaluate full-robot deployment on hardware, testing whether the identified models enable robust tracking.
We further analyze the two platforms
\robot
Tytan and
\robot
ANYmal on long-duration trials. We quantify locomotion efficiency and decompose the total energy consumption into contributions from locomotion, inverter switching losses, compute and sensors overhead.
Using the identified parameters, we train blind locomotion policies as in Section
2.3
without dynamics randomization. Hyperparameters and robot specific parameters are listed in Table
7
and Table
2
respectively.
Table 2
:
Reward scales and PD gains used for locomotion.
\robot
Tytan
\robot
ANYmal
\robot
Minimal
Velocity tracking
0.2
0.2
0.2
0.2
0.2
0.2
Energy [
​
10
−
5
{10}^{-5}
]
−
16
-\,$16$
−
16
-\,$16$
−
128
-\,$128$
Collisions
−
1.0
-\,1.0
−
1.0
-\,1.0
−
1.0
-\,1.0
FTD
−
0.1
-\,0.1
−
0.1
-\,0.1
−
0.1
-\,0.1
P
τ
P_{\tau}
60
\,60
85
\,85
4
\,4
D
τ
D_{\tau}
2
\,2
0.6
\,0.6
0.05
\,0.05
3.3.1
Tytan
We train a single policy and compare real vs. simulation by replaying the real commanded base-velocity trajectory in open loop. The robot is manually driven within a
\qty
3
×
\times
\qty
3 area; total duration is
∼
\qty
​
90
\sim\!\qty{90}{}
(SignalLogger buffer limit).
(a)
Real-world deployment of
\robot
Tytan during continuous deployment.
(b)
Corresponding simulation rollout under identical base velocity targets and conditions.
(c)
Tracking performance. Top: commanded and measured forward velocity. Bottom: commanded and measured angular velocity around the yaw axis.
(d)
Joint torque tracking on the left front leg. Shown are policy-commanded and measured torques for the HAA, HFE, and KFE joints (left to right).
Figure 6
:
Sim-to-real evaluation of
\robot
Tytan using the proposed
\ac
pace framework.
3.3.2
ANYmal
We train three policies—(i) no model (URDF-only), (ii) actuator network
hwangbo2019learning
, and (iii)
\ac
pace—with identical rewards, no dynamics randomization, and the same environment. We compare step responses at
\qty
1
\per
(forward/sideways over
\qty
4) and yaw steps at
\qty
2
\per
.
(a)
Real-world deployment without any actuator model.
(b)
Real-world deployment with a learned actuator network
hwangbo2019learning
.
(c)
Real-world deployment with the proposed method.
(d)
Phase portrait of the LF HFE joint over multiple gait cycles.
(e)
Forward velocity tracking. Left: commanded and measured velocities over time. Right: distribution of steady-state velocity errors across methods.
Figure 7
:
Sim-to-real evaluation on
\robot
ANYmal. Panels (a–c) illustrate real-world deployments. Panel (d) shows the corresponding phase portraits. Panel (e) presents commanded and measured forward velocities (left) and steady-state velocity error distributions across methods (right).
3.3.3
Energetic running-track evaluations
(a)
Representative stills from one step cycle during running-track evaluation.
(b)
Battery state of charge (SoC) as a function of distance traveled.
(c)
Decomposition of the Cost of Transport (CoT).
(d)
Limit cycle of the
\robot
Tytan knee joint over a
10
s
10\text{\,}\mathrm{s}
window.
(e)
Frequency spectrum of the LF HFE joint during steady locomotion.
Figure 8
:
Running-track evaluations across robotic platforms.
(a) Representative stills from one step cycle.
(b) Battery consumption over distance (SoC).
(c) Cost of transport (CoT) with contributions from electronics (CoE, diagonal hatching) and drives’ inverter switching (CoD, horizontal hatching).
(d) Limit cycle of the
\robot
Tytan knee joint over a
10
s
10\text{\,}\mathrm{s}
window.
(e) Frequency spectrum of the LF HFE joint during steady locomotion.
With dynamics fitted by
\ac
pace, policies do not require dynamics randomization and thus exploit the machine dynamics. We therefore perform long-duration endurance tests on a standard
\qty
400 track, comparing to our prior actuator-network results
bjelonic2023learning
. The same
\robot
Tytan and
\robot
ANYmal policies from Secs.
3.3.1
and
3.3.2
are used.
All policies are also capable of traversing standard stairs (cf. Figure
3
).
Both platforms use the same battery type as in
bjelonic2023learning
, with capacity
E
B
=
\qty
​
907.2
E_{B}=\qty{907.2}{}
.
We decompose the classical cost of transport (CoT) into electronics (CoE), drives (CoD), and locomotion (CoL):
CoT
=
CoE
+
CoD
+
CoL
.
\displaystyle\mathrm{CoT}\;=\;\mathrm{CoE}+\mathrm{CoD}+\mathrm{CoL}.
(24)
To estimate resting powers, we repeat full-charge discharge tests with (i) robot on a crane, drives enabled but zero current target (
rest
), and (ii) robot on ground, drives off (
off
). With measured durations
t
i
t_{i}
and
\ac
soc windows
(
SoC
i
max
−
SoC
i
min
)
(\mathrm{SoC}_{i}^{\text{max}}-\mathrm{SoC}_{i}^{\text{min}})
, the average powers are
P
rest
\displaystyle P_{\text{rest}}
=
E
B
​
(
SoC
rest
max
−
SoC
rest
min
)
t
rest
,
\displaystyle=\frac{E_{B}\big{(}\mathrm{SoC}_{\text{rest}}^{\text{max}}-\mathrm{SoC}_{\text{rest}}^{\text{min}}\big{)}}{t_{\text{rest}}},
(25)
P
off
\displaystyle P_{\text{off}}
=
E
B
​
(
SoC
off
max
−
SoC
off
min
)
t
off
.
\displaystyle=\frac{E_{B}\big{(}\mathrm{SoC}_{\text{off}}^{\text{max}}-\mathrm{SoC}_{\text{off}}^{\text{min}}\big{)}}{t_{\text{off}}}.
(26)
For a locomotion experiment of duration
t
track
t_{\text{track}}
and distance
Δ
​
s
\Delta s
, the total CoT is
CoT
=
E
B
​
(
SoC
track
max
−
SoC
track
min
)
m
​
g
​
Δ
​
s
.
\displaystyle\mathrm{CoT}\;=\;\frac{E_{B}\big{(}\mathrm{SoC}_{\text{track}}^{\text{max}}-\mathrm{SoC}_{\text{track}}^{\text{min}}\big{)}}{mg\,\Delta s}.
(27)
Electronics and drives contributions are
CoE
\displaystyle\mathrm{CoE}
=
P
off
​
t
track
m
​
g
​
Δ
​
s
,
CoD
=
(
P
rest
−
P
off
)
​
t
track
m
​
g
​
Δ
​
s
,
\displaystyle=\frac{P_{\text{off}}\,t_{\text{track}}}{mg\,\Delta s},\qquad\mathrm{CoD}\;=\;\frac{\big{(}P_{\text{rest}}-P_{\text{off}}\big{)}\,t_{\text{track}}}{mg\,\Delta s},
(28)
and the locomotion term follows as
CoL
=
CoT
−
P
rest
​
t
track
m
​
g
​
Δ
​
s
.
\displaystyle\mathrm{CoL}\;=\;\mathrm{CoT}-\frac{P_{\text{rest}}\,t_{\text{track}}}{mg\,\Delta s}.
(29)
By construction, CoL captures mechanical and electrical losses due to motion (friction, damping, Joule heating); CoE and CoD capture compute/sensing and inverter switching overheads.
4
Results
We report identification progress, fitted parameters, and sim2real performance across
\robot
Tytan,
\robot
ANYmal, and
\robot
Minimal. Optimization traces (
\ac
cmaes scores per iteration) are shown in Figure
9
; final parameter sets appear in Table
5
. For
\robot
ANYmal, the same fitted set can be used across units, though hardware wear, varying firmware and changing simulator may introduce small residual gaps.
Figure 9
:
CMA-ES optimization score as a function of iteration (logarithmic scale) for the three main robotic platforms.
Across robots, same-type joints fit to similar armature
I
a
I_{a}
and damping
d
d
. Both
\robot
ANYmal and
\robot
Tytan identify a global delay of
\qty
7.5
\milli
.
\robot
ANYmal shows the largest fitted damping (order
\qty
5
\per
); pseudo-direct variants (
\robot
Tytan,
\robot
Minimal) fit substantially lower
d
d
. Despite higher damping on
\robot
ANYmal, fitted Coulomb-like friction is of similar order to
\robot
Tytan. With
N
=
4096
N\!=\!4096
environments, end-to-end optimization for each platform converges within
\qtyrange
1024, depending on trajectory length, GPU throughput, and parameter bounds.
4.1
In-air evaluation and validation
4.1.1
Single drive
Current loop validation.
From current-chirp data (
\qtyrange
11250,
\qty
2,
\qty
25) we estimate the motor inner-loop transfer
H
i
​
(
s
)
=
τ
m
/
τ
^
m
H_{i}(s)=\tau_{m}/\hat{\tau}_{m}
. Measured Bode plots (Figure
16
) match the analytical model up to
>
\qty
​
100
>\!\qty{100}{}
. Magnitude stays near
\qty
0 (unity gain) with
≤
\qty
​
5
\leq\!\qty{5}{}
phase lag up to
\qty
25; the
−
1
​
dB
-1\,$\mathrm{dB}$
point occurs near
\qty
50, and we observe a control bandwidth of
≈
\qty
​
346
\approx\!\qty{346}{}
or
\qty
0.58 of the pulse-width modulation frequency. A dead time of
≈
\qty
​
400
​
\micro
\approx\!\qty{400}{\micro}
is inferred from the phase slope. Above
∼
\qty
​
350
\sim\!\qty{350}{}
, signal-to-noise ratio limits the estimate.
Mechanical loop identification.
We fit
{
I
a
,
d
,
τ
f
}
\{I_{a},d,\tau_{f}\}
from position-chirp data (
\qtyrange
0.110,
\qty
2.5
\kilo
logging) over five load cases and three PD configurations (Section
3.2.1
). In the no-compensation firmware mode, fitted
I
a
I_{a}
agrees with analytic targets within
\qty
2 (rotor),
\qty
6 (interface), and
\qty
15 (added mass at radius); across PD settings, the standard deviation is
\qty
0.67,
\qty
3.7, and
\qty
14, respectively (Figure
5(b)
). One high-gain/high-mass trial failed mechanically (the lever broke) and is excluded.
With firmware compensations enabled, all fits exhibit an approximately constant offset
I
comp
=
\qty
​
8.1
​
e
−
3
​
\squared
I_{\mathrm{comp}}\!=\!\qty{8.1e-3}{\squared}
(virtual inertia, cf. Figure
17
).
Assuming unit inner-loop gain, the outer-loop single-joint transfer from
q
^
\hat{q}
to
q
q
matches the second-order model in Equation
1
. For the high-load, locomotion-gain case (
P
τ
=
\qty
​
60
​
\per
P_{\tau}\!=\!\qty{60}{\per}
,
D
τ
=
\qty
​
2
​
\per
D_{\tau}\!=\!\qty{2}{\per}
), the dominant complex conjugate yields a gain drop near
\qty
4.35 and
≈
−
14
​
dB
\approx\!-14\,$\mathrm{d}\mathrm{B}$
at
\qty
10 (cf. Figure
18
). The dominant pole pair lies within the frequency range spanned by the excitation trajectories.
4.1.2
Full robot
Representative in-air trajectory overlays for
\robot
Tytan show near-overlap between real measurements and simulated replay using fitted parameters (Figure
5(d)
). Validation at different gains (
P
τ
=
\qty
​
145
​
\per
,
D
τ
=
\qty
​
5
​
\per
P_{\tau}\!=\!\qty{145}{\per},D_{\tau}\!=\!\qty{5}{\per}
) preserves the match (Figure
15
).
A comparison against the single-drive baseline highlights a clear discrepancy.
For the LF–HFE joint of
\robot
Tytan, the fitted inertia is
I
a
=
\qty
​
10.6
​
e
−
2
​
\squared
I_{a}=\qty{10.6e-2}{\squared}
,
approximately four times higher than the expected
I
a
≈
\qty
​
2.5
​
e
−
2
​
\squared
I_{a}\approx\qty{2.5e-2}{\squared}
from rotor and compensation alone, while the fitted damping
d
=
\qty
​
0.17
​
\per
d=\qty{0.17}{\per}
remains consistent with expectation (cf. Table
5
).
On
\robot
Minimal, both chirps and random joint steps are reproduced closely; the URDF-only model deviates substantially (Appendix: Figure
20(a)
, Figure
20(b)
).
Comparison with the actuator network on ANYmal
Figure
1
compares in-air behavior for
\robot
ANYmal using: (i) no model (URDF-only), (ii) the actuator network
hwangbo2019learning
, and (iii)
\ac
pace. The trace overlays (stills) qualitatively match only for the latter two; the no-model case diverges. The delta phase portraits of LF–HAA show smallest
(
Δ
​
q
,
Δ
​
q
˙
)
(\Delta q,\Delta\dot{q})
with
\ac
pace, while the actuator network exhibits more variance and a joint position bias (
∼
\qty
​
5
\sim\!\qty{5}{}
). The deltas are calculated as
Δ
​
q
i
=
q
i
sim
−
q
i
real
\Delta q_{i}=q^{\mathrm{sim}}_{i}-q^{\mathrm{real}}_{i}
and
Δ
​
q
˙
i
=
q
˙
i
sim
−
q
˙
i
real
\Delta\dot{q}_{i}=\dot{q}^{\mathrm{sim}}_{i}-\dot{q}^{\mathrm{real}}_{i}
. The closer a trajectory to (0,0), the smaller the reality gap.
A zoomed HFE time window appears in Figure
10
. Data usage differs: our approach relied on
∼
\qty
​
20
\sim\!\qty{20}{}
of in-air data; the 2019 actuator network used
∼
\qty
​
4
\sim\!\qty{4}{}
of torqued data, using more complex data acquisition techniques, and the deployed vendor LSTM was likely trained on a larger dataset.
Figure 10
:
In-air LF–HFE joint trajectory of
\robot
ANYmal (zoomed view). Dashed gray: target trajectory. Dashed green: measured trajectory. Blue: no actuator model. Red: actuator network baseline. Orange: proposed method (
\ac
pace).
4.2
On-ground locomotion analysis
4.2.1
Tytan
Figure
6
juxtaposes real vs. simulated sequences for an open-loop replay of commanded base velocities. Both absolute and relative motion match closely (cf. supplementary video). Linear and yaw-rate tracking align between real (blue) and fitted simulation (orange), with targets in dashed black. Measured vs. commanded motor torques on the LF leg also match well; small deviations appear at KFE during liftoff.
4.2.2
ANYmal
Forward walking with URDF-only fails (caught by crane), whereas both actuator network and
\ac
pace succeed (Figure
7
). During steady state, both achieve a mean
≈
\qty
​
0.85
​
\per
\approx\!\qty{0.85}{\per}
(violin plot), with similar limit cycles; the no-model case diverges by
∼
\qty
​
3
\sim\!\qty{3}{}
. Sideways and yaw tests are provided in the Appendix (Figure
21
); the URDF-only policy can execute yaw steps at
\qty
2
\per
but remains unreliable for
\qty
1
\per
forward and sideways walking.
4.2.3
Energetic running-track evaluations
We replicate the
\qty
400 track protocol of Bjelonic et al.
bjelonic2023learning
and extend it with
\robot
ANYmal D and
\robot
Tytan. Summary statistics are listed in Table
3
. State-of-charge vs. distance, CoT decomposition, frequency spectra, and a knee limit-cycle analysis are in Figure
8
.
Table 3
:
Running-track performance (per full battery charge).
\robot
ANYmal C
bjelonic2023learning
\robot
AoPS
bjelonic2023learning
\robot
ANYmal D
\robot
Tytan
Rounds [–]
6.6
6.6
7.5
7.5
10.3
10.3
15.25
15.25
Distance
s
s
[
m
\mathrm{m}
]
2640
2640
3000
3000
4120
4120
6100
6100
Initial SoC [
%
\mathrm{\char 37\relax}
]
89
89
92
92
95
95
98
98
Final SoC [
%
\mathrm{\char 37\relax}
]
10
10
11
11
15
15
5
5
Time
t
t
[
min
\mathrm{min}
]
59
59
68
68
82
82
104
104
Commanded
v
^
x
,
B
\hat{v}_{x,B}
[
m
s
−
1
\mathrm{m}\text{\,}{\mathrm{s}}^{-1}
]
1.0
1.0
1.0
1.0
1.0
1.0
1.0
1.0
Average
v
¯
x
,
B
\bar{v}_{x,B}
[
m
s
−
1
\mathrm{m}\text{\,}{\mathrm{s}}^{-1}
]
0.740
0.740
0.735
0.735
0.850
0.850
0.978
0.978
Efficiency
η
\eta
[
%
\mathrm{\char 37\relax}
]
100
100
111
111
132
132
196
196
Ambient Temp.
Θ
\Theta
[
°C
\mathrm{\SIUnitSymbolCelsius}
]
26
26
31
31
10
10
18
18
Avg. power
P
full
P_{\mathrm{full}}
[
W
\mathrm{W}
]
723
723
646
646
556
556
504
504
We decompose the cost of transport (CoT) into electronics (CoE), drives (CoD), and locomotion (CoL) using the procedures in Section
3.3.3
. Table
4
summarizes the contributions. For
\robot
ANYmal D,
\ac
cot reduces to
1.27
1.27
(vs.
1.86
1.86
for
\robot
ANYmal C);
\robot
Tytan reaches
0.97
0.97
. In both platforms, less than half of the energy is attributed to CoL. Biological dogs reach similar distribution at half the
\ac
cot
bryce2017comparative
at
\qty
30 weight compared to
\robot
Tytan.
Table 4
:
CoT decomposition on the track.
CoE
CoD
CoL
CoT
ANYmal C
(
0.50
0.50
)
(
0.24
0.24
)
1.12
1.12
1.86
1.86
ANYmal D
0.50
0.50
0.24
0.24
0.53
0.53
1.27
1.27
Tytan
0.29
0.29
0.21
0.21
0.47
0.47
0.97
0.97
Dogs
bryce2017comparative
0.25
0.25
0.23
0.23
0.48
0.48
4.2.4
Dynamic-limit demonstrations
Figure
11
shows three policies trained in fitted simulation and deployed zero-shot. On
\robot
ANYmal, we demonstrate two-legged balancing (orientation tracking) and running. In these running trials, simulation attained
≈
\qty
​
4
​
\per
\approx\!\qty{4}{\per}
; hardware peaked near
\qty
3
\per
, limited by battery current (
\qty
32 threshold; Figure
12
). On
\robot
Minimal (
\qty
4), the policy climbs standard
\qty
18
\centi
stairs continuously (20 steps in
\qty
46).
Figure 11
:
Dynamic-limit demonstrations. Top: two-legged balance with
\robot
ANYmal. Center: running with
\robot
ANYmal. Bottom: stair climbing with
\robot
Minimal.
Figure 12
:
Running experiments with
\robot
ANYmal. Top: commanded forward velocity versus measured base velocity (state estimation). Bottom: battery current profile during the same run, including measured current and the saturated peak limit of
\qty
32 imposed by the battery.
5
Discussion
Inner loop as near–unit torque source.
The electrical inner loop tracks torque at high bandwidth (measured
≈
\approx
\qty
346), so within the policy’s frequency range it can be treated as a near–unit-gain source (Section
4.1.1
). With more advanced control, bandwidths up to
∼
\sim
\qty
3.5
\kilo
are feasible
springob2002high
.
Physicality of fitted parameters and
virtual
inertia.
On a single drive,
\ac
pace recovers output inertias that match analytic expectations, and the drive behaves closely like a second–order LTI system. Because the mechanical bandwidth falls in the excitation range,
I
a
I_{a}
and
d
d
can be reliably identified and directly relate to the dynamics shaped by the PD gains (cf. Section
2.1
). When firmware compensations are enabled, we observe an additive, load–independent
virtual
inertia that shifts the effective dynamics (Section
4.1.1
); identification and deployment should therefore use identical firmware modes.
Interpreting
I
a
I_{a}
and
d
d
.
Across abstraction levels, the fitted armature term
I
a
I_{a}
aggregates rotor inertia, compensation effects, and inaccuracies in link and load modeling, while the fitted damping term
d
d
captures contributions from the motor, gearbox, and compensations. Both scale with the ratio
k
i
/
k
i
,
real
k_{i}/k_{i,\mathrm{real}}
, and are handled end–to–end by
\ac
pace.
The unexpectedly larger full-robot
I
a
I_{a}
(about fourfold relative to the single-drive baseline) is consistent with three effects: (i) underestimation of link inertia in CAD, particularly in the thigh, (ii) additional apparent inertia introduced by compensation, and (iii) rotor inertia.
A diagnostic single-leg-segment analysis confirms this explanation (Appendix
C
), but it is not central to our contribution.
Generalization across gains and trajectories.
Fitted simulators reproduce in–air trajectories at unseen PD gains and for unseen trajectories with higher fidelity than the actuator–network baseline, indicating a consistent physical model rather than an overfit function approximator (cf. Figure
5(d)
, Figure
10
and Figure
20
).
In–air data suffices for contact tasks.
Identifying joint–space dynamics (Section
2.2
) is sufficient for zero–shot locomotion.
Zero–model failure and the role of speed.
Policies trained without actuator models can fail even at low speeds (Section
4.2
); covering the resulting gap would require heavy dynamics randomization, whose scope grows rapidly with step frequency (Figure
19
). Thus, achievable speed (step frequency) is a sensitive proxy for the reality gap.
Sim2real tracking and dynamic behaviors.
Open–loop base–velocity replays match between real and simulation (Figure
6
), and
\ac
pace is competitive with the actuator network on
\robot
ANYmal (Figure
7
). The same modeling enabled two–leg balancing and high–speed running up to the platform’s electrical limit (Section
4.2.4
) and continuous stair-climbing on
\robot
Minimal and the fastest
\robot
ANYmal ever reportedly ran. The
\qty
3
\per
ceiling on
\robot
ANYmal is set by battery current limits (
\qty
32A, Figure
12
). Notably,
\ac
pace uses only
∼
\sim
\qty
20s of encoder–only in–air data per robot, versus minutes of torque–instrumented data for actuator networks—broadening applicability to systems without torque sensors.
Energy shaping and straight–leg gaits.
With compensation and
k
i
k_{i}
drift folded into
d
d
, the energy reward (Section
2.3.3
) penalizes dissipations that correlate with track–side CoT reductions (Table
4
). Precise simulator fitting through parameter fitting eliminates the need for dynamics randomization, enabling
\robot
ANYmal to adopt straighter–knee gaits. This reduces the total CoT by
∼
\sim
\qty
32 and halving locomotion losses. Despite
\robot
Tytan’s efficient drives and low leg inertia, its inability to fully straighten the knee keeps its locomotion cost only
∼
\sim
\qty
11 lower than
\robot
ANYmal’s in this regime, underscoring the energetic importance of full knee extension on flat terrain (Table
4.2.3
).
Where the watts go.
For both
\robot
ANYmal and
\robot
Tytan, less than half of the energy budget is spent on locomotion itself; electronics and inverter switching account for the remainder. Algorithmic efficiency alone cannot close the full
\ac
cot gap to biology. We need to build robots based on more efficient sensors, computers, and power electronics.
Generalization across robots.
The same parameterization
{
𝐈
a
,
𝐝
,
𝝉
f
,
𝐪
~
b
,
T
d
}
\{\mathbf{I}_{a},\mathbf{d},\boldsymbol{\tau}_{f},\tilde{\mathbf{q}}_{b},T_{d}\}
transfers from open (
\robot
Tytan,
\robot
Minimal) to closed (
\robot
ANYmal) platforms—and beyond—suggesting a broadly useful basis.
Future extensions.
•
Hybrid CMA–ES with local gradient refinements in differentiable physics.
•
Contact–parameter refinement when foot–force sensing or plates are available.
•
Lightweight online adaptation (global scale on
{
I
a
,
d
}
\{I_{a},d\}
) to track wear/temperature over time
lee2020learning
.
6
Conclusion
We presented
\acf
pace
, an alignment of joint-space dynamics that enables zero-shot sim2real locomotion without dynamics randomization. The key idea is to fit a parameterization—per-joint armature, viscous damping, Coulomb friction, joint bias, and a global delay—directly using
∼
\sim
\qty
20s of in-air, encoder-only trajectories and massively parallel evolutionary search.
Across three platforms (
\robot
Tytan,
\robot
ANYmal,
\robot
Minimal), the fitted simulators reproduce in-air joint trajectories with near overlap and generalize across PD gains and trajectories; on
\robot
ANYmal they close the gap where URDF-only models fail and match actuator-network fidelity while requiring less data and no torque sensors. The fitted dynamics translates to the ground: blind locomotion policies train in fitted simulation and deploy zero-shot, yielding accurate base-velocity tracking and endurance improvements on a
\qty
400m track (e.g.,
\robot
ANYmal D: CoT
1.27
1.27
and
\qty
4.12
\kilo
;
\robot
Tytan: CoT
0.97
0.97
and
\qty
6.10
\kilo
). Dynamic-limit demonstrations (two-legged balance,
∼
\sim
\qty
3m/s running bounded by a
\qty
32A battery limit, and stair climbing on a
\qty
4 robot) further indicate that residual constraints are hardware–power and sensing limited rather than model limited.
The approach is data- and compute-efficient (
\qty
20 trajectories, single-GPU,
\qtyrange
1024h per robot with
N
=
4096
N\!=\!4096
environments) and applies to both open and closed platforms, provided firmware compensation modes and filters are consistent between identification and deployment. Current limitations include finite excitation bandwidth on suspended setups and temperature/aging dependencies that shift effective parameters. Our approach fails if any of the assumptions from Section
2.4
are violated, which fully online and model-free approaches might be able to catch.
Future work will focus on (i) accurately identifying
electrical
constraints—bus-voltage (Appendix
A
) and current limits (Section
12
), and inverter switching behavior—and embedding them into the fit, and (ii) explicitly modeling
compliance
(joint/foot stiffness, link flexibilities, series elasticity) within the optimization. Together, these additions should enable higher step-frequency gaits that deliberately exploit hardware dynamics rather than fight them. As simulators expose richer effects, we will also extend the parameterization beyond acceleration to include higher-order motion terms—jerk
q
˙˙˙
\dddot{q}
, snap
q
(
4
)
q^{(4)}
, crackle
q
(
5
)
q^{(5)}
, and pop
q
(
6
)
q^{(6)}
—to better capture bandwidth limits, electro-magnetic effects, mitigate wear, and shape smooth, high-frequency locomotion.
Takeaway.
A small, physically meaningful parameter set (
4
​
n
+
1
4n{+}1
), identified from in-air experiments using joint-encoders, is sufficient to eliminate dynamics randomization for blind quadrupedal locomotion and to translate simulation capabilities to hardware in a single shot.
{an}
ORCIDs
Filip Bjelonic
0000-0002-4890-3132
Fabian Tischhauser
0009-0009-8821-3994
Marco Hutter
0000-0002-4285-4990
{acks}
The authors would like to thank the RSL Learning Group for many insightful discussions throughout this work. We are grateful to Konrad and Matthias for valuable discussions on electronics. We also thank Zichong, Stephan, Efe, and Yuntao, René, Clemens, Ryo, Alexander, Fabio for employing and extending our approach in their own research. Finally, we acknowledge the use of OpenAI’s GPT-5, which assisted in refining the manuscript language. All technical content, analyses, and citations were generated and verified by the authors. We note that large language models may exhibit biases, errors, or omissions, and we take full responsibility for the accuracy and appropriateness of the manuscript.
{AuthorContribution}
The authors confirm contribution to the paper as follows:
study conception and design: F. Bjelonic, F. Tischhauser, M. Hutter;
data collection: F. Bjelonic;
analysis and interpretation of results: F. Bjelonic;
mechanical experimental setups and electronics support: F. Tischhauser;
manuscript preparation: F. Bjelonic, M. Hutter.
M. Hutter provided funding, supervision and critical feedback.
All authors reviewed the results and approved the final version of the manuscript.
{dci}
The authors declare that there are no potential conflicts of interest with respect to the research, authorship, or publication of this article.
{funding}
This work was supported by the European Union’s Horizon Europe Framework Programme (Grant Agreement No. 101070596).
{das}
The datasets supporting the findings of this study are available at
http://hdl.handle.net/20.500.11850/783505
, and the source code will be released soon at
https://github.com/leggedrobotics/pace-sim2real
.
References
(1)
Hutter, M., C. Gehring, A. Lauber, F. Gunther, C. D. Bellicoso, V. Tsounis, P. Fankhauser, R. Diethelm, S. Bachmann, M. Blösch, et al.
Anymal-toward legged robots for harsh environments.
Advanced Robotics
, Vol. 31, No. 17, 2017, pp. 918–931.
(2)
Unitree Robotics.
Motor SDK Development Guide.
Online:
https://support.unitree.com/home/en/Motor_SDK_Dev_Guide/overview
, 2025.
Accessed: Aug. 21, 2025.
(3)
Aractingi, M., P.-A. Léziart, T. Flayols, J. Perez, T. Silander, and P. Souères.
Controlling the solo12 quadruped robot with deep reinforcement learning.
scientific Reports
, Vol. 13, No. 1, 2023, p. 11945.
(4)
Katz, B. G.
A low cost modular actuator for dynamic robots
.
Ph.D. thesis, Massachusetts Institute of Technology, 2018.
(5)
Liu, D., F. Yang, X. Liao, and X. Lyu.
Diablo: A 6-dof wheeled bipedal robot composed entirely of direct-drive joints.
In
2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)
. IEEE, 2024, pp. 3605–3612.
(6)
Derammelaere, S., M. Haemers, J. De Viaene, F. Verbelen, and K. Stockman.
A quantitative comparison between BLDC, PMSM, brushed DC and stepping motor technologies.
In
2016 19th International Conference on Electrical Machines and Systems (ICEMS)
. Ieee, 2016, pp. 1–5.
(7)
Gamazo-Real, J. C., E. Vázquez-Sánchez, and J. Gómez-Gil.
Position and speed control of brushless DC motors using sensorless techniques and application trends.
sensors
, Vol. 10, No. 7, 2010, pp. 6901–6947.
(8)
Xie, Z., P. Clary, J. Dao, P. Morais, J. Hurst, and M. Panne.
Learning locomotion skills for cassie: Iterative design and sim-to-real.
In
Conference on Robot Learning
. PMLR, 2020, pp. 317–329.
(9)
Li, Y., J. Li, W. Fu, and Y. Wu.
Learning agile bipedal motions on a quadrupedal robot.
In
2024 IEEE International Conference on Robotics and Automation (ICRA)
. IEEE, 2024, pp. 9735–9742.
(10)
Bellegarda, G., Y. Chen, Z. Liu, and Q. Nguyen.
Robust high-speed running for quadruped robots via deep reinforcement learning.
In
2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)
. IEEE, 2022, pp. 10364–10370.
(11)
Peng, X. B., E. Coumans, T. Zhang, T.-W. Lee, J. Tan, and S. Levine.
Learning agile robotic locomotion skills by imitating animals.
arXiv preprint arXiv:2004.00784
.
(12)
Handa, A., A. Allshire, V. Makoviychuk, A. Petrenko, R. Singh, J. Liu, D. Makoviichuk, K. Van Wyk, A. Zhurkevich, B. Sundaralingam, et al.
Dextreme: Transfer of agile in-hand manipulation from simulation to reality.
In
2023 IEEE International Conference on Robotics and Automation (ICRA)
. IEEE, 2023, pp. 5977–5984.
(13)
Mehta, B., M. Diaz, F. Golemo, C. J. Pal, and L. Paull.
Active domain randomization.
In
Conference on Robot Learning
. PMLR, 2020, pp. 1162–1176.
(14)
Ramos, F., R. C. Possas, and D. Fox.
Bayessim: adaptive domain randomization via probabilistic inference for robotics simulators.
arXiv preprint arXiv:1906.01728
.
(15)
Shi, F., C. Zhang, T. Miki, J. Lee, M. Hutter, and S. Coros.
Rethinking robustness assessment: Adversarial attacks on learning-based quadrupedal locomotion controllers.
arXiv preprint arXiv:2405.12424
.
(16)
Tiboni, G., K. Arndt, and V. Kyrki.
DROPO: Sim-to-real transfer with offline domain randomization.
Robotics and Autonomous Systems
, Vol. 166, 2023, p. 104432.
(17)
Ajay, A., J. Wu, N. Fazeli, M. Bauza, L. P. Kaelbling, J. B. Tenenbaum, and A. Rodriguez.
Augmenting physical simulators with stochastic neural networks: Case study of planar pushing and bouncing.
In
2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)
. IEEE, 2018, pp. 3066–3073.
(18)
Jiang, Y., T. Zhang, D. Ho, Y. Bai, C. K. Liu, S. Levine, and J. Tan.
Simgan: Hybrid simulator identification for domain adaptation via adversarial reinforcement learning.
In
2021 IEEE International Conference on Robotics and Automation (ICRA)
. IEEE, 2021, pp. 2884–2890.
(19)
Golemo, F., A. A. Taiga, A. Courville, and P.-Y. Oudeyer.
Sim-to-real transfer with neural-augmented robot simulation.
In
Conference on Robot Learning
. PMLR, 2018, pp. 817–828.
(20)
Hwangbo, J., J. Lee, A. Dosovitskiy, D. Bellicoso, V. Tsounis, V. Koltun, and M. Hutter.
Learning agile and dynamic motor skills for legged robots.
Science Robotics
, Vol. 4, No. 26, 2019, p. eaau5872.
(21)
Fey, N., G. B. Margolis, M. Peticco, and P. Agrawal.
Bridging the Sim-to-Real Gap for Athletic Loco-Manipulation.
arXiv preprint arXiv:2502.10894
.
(22)
Miller, A., F. Yu, M. Brauckmann, and F. Farshidian.
High-Performance Reinforcement Learning on Spot: Optimizing Simulation Parameters with Distributional Measures.
arXiv preprint arXiv:2504.17857
.
(23)
He, T., J. Gao, W. Xiao, Y. Zhang, Z. Wang, J. Wang, Z. Luo, G. He, N. Sobanbab, C. Pan, et al.
Asap: Aligning simulation and real-world physics for learning agile humanoid whole-body skills.
arXiv preprint arXiv:2502.01143
.
(24)
Sobanbabu, N., G. He, T. He, Y. Yang, and G. Shi.
Sampling-based system identification with active exploration for legged robot sim2real learning.
arXiv preprint arXiv:2505.14266
.
(25)
Tsai, Y.-Y., H. Xu, Z. Ding, C. Zhang, E. Johns, and B. Huang.
Droid: Minimizing the reality gap using single-shot human demonstration.
IEEE Robotics and Automation Letters
, Vol. 6, No. 2, 2021, pp. 3168–3175.
(26)
Sontakke, N., H. Chae, S. Lee, T. Huang, D. W. Hong, and S. Hal.
Residual physics learning and system identification for sim-to-real transfer of policies on buoyancy assisted legged robots.
In
2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)
. IEEE, 2023, pp. 392–399.
(27)
Wu, P., A. Escontrela, D. Hafner, P. Abbeel, and K. Goldberg.
Daydreamer: World models for physical robot learning.
In
Conference on robot learning
. PMLR, 2023, pp. 2226–2240.
(28)
Li, C., A. Krause, and M. Hutter.
Offline Robotic World Model: Learning Robotic Policies without a Physics Simulator.
arXiv preprint arXiv:2504.16680
.
(29)
Hanna, J. P., S. Desai, H. Karnan, G. Warnell, and P. Stone.
Grounded action transformation for sim-to-real reinforcement learning.
Machine Learning
, Vol. 110, No. 9, 2021, pp. 2469–2499.
(30)
Smith, L., J. C. Kew, X. B. Peng, S. Ha, J. Tan, and S. Levine.
Legged robots that keep on learning: Fine-tuning locomotion policies in the real world.
In
2022 international conference on robotics and automation (ICRA)
. IEEE, 2022, pp. 1593–1599.
(31)
Song, X., Y. Yang, K. Choromanski, K. Caluwaerts, W. Gao, C. Finn, and J. Tan.
Rapidly adaptable legged robots via evolutionary meta-learning.
In
2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)
. IEEE, 2020, pp. 3769–3776.
(32)
Lee, J., J. Hwangbo, L. Wellhausen, V. Koltun, and M. Hutter.
Learning quadrupedal locomotion over challenging terrain.
Science robotics
, Vol. 5, No. 47, 2020, p. eabc5986.
(33)
Muratore, F., F. Ramos, G. Turk, W. Yu, M. Gienger, and J. Peters.
Robot learning from randomized simulations: A review.
Frontiers in Robotics and AI
, Vol. 9, 2022, p. 799893.
(34)
Ju, H., R. Juan, R. Gomez, K. Nakamura, and G. Li.
Transferring policy of deep reinforcement learning from simulation to reality for robotics.
Nature Machine Intelligence
, Vol. 4, No. 12, 2022, pp. 1077–1087.
(35)
Grandia, R., F. Jenelten, S. Yang, F. Farshidian, and M. Hutter.
Perceptive locomotion through nonlinear model-predictive control.
IEEE Transactions on Robotics
, Vol. 39, No. 5, 2023, pp. 3402–3421.
(36)
Miki, T., J. Lee, J. Hwangbo, L. Wellhausen, V. Koltun, and M. Hutter.
Learning robust perceptive locomotion for quadrupedal robots in the wild.
Science Robotics
, Vol. 7, No. 62, 2022, p. eabk2822.
(37)
Xue, H., C. Pan, Z. Yi, G. Qu, and G. Shi.
Full-order sampling-based mpc for torque-level locomotion control via diffusion-style annealing.
arXiv preprint arXiv:2409.15610
.
(38)
Rudin, N., J. He, J. Aurand, and M. Hutter.
Parkour in the Wild: Learning a General and Extensible Agile Locomotion Policy Using Multi-expert Distillation and RL Fine-tuning.
arXiv preprint arXiv:2505.11164
.
(39)
Schulman, J., F. Wolski, P. Dhariwal, A. Radford, and O. Klimov.
Proximal policy optimization algorithms.
arXiv preprint arXiv:1707.06347
.
(40)
Liu, Y., J. Ding, and X. Liu.
Ipo: Interior-point policy optimization under constraints.
In
Proceedings of the AAAI conference on artificial intelligence
, Vol. 34. 2020, pp. 4940–4947.
(41)
Kim, Y., H. Oh, J. Lee, J. Choi, G. Ji, M. Jung, D. Youm, and J. Hwangbo.
Not only rewards but also constraints: Applications on legged robot locomotion.
IEEE Transactions on Robotics
, Vol. 40, 2024, pp. 2984–3003.
(42)
Kaelbling, L. P., M. L. Littman, and A. R. Cassandra.
Planning and acting in partially observable stochastic domains.
Artificial intelligence
, Vol. 101, No. 1-2, 1998, pp. 99–134.
(43)
Hausknecht, M. J. and P. Stone.
Deep Recurrent Q-Learning for Partially Observable MDPs.
In
AAAI fall symposia
, Vol. 45. 2015, p. 141.
(44)
Pinto, L., M. Andrychowicz, P. Welinder, W. Zaremba, and P. Abbeel.
Asymmetric actor critic for image-based robot learning.
arXiv preprint arXiv:1710.06542
.
(45)
Rudin, N., D. Hoeller, P. Reist, and M. Hutter.
Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning.
In
Proceedings of the 5th Conference on Robot Learning
,
Proceedings of Machine Learning Research
, Vol. 164 (A. Faust, D. Hsu, and G. Neumann, eds.). PMLR, 2022, pp. 91–100.
URL
https://proceedings.mlr.press/v164/rudin22a.html
.
(46)
Ji, G., J. Mun, H. Kim, and J. Hwangbo.
Concurrent training of a control policy and a state estimator for dynamic and robust legged locomotion.
IEEE Robotics and Automation Letters
, Vol. 7, No. 2, 2022, pp. 4630–4637.
(47)
Shin, Y.-H., T.-G. Song, G. Ji, and H.-W. Park.
Actuator-constrained reinforcement learning for high-speed quadrupedal locomotion.
arXiv preprint arXiv:2312.17507
.
(48)
Shafiee, M., G. Bellegarda, and A. Ijspeert.
Manyquadrupeds: Learning a single locomotion policy for diverse quadruped robots.
In
2024 IEEE International Conference on Robotics and Automation (ICRA)
. IEEE, 2024, pp. 3471–3477.
(49)
Ma, Y. J., W. Liang, G. Wang, D.-A. Huang, O. Bastani, D. Jayaraman, Y. Zhu, L. Fan, and A. Anandkumar.
Eureka: Human-level reward design via coding large language models.
arXiv preprint arXiv:2310.12931
.
(50)
Eysenbach, B., A. Gupta, J. Ibarz, and S. Levine.
Diversity is all you need: Learning skills without a reward function.
arXiv preprint arXiv:1802.06070
.
(51)
Yang, Y., T. Zhang, E. Coumans, J. Tan, and B. Boots.
Fast and efficient locomotion via learned gait transitions.
In
Conference on robot learning
. PMLR, 2022, pp. 773–783.
(52)
Wensing, P. M., A. Wang, S. Seok, D. Otten, J. Lang, and S. Kim.
Proprioceptive actuator design in the mit cheetah: Impact mitigation and high-bandwidth physical interaction for dynamic legged robots.
Ieee transactions on robotics
, Vol. 33, No. 3, 2017, pp. 509–522.
(53)
Fadini, G., T. Flayols, A. Del Prete, N. Mansard, and P. Souères.
Computational design of energy-efficient legged robots: Optimizing for size and actuators.
In
2021 IEEE International Conference on Robotics and Automation (ICRA)
. IEEE, 2021, pp. 9898–9904.
(54)
Roux, C., E. Chane-Sane, L. De Matteïs, T. Flayols, J. Manhes, O. Stasse, and P. Souères.
Constrained Reinforcement Learning for Unstable Point-Feet Bipedal Locomotion Applied to the Bolt Robot.
arXiv preprint arXiv:2508.02194
.
(55)
Ferrari, S., P. Ragazzo, G. Dilevrano, and G. Pellegrino.
Flux and loss map based evaluation of the efficiency map of synchronous machines.
IEEE Transactions on Industry Applications
, Vol. 59, No. 2, 2022, pp. 1500–1509.
(56)
Valsecchi, G., A. Vicari, F. Tischhauser, M. Garabini, and M. Hutter.
Accurate power consumption estimation method makes walking robots energy efficient and quiet.
In
2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)
. IEEE, 2024, pp. 13282–13288.
(57)
Nomura, M. and M. Shibata.
cmaes : A Simple yet Practical Python Library for CMA-ES.
arXiv preprint arXiv:2402.01373
.
(58)
Telikani, A., A. Tahmassebi, W. Banzhaf, and A. H. Gandomi.
Evolutionary machine learning: A survey.
ACM Computing Surveys (CSUR)
, Vol. 54, No. 8, 2021, pp. 1–35.
(59)
Hansen, N.
The CMA evolution strategy: A tutorial.
arXiv preprint arXiv:1604.00772
.
(60)
Hansen, N. and A. Ostermeier.
Completely derandomized self-adaptation in evolution strategies.
Evolutionary computation
, Vol. 9, No. 2, 2001, pp. 159–195.
(61)
Valsecchi, G., N. Rudin, L. Nachtigall, K. Mayer, F. Tischhauser, and M. Hutter.
Barry: a high-payload and agile quadruped robot.
IEEE Robotics and Automation Letters
, Vol. 8, No. 11, 2023, pp. 6939–6946.
(62)
Hutter, M., C. Gehring, D. Jud, A. Lauber, C. D. Bellicoso, V. Tsounis, J. Hwangbo, K. Bodie, P. Fankhauser, M. Bloesch, et al.
Anymal-a highly mobile and dynamic quadrupedal robot.
In
IEEE/RSJ International conference on intelligent robots and systems (IROS)
. 2016, pp. 38–44.
(63)
Watanabe, R., T. Miki, F. Shi, Y. Kadokawa, F. Bjelonic, K. Kawaharazuka, A. Cramariuc, and M. Hutter.
Learning quiet walking for a small home robot.
arXiv preprint arXiv:2502.10983
.
(64)
Bellicoso, C. D., K. Krämer, M. Stäuble, D. Sako, F. Jenelten, M. Bjelonic, and M. Hutter.
Alma-articulated locomotion and manipulation for a torque-controllable robot.
In
2019 International conference on robotics and automation (ICRA)
. IEEE, 2019, pp. 8477–8483.
(65)
Ma, Y., A. Cramariuc, F. Farshidian, and M. Hutter.
Learning coordinated badminton skills for legged manipulators.
Science Robotics
, Vol. 10, No. 102, 2025, p. eadu3922.
(66)
Spiridonov, A., F. Buehler, M. Berclaz, V. Schelbert, J. Geurts, E. Krasnova, E. Steinke, J. Toma, J. Wuethrich, R. Polat, et al.
Spacehopper: A small-scale legged robot for exploring low-gravity celestial bodies.
In
2024 IEEE International Conference on Robotics and Automation (ICRA)
. IEEE, 2024, pp. 3464–3470.
(67)
Arnold, M., L. Hildebrandt, K. Janssen, E. Ongan, P. Bürge, Á. G. Gábriel, J. Kennedy, R. Lolla, Q. Oppliger, M. Schaaf, et al.
LEVA: A high-mobility logistic vehicle with legged suspension.
arXiv preprint arXiv:2503.10028
.
(68)
Leuthard, S., T. Eugster, N. Faesch, R. Feingold, C. Flynn, M. Fritsche, N. Hürlimann, E. Morbach, F. Tischhauser, M. Müller, et al.
Magnecko: Design and Control of a Quadrupedal Magnetic Climbing Robot.
In
Climbing and Walking Robots Conference
. Springer, 2024, pp. 55–67.
(69)
He, J., C. Zhang, F. Jenelten, R. Grandia, M. BÄcher, and M. Hutter.
Attention-Based Map Encoding for Learning Generalized Legged Locomotion.
arXiv preprint arXiv:2506.09588
.
(70)
Hottiger, G., C. Gehring, and D. Bellicoso.
signal_logger: Signal logging and plotting tools for robotics.
https://github.com/ANYbotics/signal_logger
, 2020.
Accessed: 2025-07-09.
(71)
Bjelonic, F., J. Lee, P. Arm, D. Sako, D. Tateo, J. Peters, and M. Hutter.
Learning-based design and control for quadrupedal robots with parallel-elastic actuators.
IEEE Robotics and Automation Letters
, Vol. 8, No. 3, 2023, pp. 1611–1618.
(72)
Bryce, C. M. and T. M. Williams.
Comparative locomotor costs of domestic dogs reveal energetic economy of wolf-like breeds.
Journal of Experimental Biology
, Vol. 220, No. 2, 2017, pp. 312–321.
(73)
Springob, L. and J. Holtz.
High-bandwidth current control for torque-ripple compensation in PM synchronous machines.
IEEE Transactions on industrial electronics
, Vol. 45, No. 5, 2002, pp. 713–721.
Figure 13
:
Idealized PMSM saturation envelope for the
\robot
Tytan hip motor, showing motor velocity
ω
\omega
(x-axis) versus torque
τ
\tau
(y-axis). The solid blue curve indicates the enforced torque limits in simulation: commanded torques inside the envelope are applied directly, while those outside are saturated to the boundary. This boundary is determined by two constraints: the maximum torque limit (dot-dashed gray), representing effects such as magnet demagnetization, and the back-EMF–limited torque (dashed gray). Orange dots mark experimental measurements from Section
3.2.1
.
Figure 14
:
Simplified block diagram of the motor‐driven joint actuator, illustrating the electronics (left, yellow) and rigid‐body dynamics (right, red) subsystems. In the electronics path, a PD position controller with gains
P
τ
P_{\tau}
and
D
τ
D_{\tau}
drives a PI current loop—comprised of integrator
I
u
I_{u}
and feedforward
P
u
P_{u}
—to regulate the torque producing quadrature current
i
q
i_{q}
. This path is subject to disturbances from the load current, nonlinear battery dynamics, temperature variations, and unknown feed-forward compensations. In the mechanical path,
i
q
i_{q}
is converted to torque through the motor constant
k
i
k_{i}
and gear ratio
n
n
, then transmitted to the inertia
I
a
I_{a}
, friction torque
τ
f
\tau_{f}
, and viscous transmission damping
d
d
, yielding joint acceleration
q
¨
\ddot{q}
and position
q
q
. The two subsystems interact via the back‐EMF constant
k
ω
k_{\omega}
.
Table 5
:
PACE parameters found per Robot
Joint
LF HAA
LF HFE
LF KFE
RF HAA
RF HFE
RF KFE
LH HAA
LH HFE
LH KFE
RH HAA
RH HFE
RH KFE
Armature
I
a
I_{a}
[
​
10
−
3
kg
m
3
{10}^{-3}\text{\,}\mathrm{kg}\text{\,}{\mathrm{m}}^{3}
]
\robot
Tytan
140
140
106
106
3.3
3.3
120
120
120
120
3.4
3.4
140
140
110
110
3.6
3.6
140
140
110
110
3.5
3.5
\robot
ANYmal
76
76
76
76
67
67
74
74
77
77
67
67
89
89
51
51
64
64
79
79
39
39
51
51
\robot
Minimal
0.050
0.050
0.025
0.025
0.026
0.026
0.056
0.056
0.025
0.025
0.025
0.025
0.049
0.049
0.023
0.023
0.026
0.026
0.052
0.052
0.028
0.028
0.025
0.025
Damping
d
d
[
N
m
s
rad
−
1
\text{\,}\mathrm{N}\text{\,}\mathrm{m}\text{\,}\mathrm{s}\text{\,}{\mathrm{rad}}^{-1}
]
\robot
Tytan
1.7
1.7
0.17
0.17
2.1
2.1
1.7
1.7
0.22
0.22
2.3
2.3
0.50
0.50
0.63
0.63
3.9
3.9
0.70
0.70
0.83
0.83
2.3
2.3
\robot
ANYmal
4.9
4.9
4.4
4.4
5.2
5.2
4.7
4.7
4.3
4.3
5.3
5.3
4.9
4.9
4.9
4.9
5.4
5.4
5.1
5.1
5.1
5.1
5.5
5.5
\robot
Minimal
0.0
0.0
0.0
0.0
0.092
0.092
0.1911
0.1911
0.031
0.031
0.030
0.030
0.0
0.0
0.0
0.0
0.066
0.066
0.0
0.0
0.0
0.0
0.030
0.030
Friction
τ
f
\tau_{f}
[–]
\robot
Tytan
0.0093
0.0093
0.044
0.044
0.000 25
0.000\,25
0.0036
0.0036
0.036
0.036
0.0015
0.0015
0.0
0.0
0.031
0.031
0.0010
0.0010
0.0
0.0
0.035
0.035
0.000 50
0.000\,50
\robot
ANYmal
0.0054
0.0054
0.021
0.021
0.028
0.028
0.0035
0.0035
0.027
0.027
0.036
0.036
0.0032
0.0032
0.024
0.024
0.040
0.040
0.0029
0.0029
0.013
0.013
0.045
0.045
\robot
Minimal
0.038
0.038
0.084
0.084
0.34
0.34
0.033
0.033
0.070
0.070
0.32
0.32
0.051
0.051
0.091
0.091
0.31
0.31
0.048
0.048
0.075
0.075
0.39
0.39
Joint Position Bias
q
~
b
\tilde{q}_{b}
[
rad
\text{\,}\mathrm{rad}
]
\robot
Tytan
0.0017
0.0017
−
0.011
-0.011
−
0.028
-0.028
−
0.0029
-0.0029
−
0.012
-0.012
−
0.026
-0.026
−
0.0011
-0.0011
−
0.017
-0.017
−
0.026
-0.026
−
0.000 70
-0.000\,70
−
0.0148
-0.0148
−
0.0275
-0.0275
\robot
ANYmal
0.022
0.022
0.0057
0.0057
−
0.003
-0.003
0.011
0.011
−
0.0072
-0.0072
0.0094
0.0094
−
0.012
-0.012
−
0.0013
-0.0013
−
0.0095
-0.0095
−
0.016
-0.016
0.0043
0.0043
0.0045
0.0045
\robot
Minimal
−
0.018
-0.018
−
0.0030
-0.0030
−
0.0017
-0.0017
−
0.020
-0.020
0.010
0.010
−
0.018
-0.018
−
0.000 40
-0.000\,40
0.0012
0.0012
0.0027
0.0027
0.010
0.010
−
0.0065
-0.0065
0.0188
0.0188
Delay
T
d
T_{d}
[
ms
\text{\,}\mathrm{ms}
]
\robot
Tytan
7.5
7.5
\robot
ANYmal
7.5
7.5
\robot
Minimal
0.0
0.0
Figure 15
:
Validation of
\robot
Tytan on the HFE joint using proportional–derivative (PD) gains of 145/5. Shown are commanded trajectories, real measurements and the response under the near-optimal identified model.
Figure 16
:
Current tracking performance for the inner-loop transfer function
H
i
:
τ
^
m
→
τ
m
H_{i}:\hat{\tau}_{m}\rightarrow\tau_{m}
, mapping commanded motor torque
τ
^
m
\hat{\tau}_{m}
to measured torque
τ
m
\tau_{m}
.
Figure 17
:
Identified versus target inertia as a function of lever arm radius, with drive compensations disabled.
Figure 18
:
An example bode plot from the single-drive experiment for the closed-loop transfer function
H
q
:
q
^
j
→
q
j
H_{q}:\hat{q}_{j}\rightarrow q_{j}
, mapping commanded joint position
q
^
j
\hat{q}_{j}
to measured joint position
q
j
q_{j}
.
Figure 19
:
An example bode plot from the single-drive experiment for the closed-loop transfer function
H
q
:
q
^
j
→
q
j
H_{q}:\hat{q}_{j}\rightarrow q_{j}
, mapping commanded joint position
q
^
j
\hat{q}_{j}
to measured joint position
q
j
q_{j}
. The hatched region indicates the
2
​
σ
2\sigma
confidence interval from analytic estimation (Appendix
C
).
(a)
Evaluation on the training trajectories, showing commanded versus measured joint positions.
(b)
Trajectory tracking performance of the LF KFE joint on the
\robot
Minimal platform. (a) Evaluation on training trajectories. (b) Validation on test trajectories.
Figure 20
:
Minimal trajectories, LF KFE
(a)
Sideways velocity tracking. Left: commanded and measured velocities over time. Right: distribution of steady-state velocity errors across methods.
(b)
Angular velocity tracking around the yaw axis. Left: commanded and measured velocities over time. Right: distribution of steady-state velocity errors across methods.
Figure 21
:
Additional sim-to-real evaluation of
\robot
ANYmal.
(a) Sideways velocity tracking. (b) Yaw angular velocity tracking. Both panels compare commanded and measured base velocities over time (left) and summarize steady-state errors across methods (right).
Figure 22
:
Joule heating constants of different robots, derived from motor parameters over body weight. These values offer a first-order estimate for black-box robotic systems.
Appendix A
Voltage–Limited Torque Bandwidth of a PMSM
Even with perfect control, a PMSM’s current (and thus torque) loop is fundamentally limited by the available DC–bus voltage and the phase inductances. Under a low–speed / locked–rotor assumption with
i
d
≈
0
i_{d}\!\approx\!0
and negligible delay, the
q
q
-axis dynamics reduce to
v
q
​
(
t
)
\displaystyle v_{q}(t)
=
R
​
i
q
​
(
t
)
+
L
​
i
˙
q
​
(
t
)
,
\displaystyle=R\,i_{q}(t)+L\,\dot{i}_{q}(t),
(30)
⇒
I
q
​
(
s
)
V
q
​
(
s
)
\displaystyle\Rightarrow\quad\frac{I_{q}(s)}{V_{q}(s)}
=
1
L
​
s
+
R
,
\displaystyle=\frac{1}{Ls+R},
(31)
with pole at
−
R
/
L
-R/L
, time constant
τ
=
L
/
R
\tau=L/R
, and theoretical
−
3
​
dB
-3\,\mathrm{dB}
bandwidth
ω
∞
=
R
L
,
f
∞
=
1
2
​
π
​
R
L
.
\displaystyle\omega_{\infty}=\frac{R}{L},\qquad f_{\infty}=\frac{1}{2\pi}\frac{R}{L}.
(32)
Taking the available phase voltage to be limited by the DC–bus (conservatively
U
max
≈
U
bus
=
48
V
U_{\max}\!\approx\!U_{\mathrm{bus}}=$48\text{\,}\mathrm{V}$
, up to modulation factors), our drive yields
f
∞
≈
310
Hz
f_{\infty}\approx$310\text{\,}\mathrm{H}\mathrm{z}$
. Hence, even at the voltage limit the current loop can, in principle, exceed the policy Nyquist of
25
Hz
25\text{\,}\mathrm{H}\mathrm{z}
.
At nonzero electrical speed
ω
e
\omega_{e}
, the back–EMF
e
q
=
ω
e
​
λ
e_{q}=\omega_{e}\lambda
consumes part of the voltage headroom:
v
q
=
R
​
i
q
+
L
​
i
˙
q
+
e
q
⇒
L
​
i
˙
q
≤
U
max
−
R
​
i
q
−
e
q
.
\displaystyle v_{q}=R\,i_{q}+L\,\dot{i}_{q}+e_{q}\;\;\Rightarrow\;\;L\,\dot{i}_{q}\leq U_{\max}-R\,i_{q}-e_{q}.
(33)
As
ω
e
\omega_{e}
increases (or during field–weakening), the effective current (torque) bandwidth reduces because
U
max
−
e
q
U_{\max}-e_{q}
shrinks. In practice, the achievable closed–loop bandwidth is also bounded to a fraction of the PWM carrier frequency and by inverter dead times and delays.
Implication.
For our experiments—low to moderate speeds and
≤
\leq
25
Hz
25\text{\,}\mathrm{H}\mathrm{z}
policy content—the inner loop behaves as a near–unit–gain motor torque source; high–speed limits are then dominated by voltage/current headroom (e.g., battery current limiting), not by inner–loop dynamics.
Appendix B
Effective Base Inertia Comparison on ANYmal
To analyze the difference between drive-specific in-air inertia and contact-induced inertia, we apply d’Alembert’s principle of virtual displacements. We consider two scenarios: (i) vertical motion of the robot’s base and legs, and (ii) horizontal motion parallel to the ground, both starting from the nominal
agile stance
. In both cases, we analyze the dynamics in the robot’s
x
x
–
z
z
plane.
For tractability, we neglect leg masses and account only for the base mass and link inertias. We further assume equal link lengths of
\qty
0.3 for the thigh and shank and no gravity.
Vertical Motion.
This case is simplest, as no torque is applied at the HFE joint, and d’Alembert’s principle reduces to (
34
):
m
​
z
¨
​
δ
​
z
+
I
h
​
q
¨
h
​
δ
​
q
h
+
I
k
​
q
¨
k
​
δ
​
q
k
\displaystyle m\ddot{z}\delta z+I_{h}\ddot{q}_{h}\delta q_{h}+I_{k}\ddot{q}_{k}\delta q_{k}
=
2
​
τ
k
​
δ
​
q
k
\displaystyle=2\tau_{k}\delta q_{k}
(34)
s.t.
x
B
\displaystyle\text{s.t.}\quad x_{B}
=
0
,
\displaystyle=0,
(35)
where
m
m
is the base mass, and
I
h
I_{h}
and
I
k
I_{k}
denote the reflected and body inertias of the thigh and shank, respectively. The kinematic constraints yield the following dependencies between the knee joint angle
q
k
q_{k}
, the hip joint angle
q
h
q_{h}
, and the base
z
z
-position
z
B
z_{B}
:
q
k
\displaystyle q_{k}
=
2
​
q
h
\displaystyle=2q_{h}
(36)
z
B
\displaystyle z_{B}
=
2
​
l
​
cos
⁡
(
q
k
2
)
\displaystyle=2l\cos\left(\tfrac{q_{k}}{2}\right)
(37)
δ
​
z
B
\displaystyle\delta z_{B}
=
−
l
​
sin
⁡
(
q
k
2
)
​
δ
​
q
k
\displaystyle=-l\sin\left(\tfrac{q_{k}}{2}\right)\delta q_{k}
(38)
z
˙
B
\displaystyle\dot{z}_{B}
=
−
l
​
sin
⁡
(
q
k
2
)
​
q
˙
k
\displaystyle=-l\sin\left(\tfrac{q_{k}}{2}\right)\dot{q}_{k}
(39)
z
¨
B
\displaystyle\ddot{z}_{B}
=
−
l
​
sin
⁡
(
q
k
2
)
​
q
¨
k
−
l
2
​
cos
⁡
(
q
k
2
)
​
q
˙
k
2
.
\displaystyle=-l\sin\left(\tfrac{q_{k}}{2}\right)\ddot{q}_{k}-\tfrac{l}{2}\cos\left(\tfrac{q_{k}}{2}\right)\dot{q}_{k}^{2}.
(40)
Substituting these relations into (
34
) yields the nonlinear differential equation
τ
k
\displaystyle\tau_{k}
=
I
~
k
​
q
¨
k
+
1
4
​
m
​
l
2
​
sin
⁡
(
q
k
2
)
​
cos
⁡
(
q
k
2
)
​
q
˙
k
2
\displaystyle=\tilde{I}_{k}\ddot{q}_{k}+\tfrac{1}{4}ml^{2}\sin\left(\tfrac{q_{k}}{2}\right)\cos\left(\tfrac{q_{k}}{2}\right)\dot{q}_{k}^{2}
(41)
I
~
k
\displaystyle\tilde{I}_{k}
=
1
2
​
[
m
​
l
2
​
sin
⁡
(
q
k
2
)
+
1
4
​
I
h
+
I
k
]
.
\displaystyle=\tfrac{1}{2}\Big{[}ml^{2}\sin\!\left(\tfrac{q_{k}}{2}\right)+\tfrac{1}{4}I_{h}+I_{k}\Big{]}.
(42)
Assuming small velocities
q
˙
k
≪
1
\dot{q}_{k}\ll 1
, we plot the position–dependent inertia against the base position in Fig.
23(b)
, and compare it to the constant inertia observed when the legs are not in contact.
Horizontal Motion.
The horizontal case is more involved, leading to large coupled equations. We employ a differentiable symbolic toolbox to obtain the reduced dynamics. As in the vertical case, we report only the main expressions: the virtual displacement in the
x
x
direction (Eq.
43
) and the base constraint (Eq.
44
).
m
​
x
¨
​
δ
​
x
+
I
h
​
q
¨
h
​
δ
​
q
h
+
I
k
​
q
¨
k
​
δ
​
q
k
\displaystyle m\ddot{x}\delta x+I_{h}\ddot{q}_{h}\delta q_{h}+I_{k}\ddot{q}_{k}\delta q_{k}
=
2
​
τ
h
​
δ
​
q
h
\displaystyle=2\tau_{h}\delta q_{h}
(43)
s.t.
z
B
\displaystyle\text{s.t.}\quad z_{B}
=
2
​
l
\displaystyle=\sqrt{2}\,l
(44)
The relations between the base
x
x
-position
x
B
x_{B}
and the joint angles are highly nonlinear and coupled:
x
B
\displaystyle x_{B}
=
l
​
(
cos
⁡
(
q
k
−
q
h
)
+
cos
⁡
(
q
h
)
)
\displaystyle=l\!\left(\cos(q_{k}-q_{h})+\cos(q_{h})\right)
(45)
q
k
\displaystyle q_{k}
=
q
h
±
arccos
⁡
(
2
−
cos
⁡
(
q
h
)
)
.
\displaystyle=q_{h}\pm\arccos\!\left(\sqrt{2}-\cos(q_{h})\right).
(46)
Because the dependencies are strongly nonlinear, their derivatives are large, and the effective inertia varies accordingly. The results are shown in Fig.
23(c)
.
(a)
(b)
(c)
Figure 23
:
Reduced inertia comparison for
\robot
ANYmal with and without ground contact. Panel (a) shows the agile stance used for evaluation. Panels (b) and (c) depict the reduced inertia at the KFE and HFE joint respectively as a function of base position during vertical and horizontal base motion, respectively.
Appendix C
Link Inertia Estimation
To analyze the discrepancy between single-drive and full-robot inertia estimates, we performed a study on a leg-segment of
\robot
Tytan. The KFE was disconnected to minimize friction, allowing the thigh or shank to swing freely. This setup permits direct estimation of link inertias from free oscillations, which we then compare against CAD-derived values and the
\ac
pace fit.
C.1
Pendulum Model
Considering small oscillations around a pivot point
P
P
, the dynamics of the harmonic oscillator are
I
p
​
q
¨
+
m
​
r
​
g
​
q
\displaystyle I_{p}\ddot{q}+mrgq
=
0
,
\displaystyle=0,
(47)
with link mass
m
m
, CoM distance
r
r
, and gravitational acceleration
g
g
. The standard form is
q
¨
+
ω
2
​
q
\displaystyle\ddot{q}+\omega^{2}q
=
0
,
\displaystyle=0,
(48)
where
ω
=
2
​
π
​
f
\omega=2\pi f
is the eigenfrequency. By comparing (
47
) and (
48
), the inertia about the pivot
P
P
is
I
P
=
m
​
r
​
g
(
2
​
π
​
f
)
2
.
\displaystyle I_{P}\;=\;\frac{mrg}{(2\pi f)^{2}}.
(49)
Using the parallel-axis theorem, the inertia about the CoM follows as
I
CoM
=
I
P
−
m
​
r
2
.
\displaystyle I_{\text{CoM}}\;=\;I_{P}-mr^{2}.
(50)
C.2
Measured Values and Uncertainties
We assume negligible uncertainty in
m
m
and
g
g
, while
r
r
and
f
f
are measured with uncertainties of
\qty
1
\centi
and
\qty
0.3 (ten-cycle timing). Error propagation is computed via
σ
2
=
∑
i
(
∂
f
∂
x
i
)
2
​
σ
x
i
2
,
\displaystyle\sigma^{2}\;=\;\sum_{i}\left(\frac{\partial f}{\partial x_{i}}\right)^{2}\sigma_{x_{i}}^{2},
(51)
and applied to both thigh and shank. The resulting inertias and variances are summarized in Table
6
, which also contrasts experimental and CAD values.
C.3
Results
For the thigh, the inertia about the CoM is
I
T
,
CoM
\displaystyle I_{T,\mathrm{CoM}}
=
0.0786
kg
m
2
±
0.0318
kg
m
2
,
\displaystyle=$0.0786\text{\,}\mathrm{kg}\text{\,}{\mathrm{m}}^{2}$\pm$0.0318\text{\,}\mathrm{kg}\text{\,}{\mathrm{m}}^{2}$,
(52)
while the shank yields
I
S
,
CoM
\displaystyle I_{S,\mathrm{CoM}}
=
0.0085
kg
m
2
±
0.0028
kg
m
2
.
\displaystyle=$0.0085\text{\,}\mathrm{kg}\text{\,}{\mathrm{m}}^{2}$\pm$0.0028\text{\,}\mathrm{kg}\text{\,}{\mathrm{m}}^{2}$.
(53)
Figure
24
illustrates the resulting dynamics envelopes, including
2
​
σ
2\sigma
uncertainty bounds. Compared to CAD measurements (Table
6
), the thigh inertia is noticeably higher, while the shank remains consistent.
These results explain part of the fourfold increase observed in the full-robot
I
a
I_{a}
and confirm that the fitted inertia represents the combined effects of rotor, link, and compensation dynamics. Figure
19
further visualizes the parameter spread under uncertainty, showing how these variations remain fully captured by
\ac
pace.
Figure 24
:
Breakdown of reduced inertia contributions for the full robot configuration for
\robot
Tytan in Section
3.2.2
. The bar plot shows the distribution of components shaping the fitted joint inertia estimate
I
a
I_{a}
(dotted black line) obtained with
\ac
pace. The solid line denotes the
±
1
​
σ
\pm 1\sigma
confidence interval from analytic measurements in Appendix
C
.
Table 6
:
Dynamics Properties for the Leg from CAD, measured as well as absolute error margins.
CAD
Measured
Difference %
Measurement Error
‖
r
1
‖
\parallel r_{1}\parallel
[
m
\mathrm{m}
]
–
0.30
0.30
–
±
0.01
\pm 0.01
‖
r
2
‖
\parallel r_{2}\parallel
[
m
\mathrm{m}
]
0.24
0.24
0.25
0.25
4.2
4.2
±
0.01
\pm 0.01
‖
r
3
‖
\parallel r_{3}\parallel
[
m
\mathrm{m}
]
0.075
0.075
0.105
0.105
40
40
±
0.01
\pm 0.01
Gravity
g
g
[
m
s
−
2
\mathrm{m}\text{\,}{\mathrm{s}}^{-2}
]
9.81
9.81
9.806
9.806
0.0
0.0
–
Thigh Mass
m
T
m_{T}
[
kg
\mathrm{kg}
]
3.612
3.612
3.775
3.775
4.5
4.5
–
Shank Mass
m
S
m_{S}
[
kg
\mathrm{kg}
]
0.4535
0.4535
0.4803
0.4803
5.9
5.9
–
Thigh Eigenfrequency
f
T
f_{T}
[
Hz
\mathrm{Hz}
]
–
0.82
0.82
–
±
0.03
\pm 0.03
Shank Eigenfrequency
f
S
f_{S}
[
Hz
\mathrm{Hz}
]
–
0.88
0.88
–
±
0.03
\pm 0.03
Thigh Inertia
I
T
I_{T}
[
kg
m
2
\mathrm{kg}\text{\,}{\mathrm{m}}^{2}
]
0.0228
0.0228
0.0786
0.0786
240
240
±
0.0318
\pm 0.0318
Shank Inertia
I
S
I_{S}
[
kg
m
2
\mathrm{kg}\text{\,}{\mathrm{m}}^{2}
]
0.0107
0.0107
0.008 50
0.008\,50
−
21
-21
±
0.002 76
\pm 0.002\,76
CoM induced Thigh Inertia [
kg
m
2
\mathrm{kg}\text{\,}{\mathrm{m}}^{2}
]
0.0203
0.0203
0.0416
0.0416
100
100
±
0.007 93
\pm 0.007\,93
Figure 25
:
Full battery depletion time of
\robot
ANYmal (Top) and
\robot
Tytan (bottom) at walking in blue, resting (Electronics & drives) in orange, and lying on the ground in green (Electronics).
Table 7
:
PPO Hyperparameters for RL Pipeline
Parameter
Value
Empirical normalization
True
Number of iterations
30 000
30\,000
Value loss coefficient
c
v
c_{v}
1.0
1.0
Clipped value loss
True
Clipping parameter
ϵ
\epsilon
0.2
0.2
Entropy coefficient
α
\alpha
(initial)
2
×
10
−
3
2\text{\times}{10}^{-3}
Entropy coefficient
α
\alpha
(final)
5
×
10
−
4
5\text{\times}{10}^{-4}
Entropy decay turn-over point
20 000
20\,000
Number of learning epochs
N
epoch
N_{\text{epoch}}
5
5
Number of mini-batches
N
mb
N_{\text{mb}}
10
10
Learning rate
η
\eta
​
10
−
3
{10}^{-3}
Learning rate schedule
adaptive
Discount factor
γ
\gamma
0.99
0.99
GAE parameter
λ
\lambda
0.95
0.95
Desired KL divergence
D
KL
target
D_{\text{KL}}^{\text{target}}
​
10
−
2
{10}^{-2}
Max gradient norm
‖
∇
‖
max
\|\nabla\|_{\text{max}}
1.0
1.0
Actor hidden nodes
[
256
,
256
,
256
,
128
]
[256,\,256,\,256,\,128]
Critic hidden nodes
[
256
,
256
,
256
,
128
]
[256,\,256,\,256,\,128]
Initial policy std
σ
0
\sigma_{0}
1.5
1.5
Steps per environment
N
step
N_{\text{step}}
24
24
Activation function
exponential linear unit (ELU)