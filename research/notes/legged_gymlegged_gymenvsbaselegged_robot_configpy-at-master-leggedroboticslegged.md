---
title: legged_gym/legged_gym/envs/base/legged_robot_config.py at master · leggedrobotics/legged_gym
  · GitHub
id: legged_gymlegged_gymenvsbaselegged_robot_configpy-at-master-leggedroboticslegged
tags:
- legged-rl-budgets
created: '2026-05-06T07:32:13.798312Z'
updated: '2026-05-06T07:38:17.020906Z'
source: https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot_config.py
source_domain: github.com
fetched_at: '2026-05-06T07:32:13.797312Z'
fetch_provider: builtin
status: draft
type: note
tier: ground_truth
content_type: code
deprecated: false
summary: 'legged_gym base config file — contains verbatim default training hyperparameters
  for the Rudin 2021 legged_gym PPO setup. Key values: num_envs=4096, num_steps_per_env=24
  (rollout length), max_iterations=1500, episode_length_s=20s, num_learning_epochs=5,
  num_mini_batches=4, learning_rate=1e-3. Derived total steps: 4096 x 24 x 1500 =
  147.5M. This is the ground-truth source for the legged_gym default training budget.'
---

*Suggested by [[github-leggedroboticslegged_gym-isaac-gym-environments-for-legged-robots-github]] — Base legged robot training config with num_envs and iterations defaults*

legged_gym/legged_gym/envs/base/legged_robot_config.py at master · leggedrobotics/legged_gym · GitHub
Skip to content
You signed in with another tab or window.
Reload
to refresh your session.
You signed out in another tab or window.
Reload
to refresh your session.
You switched accounts on another tab or window.
Reload
to refresh your session.
Dismiss alert
leggedrobotics
/
legged_gym
Public
Notifications
You must be signed in to change notification settings
Fork
566
Star
2.9k
Files
Expand file tree
master
/
legged_robot_config.py
Copy path
Blame
More file actions
Blame
More file actions
Latest commit
History
History
History
244 lines (224 loc) · 9.82 KB
master
/
legged_robot_config.py
Top
File metadata and controls
Code
Blame
244 lines (224 loc) · 9.82 KB
Raw
Copy raw file
Download raw file
Open symbols panel
Edit and raw actions
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40
41
42
43
44
45
46
47
48
49
50
51
52
53
54
55
56
57
58
59
60
61
62
63
64
65
66
67
68
69
70
71
72
73
74
75
76
77
78
79
80
81
82
83
84
85
86
87
88
89
90
91
92
93
94
95
96
97
98
99
100
101
102
103
104
105
106
107
108
109
110
111
112
113
114
115
116
117
118
119
120
121
122
123
124
125
126
127
128
129
130
131
132
133
134
135
136
137
138
139
140
141
142
143
144
145
146
147
148
149
150
151
152
153
154
155
156
157
158
159
160
161
162
163
164
165
166
167
168
169
170
171
172
173
174
175
176
177
178
179
180
181
182
183
184
185
186
187
188
189
190
191
192
193
194
195
196
197
198
199
200
201
202
203
204
205
206
207
208
209
210
211
212
213
214
215
216
217
218
219
220
221
222
223
224
225
226
227
228
229
230
231
232
233
234
235
236
237
238
239
240
241
242
243
244
# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin
from
.
base_config
import
BaseConfig
class
LeggedRobotCfg
(
BaseConfig
):
class
env
:
num_envs
=
4096
num_observations
=
235
num_privileged_obs
=
None
# if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise
num_actions
=
12
env_spacing
=
3.
# not used with heightfields/trimeshes
send_timeouts
=
True
# send time out information to the algorithm
episode_length_s
=
20
# episode length in seconds
class
terrain
:
mesh_type
=
'trimesh'
# "heightfield" # none, plane, heightfield or trimesh
horizontal_scale
=
0.1
# [m]
vertical_scale
=
0.005
# [m]
border_size
=
25
# [m]
curriculum
=
True
static_friction
=
1.0
dynamic_friction
=
1.0
restitution
=
0.
# rough terrain only:
measure_heights
=
True
measured_points_x
=
[
-
0.8
,
-
0.7
,
-
0.6
,
-
0.5
,
-
0.4
,
-
0.3
,
-
0.2
,
-
0.1
,
0.
,
0.1
,
0.2
,
0.3
,
0.4
,
0.5
,
0.6
,
0.7
,
0.8
]
# 1mx1.6m rectangle (without center line)
measured_points_y
=
[
-
0.5
,
-
0.4
,
-
0.3
,
-
0.2
,
-
0.1
,
0.
,
0.1
,
0.2
,
0.3
,
0.4
,
0.5
]
selected
=
False
# select a unique terrain type and pass all arguments
terrain_kwargs
=
None
# Dict of arguments for selected terrain
max_init_terrain_level
=
5
# starting curriculum state
terrain_length
=
8.
terrain_width
=
8.
num_rows
=
10
# number of terrain rows (levels)
num_cols
=
20
# number of terrain cols (types)
# terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
terrain_proportions
=
[
0.1
,
0.1
,
0.35
,
0.25
,
0.2
]
# trimesh only:
slope_treshold
=
0.75
# slopes above this threshold will be corrected to vertical surfaces
class
commands
:
curriculum
=
False
max_curriculum
=
1.
num_commands
=
4
# default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
resampling_time
=
10.
# time before command are changed[s]
heading_command
=
True
# if true: compute ang vel command from heading error
class
ranges
:
lin_vel_x
=
[
-
1.0
,
1.0
]
# min max [m/s]
lin_vel_y
=
[
-
1.0
,
1.0
]
# min max [m/s]
ang_vel_yaw
=
[
-
1
,
1
]
# min max [rad/s]
heading
=
[
-
3.14
,
3.14
]
class
init_state
:
pos
=
[
0.0
,
0.0
,
1.
]
# x,y,z [m]
rot
=
[
0.0
,
0.0
,
0.0
,
1.0
]
# x,y,z,w [quat]
lin_vel
=
[
0.0
,
0.0
,
0.0
]
# x,y,z [m/s]
ang_vel
=
[
0.0
,
0.0
,
0.0
]
# x,y,z [rad/s]
default_joint_angles
=
{
# target angles when action = 0.0
"joint_a"
:
0.
,
"joint_b"
:
0.
}
class
control
:
control_type
=
'P'
# P: position, V: velocity, T: torques
# PD Drive parameters:
stiffness
=
{
'joint_a'
:
10.0
,
'joint_b'
:
15.
}
# [N*m/rad]
damping
=
{
'joint_a'
:
1.0
,
'joint_b'
:
1.5
}
# [N*m*s/rad]
# action scale: target angle = actionScale * action + defaultAngle
action_scale
=
0.5
# decimation: Number of control action updates @ sim DT per policy DT
decimation
=
4
class
asset
:
file
=
""
name
=
"legged_robot"
# actor name
foot_name
=
"None"
# name of the feet bodies, used to index body state and contact force tensors
penalize_contacts_on
=
[]
terminate_after_contacts_on
=
[]
disable_gravity
=
False
collapse_fixed_joints
=
True
# merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
fix_base_link
=
False
# fixe the base of the robot
default_dof_drive_mode
=
3
# see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
self_collisions
=
0
# 1 to disable, 0 to enable...bitwise filter
replace_cylinder_with_capsule
=
True
# replace collision cylinders with capsules, leads to faster/more stable simulation
flip_visual_attachments
=
True
# Some .obj meshes must be flipped from y-up to z-up
density
=
0.001
angular_damping
=
0.
linear_damping
=
0.
max_angular_velocity
=
1000.
max_linear_velocity
=
1000.
armature
=
0.
thickness
=
0.01
class
domain_rand
:
randomize_friction
=
True
friction_range
=
[
0.5
,
1.25
]
randomize_base_mass
=
False
added_mass_range
=
[
-
1.
,
1.
]
push_robots
=
True
push_interval_s
=
15
max_push_vel_xy
=
1.
class
rewards
:
class
scales
:
termination
=
-
0.0
tracking_lin_vel
=
1.0
tracking_ang_vel
=
0.5
lin_vel_z
=
-
2.0
ang_vel_xy
=
-
0.05
orientation
=
-
0.
torques
=
-
0.00001
dof_vel
=
-
0.
dof_acc
=
-
2.5e-7
base_height
=
-
0.
feet_air_time
=
1.0
collision
=
-
1.
feet_stumble
=
-
0.0
action_rate
=
-
0.01
stand_still
=
-
0.
only_positive_rewards
=
True
# if true negative total rewards are clipped at zero (avoids early termination problems)
tracking_sigma
=
0.25
# tracking reward = exp(-error^2/sigma)
soft_dof_pos_limit
=
1.
# percentage of urdf limits, values above this limit are penalized
soft_dof_vel_limit
=
1.
soft_torque_limit
=
1.
base_height_target
=
1.
max_contact_force
=
100.
# forces above this value are penalized
class
normalization
:
class
obs_scales
:
lin_vel
=
2.0
ang_vel
=
0.25
dof_pos
=
1.0
dof_vel
=
0.05
height_measurements
=
5.0
clip_observations
=
100.
clip_actions
=
100.
class
noise
:
add_noise
=
True
noise_level
=
1.0
# scales other values
class
noise_scales
:
dof_pos
=
0.01
dof_vel
=
1.5
lin_vel
=
0.1
ang_vel
=
0.2
gravity
=
0.05
height_measurements
=
0.1
# viewer camera:
class
viewer
:
ref_env
=
0
pos
=
[
10
,
0
,
6
]
# [m]
lookat
=
[
11.
,
5
,
3.
]
# [m]
class
sim
:
dt
=
0.005
substeps
=
1
gravity
=
[
0.
,
0.
,
-
9.81
]
# [m/s^2]
up_axis
=
1
# 0 is y, 1 is z
class
physx
:
num_threads
=
10
solver_type
=
1
# 0: pgs, 1: tgs
num_position_iterations
=
4
num_velocity_iterations
=
0
contact_offset
=
0.01
# [m]
rest_offset
=
0.0
# [m]
bounce_threshold_velocity
=
0.5
#0.5 [m/s]
max_depenetration_velocity
=
1.0
max_gpu_contact_pairs
=
2
**
23
#2**24 -> needed for 8000 envs and more
default_buffer_size_multiplier
=
5
contact_collection
=
2
# 0: never, 1: last sub-step, 2: all sub-steps (default=2)
class
LeggedRobotCfgPPO
(
BaseConfig
):
seed
=
1
runner_class_name
=
'OnPolicyRunner'
class
policy
:
init_noise_std
=
1.0
actor_hidden_dims
=
[
512
,
256
,
128
]
critic_hidden_dims
=
[
512
,
256
,
128
]
activation
=
'elu'
# can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
# only for 'ActorCriticRecurrent':
# rnn_type = 'lstm'
# rnn_hidden_size = 512
# rnn_num_layers = 1
class
algorithm
:
# training params
value_loss_coef
=
1.0
use_clipped_value_loss
=
True
clip_param
=
0.2
entropy_coef
=
0.01
num_learning_epochs
=
5
num_mini_batches
=
4
# mini batch size = num_envs*nsteps / nminibatches
learning_rate
=
1.e-3
#5.e-4
schedule
=
'adaptive'
# could be adaptive, fixed
gamma
=
0.99
lam
=
0.95
desired_kl
=
0.01
max_grad_norm
=
1.
class
runner
:
policy_class_name
=
'ActorCritic'
algorithm_class_name
=
'PPO'
num_steps_per_env
=
24
# per iteration
max_iterations
=
1500
# number of policy updates
# logging
save_interval
=
50
# check for potential saves every this many iterations
experiment_name
=
'test'
run_name
=
''
# load and resume
resume
=
False
load_run
=
-
1
# -1 = last run
checkpoint
=
-
1
# -1 = last saved model
resume_path
=
None
# updated from load_run and chkpt
You can’t perform that action at this time.