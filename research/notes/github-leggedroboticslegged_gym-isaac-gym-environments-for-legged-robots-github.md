---
title: 'GitHub - leggedrobotics/legged_gym: Isaac Gym Environments for Legged Robots
  · GitHub'
id: github-leggedroboticslegged_gym-isaac-gym-environments-for-legged-robots-github
tags:
- legged-rl-budgets
- quadruped
- ppo
- config
- canonical-anchor
created: '2026-05-06T07:31:04.528553Z'
updated: '2026-05-06T07:57:53.578255Z'
source: https://github.com/leggedrobotics/legged_gym
source_domain: github.com
fetched_at: '2026-05-06T07:31:04.528553Z'
fetch_provider: builtin
status: draft
type: note
tier: ground_truth
content_type: code
deprecated: false
summary: 'leggedrobotics/legged_gym GitHub repo README (2.9k stars). Isaac Gym environments
  for legged robots by Nikita Rudin/ETH Zurich. Default training config from base
  class: num_envs=4096, num_steps_per_env=24 (rollout length), max_iterations=1500,
  episode_length=20s, 5 PPO learning epochs, 4 mini-batches, lr=1e-3. Total steps
  = 4096 x 24 x 1500 = 147,456,000 (~147M steps). ANYmal C flat override: max_iterations=300
  (flat terrain converges faster = ~29M steps). ANYmal C rough: max_iterations=1500
  (default ~147M). The paper behind this code is Rudin et al. 2021 arXiv:2109.11978.'
---

GitHub - leggedrobotics/legged_gym: Isaac Gym Environments for Legged Robots · GitHub
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
master
Branches
Tags
Go to file
Code
Open more actions menu
Folders and files
Name
Name
Last commit message
Last commit date
Latest commit
History
18 Commits
18 Commits
.github/
ISSUE_TEMPLATE
.github/
ISSUE_TEMPLATE
legged_gym
legged_gym
licenses
licenses
resources
resources
.gitattributes
.gitattributes
.gitignore
.gitignore
LICENSE
LICENSE
README.md
README.md
setup.py
setup.py
View all files
Repository files navigation
Isaac Gym Environments for Legged Robots
This repository provides the environment used to train ANYmal (and other robots) to walk on rough terrain using NVIDIA's Isaac Gym.
It includes all components needed for sim-to-real transfer: actuator network, friction & mass randomization, noisy observations and random pushes during training.
Maintainer
: Nikita Rudin
Affiliation
: Robotic Systems Lab, ETH Zurich
Contact
:
rudinn@ethz.ch
🔔 Announcement (09.01.2024)
With the shift from Isaac Gym to Isaac Sim at NVIDIA, we have migrated all the environments from this work to
Isaac Lab
. Following this migration, this repository will receive limited updates and support. We encourage all users to migrate to the new framework for their applications.
Information about this work's locomotion-related tasks in Isaac Lab is available
here
.
Useful Links
Project website:
https://leggedrobotics.github.io/legged_gym/
Paper:
https://arxiv.org/abs/2109.11978
Installation
Create a new python virtual env with python 3.6, 3.7 or 3.8 (3.8 recommended)
Install pytorch 1.10 with cuda-11.3:
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
Install Isaac Gym
Download and install Isaac Gym Preview 3 (Preview 2 will not work!) from
https://developer.nvidia.com/isaac-gym
cd isaacgym/python && pip install -e .
Try running an example
cd examples && python 1080_balls_of_solitude.py
For troubleshooting check docs
isaacgym/docs/index.html
)
Install rsl_rl (PPO implementation)
Clone
https://github.com/leggedrobotics/rsl_rl
cd rsl_rl && git checkout v1.0.2 && pip install -e .
Install legged_gym
Clone this repository
cd legged_gym && pip install -e .
CODE STRUCTURE
Each environment is defined by an env file (
legged_robot.py
) and a config file (
legged_robot_config.py
). The config file contains two classes: one containing  all the environment parameters (
LeggedRobotCfg
) and one for the training parameters (
LeggedRobotCfgPPo
).
Both env and config classes use inheritance.
Each non-zero reward scale specified in
cfg
will add a function with a corresponding name to the list of elements which will be summed to get the total reward.
Tasks must be registered using
task_registry.register(name, EnvClass, EnvConfig, TrainConfig)
. This is done in
envs/__init__.py
, but can also be done from outside of this repository.
Usage
Train:
python legged_gym/scripts/train.py --task=anymal_c_flat
To run on CPU add following arguments:
--sim_device=cpu
,
--rl_device=cpu
(sim on CPU and rl on GPU is possible).
To run headless (no rendering) add
--headless
.
Important
: To improve performance, once the training starts press
v
to stop the rendering. You can then enable it later to check the progress.
The trained policy is saved in
issacgym_anymal/logs/<experiment_name>/<date_time>_<run_name>/model_<iteration>.pt
. Where
<experiment_name>
and
<run_name>
are defined in the train config.
The following command line arguments override the values set in the config files:
--task TASK: Task name.
--resume:   Resume training from a checkpoint
--experiment_name EXPERIMENT_NAME: Name of the experiment to run or load.
--run_name RUN_NAME:  Name of the run.
--load_run LOAD_RUN:   Name of the run to load when resume=True. If -1: will load the last run.
--checkpoint CHECKPOINT:  Saved model checkpoint number. If -1: will load the last checkpoint.
--num_envs NUM_ENVS:  Number of environments to create.
--seed SEED:  Random seed.
--max_iterations MAX_ITERATIONS:  Maximum number of training iterations.
Play a trained policy:
python legged_gym/scripts/play.py --task=anymal_c_flat
By default, the loaded policy is the last model of the last run of the experiment folder.
Other runs/model iteration can be selected by setting
load_run
and
checkpoint
in the train config.
Adding a new environment
The base environment
legged_robot
implements a rough terrain locomotion task. The corresponding cfg does not specify a robot asset (URDF/ MJCF) and has no reward scales.
Add a new folder to
envs/
with
'<your_env>_config.py
, which inherit from an existing environment cfgs
If adding a new robot:
Add the corresponding assets to
resources/
.
In
cfg
set the asset path, define body names, default_joint_positions and PD gains. Specify the desired
train_cfg
and the name of the environment (python class).
In
train_cfg
set
experiment_name
and
run_name
(If needed) implement your environment in <your_env>.py, inherit from an existing environment, overwrite the desired functions and/or add your reward functions.
Register your env in
isaacgym_anymal/envs/__init__.py
.
Modify/Tune other parameters in your
cfg
,
cfg_train
as needed. To remove a reward set its scale to zero. Do not modify parameters of other envs!
Troubleshooting
If you get the following error:
ImportError: libpython3.8m.so.1.0: cannot open shared object file: No such file or directory
, do:
sudo apt install libpython3.8
. It is also possible that you need to do
export LD_LIBRARY_PATH=/path/to/libpython/directory
/
export LD_LIBRARY_PATH=/path/to/conda/envs/your_env/lib
(for conda user. Replace /path/to/ to the corresponding path.).
Known Issues
The contact forces reported by
net_contact_force_tensor
are unreliable when simulating on GPU with a triangle mesh terrain. A workaround is to use force sensors, but the force are propagated through the sensors of consecutive bodies resulting in an undesirable behaviour. However, for a legged robot it is possible to add sensors to the feet/end effector only and get the expected results. When using the force sensors make sure to exclude gravity from the reported forces with
sensor_options.enable_forward_dynamics_forces
. Example:
sensor_pose = gymapi.Transform()
    for name in feet_names:
        sensor_options = gymapi.ForceSensorProperties()
        sensor_options.enable_forward_dynamics_forces = False # for example gravity
        sensor_options.enable_constraint_solver_forces = True # for example contacts
        sensor_options.use_world_frame = True # report forces in world frame (easier to get vertical components)
        index = self.gym.find_asset_rigid_body_index(robot_asset, name)
        self.gym.create_asset_force_sensor(robot_asset, index, sensor_pose, sensor_options)
    (...)

    sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
    self.gym.refresh_force_sensor_tensor(self.sim)
    force_sensor_readings = gymtorch.wrap_tensor(sensor_tensor)
    self.sensor_forces = force_sensor_readings.view(self.num_envs, 4, 6)[..., :3]
    (...)

    self.gym.refresh_force_sensor_tensor(self.sim)
    contact = self.sensor_forces[:, :, 2] > 1.
About
Isaac Gym Environments for Legged Robots
Resources
Readme
License
View license
Uh oh!
There was an error while loading.
Please reload this page
.
Activity
Custom properties
Stars
2.9k
stars
Watchers
47
watching
Forks
566
forks
Report repository
Releases
No releases published
Packages
0
Uh oh!
There was an error while loading.
Please reload this page
.
Uh oh!
There was an error while loading.
Please reload this page
.
Contributors
Uh oh!
There was an error while loading.
Please reload this page
.
Languages
Python
100.0%
You can’t perform that action at this time.