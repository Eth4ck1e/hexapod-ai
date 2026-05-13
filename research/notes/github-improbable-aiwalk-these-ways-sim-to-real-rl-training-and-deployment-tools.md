---
title: 'GitHub - Improbable-AI/walk-these-ways: Sim-to-real RL training and deployment
  tools for the Unitree Go1 robot. · GitHub'
id: github-improbable-aiwalk-these-ways-sim-to-real-rl-training-and-deployment-tools
tags:
- legged-rl-budgets
created: '2026-05-06T07:32:51.935137Z'
source: https://github.com/Improbable-AI/walk-these-ways
source_domain: github.com
fetched_at: '2026-05-06T07:32:51.935137Z'
fetch_provider: builtin
status: draft
type: note
tier: ground_truth
content_type: code
deprecated: false
---

*Suggested by [[adversarial-motion-priors-make-good-substitutes-for-complex-reward-functions-ale]] — Code repository linked from Escontrela AMP project page*

GitHub - Improbable-AI/walk-these-ways: Sim-to-real RL training and deployment tools for the Unitree Go1 robot. · GitHub
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
Improbable-AI
/
walk-these-ways
Public
Notifications
You must be signed in to change notification settings
Fork
217
Star
1.3k
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
25 Commits
25 Commits
LICENSES
LICENSES
go1_gym
go1_gym
go1_gym_deploy
go1_gym_deploy
go1_gym_learn
go1_gym_learn
logs/
example_experiment/
2022/
11_01/
16_01_50_0
logs/
example_experiment/
2022/
11_01/
16_01_50_0
media
media
resources
resources
runs/
gait-conditioned-agility/
pretrain-v0/
train/
025417.456545
runs/
gait-conditioned-agility/
pretrain-v0/
train/
025417.456545
scripts
scripts
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
Go1 Sim-to-Real Locomotion Starter Kit
Table of contents
Overview
System Requirements
Training a Model
Installation
Environment and Model Configuration
Training and Logging
Analyzing the Policy
Deploying a Model
Installing the Deployment Utility
Running the Controller
RC Configuration
Deploying a Custom Model
Deployment and Logging
Analyzing Real-world Performance
Debugging Common Errors
Overview
This repository provides an implementation of the paper:
Walk these Ways: Tuning Robot Control for Generalization with Multiplicity of Behavior
Gabriel B. Margolis
and
Pulkit Agrawal
Conference on Robot Learning
, 2022
paper
/
project page
If you use this repository in your work, consider citing:
@article{margolis2022walktheseways,
    title={Walk These Ways: Tuning Robot Control for Generalization with Multiplicity of Behavior},
    author={Margolis, Gabriel B and Agrawal, Pulkit},
    journal={Conference on Robot Learning},
    year={2022}
}
This environment builds on the
legged gym environment
by Nikita
Rudin, Robotic Systems Lab, ETH Zurich (Paper:
https://arxiv.org/abs/2109.11978
) and the Isaac Gym simulator from
NVIDIA (Paper:
https://arxiv.org/abs/2108.10470
). Training code builds on the
rsl_rl
repository, also by Nikita
Rudin, Robotic Systems Lab, ETH Zurich. All redistributed code retains its
original
license
.
Our initial release provides the following features:
Train reinforcement learning policies for the Go1 robot using PPO, IsaacGym, Domain Randomization, and Multiplicity of Behavior (MoB).
Evaluate a pretrained MoB policy in simulation.
Deploy learned policies on the Go1 using the
unitree_legged_sdk
.
System Requirements
Simulated Training and Evaluation
: Isaac Gym requires an NVIDIA GPU. To train in the default configuration, we recommend a GPU with at least 10GB of VRAM. The code can run on a smaller GPU if you decrease the number of parallel environments (
Cfg.env.num_envs
). However, training will be slower with fewer environments.
Hardware Deployment
: We provide deployment code for the Unitree Go1 Edu robot. This relatively low-cost, commercially available quadruped can be purchased here:
https://shop.unitree.com/
. You will need the Edu version of the robot to run and customize your locomotion controller.
Training a Model
Installation
Install pytorch 1.10 with cuda-11.3:
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
Install Isaac Gym
Download and install Isaac Gym Preview 4 from
https://developer.nvidia.com/isaac-gym
unzip the file via:
tar -xf IsaacGym_Preview_4_Package.tar.gz
now install the python package
cd
isaacgym/python
&&
pip install -e
.
Verify the installation by try running an example
python examples/1080_balls_of_solitude.py
For troubleshooting check docs
isaacgym/docs/index.html
Install the
go1_gym
package
In this repository, run
pip install -e .
Verifying the Installation
If everything is installed correctly, you should be able to run the test script with:
python scripts/test.py
The script should print
Simulating step {i}
.
The GUI is off by default. To turn it on, set
headless=False
in
test.py
's main function call.
Environment and Model Configuration
CODE STRUCTURE
The main environment for simulating a legged robot is
in
legged_robot.py
. The default configuration parameters including reward
weightings are defined in
legged_robot_config.py::Cfg
.
There are three scripts in the
scripts
directory:
scripts
├── __init__.py
├── play.py
├── test.py
└── train.py
You can run the
test.py
script to verify your environment setup. If it runs then you have installed the gym
environments correctly. To train an agent, run
train.py
. To evaluate a pretrained agent, run
play.py
. We provie a
pretrained agent checkpoint in the
./runs/pretrain-v0
directory.
Training and Logging
To train the Go1 controller from
Walk these Ways
, run:
python scripts/train.py
After initializing the simulator, the script will print out a list of metrics every ten training iterations.
Training with the default configuration requires about 12GB of GPU memory. If you have less memory available, you can
still train by reducing the number of parallel environments used in simulation (the default is
Cfg.env.num_envs = 4000
).
To visualize training progress, first start the ml_dash frontend app:
python -m ml_dash.app
then start the ml_dash backend server by running this command in the parent directory of the
runs
folder:
python -m ml_dash.server
.
Finally, use a web browser to go to the app IP (defaults to
localhost:3001
)
and create a new profile with the credentials:
Username:
runs
API: [server IP] (defaults to
localhost:8081
)
Access Token: [blank]
Now, clicking on the profile should yield a dashboard interface visualizing the training runs.
Analyzing the Policy
To evaluate the most recently trained model, run:
python scripts/play.py
The robot is commanded to run forward at 3m/s for 5 seconds. After completing the simulation,
the script plots the robot's velocity and joint angles.
The GUI is on by default.
If it does not appear, and you're working in docker, make sure you haven't forgotten to run
bash docker/visualize_access.bash
.
Deploying a Model
Safety Recommendations
Users are advised to follow Unitree's recommendations for safety while using the Go1 in low-level control mode.
This means hanging up the robot and keeping it away from people and obstacles.
In practice, the main safety consideration we've found important has been not plug anything into the robot's back (ethernet cable, USB) during the initial calibration or when testing a new policy because it can hurt the robot in case of a fall.
Our code implements the safety layer from Unitree's
unitree_legged_sdk
with PowerProtect level 9. This will cut off power to the motors if the joint torque is too high (could happen sometimes during fast running)
This is research code; use at your own risk; we do not take responsibility for any damage.
Installing the Deployment Utility
The first step is to connect your development machine to the robot using ethernet. You should ping the robot to verify the connection:
ping 192.168.123.15
should return
x packets transmitted, x received, 0% packet loss
.
Once you have confirmed the robot is connected, run the following command on your computer to transfer files to the robot. The first time you run it, the script will download and transfer the zipped docker image for development on the robot (
deployment_image.tar
). This file is quite large (3.5GB), but it only needs to be downloaded and transferred once.
cd go1_gym_deploy/scripts && ./send_to_unitree.sh
Next, you will log onto the robot's onboard computer and install the docker environment. To enter the onboard computer, the command is:
ssh unitree@192.168.123.15
Now, run the following commands on the robot's onboard computer:
chmod +x installer/install_deployment_code.sh
cd ~/go1_gym/go1_gym_deploy/scripts
sudo ../installer/install_deployment_code.sh
The installer will automatically unzip and install the docker image containing the deployment environment.
Running the Controller
Place the robot into damping mode. The control sequence is: [L2+A], [L2+B], [L1+L2+START]. After this, the robot should sit on the ground and the joints should move freely.
Now, ssh to
unitree@192.168.123.15
and run the following two commands to start the controller.
This will operate the robot in low-level control mode. Make sure your Go1 is hung up.
First:
cd ~/go1_gym/go1_gym_deploy/autostart
./start_unitree_sdk.sh
Second:
cd ~/go1_gym/go1_gym_deploy/docker
sudo make autostart
The robot will wait for you to press [R2], then calibrate, then wait for a second press of [R2] before running the control loop.
The RC Mapping
The RC mapping is depicted above.
Deploying a Custom Model
After training a custom model, it will be saved in the
runs
folder (
https://github.com/Improbable-AI/walk-these-ways/tree/master/runs/
). Note the relative location of your custom model of the
train
folder (for the default policy), it's
gait-conditioned-agility/pretrain-v0/train
. We'll denote this as
$PDIR
.
To play the custom model in simulation first, replace the line
https://github.com/Improbable-AI/walk-these-ways/blob/master/scripts/play.py#L97
with
label = "$PDIR"
.
To deploy on the robot, replace the line
https://github.com/Improbable-AI/walk-these-ways/blob/master/go1_gym_deploy/scripts/deploy_policy.py#L73
with
label = "$PDIR"
. Then re-run the
send_to_unitree.sh
script to update the files on the robot.
Logging and Debugging
Coming soon
Analyzing Real-world Performance
Coming soon
Debugging Common Errors
Bug
Solution
First report
Out of disk space
If you run out of disk space during
cd ~/go1_gym/go1_gym_deploy/installer && ./install_deployment_code.sh
consider changing the script to use
192.168.123.13
instead (at least in my Go1 Edu with 3 Jetson nano, I only had the required disk space to copy the tar and extract the image in only
192.168.123.13
). Alternatively, consider deploying on an external PC.
#7
lcm_position
syntax error
When deploying with
sudo ./start_unitree_sdk.sh
on an external PC/NUC, if you get the following error:
./lcm_position: 1: Syntax error: word unexpected (expecting ")")
, It is likely because the ./lcm_position has been compiled for ARM aarch64 (to run on the jetson), please recompile it for your architecture(external PC/ NUC) using
https://github.com/Improbable-AI/unitree_legged_sdk
.
#7
About
Sim-to-real RL training and deployment tools for the Unitree Go1 robot.
gmargo11.github.io/walk-these-ways/
Topics
reinforcement-learning
robotics
go1
sim-to-real
unitree
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
1.3k
stars
Watchers
14
watching
Forks
217
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
95.8%
C++
1.7%
Dockerfile
1.7%
Other
0.8%
You can’t perform that action at this time.