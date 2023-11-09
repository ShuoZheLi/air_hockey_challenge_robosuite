# Welcome to RoboAirHockey!

lol

# Install
git clone git@github.com:ShuoZheLi/air_hockey_challenge_robosuite.git
cd air_hockey_challenge_robosuite
git checkout -b reaching_policy
git pull origin reaching_policy
conda create -n air python=3.9
pip install -e .
python3 robosuite/demos/demo_device_control.py --robots UR5e --environment AirHockey

<!-- cd into the repo folder

    pip3 install -e . -->

# Testing Your Code

You can test the IK with this command.

    python3 robosuite/demos/demo_device_control.py --robots UR5e --environment AirHockey

If you are adding in new controller, take a look at the code at `robosuite/demos/demo_device_control.py` .

# Plug in RL

    python3 train_example.py
