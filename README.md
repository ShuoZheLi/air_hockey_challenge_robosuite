# Welcome to RoboAirHockey!

# Install
cd into the repo folder, make sure Python version=3.9 for the environment

    pip3 install -e .
    pip3 install -r requirements-extra.txt
    pip3 install torch torchvision torchaudio (for Mac/Windows, check PyTorch for installation on Linux)
    pip3 install pygame
    pip3 install imageio
    pip3 install matplolib
    pip3 install PyYAML
    pip3 install xmltodict
    pip3 install foxglove_websocket
    pip3 install tensorboard

# Testing Your Code

You can test the IK with this command.

    python3 robosuite/demos/demo_device_control.py --robots UR5e --environment AirHockey

If you are adding in new controller, take a look at the code at `robosuite/demos/demo_device_control.py` .

# YAML Parameters that can be modified
    length: Table Length
    width: Table Width
    puck_radius: Puck Radius (m)
    puck_damping: Puck Damping

# Scripts
Running Browse_mjcf_model script - Loads MJCF XML models from file and renders it on screen. Example:

    python browse_mjcf_model.py --filepath ../models/assets/arenas/table_arena.xml

Running Collect_human_demonstrations script - collects a batch of human demonstrations. All params are optional.
Example:

    python collect_human_demonstrations.py --directory ... --environment ... --robots ... --config ... --arm ... --camera ... --controller ... --device ... --pos-sensitivity ... --rot-sensitivity ...

Running Compile_mjcf_model script - Loads a raw mjcf file and saves a compiled mjcf file. Example:

    python compile_mjcf_model.py source_mjcf.xml target_mjcf.xml

Running Playback_demonstrations_from_hdf5 script - playbacks random demonstrations from
a set of demonstrations stored in a hdf5 file. Example:

    python playback_demonstrations_from_hdf5.py --folder ../models/assets/demonstrations/lift/

Running make_reset_video script - makes a video out of initial environment configurations. This can be a useful debugging tool to understand what different sampled environment configurations look like. All params are optional.
Example:

    python make_reset_video.py --camera ... --frames ... --output ...

Running ppo_continuous_action script - runs the proximal policy optimization code for a continuous action space. All params are optional.

    python ppo_continuous_action.py --exp-name ... --seed ... --torch-deterministic ... --cuda ... --track ...--wandb-project-name ... --wandb-entity ... --capture-video ... --save-model ... --upload-model ... --hf-entity ... --checkpoint-path ... --env-id ... --task ... --total-timesteps ... --learning-rate ... --num-envs ... --num-steps ... --anneal-lr ... --gamma ... --gae-lambda ... --num-minibatches ... --update-epochs ... --norm-adv ... --clip-coef ... --clip-vloss ... --ent-coef ... --vf-coef ... --max-grad-norm ... --target-kl ...

Running setup_macros script - Sets up a private macros file. The private macros file (macros_private.py) is not tracked by git, allowing user-specific settings that are not tracked by git. If applicable, it creates the private macros at robosuite/macros_private.py. Run:

    python setup_macros.py

Running teleop script - Script to run teleoperations with the robot in the simulation environment, and can also
collect data while performing teleoperations. Params are optional. Example:

    python teleop.py --collect-data ... --save-folder-path ... --foxglove-log ...

Running train_example script - Runs an example of the UR5e robot in the air hockey environment.

    python train_example.py

Running tune_camera script - tunes a camera view in a mujoco environment. Allows keyboard presses to move a camera around in the viewer, and then prints the final position and quaternion you should set for your camera in the mujoco XML file. Params are optional. Example:

    python tune_camera.py --env ... --robots ...

Running tune_joints script - tunes a robot's joint positions in a mujoco environment. Allows keyboard presses to move specific robot joints around in the viewer, and then prints the current joint parameters upon an inputted command. Params are optional. Example:

    python tune_joints.py --env ... --robots ... --init_qpos ...

    RELEVANT KEY PRESSES:
        '1 - n' : Sets the active robot joint being tuned to this number. Maximum
            is n which is the number of robot joints
        't' : Toggle between robot arms being tuned (only applicable for multi-arm environments)
        'r' : Resets the active joint values to 0
        'UP_ARROW' : Increment the active robot joint position
        'DOWN_ARROW' : Decrement the active robot joint position
        'RIGHT_ARROW' : Increment the delta joint position change per keypress
        'LEFT_ARROW' : Decrement the delta joint position change per keypress
    
Running visualize_model script - Loads model from checkpoint path (change accordingly) and visualizes the robot's actions in the environment based on the trained model. Example:

    python visualize_model.py

# Utils
binding_utils.py: 
- Useful classes for supporting DeepMind MuJoCo binding. 
- Many methods for retrieving information from the MuJoCo environment, such as position, orientation, velocity, acceleration, etc. Look for more information in the file.

buffers.py: 
- Collection of Buffer objects with general functionality

camera_utils.py: 
- Utility classes for modifying sim cameras. 
- Utility functions for performing common camera operations such as retrieving camera matrices and transforming from world to camera frame or vice-versa. 

collect_data.py: Utility class to perform data collection for a specific task. Saves joint positions and velocities, end effector positions, puck positions, and other observations. --data-length param optional. Example:

    python collect_data.py --model-path ... --save-folder-path ... --task ... --data-length ...

control_utils.py:
- Utility class for calculating goal position, goal orientation, nullspace torques to keep the robot joints in given positions, relevant matrices for the operational space control, and orientation error.
- Methods include nullspace_torques, opspace_matrices, orientation_error, set_goal_position, set_goal_orientation. Check for more information in the file.

dataset_examine.py: Examines the dataset by comparing the total size currently to the desired size.

    python dataset_examine.py --data-path ...

    DATA FORMAT
    data: (mouse_x, mouse_y, puck_pos, puck_vel, gripper_pos, gripper_vel, joint_pos, joint_vel, time, delta_time)

errors.py:
- Exception handling for errors due to Robosuite, XML, runtime, or RNG.

foxglove_logging.py:
- Creates a logger that sends and receive messages from a client foxglove server. 

input_utils.py:
- Utility functions for grabbing user inputs. Allows to choose environment, controller, multi_arm_config, robots, and converts inputs to a valid action sequence that can be fed into an env.step() call. 

log_utils.py:
- This file contains utility classes and functions for logging to stdout and stderr. Selects specifc formats for the Logger which is returned to be used to log to the console or to a file.

mjcf_utils.py:
- Utility functions for manipulating MJCF XML models. Can be used to render new custom elements into the MuJoCo environment. See more information in the file.

mjmod.py:
- Modder classes used for domain randomization. Largely based off of the mujoco-py
implementation below. See more information in the file.

numba.py:
- Numba utils.

observables.py:
- Utils class to define Observables and methods to set sensors, enable an observable, reset an observable, update an observable, etc.

opencv_renderer.py:
- Opencv render class, sets up the camera and renders the frames of the camera.

placement_samplers.py:
- Utils class for placement of objects inside the environment. Contains a base object placement sampler class, a uniform placement sampler class, and sequential placement sampler class. See more information in the file. 

recreate_scene.py: Recreates the scenes in the environment based on data given as an argument. 

    python recreate_scene.py --data-path ...

RobosuiteTransforms.py:
- Utils class for calculating z position, line intersections, transforms pixels to relative coordinates, transforms relative to world coordinates, and transforms pixels to world coordinates.

robot_utils.py:
- Utilities functions for working with robots

sim_utils.py:
- Collection of useful simulation utilities. Checks for any contact between geom objects in the environment.

transform_utils.py:
- Utils class for transformations and operations for matrices, quaternions, rotation matrices, euler angles, axis angles, pose, etc. 

validation.py: validate observations, such as End Effector Orientation, Position, Translational Velocity, and Rotational Velocity from the Robosuite environemnt. Params are required. Example:

    python validation.py --fileNamePath ValidationResultsTest1 --dataset ../data/dataset.npy

# Operational Space Controller (osc.py)

For parameter specifications and initialization, please take a look at the detailed documentation at the header of the Operational Space Controller class.

The set_goal method sets goal based on input @action. If self.impedance_mode is not "fixed", then the input will be parsed into the delta values to update the goal position / pose and the kp and/or damping_ratio values to be immediately updated internally before executing the proceeding control loop.

The run_controller method calculates the torques and forces required to reach the desired setpoint with either position or both position and orientation. Performs computations in the operational space matrices.

The update_initial_joints updates all the joint positions to their initial state and also calls reset_goal.

The reset_goal method resets the goal to the current state of the robot and resets the interpolators if necessary. It also calculates the goal_orientation error.

The control_limits method returns the limits over this controller's action space, overrides the superclass property.