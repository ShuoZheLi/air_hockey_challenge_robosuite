from .controller_factory import controller_factory, load_controller_config, reset_controllers, get_pybullet_server
from .air_hockey_osc import AirHockeyOperationalSpaceController
from .osc import OperationalSpaceController
from .joint_pos import JointPositionController
from .joint_vel import JointVelocityController
from .joint_tor import JointTorqueController


CONTROLLER_INFO = {
    "JOINT_VELOCITY": "Joint Velocity",
    "JOINT_TORQUE": "Joint Torque",
    "JOINT_POSITION": "Joint Position",
    "OSC_POSITION": "Operational Space Control (Position Only)",
    "OSC_POSE": "Operational Space Control (Position + Orientation)",
    "AIR_HOCKEY_OSC_POSE": "Customized Operational Space Control (keeps eef on table) (Position + Orientation)",
    "AIR_HOCKEY_OSC_POSITION": "Customized Operational Space Control (keeps eef on table) (Position + Orientation)",
    "IK_POSE": "Inverse Kinematics Control (Position + Orientation) (Note: must have PyBullet installed)",
}

ALL_CONTROLLERS = CONTROLLER_INFO.keys()
