import numpy as np
import math

import robosuite as suite
from robosuite.wrappers import GymWrapper
from robosuite.wrappers.visualization_wrapper import VisualizationWrapper
from robosuite.utils.transform_utils import _AXES2TUPLE, mat2quat, quat2mat, mat2euler, convert_quat, euler2mat

# config = {'env_name': 'AirHockey',
#           'robots': ['UR5e'],
#           'controller_configs':
#                 {'type': 'OSC_POSITION',
#                 'interpolation': None,
#                 "impedance_mode" : "fixed"},
#         'gripper_types': 'Robotiq85Gripper',}



config = {'env_name': 'AirHockey',
              'robots': ['UR5e'],
              'controller_configs':
                  {'type': 'OSC_POSE',
                   "kp": [0,0,0,0,0,0],
                   "damping_ratio": [1, 1, 1, 1, 1, 1],
                   'interpolation': 'linear',
                   "impedance_mode": "fixed",
                   "control_delta": False,
                   "ramp_ratio": 1,
                   "kp_limits": (0, 10000000),
                   "uncouple_pos_ori": False,},
              'gripper_types': 'Robotiq85Gripper'}

env = suite.make(
        **config,
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera="sideview",
        # render_camera="birdview",
        use_camera_obs=False,
        control_freq=20,
    )

env = VisualizationWrapper(env)
env = GymWrapper(env, keys=['robot0_eef_pos',
                            'robot0_eef_quat',
                            ])

data = np.load('./Datasets/dataset1.npy')
env.reset()

counter = 0
print (data.shape)
# data: (mouse_x, mouse_y, puck_pos, puck_vel, gripper_pos, gripper_vel, joint_pos, joint_vel)

# {'puck_pos': array([0.79894989, 0.1171991 , 1.19990523]),
#  'puck_vel': array([-0.04367879,  0.        , -0.00394174]),
#  'gripper_pos': array([0.20976306, 0.00388718, 1.19852653]),
#  'gripper_vel': array([-2.25057362,  0.09408171, -1.32536082]),
#  'joint_pos': [-0.2562968815786609, -1.4037016841276384, 1.9696240174529724, -2.1944579935531774, -1.539114583987515, -2.044264439362114],
#  'joint_vel': [-0.8948479668025617, -3.1744521848935854, 7.548211552338913, -3.4043974127948538, -0.1803107993539623, -12.451709298965682]}

# info["puck_pos"] = self.sim.data.get_body_xpos("puck")
# info["puck_vel"] = self.sim.data.get_body_xvelp("puck")
# info["gripper_pos"] = self.sim.data.site_xpos[self.robots[0].eef_site_id]
# info["gripper_vel"] = self.sim.data.get_body_xvelp("gripper0_eef")
# self.robot_joints = self.robots[0].robot_model.joints
# self._ref_joint_pos_indexes = [self.sim.model.get_joint_qpos_addr(x) for x in self.robot_joints]
# self._ref_joint_vel_indexes = [self.sim.model.get_joint_qvel_addr(x) for x in self.robot_joints]
# info["joint_pos"] = [self.sim.data.qpos[x] for x in self._ref_joint_pos_indexes]
# info["joint_vel"] = [self.sim.data.qvel[x] for x in self._ref_joint_vel_indexes]

for i, d in enumerate(data):
    # action = np.array([d[0], d[1], 0.])
    action = np.zeros(6)



    '''PUCK STUFF'''
    # set the puck position
    puck_pos = np.array([d[2], d[3], d[4]])
    puck_vel = np.array([d[5], d[6], d[7]])


    sid = env.sim.model.site_name2id("fake_puck")
    # print(sid)
    env.sim.model.site_pos[sid] = puck_pos

    # Get the id of the puck geom
    puck_geom_id = env.sim.model.geom_name2id('puck')

    # Set the RGBA values to make the puck fully transparent
    env.sim.model.geom_rgba[puck_geom_id] = [1, 1, 1, 0]

    # env.sim.data.set_joint_qpos("puck_x", puck_pos[0])
    # env.sim.data.set_joint_qpos("puck_y", puck_pos[1])

    # env.sim.data.set_joint_qvel("puck_x", puck_vel[0])
    # env.sim.data.set_joint_qvel("puck_y", puck_vel[1])

    # env.sim.data.set_mocap_pos("puck_mocap", puck_pos)
    # env.sim.data.set_mocal_vel("puck_mocap", puck_vel)

    '''ARM STUFF'''
    arm_pos = np.array([d[14], d[15], d[16], d[17], d[18], d[19]])
    arm_vel = np.array([d[20], d[21], d[22], d[23], d[24], d[25]])

    robot_joints = env.robots[0].robot_model.joints
    # print(robot_joints)

    for j, joint in enumerate(robot_joints):
        env.sim.data.set_joint_qpos(joint, arm_pos[j])
        env.sim.data.set_joint_qvel(joint, arm_vel[j])

    env.step(action)
    print("executing: ", i)
    env.render()