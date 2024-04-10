import numpy as np
import math
import time
import argparse

import robosuite as suite
from robosuite.wrappers import GymWrapper
from robosuite.wrappers.visualization_wrapper import VisualizationWrapper
from robosuite.utils.transform_utils import _AXES2TUPLE, mat2quat, quat2mat, mat2euler, convert_quat, euler2mat

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default=None, help="File path for the data to recreate.")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    if args.data_path:
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
                use_camera_obs=False,
                control_freq=20,
            )

        env = VisualizationWrapper(env)
        env = GymWrapper(env, keys=['robot0_eef_pos',
                                    'robot0_eef_quat',
                                    ])

        data = np.load(args.data_path)
        env.reset()

        counter = 0
        print (data.shape)

        startTime = time.time()
        currTime = (time.time() - startTime) * 1000
        lastTime = (time.time() - startTime) * 1000
        print(startTime)
        for i, d in enumerate(data):
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

            '''ARM STUFF'''
            arm_pos = np.array([d[14], d[15], d[16], d[17], d[18], d[19]])
            arm_vel = np.array([d[20], d[21], d[22], d[23], d[24], d[25]])

            robot_joints = env.robots[0].robot_model.joints

            for j, joint in enumerate(robot_joints):
                env.sim.data.set_joint_qpos(joint, arm_pos[j])
                env.sim.data.set_joint_qvel(joint, arm_vel[j])

            lastTime = currTime # time from last frame in ms
            currTime = (time.time() - startTime) * 1000 # time since start in ms
            deltaTime = currTime - lastTime # time since last frame in ms
            sleepTime = max(0, ((d[27] - deltaTime) / 1000))
            if (sleepTime > 0 and i > 0):
                time.sleep((d[27] - deltaTime) / 1000) # sleep for delta time
            env.step(action)
            print("executing frame: ", i)
            env.render()
    else:
        print("Provide data path using the command line argument --data-path")