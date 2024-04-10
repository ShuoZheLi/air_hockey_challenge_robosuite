import numpy as np
import math
import os
import argparse
from colorama import Fore, Style

'''
DATA FORMAT

data: (mouse_x, mouse_y, puck_pos, puck_vel, gripper_pos, gripper_vel, joint_pos, joint_vel, time, delta_time)

{'mouse_x',
 'mouse_y',
 'puck_pos': array([0.79894989, 0.1171991 , 1.19990523]),
 'puck_vel': array([-0.04367879,  0.        , -0.00394174]),
 'gripper_pos': array([0.20976306, 0.00388718, 1.19852653]),
 'gripper_vel': array([-2.25057362,  0.09408171, -1.32536082]),
 'joint_pos': [-0.2562968815786609, -1.4037016841276384, 1.9696240174529724, -2.1944579935531774, -1.539114583987515, -2.044264439362114],
 'joint_vel': [-0.8948479668025617, -3.1744521848935854, 7.548211552338913, -3.4043974127948538, -0.1803107993539623, -12.451709298965682],
 'time',
 'delta_time'}

info["puck_pos"] = self.sim.data.get_body_xpos("puck")
info["puck_vel"] = self.sim.data.get_body_xvelp("puck")
info["gripper_pos"] = self.sim.data.site_xpos[self.robots[0].eef_site_id]
info["gripper_vel"] = self.sim.data.get_body_xvelp("gripper0_eef")
self.robot_joints = self.robots[0].robot_model.joints
self._ref_joint_pos_indexes = [self.sim.model.get_joint_qpos_addr(x) for x in self.robot_joints]
self._ref_joint_vel_indexes = [self.sim.model.get_joint_qvel_addr(x) for x in self.robot_joints]
info["joint_pos"] = [self.sim.data.qpos[x] for x in self._ref_joint_pos_indexes]
info["joint_vel"] = [self.sim.data.qvel[x] for x in self._ref_joint_vel_indexes]
'''

def sort_key(filename):
    # Split the filename on underscores
    parts = filename.split('_')

    # Get the dataset number and the timestamp
    dataset_number = int(parts[1])
    timestamp = parts[-1].split('.')[0]

    # Return a tuple of the dataset number and the timestamp
    return (dataset_number, timestamp)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default=None, help="File path for the dataset to examine.")
    args = parser.parse_args()
    return args

totalSize = 0

if __name__ == '__main__':
    
    args = parse_args()

    if args.data_path:
        try:
            data = np.load(args.data_path)
            totalSize += len(data)
            print(f'{Style.RESET_ALL}Dataset Size: {Fore.GREEN}{len(data):<10} {Style.RESET_ALL}Filename: {Fore.RED}{args.data_path}{Style.RESET_ALL}') # fancy colors
            print("-------------------------------------------------")
            print(f'{Style.RESET_ALL}Total Current Size vs Desired: {Fore.BLUE}{format(totalSize, ",")}{Style.RESET_ALL} / {Fore.YELLOW}{format(1000 * 300, ",")}{Style.RESET_ALL}') # fancy colors

        except Exception as e:
            print(f"Error occured while loading file: {e}.")
    
    else:
        print("Provide a data path using the command line argument --data-path")