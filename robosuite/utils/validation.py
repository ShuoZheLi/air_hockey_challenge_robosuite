import numpy as np
import math

import robosuite as suite
from robosuite.wrappers import GymWrapper
from robosuite.wrappers.visualization_wrapper import VisualizationWrapper
from robosuite.utils.transform_utils import quat2mat, mat2euler, convert_quat

config = {'env_name': 'AirHockey',
          'robots': ['UR5e'],
          'controller_configs':
                {'type': 'OSC_POSITION',
                'interpolation': None,
                "impedance_mode" : "fixed"},
        'gripper_types': 'Robotiq85Gripper',}

env = suite.make(
        **config,
        has_renderer=True,
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

#Angle Error Difference Calculator for row, pitch, yaw
def errInAngle(xyz):
    xyz = np.array(xyz)
    #Fixed Orientation from OSC
    fixed_ori = np.array([0, (math.pi - 0.05), 0])
    result = np.array([])
    counter = 0
    for i in xyz:
        #Calculate Minimum Angle Difference from 0 to 180 degrees
        result = np.append(result, math.degrees(abs(math.atan2(math.sin(fixed_ori[counter] - xyz[counter]), math.cos(fixed_ori[counter] - xyz[counter])))))
        counter += 1
    return result

data = np.load('dataset.npy')
#Initialize Data Arrays
resultXYZ = np.array([[0., 0., 0.]])
resultOri = np.array([[0., 0., 0., 0.]])
resultTransVel = np.array([[0., 0., 0.]])
resultRotVel = np.array([[0., 0., 0.]])
env.reset()

counter = 0
for x, y in data:
    action = np.array([x,  y, 0.])
    obs, reward, done, _, info= env.step(action)
    env.render()
    #Get Position, Orientation, Translational Velocity, Rotational Velocity from Info
    if(info["validation_data"]):
        resultXYZ = np.append(resultXYZ, [info["validation_data"][0]], axis=0)
        resultOri = np.append(resultOri, [info["validation_data"][1]], axis=0)
        resultTransVel = np.append(resultTransVel, [info["validation_data"][2]], axis=0)
        resultRotVel = np.append(resultRotVel, [info["validation_data"][3]], axis=0)
    # env.reset()
#Formatting, delete first row of zeros
npXYZ = np.delete(resultXYZ, (0), axis = 0)
resultOri = np.delete(resultOri, (0), axis = 0)
npResRotVal = np.delete(resultRotVel, (0), axis = 0)
npResTVal = np.delete(resultTransVel, (0), axis = 0)

xyzAngle = np.array([[0., 0., 0.]])  #Formatting
xyzAngleErr = np.array([[0., 0., 0.]]) #Formatting
#Transform quaternion to rotation matrix to euler angles
for oriArray in resultOri:
    xyzw = convert_quat(oriArray, to="xyzw")
    quatMat = quat2mat(xyzw)
    aXYZ = mat2euler(quatMat, axes="sxyx")
    #Store Euler Angles and their Differences with Fixed Orientation
    xyzAngle = np.append(xyzAngle, [aXYZ], axis = 0)
    xyzAngleErr = np.append(xyzAngleErr, [errInAngle(aXYZ)], axis = 0)
#Formatting
xyzAngleErr = np.delete(xyzAngleErr, (0), axis = 0)
xyzAngle = np.delete(xyzAngle, (0), axis = 0)

#Orientation Calculations
xyzAngleMean = np.mean(xyzAngle, axis = 0) * 180/math.pi
xyzAngleMax = np.max(abs(xyzAngle), axis = 0) * 180/math.pi
xyzAngleMin = np.min(abs(xyzAngle), axis = 0) * 180/math.pi

#Orientation Error Calculations
diffAngleErr = np.mean(xyzAngleErr, axis=0)
diffAngleMaxErr = np.max(xyzAngleErr, axis=0)
diffAngleMinErr =  np.min(xyzAngleErr, axis=0)

#Position Stats Calculations
xyzMean = np.mean(npXYZ, axis = 0)
xyzMax = np.max(npXYZ, axis = 0)
xyzMin = np.min(npXYZ, axis = 0)

#Translational Velocity Calculations
resTValMean = np.mean(npResTVal, axis = 0)
resTValMax = np.max(abs(npResTVal), axis = 0)
resTValMin = np.min(abs(npResTVal), axis = 0)

#Rotational Velocity Calculations
resRValMean = np.mean(npResRotVal, axis = 0)
resRValMax = np.max(abs(npResRotVal), axis = 0)
resRValMin = np.min(abs(npResRotVal), axis = 0)

#Write to Output File and Save Results
f = open("validationResults.txt", "x")
s = "Orientation avg angle: " + str(xyzAngleMean) + "\n"
f.write(s)
s = "Orientation max angle: " + str(xyzAngleMax) + "\n"
f.write(s)
s = "Orientation min angle: " + str(xyzAngleMin) + "\n"
f.write(s)
s = "Difference in orientation avg angle: " + str(diffAngleErr) + "\n"
f.write(s)
s =  "Difference in orientation max angle: " + str(diffAngleMaxErr) + "\n"
f.write(s)
s = "Difference in orientation min angle: " + str(diffAngleMinErr) + "\n"
f.write(s)
s = "Average values for x, y, z: " + str(xyzMean) + "\n"
f.write(s)
s = "Max values for x, y, z: " + str(xyzMax) + "\n"
f.write(s)
s = "Min values for x, y, z: " + str(xyzMin) + "\n"
f.write(s)
s = "Average values for translationalVel: " + str(resTValMean) + "\n"
f.write(s)
s = "Max values for translationalVel: " + str(resTValMax) + "\n"
f.write(s)
s = "Min values for translationalVel: " + str(resTValMin) + "\n"
f.write(s)
s = "Average values for rotationalVel: " + str(resRValMean) + "\n"
f.write(s)
s = "Max values for rotationalVel: " + str(resRValMax) + "\n"
f.write(s)
s = "Min values for rotationalVel: " + str(resRValMin) + "\n"
f.write(s)