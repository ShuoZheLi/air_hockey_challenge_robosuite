import numpy as np
import math

import robosuite as suite
from robosuite.wrappers import GymWrapper
from robosuite.wrappers.visualization_wrapper import VisualizationWrapper
from robosuite.utils.transform_utils import quat2mat, mat2euler, convert_quat, euler2mat

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
        has_offscreen_renderer=False,
        render_camera="sideview",
        use_camera_obs=False,
        control_freq=20,
    )

env = VisualizationWrapper(env)
env = GymWrapper(env, keys=['robot0_eef_pos',
                            'robot0_eef_quat',
                            ])

def euler_from_quaternion(xyzw):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        w, x, y, z = xyzw
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
        print(roll_x, pitch_y, yaw_z)
        return np.array([roll_x, pitch_y, yaw_z])

def errInAngle(xyz):
    xyz = np.array(xyz)
    fixed_ori = np.array([0, (math.pi - 0.05), 0])
    result = np.array([])
    counter = 0
    for i in xyz:
        result = np.append(result, math.degrees(abs(math.atan2(math.sin(fixed_ori[counter] - xyz[counter]), math.cos(fixed_ori[counter] - xyz[counter])))))
        counter += 1
    return result

def quaternion_rotation_matrix(Q):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.

    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3)

    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix.
             This rotation matrix converts a point in the local reference
             frame to a point in the global reference frame.
    """
    # Extract the values from Q
    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]

    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)

    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)

    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1

    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])

    return rot_matrix

data = np.load('dataset.npy')
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
    if(info["validation_data"]):
        resultXYZ = np.append(resultXYZ, [info["validation_data"][0]], axis=0)
        resultOri = np.append(resultOri, [info["validation_data"][1]], axis=0)
        resultTransVel = np.append(resultTransVel, [info["validation_data"][2]], axis=0)
        resultRotVel = np.append(resultRotVel, [info["validation_data"][3]], axis=0)
    # env.reset()
npXYZ = np.delete(resultXYZ, (0), axis = 0)
resultOri = np.delete(resultOri, (0), axis = 0)
npResRotVal = np.delete(resultRotVel, (0), axis = 0)
npResTVal = np.delete(resultTransVel, (0), axis = 0)
xyzAngle = np.array([[0., 0., 0.]])  #Formatting
xyzAngleErr = np.array([[0., 0., 0.]])
#Formatting
for oriArray in resultOri:
    xyzw = convert_quat(oriArray, to="xyzw")
    quatMat = quat2mat(xyzw)
    aXYZ = mat2euler(quatMat, axes="rxyz")
    xyzAngle = np.append(xyzAngle, [aXYZ], axis = 0)
    xyzAngleErr = np.append(xyzAngleErr, [errInAngle(aXYZ)], axis = 0)
xyzAngleErr = np.delete(xyzAngleErr, (0), axis = 0)
xyzAngle = np.delete(xyzAngle, (0), axis = 0)
file = open("AllOrientations.txt", "w+")
# Saving the array in a text file
randomOrientations = xyzAngle[np.random.choice(xyzAngle.shape[0], 30, replace=False), :]
content = str(randomOrientations) + "\n"
file.write(content)
for i in randomOrientations:
    content = str(errInAngle(i)) + "\n"
    file.write(content)
file.close()
#Orientation Calculations
xyzAngleMean = np.mean(xyzAngle, axis = 0)
xyzAngleMax = np.max(abs(xyzAngle), axis = 0)
xyzAngleMin = np.min(abs(xyzAngle), axis = 0)
#Orientation Error Calculations
diffAngleErr = errInAngle(xyzAngleMean)
diffAngleMaxErr = np.max(xyzAngleErr, axis=0) #Check again
diffAngleMinErr =  np.min(xyzAngleErr, axis=0) #Check again
#Position Stats Calculations
xyzMean = np.mean(npXYZ, axis = 0)
xyzMax = np.max(npXYZ, axis = 0)
xyzMin = np.min(npXYZ, axis = 0)
#Translational Velocity Calculations
resTValMean = np.mean(npResTVal, axis = 0)
resTValMax = np.max(npResTVal, axis = 0)
resTValMin = np.min(npResTVal, axis = 0)
#Rotational Velocity Calculations
resRValMean = np.mean(npResRotVal, axis = 0)
resRValMax = np.max(npResRotVal, axis = 0)
resRValMin = np.min(npResRotVal, axis = 0)
f = open("validationResults.txt", "x")
s = "Orientation avg angle: " + str(xyzAngleMean) + " Difference in orientation avg angle: " + str(diffAngleErr) + "\n"
f.write(s)
s = "Orientation max angle: " + str(xyzAngleMax) + " Difference in orientation max angle: " + str(diffAngleMaxErr) + "\n"
f.write(s)
s = "Orientation min angle: " + str(xyzAngleMin) + " Difference in orientation min angle: " + str(diffAngleMinErr) + "\n"
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