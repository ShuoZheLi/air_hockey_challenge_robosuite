import numpy as np
import math
 
import robosuite as suite
from robosuite.wrappers import GymWrapper

config = {'env_name': 'AirHockey', 
          'robots': ['UR5e'], 
          'controller_configs': 
                {'type': 'OSC_POSITION', 
                'interpolation': None, 
                "impedance_mode" : "fixed"}, 
        'gripper_types': 'Robotiq85Gripper',}

env = suite.make(
        **config,
        has_renderer=False,
        has_offscreen_renderer=False,
        render_camera="sideview",
        use_camera_obs=False,
        control_freq=20,
    )

env = GymWrapper(env, keys=['robot0_eef_pos',
                            'robot0_eef_quat',
                            ])

def quat2axisangle(quat):
    """
    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    quat = np.array(quat)
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den                      

data = np.load('dataset.npy')
result = []
resultXYZ = []
resultOri = []
resultTransVel = []
resultRotVel = []
env.reset()

counter = 0
for x, y in data:
    action = np.array([x,  y, 0])
    obs, reward, done, info, _= env.step(action)
    if(_):
        resultXYZ.append([_[0]])
        resultOri.append([_[1]])
        resultTransVel.append(_[2])
        resultRotVel.append(_[3])
    env.reset()
    counter += 1

resXYZ = np.asarray(resultXYZ)
npXYZ = np.array([[0., 0., 0.]])
counter = 0
for xyzArr in resXYZ:
    xyz = resXYZ[counter][0]
    npXYZ = np.append(npXYZ, [xyz], axis=0)
    counter += 1 
npXYZ = np.delete(npXYZ, (0), axis = 0)
resTVal = np.asarray(resultTransVel)
npResTVal = np.array([[0., 0., 0.]])
counter = 0
for tval in resTVal:
    xyz = resTVal[counter]
    npResTVal = np.append(npResTVal, [xyz], axis=0)
    counter += 1
npResTVal = np.delete(npResTVal, (0), axis = 0)
resRVal = np.asarray(resultRotVel)
npResRotVal = np.array([[0., 0., 0.]])
counter = 0
for Rval in resRVal:
    xyz = resRVal[counter]
    npResRotVal = np.append(npResRotVal, [xyz], axis=0)
    counter += 1
npResRotVal = np.delete(npResRotVal, (0), axis = 0)
xyzMean = np.mean(npXYZ, axis = 0)
xyzMax = np.max(npXYZ, axis = 0)
xyzMin = np.min(npXYZ, axis = 0)
resOri = np.asarray(resultOri)
xyzAngle = np.array([[0., 0., 0.]])
counter = 0
for oriArray in resOri:
    aXYZ = quat2axisangle(resOri[counter][0])
    xyzAngle = np.append(xyzAngle, [aXYZ], axis = 0)
    counter += 1
xyzAngle = np.delete(xyzAngle, (0), axis = 0)
xyzAngleMean = np.mean(xyzAngle, axis = 0)
fixed_ori = np.array([0,   math.pi - 0.05, 0])
diffAngleErr = xyzAngleMean - fixed_ori
print(diffAngleErr)
resTValMean = np.mean(npResTVal, axis = 0)
resTValMax = np.max(npResTVal, axis = 0)
resTValMin = np.min(npResTVal, axis = 0)
resRValMean = np.mean(npResRotVal, axis = 0)
resRValMax = np.max(npResRotVal, axis = 0)
resRValMin = np.min(npResRotVal, axis = 0)
f = open("validationResults.txt", "x")
s = "Orientation avg angle: " + str(xyzAngleMean) + " Difference in orientation avg angle: " + str(diffAngleErr) + "\n"
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


