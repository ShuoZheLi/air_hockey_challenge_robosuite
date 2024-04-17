import numpy as np

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
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera="sideview",
        use_camera_obs=False,
        control_freq=20,
    )

env = GymWrapper(env, keys=['robot0_joint_pos_cos', 
                            'robot0_joint_pos_sin', 
                            'robot0_joint_vel',
                            'robot0_eef_pos'
                            ])



# state dim = 21, all between -1 to 1, except eef_pos
# action dim = 3, should be 2, just put 0 into z-axis
# delta position
x_pos = 0.2
y_pos = 0
goal_z = lambda x_pos: 0.99 + np.tan(0.26) * x_pos
# env.reset(goal_pos=[x_pos, y_pos, goal_z(x_pos)])
env.reset()

while True:
    action = np.array([0.,  0.,  0])
    obs, reward, done, info, _= env.step(action)
    env.render()
    if done:
        x_pos = 0.02
        y_pos = 0
        goal_z = lambda x_pos: 0.99 + np.tan(0.26) * x_pos
        env.reset()