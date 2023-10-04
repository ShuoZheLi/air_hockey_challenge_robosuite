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

env = GymWrapper(env)
x_pos = 0.02
env.reset(goal_pos=[x_pos, 0, 0.99 + np.tan(0.26) * x_pos])

while True:
    action = np.array([0.,  0.,  -0.001])
    obs, reward, done, info, _= env.step(action)
    env.render()
    if done:
        x_pos = 0.02
        env.reset(goal_pos=[x_pos, 0, 0.99 + np.tan(0.26) * x_pos])