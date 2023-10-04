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

print("env.action_spec: ", env.action_spec)
env.reset()

while True:
    # action = env.action_space.sample()
    # action = np.append(action, [100,100,100])
    # action = np.array([0.,  0.,  0.,  0.,  0.,  0., -1., -1., -1, 0.,  0.,  0.,])

    action = np.array([0.,  0.,  0.])
    # print("action: ", action)
    obs, reward, done, info, _= env.step(action)
    # print(env.step(action))
    # print("obs: ", obs)
    # print("reward: ", reward)
    # print("done: ", done)
    # print("info: ", info)
    env.render()
    # if done:
    #     break