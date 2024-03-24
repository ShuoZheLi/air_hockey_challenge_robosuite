import gymnasium as gym
import numpy as np
import robosuite
import torch
from robosuite import load_controller_config
from ppo_continuous_action import Agent
from robosuite.wrappers import GymWrapper
from torch import nn
from torch.distributions.normal import Normal
import os
import pickle

if __name__ == '__main__':

    SAVED_DATA = {
        "states": [],
        "actions": [],
        "next_states": [],
        "rewards": [],
        "dones": [],
        "terminates": [],
        "infos": []
    }

    DATA_LENGTH = 1_000_000
    task = "JUGGLE_PUCK"
    save_path = f"data/{task}.pkl"
    model_path = "runs/AirHockeyHIT__Juggling1__1__2024-03-23_21-16-48/Juggling1.cleanrl_model"
    
    assert task in ["MIN_UPWARD_VELOCITY", "GOAL_REGION", "GOAL_REGION_DESIRED_VELOCITY", "JUGGLE_PUCK",
                        "POSITIVE_REGION"]

    # Create the environment
    def thunk():
        config = {'env_name': 'AirHockey', 
            'robots': ['UR5e'], 
            'controller_configs': 
                    {'type': 'OSC_POSITION', 
                    'interpolation': None, 
                    "impedance_mode" : "fixed"}, 
            'gripper_types': 'Robotiq85Gripper',
            "task": task
            }

        environment = robosuite.make(
            **config,
            has_renderer=True,
            has_offscreen_renderer=False,
            render_camera="sideview",
            use_camera_obs=False,
            control_freq=20,
        )
        environment = GymWrapper(environment, keys=['robot0_joint_pos_cos', 
                        'robot0_joint_pos_sin', 
                        'robot0_joint_vel',
                        'robot0_eef_pos',
                        'puck_pos'
                        ])
        environment = gym.wrappers.FlattenObservation(environment)  # deal with dm_control's Dict observation space
        environment = gym.wrappers.RecordEpisodeStatistics(environment)
        environment = gym.wrappers.ClipAction(environment)
        return environment

    # Load the pytorch model stored at a file path and then visualize its performance using the renderer
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
    elapsed_steps = 0
    while elapsed_steps < DATA_LENGTH:
        env = thunk()
        # Set the camera
        env.viewer.set_camera(camera_id=0)
        envs = gym.vector.SyncVectorEnv(
            [thunk]
        )

        obs, _ = env.reset()
        obs = torch.Tensor(obs)

    
        agent = Agent(envs)
        agent.load_state_dict(checkpoint)

        # Perform rollouts and render performance using agent
        done = False
        ret = 0.
        while not done:
            # add dimension to obs
            obs = obs[None, :]

            action, _, _, _ = agent.get_action_and_value(torch.Tensor(obs))
            SAVED_DATA['states'].append(obs)
            SAVED_DATA['actions'].append(action)
            
            # in our env, done determines end, there is no truncated (it just returns False in the wrapper)
            obs, reward, done, _, info = env.step(action.numpy().squeeze())  # play action
            
            terminated = False
            if 'terminated_reason' in info.keys():
                terminated = False
            elif done:
                terminated = True

            SAVED_DATA['next_states'].append(obs)
            SAVED_DATA['rewards'].append(reward)
            SAVED_DATA['dones'].append(done)
            SAVED_DATA['terminates'].append(terminated)
            SAVED_DATA['infos'].append(info)
            ret += reward
            elapsed_steps += 1

    
    # Close the environment
    env.close()

    os.makedirs(save_path, exist_ok=True)
    with open(save_path, 'wb') as file:
        pickle.dump(SAVED_DATA, file)
    
    print(f"Data has been saved to {save_path}")
