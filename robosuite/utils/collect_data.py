import gymnasium as gym
import numpy as np
import robosuite
import torch
from robosuite import load_controller_config
from robosuite.scripts.ppo_continuous_action import Agent
from robosuite.wrappers import GymWrapper
from torch import nn
from torch.distributions.normal import Normal
import os
import pickle
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=None, help="the path of the checkpoint to use")
    parser.add_argument("--save-folder-path", type=str, default=None, help="the folder to save the .pkl file to")
    parser.add_argument("--task", type=str, default=None, help="The task to perform: TOUCHING_PUCK, REACHING, MIN_UPWARD_VELOCITY, GOAL_REGION, GOAL_X, GOAL_REGION_DESIRED_VELOCITY, JUGGLE_PUCK, POSITIVE_REGION, HITTING")
    parser.add_argument("--data-length", type=int, default=1_000_000, help="The length of the data to save, defaults to 1 million")
    
    return parser.parse_args()
    


if __name__ == '__main__':
    
    args = parse_args()

    saved_data = {
        "observations": [],
        "actions": [],
        "next_observations": [],
        "rewards": [],
        "terminals": [],
        "terminates": [],
        "infos": []
    }
    
    data_length = 1_000_000
    task = args.task

    save_path = f"{args.save_folder}/{task}.pkl"
    model_path = args.model_path
    
    assert task in ["MIN_UPWARD_VELOCITY", "GOAL_REGION", "GOAL_REGION_DESIRED_VELOCITY", "JUGGLE_PUCK",
                        "POSITIVE_REGION"]

    # Create the environment
    def thunk():
        config = {'env_name': 'AirHockey', 
            'robots': ['UR5e'], 
            'controller_configs': 
                    {'type': 'AIR_HOCKEY_OSC_POSITION', 
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
    while elapsed_steps < data_length:
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
            saved_data['observations'].append(obs)
            saved_data['actions'].append(action)
            
            # in our env, done determines end, there is no truncated (it just returns False in the wrapper)
            obs, reward, done, _, info = env.step(action.numpy().squeeze())  # play action
            
            terminated = False
            if 'terminated_reason' in info.keys():
                terminated = False
            elif done:
                terminated = True

            saved_data['next_observations'].append(obs)
            saved_data['rewards'].append(reward)
            saved_data['terminals'].append(done)
            saved_data['terminates'].append(terminated)
            saved_data['infos'].append(info)
            ret += reward
            elapsed_steps += 1

    
    # Close the environment
    env.close()

    os.makedirs(save_path, exist_ok=True)
    with open(save_path, 'wb') as file:
        for key in saved_data.keys():
            saved_data[key] = np.array(saved_data[key])
        pickle.dump(saved_data, file)
    
    print(f"Data has been saved to {save_path}")
