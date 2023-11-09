"""
This file implements a wrapper for facilitating compatibility with OpenAI gym.
This is useful when using these environments with code that assumes a gym-like
interface.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces, Env
from robosuite.wrappers import Wrapper

import xml.etree.ElementTree as ET
from copy import deepcopy
from robosuite.utils.mjcf_utils import new_body, new_geom, new_site
from robosuite.wrappers import Wrapper

class GymWrapper(Wrapper, gym.Env):
    metadata = None
    render_mode = None
    """
    Initializes the Gym wrapper. Mimics many of the required functionalities of the Wrapper class
    found in the gym.core module

    Args:
        env (MujocoEnv): The environment to wrap.
        keys (None or list of str): If provided, each observation will
            consist of concatenated keys from the wrapped environment's
            observation dictionary. Defaults to proprio-state and object-state.

    Raises:
        AssertionError: [Object observations must be enabled if no keys]
    """

    def __init__(self, env, keys=None):
        # Run super method
        super().__init__(env=env)
        # Create name for gym
        robots = "".join([type(robot.robot_model).__name__ for robot in self.env.robots])
        self.name = robots + "_" + type(self.env).__name__

        # Get reward range
        self.reward_range = (0, self.env.reward_scale)

        self.keys = ['robot0_joint_pos_cos', 
                    'robot0_joint_pos_sin', 
                    'robot0_joint_vel',
                    'robot0_eef_pos'
                    ]

        # Gym specific attributes
        self.env.spec = None

        # set up observation and action spaces
        obs = self.env.reset()
        # self.modality_dims = {key: obs[key].shape for key in self.keys}
        flat_ob = self._flatten_obs(obs)
        self.obs_dim = flat_ob.size
        high = np.inf * np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low, high)
        low, high = self.env.action_spec
        self.action_space = spaces.Box(low, high)

    def _flatten_obs(self, obs_dict, verbose=False):
        """
        Filters keys of interest out and concatenate the information.

        Args:
            obs_dict (OrderedDict): ordered dictionary of observations
            verbose (bool): Whether to print out to console as observation keys are processed

        Returns:
            np.array: observations flattened into a 1d array
        """
        ob_lst = []
        for key in self.keys:
            if key in obs_dict:
                if verbose:
                    print("adding key: {}".format(key))
                ob_lst.append(np.array(obs_dict[key]).flatten())
        return np.concatenate(ob_lst)


    def _add_indicators_to_model(self, xml):
        """
        Adds indicators to the mujoco simulation model

        Args:
            xml (string): MJCF model in xml format, for the current simulation to be loaded
        """
        if self.indicator_configs is not None:
            root = ET.fromstring(xml)
            worldbody = root.find("worldbody")

            for indicator_config in self.indicator_configs:
                config = deepcopy(indicator_config)
                indicator_body = new_body(name=config["name"] + "_body")
                indicator_body.append(new_site(**config))
                worldbody.append(indicator_body)

            xml = ET.tostring(root, encoding="utf8").decode("utf8")

        return xml

    def reset(self, seed=None, options=None, goal_pos=[0,0,1], axisangle=[0, 1, 0, -0.26]):
        """
        Extends env reset method to return flattened observation instead of normal OrderedDict and optionally resets seed

        Returns:
            np.array: Flattened environment observation space after reset occurs
        """
        if seed is not None:
            if isinstance(seed, int):
                np.random.seed(seed)
            else:
                raise TypeError("Seed must be an integer type!")
            
        COM_indicator_config = {
            "name": "indicator1",
            "type": "box",
            "size": [0.02,0.02],
            "rgba": [1, 0, 0, 1],
            "pos": goal_pos,
            "axisangle": axisangle,
        }
        self.indicator_configs = []
        self.indicator_configs.append(COM_indicator_config)
        self.env.set_xml_processor(processor=self._add_indicators_to_model)

        ob_dict = self.env.reset(goal_pos)
        return self._flatten_obs(ob_dict), {}

    def step(self, action):
        """
        Extends vanilla step() function call to return flattened observation instead of normal OrderedDict.

        Args:
            action (np.array): Action to take in environment

        Returns:
            4-tuple:

                - (np.array) flattened observations from the environment
                - (float) reward from the environment
                - (bool) episode ending after reaching an env terminal state
                - (bool) episode ending after an externally defined condition
                - (dict) misc information
        """
        ob_dict, reward, terminated, info = self.env.step(action)
        return self._flatten_obs(ob_dict), reward, terminated, False, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        """
        Dummy function to be compatible with gym interface that simply returns environment reward

        Args:
            achieved_goal: [NOT USED]
            desired_goal: [NOT USED]
            info: [NOT USED]

        Returns:
            float: environment reward
        """
        # Dummy args used to mimic Wrapper interface
        return self.env.reward()
