from collections import OrderedDict

import numpy as np
import math
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import AirHockeyTableArena
from robosuite.models.objects import BoxObject, CylinderObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler
import robosuite.utils.transform_utils as T
from robosuite.utils.transform_utils import convert_quat
from robosuite.utils.mjmod import DynamicsModder


class AirHockey(SingleArmEnv):
    """
    This class corresponds to the lifting task for a single robot arm.

    Args:
        robots (str or list of str): Specification for specific robot arm(s) to be instantiated within this env
            (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms)
            Note: Must be a single single-arm robot!

        env_configuration (str): Specifies how to position the robots within the environment (default is "default").
            For most single arm environments, this argument has no impact on the robot setup.

        controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
            custom controller. Else, uses the default controller for this specific task. Should either be single
            dict if same controller is to be used for all robots or else it should be a list of the same length as
            "robots" param

        gripper_types (str or list of str): type of gripper, used to instantiate
            gripper models from gripper factory. Default is "default", which is the default grippers(s) associated
            with the robot(s) the 'robots' specification. None removes the gripper, and any other (valid) model
            overrides the default gripper. Should either be single str if same gripper type is to be used for all
            robots or else it should be a list of the same length as "robots" param

        initialization_noise (dict or list of dict): Dict containing the initialization noise parameters.
            The expected keys and corresponding value types are specified below:

            :`'magnitude'`: The scale factor of uni-variate random noise applied to each of a robot's given initial
                joint positions. Setting this value to `None` or 0.0 results in no noise being applied.
                If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
            :`'type'`: Type of noise to apply. Can either specify "gaussian" or "uniform"

            Should either be single dict if same noise value is to be used for all robots or else it should be a
            list of the same length as "robots" param

            :Note: Specifying "default" will automatically use the default noise settings.
                Specifying None will automatically create the required dict with "magnitude" set to 0.0.

        table_full_size (3-tuple): x, y, and z dimensions of the table.

        table_friction (3-tuple): the three mujoco friction parameters for
            the table.

        use_camera_obs (bool): if True, every observation includes rendered image(s)

        use_object_obs (bool): if True, include object (cube) information in
            the observation.

        reward_scale (None or float): Scales the normalized reward function by the amount specified.
            If None, environment reward remains unnormalized

        reward_shaping (bool): if True, use dense rewards.

        placement_initializer (ObjectPositionSampler): if provided, will
            be used to place objects on every reset, else a UniformRandomSampler
            is used by default.

        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.

        has_offscreen_renderer (bool): True if using off-screen rendering

        render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
            will result in the default angle being applied, which is useful as it can be dragged / panned by
            the user using the mouse

        render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.

        render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.

        render_gpu_device_id (int): corresponds to the GPU device id to use for offscreen rendering.
            Defaults to -1, in which case the device will be inferred from environment variables
            (GPUS or CUDA_VISIBLE_DEVICES).

        control_freq (float): how many control signals to receive in every second. This sets the amount of
            simulation time that passes between every action input.

        horizon (int): Every episode lasts for exactly @horizon timesteps.

        ignore_done (bool): True if never terminating the environment (ignore @horizon).

        hard_reset (bool): If True, re-loads model, sim, and render object upon a reset call, else,
            only calls sim.reset and resets all robosuite-internal variables

        camera_names (str or list of str): name of camera to be rendered. Should either be single str if
            same name is to be used for all cameras' rendering or else it should be a list of cameras to render.

            :Note: At least one camera must be specified if @use_camera_obs is True.

            :Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"), use the
                convention "all-{name}" (e.g.: "all-robotview") to automatically render all camera images from each
                robot's camera list).

        camera_heights (int or list of int): height of camera frame. Should either be single int if
            same height is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_widths (int or list of int): width of camera frame. Should either be single int if
            same width is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_depths (bool or list of bool): True if rendering RGB-D, and RGB otherwise. Should either be single
            bool if same depth setting is to be used for all cameras or else it should be a list of the same length as
            "camera names" param.

        camera_segmentations (None or str or list of str or list of list of str): Camera segmentation(s) to use
            for each camera. Valid options are:

                `None`: no segmentation sensor used
                `'instance'`: segmentation at the class-instance level
                `'class'`: segmentation at the class level
                `'element'`: segmentation at the per-geom level

            If not None, multiple types of segmentations can be specified. A [list of str / str or None] specifies
            [multiple / a single] segmentation(s) to use for all cameras. A list of list of str specifies per-camera
            segmentation setting(s) to use.

    Raises:
        AssertionError: [Invalid number of robots specified]
    """

    def __init__(
            self,
            robots,
            env_configuration="default",
            controller_configs=None,
            gripper_types="default",
            initialization_noise="default",
            table_full_size=(0.8, 0.8, 0.05),
            table_friction=(1.0, 5e-3, 1e-4),
            use_camera_obs=True,
            use_object_obs=True,
            reward_scale=1.0,
            reward_shaping=False,
            placement_initializer=None,
            has_renderer=False,
            has_offscreen_renderer=True,
            render_camera="frontview",
            render_collision_mesh=False,
            render_visual_mesh=True,
            render_gpu_device_id=-1,
            control_freq=20,
            horizon=1000,
            ignore_done=False,
            hard_reset=True,
            camera_names="agentview",
            camera_heights=256,
            camera_widths=256,
            camera_depths=False,
            camera_segmentations=None,  # {None, instance, class, element}
            renderer="mujoco",
            renderer_config=None,
            initial_qpos=[-0.265276, -1.383369, 2.326823, -2.601113, -1.547214, -3.405865],
            task="JUGGLE_PUCK"
    ):

        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer

        # gripper_types = "WipingGripper"
        gripper_types = "RoundGripper"

        self.arm_limit_collision_penalty = -10
        self.success_reward = 50
        self.old_puck_pos = None
        self.goal_region_world = None
        self.goal_region = None
        self.goal_vel = None
        self.positive_regions = None

        self.table_tilt = 0.09
        self.table_elevation = 1
        self.table_x_start = 0.8
        self.transform_z = lambda x: self.table_tilt * (x - self.table_x_start) + self.table_elevation

        table_q = T.axisangle2quat(np.array([0, self.table_tilt, 0]))
        self.table_transform = T.quat2mat(table_q)

        assert task in ["TOUCHING_PUCK", "REACHING", "MIN_UPWARD_VELOCITY", "GOAL_REGION",
                        "GOAL_REGION_DESIRED_VELOCITY", "JUGGLE_PUCK", "POSITIVE_REGION"]
        self.task = task

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types="default",
            gripper_types=gripper_types,
            initial_qpos=initial_qpos,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
        )

        if "GOAL_REGION" in task or task == "REACHING":
            self.randomize_goal_location(reaching=task=="REACHING")

        if task == "GOAL_REGION_DESIRED_VELOCITY":
            self.randomize_goal_vel()

        if task == "POSITIVE_REGION":
            self.randomize_positive_regions()

    # function bank
    # towards robot negative
    # print("puck_x pos:", self.sim.data.get_joint_qpos("puck_x"))
    # -left, +right
    # print("puck_y pos:", self.sim.data.get_joint_qpos("puck_y"))
    # print("puck_yaw pos:", self.sim.data.get_joint_qpos("puck_yaw"))

    # set puck position in sim
    # self.sim.data.set_joint_qpos("puck_x", 0)
    # self.sim.data.set_joint_qpos("puck_y", 0)
    # self.sim.data.set_joint_qpos("puck_yaw", 0)
    # self.sim.data.set_body_xpos("puck", [0.996, -0.300, 0.998])
    # self.sim.forward()

    # print(self.sim.data.model._body_name2id.keys())
    # self.modder = DynamicsModder(sim=self.sim)
    # self.modder.mod_position("puck", [1, -0.3, 1])
    # self.modder.mod_position("puck_main", [1, -0.3, 1])
    # self.modder.update()  

    # gripper position
    # print(self.sim.data.site_xpos[self.robots[0].eef_site_id])

    def randomize_goal_location(self, reaching=False):
        site = self.sim.model.site_name2id("goal_region")

        if reaching:
            spawn_pos_min = np.array([0.1, -0.3])
            spawn_pos_max = np.array([0.25, 0.3])
        else:
            spawn_pos_min = np.array([0.4, -0.3])
            spawn_pos_max = np.array([0.6, 0.3])

        spawn_pos = np.random.uniform(spawn_pos_min, spawn_pos_max)
        spawn_pos = np.array([spawn_pos[0], spawn_pos[1], self.transform_z(spawn_pos[0])])
        self.goal_region_world = np.array([spawn_pos[0] + 0.025, spawn_pos[1], self.transform_z(spawn_pos[0] + 0.025)])

        self.sim.model.site_pos[site] = spawn_pos.tolist()
        self.sim.model.site_rgba[site] = [1, 0, 0, 0.3]

        self.goal_region = np.dot(self.table_transform, spawn_pos)

    def randomize_goal_vel(self):
        site = self.sim.model.site_name2id("desired_vel")

        vel_min = np.array([0, -0.5])
        vel_max = np.array([2, 0.5])

        vel = np.random.uniform(vel_min, vel_max)
        self.goal_vel = np.array([vel[0] * np.cos(vel[1]), vel[0] * np.sin(vel[1]), 0])

        self.sim.model.site_rgba[site] = [0, 0, 1, 0.3]

        quat = T.axisangle2quat(1.480796326794896 * np.array([np.sin(vel[1]), np.cos(vel[1]), 0]))
        self.sim.model.site_quat[site][0] = quat[3]
        self.sim.model.site_quat[site][1:] = quat[:3]

        self.sim.model.site_pos[site] = self.goal_region_world

    def randomize_positive_regions(self):
        self.positive_regions = np.random.choice([0, 1], size=10, p=[0.5, 0.5])

    def reset(self, ):
        obs = super().reset()
        if "GOAL_REGION" in self.task or self.task == "REACHING":
            self.randomize_goal_location(reaching=self.task == "REACHING")

        if self.task == "GOAL_REGION_DESIRED_VELOCITY":
            self.randomize_goal_vel()

        if self.task == "POSITIVE_REGION":
            self.randomize_positive_regions()

        return obs

    def reward(self, action=None):
        """
        Reward function for the task.

        Sparse un-normalized reward:

            - a discrete reward of 2.25 is provided if the cube is lifted

        Un-normalized summed components if using reward shaping:

            - Reaching: in [0, 1], to encourage the arm to reach the cube
            - Grasping: in {0, 0.25}, non-zero if arm is grasping the cube
            - Lifting: in {0, 1}, non-zero if arm has lifted the cube

        The sparse reward only consists of the lifting component.

        Note that the final reward is normalized and scaled by
        reward_scale / 2.25 as well so that the max score is equal to reward_scale

        Args:
            action (np array): [NOT USED]

        Returns:
            float: reward value
        """

        # print(self.sim.model._site_name2id.keys())

        # print(self.sim.model._site_name2id.keys())

        # 
        # print("puck: ", self.sim.data.get_body_xpos("puck"))

        # print("puck_x:", self.sim.data.get_joint_qpos("puck_x"))
        # print("puck_y:", self.sim.data.get_joint_qpos("puck_y"))
        # print("puck_yaw:", self.sim.data.get_joint_qpos("puck_yaw"))

        # eef_ori = self.sim.data.get_body_xquat("gripper0_eef")
        # eef_angle = self.quat2axisangle([eef_ori[1],eef_ori[2],eef_ori[3], eef_ori[0]])/math.pi*180
        # print(eef_angle)

        # gripper0_wiping_gripper position
        # print(self.sim.data.get_body_xpos("gripper0_wiping_gripper"))
        # gripper_pos = gripper_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
        # puck_vel = self.sim.data.get_body_xvelp("puck")
        # if (puck_vel[0] < 0):
        #     reward = puck_vel[0] / 5
        # else:
        #     reward = puck_vel[0] * 20
        #
        # gripper_pos = gripper_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
        # goal_pos = self.sim.data.get_body_xpos("puck")
        # reward -= 0.01

        puck_pos = np.dot(self.table_transform, self.sim.data.get_body_xpos("puck"))
        puck_vel = np.dot(self.table_transform, self.sim.data.get_body_xvelp("puck"))
        gripper_pos = np.dot(self.table_transform, self.sim.data.site_xpos[self.robots[0].eef_site_id])

        reward = 0
        if self.task == "TOUCHING_PUCK":
            reward = 10 if np.linalg.norm(puck_pos[:2] - gripper_pos[:2]) < 0.1 else 0
        elif self.task == "REACHING":
            reward = 20 if np.linalg.norm((gripper_pos - self.goal_region)[:2]) <= 0.05 else 0
        elif self.task == "MIN_UPWARD_VELOCITY":
            reward = 20 if puck_vel[0] > 2 else 0
        elif self.task == "GOAL_REGION":
            reward = 20 if np.linalg.norm((puck_pos - self.goal_region)[:2]) <= 0.05 else 0
        elif self.task == "GOAL_REGION_DESIRED_VELOCITY":
            condition = (np.linalg.norm(
                (puck_pos - self.goal_region)[:2]) <= 0.05 and  # Checks if the puck is in the correct region
                         np.abs(np.linalg.norm(puck_vel[:2]) - np.linalg.norm(
                             self.goal_vel[:2])) < 0.3  # Checks if puck velocity has similar magnitude
                         and np.dot(puck_vel[:2], self.goal_vel[
                                                  :2]) /  # Checks if puck velocity is at a similar angle to the goal velocity
                         (np.linalg.norm(puck_vel[:2]) * np.linalg.norm(self.goal_vel[:2])) >= 0.8)

            return 30 if condition else 0
        elif self.task == "JUGGLE_PUCK":
            if puck_vel[0] > 0.75:
                reward = puck_vel[0]
            elif puck_vel[0] > 1.5:
                reward = puck_vel[0] * 2
            elif puck_vel[0] > 3:
                reward = puck_vel[0] * 3
            elif puck_vel[0] > 6:
                reward = puck_vel[0] * 4
            elif puck_vel[0] > 12:
                reward = puck_vel[0] * 10
            if puck_vel[0] <= 0:
                reward = -0.2
                reward = -puck_vel[0]                
        elif self.task == "POSITIVE_REGION":
            reward = 0
        else:
            return 0

        return reward

    def _post_action(self, action):
        """
        In addition to super method, add additional info if requested
        Args:
            action (np.array): Action to execute within the environment
        Returns:
            3-tuple:
                - (float) reward from the environment
                - (bool) whether the current episode is completed or not
                - (dict) info about current env step
        """
        reward, done, info = super()._post_action(action)

        # end_effector_pos = self.sim.data.get_body_xpos('robot0_right_hand')
        # # Update the position of the sphere to match the end effector
        # self.sim.model.body_pos[self.sim.model.body_name2id('robot0_end_effector_sphere')] = end_effector_pos

        done, reward = self._check_terminated(done, reward, info)
        return reward, done, info

    def _check_terminated(self, done, reward, info):
        """
        Check if the task has completed one way or another. The following conditions lead to termination:
            - Collision
            - Task completion (pushing succeeded)
            - Joint Limit reached
        Returns:
            bool: True if episode is terminated
        """
        # terminated = False
        # Prematurely terminate if contacting the table with the arm
        if self.check_contact(self.robots[0].robot_model):
            reward = self.arm_limit_collision_penalty
            # print("arm collision happens")
            info["terminated_reason"] = "arm_hit_table"
            done = True
        if self.check_contact("gripper0_hand_collision"):
            reward = self.arm_limit_collision_penalty
            # print("gripper hand collision happens")
            info["terminated_reason"] = "gripper_hit_table"
            done = True

        if self.robots[0].check_q_limits():
            reward = self.arm_limit_collision_penalty
            # print("reach joint limits")
            info["terminated_reason"] = "arm_limit"
            done = True

        if self.sim.data.get_body_xpos("puck")[0] < -0.1:
            reward = self.arm_limit_collision_penalty
            # print("puck out of table")
            info["terminated_reason"] = "puck_out_of_table"
            done = True

        # if np.linalg.norm(np.array(self.robots[0].recent_ee_forcetorques.current[:3])) >= 100:
        #     print("too much force: ", np.linalg.norm(np.array(self.robots[0].recent_ee_forcetorques.current[:3])))
        #     reward = self.arm_limit_collision_penalty
        #     done = True
        # Prematurely terminate if task is success
        if self._check_success():
            # reward = self.success_reward
            # print("success")
            info["terminated_reason"] = "success"
            done = True
        return done, reward

    # def _check_success(self):
    #     """
    #     Check if cube has been lifted.
    #     Returns:
    #         bool: True if cube has been lifted
    #     """

    #     # gripper_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
    #     print("checks here")
    #     return False

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        # print("xpos: ", xpos)
        xpos = (-0.48, 0, 0)
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = AirHockeyTableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            # mujoco_objects=self.puck,
        )

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        # self.cube_body_id = self.sim.model.body_name2id(self.cube.root_body)

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        # low-level object information
        if self.use_object_obs:
            # Get robot prefix and define observables modality
            pf = self.robots[0].robot_model.naming_prefix
            modality = "object"

            # cube-related observables
            @sensor(modality=modality)
            def cube_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.cube_body_id])

            @sensor(modality=modality)
            def cube_quat(obs_cache):
                return convert_quat(np.array(self.sim.data.body_xquat[self.cube_body_id]), to="xyzw")

            @sensor(modality=modality)
            def gripper_to_puck_pos(obs_cache):
                return (
                    obs_cache[f"{pf}eef_pos"] - obs_cache["goal_pos"]
                    if f"{pf}eef_pos" in obs_cache and "goal_pos" in obs_cache
                    else np.zeros(3)
                )

            @sensor(modality=modality)
            def puck_pos(obs_cache):
                return self.sim.data.get_body_xpos("puck")

            @sensor(modality=modality)
            def puck_velo(obs_cache):
                return self.sim.data.get_body_xvelp("puck")

            @sensor(modality=modality)
            def goal_pos(obs_cache):
                return self.goal_region

            @sensor(modality=modality)
            def goal_vel(obs_cache):
                return self.goal_vel

            sensors = [puck_pos, gripper_to_puck_pos]

            # ["MIN_UPWARD_VELOCITY", "GOAL_REGION", "GOAL_REGION_DESIRED_VELOCITY", "JUGGLE_PUCK"]
            if "GOAL_REGION" in self.task or self.task == "REACHING":
                sensors.append(goal_pos)

            if self.task == "GOAL_REGION_DESIRED_VELOCITY":
                sensors.append(goal_vel)

            names = [s.__name__ for s in sensors]

            # Create observables
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:
            self.modder = DynamicsModder(sim=self.sim)
            self.modder.mod_position("base", [0.8, np.random.uniform(-0.3, 0.3), 1.2])
            self.modder.update()

    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to the cube.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

        # Color the gripper visualization site according to its distance to the cube
        # if vis_settings["grippers"]:
        #     self._visualize_gripper_to_target(gripper=self.robots[0].gripper, target=self.cube)

    def _check_success(self):
        """
        Check if cube has been lifted.

        Returns:
            bool: True if cube has been lifted
        """
        # cube_height = self.sim.data.body_xpos[self.cube_body_id][2]
        # table_height = self.model.mujoco_arena.table_offset[2]

        # # cube is higher than the table top above a margin
        # return cube_height > table_height + 0.04

        puck_pos = np.dot(self.table_transform, self.sim.data.get_body_xpos("puck"))
        puck_vel = np.dot(self.table_transform, self.sim.data.get_body_xvelp("puck"))
        gripper_pos = np.dot(self.table_transform, self.sim.data.site_xpos[self.robots[0].eef_site_id])

        if self.task == "TOUCHING_PUCK":
            return True if np.linalg.norm(puck_pos[:2] - gripper_pos[:2]) < 0.1 else False
        elif self.task == "REACHING":
            return np.linalg.norm((gripper_pos - self.goal_region)[:2]) <= 0.04
        elif self.task == "MIN_UPWARD_VELOCITY":
            return False
        elif self.task == "GOAL_REGION":
            return np.linalg.norm((puck_pos - self.goal_region)[:2]) <= 0.04 and puck_vel[0] >= 0
        elif self.task == "GOAL_REGION_DESIRED_VELOCITY":
            return (np.linalg.norm((puck_pos - self.goal_region)[:2]) <= 0.05 and
                    np.abs(np.linalg.norm(puck_vel[:2]) - np.linalg.norm(self.goal_vel[:2])) < 0.3
                    and np.dot(puck_vel[:2], self.goal_vel[:2]) /
                    (np.linalg.norm(puck_vel[:2]) * np.linalg.norm(self.goal_vel[:2])) >= 0.8)
        elif self.task == "JUGGLE_PUCK":
            return False
        else:
            return True

    def quat2axisangle(self, quat):
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
