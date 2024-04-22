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
import yaml
import xmltodict
import time
import datetime

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

        use_object_obs (bool): if True, include object (puck) information in
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
            horizon=400,
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
        initial_qpos =  (math.pi / 180 * np.array([-11.4, -63.2, 82.1, -113.2, -88.92, -101.25]))
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

        gripper_types = "RoundGripper"

        self.arm_limit_collision_penalty = -20
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

        assert task in ["TOUCHING_PUCK", "REACHING", "MIN_UPWARD_VELOCITY", "GOAL_REGION", "GOAL_X",
                        "GOAL_REGION_DESIRED_VELOCITY", "JUGGLE_PUCK", "POSITIVE_REGION", "HITTING"]
        self.task = task

        self.prev_puck_goal_dist = None

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

        if self.task == "GOAL_X":
            self.goal_x = 1.4

        return obs

    def reward(self, action=None):
        """
        Reward function for the tasks.

        Each reward is customized based on the specifics of the task.

        Args:
            action (np array): [NOT USED]

        Returns:
            float: reward value
        """

        puck_pos = np.dot(self.table_transform, self.sim.data.get_body_xpos("puck"))
        puck_vel = np.dot(self.table_transform, self.sim.data.get_body_xvelp("puck"))
        gripper_pos = np.dot(self.table_transform, self.sim.data.site_xpos[self.robots[0].eef_site_id])
        gripper_vel = self.sim.data.get_body_xvelp("gripper0_eef")
        reward = 0
        if self.task == "TOUCHING_PUCK":
            reward = 10 if np.linalg.norm(puck_pos[:2] - gripper_pos[:2]) < 0.1 and gripper_vel[0] > 0.02 else 0
        elif self.task == "REACHING":
            reward = 20 if np.linalg.norm((gripper_pos - self.goal_region)[:2]) <= 0.05 and gripper_vel[0] > 0.02 else 0
        elif self.task == "MIN_UPWARD_VELOCITY":
            reward = 20 if puck_vel[0] > 2 else -1
        elif self.task == "GOAL_REGION":
            reward = 20 if np.linalg.norm((puck_pos - self.goal_region)[:2]) <= 0.05 else -np.linalg.norm((puck_pos - self.goal_region)[:2])
        elif self.task == "GOAL_X":
            if puck_pos[0] > self.goal_x:
                reward = 10
            else:
                if puck_vel[0] > 0:
                    reward = puck_vel[0] * 10
                else:
                    reward = puck_vel[0]

                reward -= 0.1 * abs(puck_pos[0] - self.goal_x)

            self.prev_puck_goal_dist = abs(puck_pos[0] - self.goal_x)

        elif self.task == "GOAL_REGION_DESIRED_VELOCITY":
            condition = (np.linalg.norm(
                (puck_pos - self.goal_region)[:2]) <= 0.05 and  # Checks if the puck is in the correct region
                         np.abs(np.linalg.norm(puck_vel[:2]) - np.linalg.norm(
                             self.goal_vel[:2])) < 0.3  # Checks if puck velocity has similar magnitude
                         and np.dot(puck_vel[:2], self.goal_vel[
                                                  :2]) /  
                         (np.linalg.norm(puck_vel[:2]) * np.linalg.norm(self.goal_vel[:2])) >= 0.8)

            return 30 if condition else -np.linalg.norm((puck_pos - self.goal_region)[:2]) + np.dot(puck_vel[:2], self.goal_vel[:2]) / (np.linalg.norm(puck_vel[:2]) * np.linalg.norm(self.goal_vel[:2]))
        elif self.task == "JUGGLE_PUCK":
            if puck_vel[0] > 0:
                reward = puck_vel[0] * 10
            else:
                reward = puck_vel[0]               
        elif self.task == "POSITIVE_REGION":
            reward = 0

        elif self.task == "HITTING":
            if puck_vel[0] > 0:
                reward = puck_vel[0] * 10
            else:
                reward = puck_vel[0]
        else:
            return 0

        return reward

    def get_transition(self, action):
        """
        Takes a step in simulation with control command @action and returns the resulting transition.
        Args:
            action (np.array): Action to execute within the environment
        Returns:
            4-tuple:
                - (OrderedDict) observations from the environment
        Raises:
            ValueError: [Steps past episode termination]
        """
        self.timestep += 1

        # Since the env.step frequency is slower than the mjsim timestep frequency, the internal controller will output
        # multiple torque commands in between new high level action commands. Therefore, we need to denote via
        # 'policy_step' whether the current step we're taking is simply an internal update of the controller,
        # or an actual policy update
        policy_step = True

        # Loop through the simulation at the model timestep rate until we're ready to take the next policy step
        # (as defined by the control frequency specified at the environment level)
        for i in range(int(self.control_timestep / self.model_timestep)):
            self.sim.forward()
            self._pre_action(action, policy_step)
            self.sim.step()
            self._update_observables()
            policy_step = False

        # Note: this is done all at once to avoid floating point inaccuracies
        self.cur_time += self.control_timestep

        if self.viewer is not None and self.renderer != "mujoco":
            self.viewer.update()

        observations = self.viewer._get_observations() if self.viewer_get_obs else self._get_observations()
        return observations

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

        # Prematurely terminate if contacting the table with the arm
        if self.check_contact(self.robots[0].robot_model):
            reward = self.arm_limit_collision_penalty
            print("arm collision happens")
            info["terminated_reason"] = "arm_hit_table"
            done = True
        if self.check_contact("gripper0_hand_collision"):
            reward = self.arm_limit_collision_penalty
            print("gripper hand collision happens")
            print("gripper hand collision happens")
            info["terminated_reason"] = "gripper_hit_table"
            done = True


        if self.robots[0].check_q_limits():
            reward = self.arm_limit_collision_penalty
            print("reach joint limits")
            print("reach joint limits")
            info["terminated_reason"] = "arm_limit"
            done = True

        # if self.sim.data.get_body_xpos("puck")[0] < -0.1:
        if self.sim.data.get_body_xpos("puck")[0] < 0.12:
            reward = self.arm_limit_collision_penalty
            print("puck out of table")
            print("puck out of table")
            info["terminated_reason"] = "puck_out_of_table"
            done = True

        puck_pos = self.sim.data.get_body_xpos("puck")
        gripper_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
        if np.allclose(puck_pos[0:2], gripper_pos[0:2], atol=0.05) and gripper_pos[2] <= puck_pos[2] - 0.2:
            reward = self.arm_limit_collision_penalty
            info["terminated_reason"] = "paddle_on_puck"
            print("paddle on puck")
            done = True

        puck_vel = np.linalg.norm(np.dot(self.table_transform, self.sim.data.get_body_xvelp("puck")))
        gripper_vel = np.linalg.norm(self.sim.data.get_body_xvelp("gripper0_eef"))

        if puck_vel <= 0.01 and gripper_vel <= 0.01:
            reward = self.arm_limit_collision_penalty
            info["terminated_reason"] = "puck_stopped"
            print("puck stopped")
            done = True

        if self._check_success():
            
            info["terminated_reason"] = "success"
            done = True
        return done, reward

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()
        YAML_PATH = "test.yaml"
        
        # Load yaml model - make this an arg later
        with open(YAML_PATH, 'r') as file:
            yaml_config = yaml.safe_load(file)
        
        sim_params = yaml_config['air_hockey']['simulator_params']
        table_length = sim_params['length']
        table_width = sim_params['width']
        puck_radius = sim_params['puck_radius']
        puck_damping = sim_params['puck_damping']
       
        with open("robosuite/models/assets/arenas/air_hockey_table.xml", "r") as file:
            xml_config = xmltodict.parse(file.read())

        # table config
        table_size = xml_config['mujoco']['worldbody']['body'][0]['body'][0]['geom']['@size']
        xml_config['mujoco']['worldbody']['body'][0]['body'][0]['geom']['@size'] = f"{table_length} {table_width} {table_size.split()[2]}"

        # puck config
        puck_size = xml_config['mujoco']['worldbody']['body'][1]['body']['geom'][0]['@size']

        xml_config['mujoco']['worldbody']['body'][1]['body']['geom'][0]['@size'] = f"{puck_radius} {puck_size.split()[1]}"
        # puck damping x
        xml_config['mujoco']['worldbody']['body'][1]['joint'][0]['@damping'] = f"{puck_damping}"
        # puck damping y
        xml_config['mujoco']['worldbody']['body'][1]['joint'][1]['@damping'] = f"{puck_damping}"
        
        # Get current timestamp
        current_time = datetime.datetime.fromtimestamp(time.time())
        formatted_time = current_time.strftime('%Y%m%d_%H%M%S')
        
        # Make new filename
        filename = f"air_hockey_table_{formatted_time}.xml"
        
        with open("robosuite/models/assets/arenas/" + filename, 'w') as file:
            file.write(xmltodict.unparse(xml_config, pretty=True))

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        
        xpos = (-0.48, 0, 0)
        self.robots[0].robot_model.set_base_xpos(xpos)
        
        # load model for table top workspace
        mujoco_arena = AirHockeyTableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
            xml=f"arenas/{filename}"
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
        )

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

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


            # @sensor(modality=modality)
            # def gripper_to_puck_pos(obs_cache):
            #     return (
            #         obs_cache[f"{pf}eef_pos"] - obs_cache["goal_pos"]
            #         if f"{pf}eef_pos" in obs_cache and "goal_pos" in obs_cache
            #         else np.zeros(3)
            #     )

            @sensor(modality=modality)
            def puck_pos(obs_cache):
                return self.sim.data.get_body_xpos("puck")
            
            @sensor(modality=modality)
            def puck_goal_dist(obs_cache):
                if not hasattr(self, 'goal_x'):
                    return np.array([np.linalg.norm(self.sim.data.get_body_xpos("puck")[0] - 0)])
                else:
                    return np.array([np.linalg.norm(self.sim.data.get_body_xpos("puck")[0] - self.goal_x)]) / 1.4

            @sensor(modality=modality)
            def puck_velo(obs_cache):
                return self.sim.data.get_body_xvelp("puck")

            @sensor(modality=modality)
            def goal_pos(obs_cache):
                if not hasattr(self, 'goal_x'):
                    return np.array([0,0])
                else:
                    return self.goal_region[:2]

            @sensor(modality=modality)
            def goal_vel(obs_cache):
                return self.goal_vel
            
            @sensor(modality=modality)
            def eef_vel(obs_cache):
                return self.sim.data.get_body_xvelp("gripper0_eef")[:2]

            sensors = [puck_pos, 
                    #    gripper_to_puck_pos, 
                       eef_vel, 
                       puck_goal_dist,
                       puck_velo]

            if "GOAL_REGION" in self.task or self.task == "REACHING":
                sensors.append(goal_pos)

            if self.task == "GOAL_REGION_DESIRED_VELOCITY":
                sensors.append(goal_vel)

            if self.task == "GOAL_X" or self.task == "GOAL_REGION":
                sensors.append(puck_goal_dist)

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
        Super call to visualize.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

    def _check_success(self):
        """
        Check if the task is complete.

        Returns:
            bool: True if the task has been completed.
        """
        
        puck_pos = np.dot(self.table_transform, self.sim.data.get_body_xpos("puck"))
        puck_vel = np.dot(self.table_transform, self.sim.data.get_body_xvelp("puck"))
        gripper_pos = np.dot(self.table_transform, self.sim.data.site_xpos[self.robots[0].eef_site_id])

        if self.task == "TOUCHING_PUCK":
            return True if np.linalg.norm(puck_pos[:2] - gripper_pos[:2]) < 0.1 else False
        elif self.task == "HITTING":
            return True if puck_vel[0] > 0.04 else False
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
        elif self.task == "GOAL_X":
            return puck_pos[0] > self.goal_x
        elif self.task == "JUGGLE_PUCK":
            return self.timestep >= self.horizon - 2
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