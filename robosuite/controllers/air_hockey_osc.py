import math

import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.controllers.base_controller import Controller
from robosuite.utils.control_utils import *
from robosuite.controllers.osc import OperationalSpaceController

# Supported impedance modes
IMPEDANCE_MODES = {"fixed", "variable", "variable_kp"}

class AirHockeyOperationalSpaceController(OperationalSpaceController):
    """
    Modified version of the Operational Space controller to control the air hockey robot and keep the end effector on
    a table incline. See OSC.py for more details on Operational Space Control.

    NOTE: Control input actions can either be taken to be relative to the current position / orientation of the
    end effector or absolute values. In either case, a given action to this controller is assumed to be of the form:
    (x, y, z, ax, ay, az) if controlling pos and ori or simply (x, y, z) if only controlling pos

    Args:
        sim (MjSim): Simulator instance this controller will pull robot state updates from

        eef_name (str): Name of controlled robot arm's end effector (from robot XML)

        joint_indexes (dict): Each key contains sim reference indexes to relevant robot joint information, namely:

            :`'joints'`: list of indexes to relevant robot joints
            :`'qpos'`: list of indexes to relevant robot joint positions
            :`'qvel'`: list of indexes to relevant robot joint velocities

        actuator_range (2-tuple of array of float): 2-Tuple (low, high) representing the robot joint actuator range

        input_max (float or Iterable of float): Maximum above which an inputted action will be clipped. Can be either be
            a scalar (same value for all action dimensions), or a list (specific values for each dimension). If the
            latter, dimension should be the same as the control dimension for this controller

        input_min (float or Iterable of float): Minimum below which an inputted action will be clipped. Can be either be
            a scalar (same value for all action dimensions), or a list (specific values for each dimension). If the
            latter, dimension should be the same as the control dimension for this controller

        output_max (float or Iterable of float): Maximum which defines upper end of scaling range when scaling an input
            action. Can be either be a scalar (same value for all action dimensions), or a list (specific values for
            each dimension). If the latter, dimension should be the same as the control dimension for this controller

        output_min (float or Iterable of float): Minimum which defines upper end of scaling range when scaling an input
            action. Can be either be a scalar (same value for all action dimensions), or a list (specific values for
            each dimension). If the latter, dimension should be the same as the control dimension for this controller

        kp (float or Iterable of float): positional gain for determining desired torques based upon the pos / ori error.
            Can be either be a scalar (same value for all action dims), or a list (specific values for each dim)

        damping_ratio (float or Iterable of float): used in conjunction with kp to determine the velocity gain for
            determining desired torques based upon the joint pos errors. Can be either be a scalar (same value for all
            action dims), or a list (specific values for each dim)

        impedance_mode (str): Impedance mode with which to run this controller. Options are {"fixed", "variable",
            "variable_kp"}. If "fixed", the controller will have fixed kp and damping_ratio values as specified by the
            @kp and @damping_ratio arguments. If "variable", both kp and damping_ratio will now be part of the
            controller action space, resulting in a total action space of (6 or 3) + 6 * 2. If "variable_kp", only kp
            will become variable, with damping_ratio fixed at 1 (critically damped). The resulting action space will
            then be (6 or 3) + 6.

        kp_limits (2-list of float or 2-list of Iterable of floats): Only applicable if @impedance_mode is set to either
            "variable" or "variable_kp". This sets the corresponding min / max ranges of the controller action space
            for the varying kp values. Can be either be a 2-list (same min / max for all kp action dims), or a 2-list
            of list (specific min / max for each kp dim)

        damping_ratio_limits (2-list of float or 2-list of Iterable of floats): Only applicable if @impedance_mode is
            set to "variable". This sets the corresponding min / max ranges of the controller action space for the
            varying damping_ratio values. Can be either be a 2-list (same min / max for all damping_ratio action dims),
            or a 2-list of list (specific min / max for each damping_ratio dim)

        policy_freq (int): Frequency at which actions from the robot policy are fed into this controller

        position_limits (2-list of float or 2-list of Iterable of floats): Limits (m) below and above which the
            magnitude of a calculated goal eef position will be clipped. Can be either be a 2-list (same min/max value
            for all cartesian dims), or a 2-list of list (specific min/max values for each dim)

        orientation_limits (2-list of float or 2-list of Iterable of floats): Limits (rad) below and above which the
            magnitude of a calculated goal eef orientation will be clipped. Can be either be a 2-list
            (same min/max value for all joint dims), or a 2-list of list (specific min/mx values for each dim)

        interpolator_pos (Interpolator): Interpolator object to be used for interpolating from the current position to
            the goal position during each timestep between inputted actions

        interpolator_ori (Interpolator): Interpolator object to be used for interpolating from the current orientation
            to the goal orientation during each timestep between inputted actions

        control_ori (bool): Whether inputted actions will control both pos and ori or exclusively pos

        control_delta (bool): Whether to control the robot using delta or absolute commands (where absolute commands
            are taken in the world coordinate frame)

        uncouple_pos_ori (bool): Whether to decouple torques meant to control pos and torques meant to control ori

        logger (foxglove logger): Whether to use foxglove's logger to record required controller values

        table_tilt (int): The incline angle of the table that the setup has

        table_elevation: The z location of the table relative to the ground

        x_start: The x value of the start of the table closest to the robot

        z_offset: How far below the table we want the end effector to attempt pushing into (used to increase force on table)

        **kwargs: Does nothing; placeholder to "sink" any additional arguments so that instantiating this controller
            via an argument dict that has additional extraneous arguments won't raise an error

    Raises:
        AssertionError: [Invalid impedance mode]
    """

    def __init__(
            self,
            sim,
            eef_name,
            joint_indexes,
            actuator_range,
            input_max=1,
            input_min=-1,
            output_max=0.5,
            output_min=-0.5,
            kp=150,
            damping_ratio=1,
            impedance_mode="variable_kp",
            kp_limits=(0, 300),
            damping_ratio_limits=(0, 100),
            policy_freq=20,
            position_limits=[[-0.1, -0.4, -10], [0.26, 0.4, 0]],
            orientation_limits=None,
            interpolator_pos=None,
            interpolator_ori=None,
            control_ori=True,
            control_delta=True,
            uncouple_pos_ori=False,
            logger=None,
            table_tilt=0.09,
            table_elevation=1,
            table_x_start=0.8,
            z_offset=0.08,
            **kwargs,  # does nothing; used so no error raised when dict is passed with extra terms used previously
    ):

        super().__init__(
            sim,
            eef_name,
            joint_indexes,
            actuator_range,
            input_max,
            input_min,
            output_max,
            output_min,
            kp,
            damping_ratio,
            impedance_mode,
            kp_limits,
            damping_ratio_limits,
            policy_freq,
            position_limits,
            orientation_limits,
            interpolator_pos,
            interpolator_ori,
            control_ori,
            control_delta,
            uncouple_pos_ori,
        )

        if logger is not None:
            self.logger = logger

        self.table_tilt = table_tilt
        self.table_elevation = table_elevation
        self.table_x_start = table_x_start

        # calculate x_offset to push controller into table orthogonal to table tilt
        self.z_offset = z_offset
        self.x_offset = self.z_offset / np.tan(self.table_tilt)

        # fixed orientation for our air hockey controller
        self.fixed_ori = trans.euler2mat(np.array([0, math.pi - table_tilt, 0]))
        self.goal_ori = np.array(self.fixed_ori)


    def transform_z(self, x):
        """
        Calculates the correct z value for the Air Hockey table given the x location.

        Args:
            x (int): The x location to determine the correct z for
        """
        return self.table_tilt * (x - self.table_x_start) + self.table_elevation - self.z_offset

    def set_goal(self, action, set_pos=None, set_ori=None):
        """
        Sets goal based on input @action. If self.impedance_mode is not "fixed", then the input will be parsed into the
        delta values to update the goal position / pose and the kp and/or damping_ratio values to be immediately updated
        internally before executing the proceeding control loop.

        Note that @action expected to be in the following format, based on impedance mode!

            :Mode `'fixed'`: [joint pos command]
            :Mode `'variable'`: [damping_ratio values, kp values, joint pos command]
            :Mode `'variable_kp'`: [kp values, joint pos command]

        Args:
            action (Iterable): Desired relative joint position goal state
            set_pos (Iterable): If set, overrides @action and sets the desired absolute eef position goal state
            set_ori (Iterable): IF set, overrides @action and sets the desired absolute eef orientation goal state
        """
        
        super().set_goal(self, action)

        self.goal_ori = self.fixed_ori
        self.goal_pos[2] = self.transform_z(self.goal_pos[0])

        if self.interpolator_pos is not None:
            self.interpolator_pos.set_goal(self.goal_pos)

        if self.interpolator_ori is not None:
            self.ori_ref = np.array(self.ee_ori_mat)  # reference is the current orientation at start
            self.interpolator_ori.set_goal(
                orientation_error(self.goal_ori, self.ori_ref)
            )  # goal is the total orientation error
            self.relative_ori = np.zeros(3)  # relative orientation always starts at 0

    def run_controller(self):
        """
        Calculates the torques required to reach the desired setpoint.

        Executes Operational Space Control (OSC) -- either position only or position and orientation.

        A detailed overview of derivation of OSC equations can be seen at:
        http://khatib.stanford.edu/publications/pdfs/Khatib_1987_RA.pdf

        Returns:
             np.array: Command torques
        """
        # Update state
        self.update()

        msg = dict()

        desired_pos = None
        # Only linear interpolator is currently supported
        if self.interpolator_pos is not None:
            # Linear case
            if self.interpolator_pos.order == 1:
                desired_pos = self.interpolator_pos.get_interpolated_goal()
            else:
                # Nonlinear case not currently supported
                pass
        else:
            desired_pos = np.array(self.goal_pos)

        desired_pos[2] = self.transform_z(desired_pos[0]) + 0.02

        if self.interpolator_ori is not None:
            # relative orientation based on difference between current ori and ref
            self.relative_ori = orientation_error(self.ee_ori_mat, self.ori_ref)

            ori_error = self.interpolator_ori.get_interpolated_goal()
        else:
            desired_ori = np.array(self.goal_ori)
            ori_error = orientation_error(desired_ori, self.ee_ori_mat)

        position_error = desired_pos - self.ee_pos
        vel_pos_error = -self.ee_pos_vel
        vel_ori_error = -self.ee_ori_vel

        position_error_in_base = T.make_pose(position_error, np.eye(3))

        table_in_base = T.make_pose(np.array([0, 0, 0]), trans.euler2mat(np.array([0, -self.table_tilt, 0])))
        base_in_table = T.pose_inv(table_in_base)

        position_error_in_table = T.pose_in_A_to_pose_in_B(position_error_in_base, base_in_table)
        velocity_error_in_table, _ = T.vel_in_A_to_vel_in_B(vel_pos_error, vel_ori_error, base_in_table)

        msg["desired_pos"] = T.pose_in_A_to_pose_in_B(T.make_pose(desired_pos, np.eye(3)), base_in_table)[:3, 3].tolist()
        msg["actual_pos"] = T.pose_in_A_to_pose_in_B(T.make_pose(self.ee_pos, np.eye(3)), base_in_table)[:3, 3].tolist()

        msg["position_error"] = position_error_in_table[:3, 3].tolist()
        msg["velocity_error"] = velocity_error_in_table.tolist()

        msg["desired_ori"] = np.rad2deg(T.mat2euler(self.goal_ori, 'rxyz')).tolist()
        msg["actual_ori"] = np.rad2deg(T.mat2euler(self.ee_ori_mat, 'rxyz')).tolist()

        msg["ori_error"] = ori_error.tolist()
        msg["vel_ori_error"] = vel_ori_error.tolist()

        desired_force = np.dot(np.dot(base_in_table[:3, :3], np.eye(3)), position_error_in_table[:3, 3]) * self.kp[:3]
        desired_force += np.dot(np.dot(base_in_table[:3, :3], np.eye(3)), velocity_error_in_table) * self.kd[:3]

        desired_torque = np.multiply(np.array(ori_error), np.array(self.kp[3:6])) + np.multiply(
            vel_ori_error, self.kd[3:6]
        )

        msg["desired_force"] = desired_force.tolist()
        msg["desired_torque"] = desired_torque.tolist()
        msg["desired_force_in_table"] = np.dot(base_in_table[:3, :3].T, desired_force).tolist()

        if self.logger is not None:
            self.logger.send_message(msg)

        # Compute nullspace matrix (I - Jbar * J) and lambda matrices ((J * M^-1 * J^T)^-1)
        lambda_full, lambda_pos, lambda_ori, nullspace_matrix = opspace_matrices(
            self.mass_matrix, self.J_full, self.J_pos, self.J_ori
        )

        # Decouples desired positional control from orientation control
        if self.uncoupling:
            decoupled_force = np.dot(lambda_pos, desired_force)
            decoupled_torque = np.dot(lambda_ori, desired_torque)
            decoupled_wrench = np.concatenate([decoupled_force, decoupled_torque])
        else:
            desired_wrench = np.concatenate([desired_force, desired_torque])
            decoupled_wrench = np.dot(lambda_full, desired_wrench)

        # Gamma (without null torques) = J^T * F + gravity compensations
        self.torques = np.dot(self.J_full.T, decoupled_wrench) + self.torque_compensation
        # Calculate and add nullspace torques (nullspace_matrix^T * Gamma_null) to final torques
        # Note: Gamma_null = desired nullspace pose torques, assumed to be positional joint control relative
        #                     to the initial joint positions
        self.torques += nullspace_torques(
            self.mass_matrix, nullspace_matrix, self.initial_joint, self.joint_pos, self.joint_vel
        )

        # Always run superclass call for any cleanups at the end
        super().run_controller()

        return self.torques

    def reset_goal(self):
        """
        Resets the goal to the current state of the robot
        """
        self.goal_ori = np.array(self.ee_ori_mat)

        self.goal_ori = self.fixed_ori
        self.goal_pos = np.array(self.ee_pos)

        self.goal_pos[2] = self.transform_z(self.goal_pos[0])

        # Also reset interpolators if required

        if self.interpolator_pos is not None:
            self.interpolator_pos.set_goal(self.goal_pos)

        if self.interpolator_ori is not None:
            self.ori_ref = np.array(self.ee_ori_mat)  # reference is the current orientation at start
            self.interpolator_ori.set_goal(
                orientation_error(self.goal_ori, self.ori_ref)
            )  # goal is the total orientation error
            self.relative_ori = np.zeros(3)  # relative orientation always starts at 0

    @property
    def control_limits(self):
        return super().control_limits()

    @property
    def name(self):
        return "AIR_HOCKEY_OSC_" + self.name_suffix
