import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.controllers.interpolators.linear_interpolator import LinearInterpolator


class AirHockeyInterpolator(LinearInterpolator):
    """
    Subclass of LinearInterpolator for use with air hockey tasks

    Args:
        ndim (int): Number of dimensions to interpolate

        controller_freq (float): Frequency (Hz) of the controller

        policy_freq (float): Frequency (Hz) of the policy model

        ramp_ratio (float): Percentage of interpolation timesteps across which we will interpolate to a goal position.

            :Note: Num total interpolation steps will be equal to np.floor(ramp_ratio * controller_freq / policy_freq)
                    i.e.: how many controller steps we get per action space update

        ori_interpolate (None or str): If set, assumes that we are interpolating angles (orientation)
            Specified string determines assumed type of input:

                `'euler'`: Euler orientation inputs
                `'quat'`: Quaternion inputs
    """

    def __init__(
        self,
        ndim,
        controller_freq,
        policy_freq,
        ramp_ratio=0.2,
        use_delta_goal=False,
        ori_interpolate=None,
    ):
        super().init(
            ndim,
            controller_freq,
            policy_freq,
            ramp_ratio=0.2,
            use_delta_goal=False,
            ori_interpolate=None,
        )

    def get_interpolated_goal(self):
        """
        Provides the next step in interpolation given the remaining steps.

        NOTE: If this interpolator is for orientation, it is assumed to be receiving either euler angles or quaternions

        Returns:
            np.array: Next position in the interpolated trajectory
        """

        if self.ori_interpolate is None:
            # This is a normal interpolation
            # Grab start position
            x = np.array(self.start)

            dx = (self.goal - x) * (self.step / self.total_steps)
            x_current = x + dx
            # Increment step if there's still steps remaining based on ramp ratio
            if self.step < self.total_steps - 1:
                self.step += 1
            return x_current
        else:
            return super().get_interpolated_goal()
