import gym

from air_hockey_challenge_robosuite.robosuite.wrappers.gym_wrapper import GymWrapper


class AirHockeyTaskWrapper(GymWrapper):
    def __init__(self, env, task):
        super().__init__(env=env)
        self.task = task

    def compute_reward(self, achieved_goal, desired_goal, info):
        """
        Task List:
        1.) Hitting at minimum upward velocity
        2.) Hitting into a goal region
        3.) Hitting into a goal region with a desired velocity vector
        4.) Juggling above a minimum height
        5.) Hitting into positive regions?

        :param achieved_goal:
        :param desired_goal:
        :param info:
        :return:
        """
