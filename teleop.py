import time

import pygame
import sys

import numpy as np
import robosuite as suite
from air_hockey_challenge_robosuite.foxglove_logging import Logger
from robosuite.wrappers.visualization_wrapper import VisualizationWrapper
from robosuite.utils.camera_utils import get_camera_extrinsic_matrix, \
    get_camera_intrinsic_matrix
from robosuite.utils.RobosuiteTransforms import RobosuiteTransforms
from robosuite.wrappers import GymWrapper

import cv2


def image_to_pygame(image):
    """
    Convert an image to a pygame surface and scale it to fit the window
    """
    pg_image = pygame.surfarray.make_surface(image)
    pg_image = pygame.transform.scale(pg_image, size)  # Scale the image to fit the window
    return pg_image


def update_window(image):
    global pixel_coord
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.MOUSEMOTION:
            pixel_coord[:2] = event.pos  # update the shared list with the mouse coordinates

    # Draw the image onto the screen
    screen.blit(image_to_pygame(image), (0, 0))

    # Update the window
    pygame.display.flip()


def get_observation_data(obs):
    return cv2.rotate(cv2.resize(np.uint8(obs[:-3].reshape(env.modality_dims['agentview_image'])), size), cv2.ROTATE_90_CLOCKWISE), obs[-3:]


if __name__ == '__main__':
    # Initialize Pygame
    pygame.init()

    # Set the size of the window
    size = (1000, 1000)

    screen_image = np.zeros((size[0], size[1], 3), np.float32)
    pixel_coord = [0, 0, 1]

    # Create a window
    screen = pygame.display.set_mode(size)

    logger = Logger()
    config = {'env_name': 'AirHockey',
              'robots': ['UR5e'],
              'controller_configs':
                  {'type': 'OSC_POSE',
                   "kp": [700, 700, 700, 700, 700, 700],
                   "damping_ratio": [1, 1, 1, 1, 1, 1],
                   'interpolation': 'linear',
                   "impedance_mode": "fixed",
                   "control_delta": False,
                   "ramp_ratio": 1,
                   "kp_limits": (0, 10000000),
                   "uncouple_pos_ori": False,
                   "logger": logger},
              'gripper_types': 'Robotiq85Gripper', }

    env = suite.make(
        **config,
        has_renderer=True,
        has_offscreen_renderer=True,
        render_camera="sideview",
        use_camera_obs=True,
        use_object_obs=False,
        control_freq=20
    )


    env = VisualizationWrapper(env)
    env = GymWrapper(env, keys=['agentview_image', 'robot0_eef_pos'])

    intrinsic_mat = get_camera_intrinsic_matrix(env.sim, 'agentview', *size)
    f = 350
    intrinsic_mat = np.array([[f, 0, 500],
                              [0, f, 500],
                              [0, 0, 1]])
    extrinsic_mat = get_camera_extrinsic_matrix(env.sim, 'agentview')
    extrinsic_mat = np.linalg.inv(extrinsic_mat)

    transforms = RobosuiteTransforms(extrinsic_mat, intrinsic_mat)

    obs, _ = env.reset()

    # Start the main loop
    coordinates = [
        [-1, -0.3, 1],  # From the 'red_dot' element
        [1, -0.3, 1],  # From the 'red_dot1' element
        [0.26, 0.3, 1],  # From the 'red_dot2' element
        [-0.03, 0.3, 1]  # From the 'red_dot3' element
    ]
    idx = 0
    delay = 0
    freq = 8
    count = 0
    try:
        while True:
            screen_image, _ = get_observation_data(obs)
            update_window(screen_image)

            relative_coord = transforms.get_relative_coord(pixel_coord)
            world_coord = transforms.pixel_to_world_coord(np.array(pixel_coord), solve_for_z=True)
            error = np.array(world_coord[:-1]) - obs[-3:]
            # print(obs[-3:], world_coord[:-1], error)
            action = np.ones(6)
            if count % freq == 0:
                idx += 1
            action = np.append(action, world_coord[:-1], axis=0)
            # action = np.append(action, np.array([coordinates[idx % len(coordinates)]]))
            action = np.append(action, np.zeros(3))
            # TODO Fix this hack

            # print(obs[-3:], action[6:9])
            # action[6:9] -= obs[-3:]
            obs, reward, done, info, _ = env.step(action[6:] if count > delay else action[6:] * 0)
            env.render()
            count += 1
            idx %= len(coordinates)
    except KeyboardInterrupt:
        print("Keyboard Interrupt")
    finally:
        logger.stop()