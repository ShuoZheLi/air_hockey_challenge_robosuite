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


def unflatten_observation(obs, env):
    reshaped_arrays = []
    start = 0
    for key in env.keys:
        # Get the shape of the current key
        shape = env.modality_dims[key]
        # Calculate the number of elements in this shape
        num_elements = np.prod(shape)
        # Reshape the corresponding part of the observation array
        reshaped_array = obs[start: start + num_elements].reshape(shape)
        # Add the reshaped array to the list
        reshaped_arrays.append(reshaped_array)
        # Move the start index to the end of the current part
        start += num_elements
    return reshaped_arrays


def reshape_image(img):
    return cv2.rotate(cv2.resize(img, size), cv2.ROTATE_90_CLOCKWISE)


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
                  {'type': 'AIR_HOCKEY_OSC_POSE',
                   "kp": [700, 700, 700, 700, 700, 700],
                   "damping_ratio": [1, 1, 1, 1, 1, 1],
                   'interpolation': 'linear',
                   "impedance_mode": "fixed",
                   "control_delta": False,
                   "ramp_ratio": 1,
                   "kp_limits": (0, 10000000),
                   "uncouple_pos_ori": False,
                   "logger": logger},
              'gripper_types': 'Robotiq85Gripper',
              'task': "REACHING"}

    env = suite.make(
        **config,
        has_renderer=True,
        has_offscreen_renderer=True,
        render_camera="sideview",
        use_camera_obs=True,
        use_object_obs=True,
        control_freq=20
    )

    env = VisualizationWrapper(env)
    env = GymWrapper(env, keys=['agentview_image', 'robot0_eef_pos', 'goal_pos'])

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
            unflattened_obs = unflatten_observation(obs, env)
            screen_image = reshape_image(unflattened_obs[0])
            update_window(screen_image)

            eef_pos = unflattened_obs[1]
            # goal_pos = unflattened_obs[2]
            # print(goal_pos)
            relative_coord = transforms.get_relative_coord(pixel_coord)
            world_coord = transforms.pixel_to_world_coord(np.array(pixel_coord), solve_for_z=True)
            error = np.array(world_coord[:-1]) - eef_pos
            # print(eef_pos, world_coord[:-1], error)
            action = np.ones(6)
            if count % freq == 0:
                idx += 1
            action = np.append(action, world_coord[:-1], axis=0)
            # action = np.append(action, np.array([coordinates[idx % len(coordinates)]]))
            action = np.append(action, np.zeros(3))

            # print(eef_pos, action[6:9])
            # action[6:9] -= eef_pos
            obs, reward, done, info, _ = env.step(action[6:] if count > delay else action[6:] * 0)
            print(reward, done)
            env.render()
            count += 1
            idx %= len(coordinates)

            if done:
                env.reset()
    except KeyboardInterrupt:
        print("Keyboard Interrupt")
    finally:
        logger.stop()
