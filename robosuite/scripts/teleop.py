from datetime import datetime
import time

import pygame
import sys

import numpy as np
import robosuite as suite
from robosuite.wrappers.visualization_wrapper import VisualizationWrapper
from robosuite.utils.camera_utils import get_camera_extrinsic_matrix, \
    get_camera_intrinsic_matrix
from robosuite.utils.RobosuiteTransforms import RobosuiteTransforms
from robosuite.wrappers import GymWrapper

import cv2
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collect-data", type=bool, default=False, help="Whether or not to collect data")
    parser.add_argument("--save-folder-path", type=str, default="./Datasets", help="The path for the folder to save the data to")
    parser.add_argument("--foxglove-log", type=bool, default=False, help="Whether or not to use the foxglove logger to log data. Must have the foxglove logger properly installed.")
    args = parser.parse_args()
    return args

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
            pixel_coord[1] += 150 # adjusted to make the eef appear closer to the mouse

    # Draw the image onto the screen
    screen.blit(image_to_pygame(image), (0, 0))

    # Update the window
    pygame.display.flip()


def get_observation_data(obs):
    return cv2.rotate(cv2.resize(np.uint8(obs[:-3].reshape(env.modality_dims['agentview_image'])), size), cv2.ROTATE_90_CLOCKWISE), obs[-3:]


if __name__ == '__main__':
    
    # Parse arguments
    args = parse_args()

    # Initialize Pygame
    pygame.init()

    # Set the size of the window
    size = (1000, 1000)

    screen_image = np.zeros((size[0], size[1], 3), np.float32)
    pixel_coord = [0, 0, 1]

    # Create a window
    screen = pygame.display.set_mode(size)

    if args.foxglove_log:
        from air_hockey_challenge_robosuite.foxglove_logging import Logger
        logger = Logger()

    config = {'env_name': 'AirHockey',
              'robots': ['UR5e'],

              'controller_configs':
                  {'type': 'OSC_POSE',
                   "kp": [1000, 1000, 1000, 1000, 1000, 1000],
                   "damping_ratio": [1, 1, 1, 1, 1, 1],
                   'interpolation': 'linear',
                   "impedance_mode": "fixed",
                   "control_delta": False,
                   "ramp_ratio": 1,
                   "kp_limits": (0, 10000000),
                   "uncouple_pos_ori": False,
                   "logger": logger if args.foxglove_log else None
                   },
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
    f = 350 # focal length
    c = 500 # center point
    intrinsic_mat = np.array([[f, 0, c],
                              [0, f, c],
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

    datasetCounter = 1

    if args.collect_data:
        for file in os.listdir(args.save_folder_path):
            if file.endswith(".npy"):
                datasetCounter += 1
        dataset = []
    try:
        startTime = time.time()
        currTime = (time.time() - startTime) * 1000
        lastTime = (time.time() - startTime) * 1000
        print(startTime)
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

            '''STORE THE TIME SINCE START AND DELTA TIME SINCE LAST FRAME'''
            lastTime = currTime # time from last frame in ms
            currTime = (time.time() - startTime) * 1000 # time since start in ms
            deltaTime = currTime - lastTime # time since last frame in ms
            # print("time since start: " + str(currTime) + " ms")
            # print("delta time: " + str(deltaTime) + " ms")

            # print(obs[-3:], action[6:9])
            # action[6:9] -= obs[-3:]
            obs, reward, done, truncated, info = env.step(action[6:] if count > delay else action[6:] * 0)
            if args.collect_data:
                dataset.append((world_coord[0], world_coord[1], *tuple(info["puck_pos"]), *tuple(info["puck_vel"]), *tuple(info["gripper_pos"]), *tuple(info["gripper_vel"]), *tuple(info["joint_pos"]), *tuple(info["joint_vel"]), currTime, deltaTime)) # what the camera thinks your mouse location is + puck position (x, y, z)
            env.render()


            count += 1
            idx %= len(coordinates)

            if (done):
                timestamp = datetime.now().strftime("%m%d%Y%H%M%S")
                if args.collect_data:
                    savePath = f"{args.save_folder_path}/dataset_{datasetCounter}_{len(dataset)}_{timestamp}"
                    if (len(dataset) > 50):
                        np.save(savePath, np.array(dataset))
                        print("Saved dataset" + str(datasetCounter))
                        datasetCounter += 1
                    dataset = []
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

            if args.collect_data and len(dataset) > 999:
                break

    except KeyboardInterrupt:
        print("Keyboard Interrupt")
    finally:
        if args.folglove_log:
            logger.stop()

    if args.collect_data:
        savePath = f"{args.save_folder_path}/dataset" + str(datasetCounter)
        if (len(dataset) > 50):
            np.save(savePath, np.array(dataset))
            print("Saved dataset" + str(datasetCounter))