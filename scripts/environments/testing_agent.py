# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to an environment test with testing action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Testing agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Base-OpenArm-DexHand-v0", help="Name of the task.")
#Isaac-Base-G1-Abs-v0, Isaac-Base-OpenArm-DexHand-v0
parser.add_argument(
    "--save_img",
    action="store_true",
    default=False,
    help="Save the data from camera RGB image.",
)
parser.add_argument("--print_obs", action="store_true", default=True, help="Print the robot observations.")
parser.add_argument("--g1_hand_type", default="inspire", choices=["trihand", "inspire"], help="DexHands type of G1.")
parser.add_argument("--joint_space", action="store_true", default=True, help="Whether to use joint space action.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# Force enable cameras for this script by modifying the parsed arguments
args_cli.enable_cameras = True

##########
# Import pinocchio before AppLauncher to force the use of the version installed by IsaacLab and not the one installed by Isaac Sim
# pinocchio is required by the Pink IK controllers and the GR1T2 retargeter
import pinocchio  # noqa: F401
##########

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

##########
#import isaaclab_tasks.manager_based.manipulation.playground_g1  # noqa: F401
##########

#import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


# PLACEHOLDER: Extension template (do not remove this comment)

import cv2
import os
import numpy as np

import carb
carb_settings_iface = carb.settings.get_settings()
carb_settings_iface.set_bool("/gr00t/use_joint_space", args_cli.joint_space)

if "G1" in args_cli.task:
    carb_settings_iface.set_string("/unitree_g1_env/hand_type", args_cli.g1_hand_type)

    
def quaternion_multiply(q_world_target, q_offset):
    """
    This applies the local rotation `q_offset` on a target quaternion `q_world_target` expressed in world frame.
    q_result = q_world_target ⊗ q_offset (active rotation)
    """
    w1,x1,y1,z1 = q_world_target[0]
    w2,x2,y2,z2 = q_offset
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return torch.tensor([[w,x,y,z]], device=q_world_target.device)

def main():
    dataset_path = "datasets/collection_test/G1_testing_dataset/"
    output_video_dir = f"{dataset_path}videos/chunk-000/observation.images.camera"
    output_data_dir = f"{dataset_path}data/chunk-000"
    os.makedirs(output_video_dir, exist_ok=True)
    os.makedirs(output_data_dir, exist_ok=True)

    episode_index = 0
    frame_count = 0

    fps = 30 
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = None
    frames = []
    obs_list = []
    action_list = []
    episode_len = 5 # second

    # create environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    obs = env.reset()

    # hand action values (LeapHand-right, dim:16)
    hand_value = torch.tensor([[ 0.0,  0.0,  0.0,  0.0,
                                 0.0,  0.0,  0.0,  0.0,
                                 0.0,  0.0,  0.0,  0.0,
                                 0.0,  0.0,  0.0,  0.0, ]])
    
    # Define arm action values (left dim:7 + right dim:7)
    if carb_settings_iface.get("/gr00t/use_joint_space"):
        left_arm_joint = torch.tensor([[-0.4, -0.2,  0.0,  0.8, 0.0,  0.0,  -1.0]])
        right_arm_joint = torch.tensor([[0.4, 0.2,  0.0,  0.8, 0.0,  0.0,  1.0]])
        arm_value = torch.cat([left_arm_joint, right_arm_joint], dim=1)
    else: # task-space(xyz)(wxyz)
        left_hand_pos = torch.tensor([[ 0.25,  0.15,  0.85]])
        left_hand_quat = torch.tensor([[ 0.707, 0, 0.707, 0]])
        right_hand_pos = torch.tensor([[ 0.25, -0.15,  0.85]])
        right_hand_quat = torch.tensor([[ 0.707, 0, 0.707, 0]])

        if "OpenArm" in args_cli.task:
            # eef frame quaternion in world frame
            LEFT_Q_IN_WORLD = [0.707, 0, -0.707, 0] 
            RIGHT_Q_IN_WORLD = [0, 0.707, 0, 0.707]
            left_hand_quat = quaternion_multiply(left_hand_quat, LEFT_Q_IN_WORLD)
            right_hand_quat = quaternion_multiply(right_hand_quat, RIGHT_Q_IN_WORLD)
            #! Currently OpenArm-LeapHand eef needs to rotate [left(Y:-90),right(X:180,Y:90)] to align with the world frame.
        arm_value = torch.cat([left_hand_pos, left_hand_quat, right_hand_pos, right_hand_quat], dim=1)
    
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # sample actions from -1 to 1
            #actions = 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
            
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            
            # replace arm action value
            actions[:, :14] = arm_value
            # replace hand joint value
            actions[:, -16:] = hand_value

            # apply actions
            obs, _, _, _, _ = env.step(actions)

            if args_cli.print_obs:
                processed_action = obs["robot_obs"]['processed_actions'].cpu()
                robot_joint_state = obs["robot_obs"]['robot_joint_pos'].cpu()
                left_eef_pos = obs["robot_obs"]['left_eef_pos'].cpu()
                left_eef_quat = obs["robot_obs"]['left_eef_quat'].cpu()
                right_eef_pos = obs["robot_obs"]['right_eef_pos'].cpu()
                right_eef_quat = obs["robot_obs"]['right_eef_quat'].cpu()
                hand_joint_state = obs["robot_obs"]['hand_joint_state'].cpu()

                print("|======Arm Obs======|")
                if not carb_settings_iface.get("/gr00t/use_joint_space"):
                    print("Arm task-space input:\n L:",arm_value[:, 0:7],"\n R:",arm_value[:, 7:14])
                print("L-joint-action:",processed_action[:, 0:7])
                print("L-joint-state :",robot_joint_state[:, 0:7])
                print("R-joint-action:",processed_action[:, 7:14])
                print("R-joint-state :",robot_joint_state[:, 7:14])
                print("L-eef-pos :",left_eef_pos)
                print("L-eef-quat:",left_eef_quat)
                print("R-eef-pos :",right_eef_pos)
                print("R-eef-quat:",right_eef_quat)
                print("|------Hand Obs------|")
                print("Hand-joint-action:",hand_value)
                print("Hand-joint-state :",hand_joint_state)

            # data collect test===================|
            if args_cli.save_img:
                rgb_image = obs["robot_obs"]["rgb_image"]  # shape: (1, 480, 640, 3)
                rgb_image = rgb_image.squeeze(0)  # remove batch dim (480, 640, 3)
                rgb_image = rgb_image.cpu().numpy()  # from cuda to cpu
                rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR) # RGB to CV2 BGR format

                # observation_state = np.concatenate([
                #     obs["robot_obs"]["robot_joint_pos"].cpu().numpy().flatten(),
                # ]).astype(float).tolist()
                
                # obs_list.append(observation_state)
                # action_list.append(actions.cpu().numpy().flatten().tolist())
                frames.append(rgb_image)
                
                # preview
                cv2.imwrite("output/frame_preview.png", rgb_image)

                # start new episode or continue current
                if frame_count % (fps * episode_len) == 0 and frame_count > 0: 
                    # create new video
                    output_path = os.path.join(output_video_dir, f"episode_{episode_index:06d}.mp4")
                    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (rgb_image.shape[1], rgb_image.shape[0]))
                    for frame in frames:
                        video_writer.write(frame)
                    episode_index += 1
                    frames = []
                    # obs_list = []
                    # action_list =[]
                    if video_writer is not None:
                        video_writer.release()

                frame_count += 1

    # Save last episode
    if frames:
        output_path = os.path.join(output_video_dir, f"episode_{episode_index:06d}.mp4")
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (rgb_image.shape[1], rgb_image.shape[0]))
        for frame in frames:
            video_writer.write(frame)
        video_writer.release()
    # close the simulator
    env.close()



if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
