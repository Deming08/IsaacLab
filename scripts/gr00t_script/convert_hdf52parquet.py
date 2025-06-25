import h5py
import os
import cv2
import numpy as np
import pandas as pd
from utils.data_collector_util import DataCollector

# Unitree G1 joint indices in whole body 43 joint.
LEFT_ARM_INDICES = [11, 15, 19, 21, 23, 25, 27]
RIGHT_ARM_INDICES = [12, 16, 20, 22, 24, 26, 28]
LEFT_HAND_INDICES = [31, 37, 41, 30, 36, 29, 35]
RIGHT_HAND_INDICES = [34, 40, 42, 32, 38, 33, 39]

JOINT_ID = LEFT_ARM_INDICES + RIGHT_ARM_INDICES + LEFT_HAND_INDICES + RIGHT_HAND_INDICES


def convert_hdf5_to_parquet_mp4(hdf5_path, output_video_dir, output_data_dir, start_idx):

    # Initialize DataCollector
    collector = DataCollector(output_video_dir, output_data_dir, fps=30)

    # Open the HDF5 file
    with h5py.File(hdf5_path, 'r') as f:
        # Iterate over each demo
        for demo_idx in range(len(f['data'])):
            demo_group = f['data/demo_{}'.format(demo_idx)]
            frames_buffer = []
            obs_list_buffer = []
            action_list_buffer = []

            # Extract RGB images from obs.rgb_image
            rgb_images = demo_group['obs/rgb_image'][start_idx:]

            for rgb_image in rgb_images:
                # Convert RGB to BGR for OpenCV
                frames_buffer.append(cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))

            # Extract joint_state from obs.robot.joint_position
            joint_states = demo_group['obs/robot_joint_pos'][start_idx:]
            selected_joint_states = np.array([state[JOINT_ID] for state in joint_states])
            obs_list_buffer.extend(selected_joint_states)

            # Extract actions from obs.processed_actions
            actions = demo_group['obs/processed_actions'][start_idx:]
            action_list_buffer.extend(actions)

            
            # Append data
            for i in range(len(frames_buffer)):
                collector.append_data(frames_buffer[i], obs_list_buffer[i], action_list_buffer[i])

            # Save the episode
            collector.save_successful_episode_data()
            collector.clear_all_buffers()

if __name__ == "__main__":
    hdf5_path = "./datasets/g1_dataset_test.hdf5"  # Replace with your HDF5 file path

    output_dataset_path = "datasets/mimic_collection/G1_dataset/"
    output_video_dir = f"{output_dataset_path}videos/chunk-000/observation.images.camera"
    output_data_dir = f"{output_dataset_path}data/chunk-000"
    start_idx = 30 # skip first 30 frame

    convert_hdf5_to_parquet_mp4(hdf5_path, output_video_dir, output_data_dir, start_idx)
    print("All demos converted to parquet and mp4.")