import h5py
import cv2
import numpy as np
import argparse
import time
import concurrent.futures
from utils.data_collector_util import DataCollector

# Unitree G1 joint indices in whole body 43 joint.
LEFT_ARM_INDICES = [11, 15, 19, 21, 23, 25, 27]
RIGHT_ARM_INDICES = [12, 16, 20, 22, 24, 26, 28]
LEFT_HAND_INDICES = [31, 37, 41, 30, 36, 29, 35]
RIGHT_HAND_INDICES = [34, 40, 42, 32, 38, 33, 39]

JOINT_ID = LEFT_ARM_INDICES + RIGHT_ARM_INDICES + LEFT_HAND_INDICES + RIGHT_HAND_INDICES


def process_demo(hdf5_path, demo_idx, joint_id, start_idx):
    with h5py.File(hdf5_path, 'r') as f:
        demo_group = f['data/demo_{}'.format(demo_idx)]
        frames_buffer = []
        obs_list_buffer = []
        action_list_buffer = []

        # Extract RGB images from obs/rgb_image
        rgb_images = demo_group['obs/rgb_image'][start_idx:]

        for rgb_image in rgb_images:
            # Convert RGB to BGR for OpenCV
            frames_buffer.append(cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))

        # Extract joint_state from obs/robot_joint_pos
        joint_states = demo_group['obs/robot_joint_pos'][start_idx:]
        selected_joint_states = np.array([state[joint_id] for state in joint_states])
        obs_list_buffer.extend(selected_joint_states.tolist())

        # Extract actions from obs/processed_actions
        actions = demo_group['obs/processed_actions'][start_idx:]
        action_list_buffer.extend(actions.tolist())

    return demo_idx, frames_buffer, obs_list_buffer, action_list_buffer


def convert_hdf5_to_parquet_mp4(hdf5_path, output_video_dir, output_data_dir, start_idx, num_workers):
    # Initialize DataCollector
    collector = DataCollector(output_video_dir, output_data_dir, fps=30)

    # Get number of demos
    with h5py.File(hdf5_path, 'r') as f:
        num_episodes = len(f['data'])

    if num_workers == 1:
        # Serial processing
        for demo_idx in range(num_episodes):
            _, frames, obs, actions = process_demo(hdf5_path, demo_idx, JOINT_ID, start_idx)
            collector.episode_index = demo_idx
            collector.save_episode(frames, obs, actions)
    else:
        # Parallel processing
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Iterate directly over results to avoid loading all in memory
            for demo_idx, frames, obs, actions in executor.map(process_demo, [hdf5_path] * num_episodes, range(num_episodes), [JOINT_ID] * num_episodes, [start_idx] * num_episodes):
                collector.episode_index = demo_idx
                collector.save_episode(frames, obs, actions)
    
    return num_episodes


def main():
    parser = argparse.ArgumentParser(description="Convert HDF5 file to Parquet and MP4.")
    parser.add_argument("--hdf5_path", type=str, default="./datasets/g1_cabinet_pour/g1_pour_generated.hdf5", help="Path to the HDF5 file.")
    parser.add_argument("--output_dataset_dir", type=str, default="datasets/mimic_collection/g1_cabinet_pour", help="Base path for output dataset.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel workers to use for processing.")
    args = parser.parse_args()

    START_IDX = 5  # skip first 5 frames
    output_video_dir = f"{args.output_dataset_dir}/videos/chunk-000/observation.images.camera"
    output_data_dir = f"{args.output_dataset_dir}/data/chunk-000"

    start_time = time.time()
    num_episodes = convert_hdf5_to_parquet_mp4(args.hdf5_path, output_video_dir, output_data_dir, START_IDX, args.num_workers)
    print(f"{num_episodes} episodes processed in {args.num_workers} workers. Elapsed time: {(time.time() - start_time):.2f} seconds.")

if __name__ == "__main__":
    main()
