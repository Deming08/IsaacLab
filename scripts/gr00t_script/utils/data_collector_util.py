import os
import cv2
import numpy as np
import pandas as pd

class DataCollector:
    def __init__(self, output_video_dir, output_data_dir, fps):
        self.output_video_dir = output_video_dir
        self.output_data_dir = output_data_dir
        self.fps = fps

        os.makedirs(self.output_video_dir, exist_ok=True)
        os.makedirs(self.output_data_dir, exist_ok=True)

        self.episode_index = 0  # Tracks the number of episodes saved by THIS collector
        self.global_frame_index_offset = 0 # Tracks the global index offset for THIS collector's frames
        self.task_index = 0  # Assuming this remains constant
        
        self.fourcc = cv2.VideoWriter.fourcc(*'mp4v')

    def save_episode(self, frames_buffer, obs_list_buffer, action_list_buffer):
        """
        Saves a single episode's data (frames, observations, actions) to the configured directories.
        This method is called after an episode completes, regardless of success or failure.
        """
        num_frames_in_episode = len(frames_buffer)
        timestamps = np.array([i / self.fps for i in range(num_frames_in_episode)], dtype=np.float32) # Ensure float32
        data = {
            "observation.state": pd.Series(obs_list_buffer, dtype=np.float32), # Explicitly cast to float32
            "action": pd.Series(action_list_buffer, dtype=np.float32),       # Explicitly cast to float32
            "timestamp": timestamps,
            "frame_index": list(range(num_frames_in_episode)),
            "episode_index": [self.episode_index] * num_frames_in_episode, # Use current episode_index before increment
            "index": list(range(self.global_frame_index_offset, self.global_frame_index_offset + num_frames_in_episode)),
            "task_index": [self.task_index] * num_frames_in_episode,
        }
        df = pd.DataFrame(data)
        parquet_path = os.path.join(self.output_data_dir, f"episode_{self.episode_index:06d}.parquet")
        df.to_parquet(parquet_path)
        print(f"Saved data to {parquet_path}")

        video_path = os.path.join(self.output_video_dir, f"episode_{self.episode_index:06d}.mp4")
        if frames_buffer:
            frame_height, frame_width = frames_buffer[0].shape[:2]
            video_writer = cv2.VideoWriter(video_path, self.fourcc, self.fps, (frame_width, frame_height))
            for frame in frames_buffer:
                video_writer.write(frame)
            video_writer.release()
            print(f"Saved video to {video_path}")
        self.global_frame_index_offset += num_frames_in_episode
        self.episode_index += 1
