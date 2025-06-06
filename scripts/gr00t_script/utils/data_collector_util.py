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

        self.frames_buffer = []
        self.obs_list_buffer = []
        self.action_list_buffer = []
        
        self.episode_index = 0  # Number of successfully saved episodes
        self.global_frame_index_offset = 0 # Tracks the starting global index for the current episode's frames
        self.task_index = 0  # Assuming this remains constant
        self.total_frames_collected_in_current_attempt = 0
        
        self.fourcc = cv2.VideoWriter.fourcc(*'mp4v')

    def append_data(self, rgb_image_bgr, data_state, data_action):
        self.frames_buffer.append(rgb_image_bgr)
        self.obs_list_buffer.append(data_state.astype(np.float32)) # Ensure float32
        self.action_list_buffer.append(data_action.astype(np.float32)) # Ensure float32
        self.total_frames_collected_in_current_attempt += 1

    def _actual_save_and_video_write(self):
        num_frames_in_episode = len(self.frames_buffer)
        timestamps = np.array([i / self.fps for i in range(num_frames_in_episode)], dtype=np.float32) # Ensure float32
        data = {
            "observation.state": pd.Series(self.obs_list_buffer, dtype=np.float32), # Explicitly cast to float32
            "action": pd.Series(self.action_list_buffer, dtype=np.float32),       # Explicitly cast to float32
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
        if self.frames_buffer:
            frame_height, frame_width = self.frames_buffer[0].shape[:2]
            video_writer = cv2.VideoWriter(video_path, self.fourcc, self.fps, (frame_width, frame_height))
            for frame in self.frames_buffer:
                video_writer.write(frame)
            video_writer.release()
            print(f"Saved video to {video_path}")
        return num_frames_in_episode

    def save_successful_episode_data(self):
        if not self.frames_buffer:
            print("No data in buffers to save for successful episode.")
            return
        print(f"Saving data for successful episode {self.episode_index} with {len(self.frames_buffer)} frames.")
        num_saved_frames = self._actual_save_and_video_write()
        self.global_frame_index_offset += num_saved_frames
        self.episode_index += 1

    def clear_all_buffers(self):
        self.frames_buffer = []
        self.obs_list_buffer = []
        self.action_list_buffer = []
        self.total_frames_collected_in_current_attempt = 0