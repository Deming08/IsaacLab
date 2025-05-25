
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import RigidObject
import isaaclab.utils.math as math_utils

# Custom Observation
def base_lin_ang_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root linear_x and angular_z velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    lin_x = asset.data.root_lin_vel_b[:, 0:1]
    ang_z = asset.data.root_ang_vel_b[:, 2:3]
    return torch.cat([lin_x, ang_z], dim=1)


def get_commands(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """The generated command from command term in the command manager with the given name."""

    command = env.command_manager.get_command(command_name)
    command[:, 2] = 0.0 # z-pos value to zero
    
    return command

    
def lidar_scan(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, directions: list, angle_range: int, res: int) -> torch.Tensor:
    ray_caster: RayCaster = env.scene.sensors[sensor_cfg.name]
    data = ray_caster.data

    # 提取 LiDAR 位置和擊中點
    pos_w = data.pos_w  # 形狀: (N, 3)
    ray_hits_w = data.ray_hits_w  # 形狀: (N, B, 3)

    # 計算距離
    pos_w_expanded = pos_w.unsqueeze(1).expand(-1, ray_hits_w.shape[1], -1)  # (N, B, 3)
    distances = torch.norm(ray_hits_w - pos_w_expanded, dim=2)  # (N, B)
    
    # 處理 inf 和 nan 值
    max_distance = 3.5  # 假設最大檢測距離為 5 米
    distances = torch.where(
        torch.isinf(distances) | torch.isnan(distances),
        torch.tensor(max_distance, device=env.device),
        distances
    )
    
    # 獲取固定角度（假設 LiDAR 朝向與機器人對齊）
    horizontal_fov_range=[-180, 180]
    horizontal_res=1.0

    start_angle, end_angle = horizontal_fov_range
    num_rays = int((end_angle - start_angle) / horizontal_res)

    # 補充缺失的射線
    num_rays_actual = ray_hits_w.shape[1]  # 例如 359
    if num_rays_actual < num_rays:
        missing_rays = num_rays - num_rays_actual
        last_distance = distances[:, -1].unsqueeze(1)  # 形狀: (N, 1)
        distances = torch.cat([distances, last_distance.repeat(1, missing_rays)], dim=1)  # 形狀: (N, 360)

    # 獲取光達的偏航角（使用 math_utils.yaw_quat）
    lidar_quat_w = data.quat_w  # 形狀: (N, 4)

    # 從偏航四元數中提取偏航角（使用 math_utils.euler_xyz_from_quat）
    roll, pitch, lidar_yaw = math_utils.euler_xyz_from_quat(lidar_quat_w)  # 形狀: (N,), (N,), (N,)，單位：弧度

    # 將偏航角轉換為度數，並計算索引偏移量
    lidar_yaw_deg = lidar_yaw * (180 / torch.pi)  # 轉換為度數
    offset = (lidar_yaw_deg / horizontal_res).long() % num_rays  # 偏移量 (N,)

    # 對距離數據進行循環移位，轉換到局部坐標系
    distances_local = torch.zeros_like(distances)

    for i in range(distances.shape[0]):  # 對每個環境
        shift = offset[i].item()
        distances_local[i] = torch.roll(distances[i], shifts=shift, dims=0)

    """
    # 前方: -45° ~ 45° ,索引 [135, 225)
    # 後方: 135° ~ 180/-180° ~ -135° ,索引 [-45, 45)
    front_index = torch.arange(180- angle_range//2, 180+ angle_range//2, device=env.device) 
    back_index = torch.arange(0- angle_range//2, 0+ angle_range//2, device=env.device)
    # 拼接前後數據，每5筆取1
    selected_index = torch.cat([front_index, back_index])[::res]
    selected_distances = distances_local[:, selected_index]
    """
    selected_distances = distances_local[:, ::res]

    return selected_distances

    """
    for direction in directions:
        if direction == "front":
            direction_indices = torch.arange(180- angle_range//2, 180+ angle_range//2, device=env.device)
        elif direction == "back":
            direction_indices = torch.arange(0- angle_range//2, 0+ angle_range//2, device=env.device)
        elif direction == "left":
            direction_indices =  torch.arange(90- angle_range//2, 90+ angle_range//2, device=env.device)
        elif direction == "right":
            direction_indices =  torch.arange(270- angle_range//2, 270+ angle_range//2, device=env.device)
        else:
            print("Invalid input direction!")
    """