# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def position_command_error_tanh(env: ManagerBasedRLEnv, std: float, command_name: str) -> torch.Tensor:
    """Reward position tracking with tanh kernel."""
    command = env.command_manager.get_command(command_name)
    #des_pos_b = command[:, :3]
    des_pos_b = command[:, :2] #only x,y
    
    distance = torch.norm(des_pos_b, dim=1)
    return 1 - torch.tanh(distance / std)


def heading_command_error_abs(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Penalize tracking orientation error."""
    command = env.command_manager.get_command(command_name)
    heading_b = command[:, 3]
    return heading_b.abs()


from isaaclab.assets import RigidObject

# Custom Reward
def goal_reached_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward for reaching the goal position."""
    command_error = env.command_manager.get_command("pose_command")
    position = torch.norm(command_error[:, :2], dim=1) < 0.1   
    heading = command_error[:, 3].abs() < 0.17
    reached = torch.logical_and(position, heading) # 距離目標小於 0.1 米,10度
    return reached.float() * 10.0  # 顯著正獎勵

def penalize_lin_y(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize non-zero lin_y actions using an exponential penalty."""
    # 獲取動作項
    action_term = env.action_manager.get_term("cmd_vel_action")  # 使用動作項名稱
    actions = action_term.raw_actions  # 提取原始動作 [lin_x, lin_y, ang_z]
    lin_y = actions[:, 1]  # 提取 lin_y
    return torch.abs(lin_y)
    #return torch.exp(torch.abs(lin_y)) - 1

def obstacle_avoidance_penalty(
    env: ManagerBasedRLEnv,
    lidar_num_rays: int,
    min_distance_threshold: float,
    penalty_scale: float,
    alpha: float = 2.0,
) -> torch.Tensor:
    """Penalty for being too close to obstacles based on LiDAR data, using normalized exponential penalty.

    This improved version uses 360-degree LiDAR data, dynamically adjusts the distance threshold based on speed,
    and considers both sides during turning to enhance obstacle avoidance, especially for smooth bypassing.

    Args:
        env: The environment instance.
        lidar_num_rays: Number of LiDAR rays (e.g., 72 for 360 degrees with 5-degree resolution).
        min_distance_threshold: Base minimum distance threshold to start penalizing (e.g., 0.45 meters).
        penalty_scale: Scaling factor for the penalty intensity.
        alpha: Controls the steepness of the exponential penalty (default: 2.0 for smoother penalty).

    Returns:
        A tensor of penalties for each environment, shape (N,), where N is the number of environments.
    """
    # 獲取光達數據，形狀為 (N, B)，N 為環境數，B 為射線數
    lidar_data = env.obs_buf["policy"][:, -lidar_num_rays:]
    # 分段光達數據：前、後、左、右各 90 度
    front_lidar_data = lidar_data[:, torch.arange(27, 45, device=env.device)]  # 前 90 度，形狀 (N, 18)
    back_lidar_data = lidar_data[:, torch.arange(-9, 9, device=env.device)]  # 後 90 度
    left_lidar_data = lidar_data[:, torch.arange(9, 27, device=env.device)]  # 左 90 度
    right_lidar_data = lidar_data[:, torch.arange(45, 63, device=env.device)]  # 右 90 度
    #print("front:",front_lidar_data,"\nright:",right_lidar_data,"\nback:",back_lidar_data,"\nleft:",left_lidar_data)
    
    # 提取機器人速度
    robot: RigidObject = env.scene["robot"]
    lin_x = robot.data.root_lin_vel_b[:, 0]  # 前進速度，形狀 (N,)
    ang_z = robot.data.root_ang_vel_b[:, 2]  # 角速度，形狀 (N,)

    # 計算總速度大小，用於動態閾值
    speed = torch.norm(robot.data.root_lin_vel_b[:, :2], dim=1)  # 形狀 (N,)
    dynamic_threshold = min_distance_threshold * (1.0 + 0.5 * speed)  # 根據速度放大閾值

    # 判斷移動方向和轉向狀態
    moving_forward = lin_x >= 0  # 形狀 (N,)，True 表示前進
    turning = torch.abs(ang_z) > 0.1 / (1.0 + speed)  # 動態轉向閾值，形狀 (N,)

    """# 基礎選擇：根據移動方向選擇前/後方數據
    selected_lidar_data = torch.where(
        moving_forward.unsqueeze(1), front_lidar_data, back_lidar_data
    )  # 形狀 (N, 18)
    
    # 轉向時考慮側方數據（左右兩側的最小距離）
    if torch.any(turning):
        side_lidar_data = torch.minimum(left_lidar_data, right_lidar_data)  # 左右側最小值，形狀 (N, 18)
        selected_lidar_data = torch.where(
            turning.unsqueeze(1),
            torch.minimum(selected_lidar_data, side_lidar_data),  # 取最近距離
            selected_lidar_data
        )"""
    
    selected_lidar_data = torch.cat([front_lidar_data, back_lidar_data], dim=1)
    
    # 計算每條射線的懲罰：當距離小於動態閾值時，使用正規化指數懲罰
    normalized_distance = selected_lidar_data / dynamic_threshold.unsqueeze(1)
    alpha = torch.tensor(alpha)
    penalty_per_ray = torch.where(
        selected_lidar_data < dynamic_threshold.unsqueeze(1),
        (1.0 - torch.exp(-alpha * (1.0 - normalized_distance))) / (1.0 - torch.exp(-alpha)),
        torch.zeros_like(selected_lidar_data)
    )

    # 對每個環境的懲罰求和，應用縮放因子
    penalty = torch.sum(penalty_per_ray, dim=1) * penalty_scale
    return penalty

def action_smoothness_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    actions = env.action_manager.action  # 當前動作
    prev_actions = env.action_manager.prev_action  # 前一步動作
    smoothness_penalty = torch.sum(torch.abs(actions - prev_actions), dim=1)
    return smoothness_penalty

def forward_movement_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    robot: RigidObject = env.scene["robot"]
    lin_x = robot.data.root_lin_vel_b[:, 0]  # 前進速度
    reward = torch.clamp(lin_x, 0.0, None)  # 僅獎勵正向速度
    return reward

