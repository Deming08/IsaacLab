# cmd_vel_action.py

from __future__ import annotations

import torch
from dataclasses import MISSING
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG, RED_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

class CmdVelAction(ActionTerm):
    """Action term for cmd_vel control of TurtleBot3.

    This action term takes cmd_vel (lin_x, lin_y, ang_z) and converts it to wheel velocities for TurtleBot3.
    Includes debug visualization with arrows for goal and current velocities.
    """

    cfg: CmdVelActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: CmdVelActionCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        # 提取單一關節索引
        left_indices = self.robot.find_joints(cfg.left_wheel_name)[0]
        right_indices = self.robot.find_joints(cfg.right_wheel_name)[0]
        if len(left_indices) != 1 or len(right_indices) != 1:
            raise ValueError("Each wheel joint name must correspond to exactly one joint.")
        self.left_idx = left_indices[0]  # 確保是單一整數
        self.right_idx = right_indices[0]  # 確保是單一整數

        self.wheel_radius = cfg.wheel_radius
        self.wheel_base = cfg.wheel_base
        self.max_lin_x = cfg.max_lin_x
        self.max_ang_z = cfg.max_ang_z
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self.raw_actions

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions

    def apply_actions(self):
        lin_x = self._raw_actions[:, 0] * self.max_lin_x
        ang_z = self._raw_actions[:, 1] * self.max_ang_z
        #lin_x = torch.clamp(self._raw_actions[:, 0], -self.max_lin_x, self.max_lin_x)  
        #ang_z = torch.clamp(self._raw_actions[:, 2], -self.max_ang_z, self.max_ang_z)  

        left_speed = (lin_x - (ang_z * self.wheel_base / 2)) / self.wheel_radius
        right_speed = (lin_x + (ang_z * self.wheel_base / 2)) / self.wheel_radius
        #print("Joint_speed L:",left_speed,"R:",right_speed)
        velocities = torch.zeros((self.num_envs, self.robot.num_joints), device=self.device)
        velocities[:, self.left_idx] = left_speed
        velocities[:, self.right_idx] = right_speed
        self.robot.set_joint_velocity_target(velocities)

        # 打印實際速度進行比較
        actual_lin_x = self.robot.data.root_lin_vel_b[:, 0]  # 基座線速度
        actual_ang_z = self.robot.data.root_ang_vel_b[:, 2]  # 基座角速度
        #print(f"Actual: lin_x={actual_lin_x[0]:.2f}, ang_z={actual_ang_z[0]:.2f}")
        actual_left_speed = self.robot.data.joint_vel[:, self.left_idx]
        actual_right_speed = self.robot.data.joint_vel[:, self.right_idx]
        #print(f"Actual Left Speed: {actual_left_speed[0]:.2f}, Right Speed: {actual_right_speed[0]:.2f}")
    
    """
    Debug visualization.
    """

    def _set_debug_vis_impl(self, debug_vis: bool):
        # 設置可視化標記的可見性
        if debug_vis:
            if not hasattr(self, "base_vel_goal_visualizer"):
                # 目標速度箭頭（綠色）
                marker_cfg = RED_ARROW_X_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Actions/velocity_goal"
                marker_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
                self.base_vel_goal_visualizer = VisualizationMarkers(marker_cfg)
                # 當前速度箭頭（藍色）
                marker_cfg = BLUE_ARROW_X_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Actions/velocity_current"
                marker_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
                self.base_vel_visualizer = VisualizationMarkers(marker_cfg)
            # 設置可見性為 True
            self.base_vel_goal_visualizer.set_visibility(True)
            self.base_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "base_vel_goal_visualizer"):
                self.base_vel_goal_visualizer.set_visibility(False)
                self.base_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # 檢查機器人是否已初始化
        if not self.robot.is_initialized:
            return
        # 獲取機器人基座位置
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5  # 將箭頭抬高以便觀察

        # 計算目標速度和當前速度的箭頭縮放和方向
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self.raw_actions[:, [0, 1]] * torch.tensor([self.max_lin_x, self.max_ang_z], device=self.device))
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, :2])

        # 顯示箭頭
        self.base_vel_goal_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.base_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

    """
    Internal helpers.
    """

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """將 XY 平面速度轉換為箭頭的方向和縮放"""
        default_scale = self.base_vel_goal_visualizer.cfg.markers["arrow"].scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0  # 縮放箭頭長度

        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)

        base_quat_w = self.robot.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)

        return arrow_scale, arrow_quat

@configclass
class CmdVelActionCfg(ActionTermCfg):
    """Configuration for cmd_vel action term."""

    class_type: type[ActionTerm] = CmdVelAction
    asset_name: str = MISSING
    left_wheel_name: str = "wheel_left_joint"
    right_wheel_name: str = "wheel_right_joint"
    wheel_radius: float = 0.033
    wheel_base: float = 0.16
    max_lin_x: float = 0.5
    max_ang_z: float = 1.0
    debug_vis: bool = True  # 默認啟用可視化