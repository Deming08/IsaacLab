# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Turtlebot3 Waffle robot in Isaac Lab."""

import torch
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.scene import SceneCfg
from isaaclab.utils import sim_utils
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

##
# Configuration
##

# 定義 Turtlebot3 Waffle 的資產配置
TURTLEBOT3_WAFFLE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/path/to/turtlebot3_waffle.usd",  # 請替換為您的 Turtlebot3 Waffle USD 文件路徑
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=10.0,
            max_angular_velocity=10.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.1),  # 初始位置 (x, y, z)
        joint_pos={"left_wheel": 0.0, "right_wheel": 0.0}  # 左右輪子的初始關節位置
    ),
    actuators={
        "left_wheel_actuator": ImplicitActuatorCfg(
            joint_names_expr=["left_wheel"],
            effort_limit=10.0,
            velocity_limit=5.0,
            stiffness=0.0,
            damping=1.0,
        ),
        "right_wheel_actuator": ImplicitActuatorCfg(
            joint_names_expr=["right_wheel"],
            effort_limit=10.0,
            velocity_limit=5.0,
            stiffness=0.0,
            damping=1.0,
        ),
    },
)

# 定義場景配置
scene_cfg = SceneCfg(
    num_envs=1024,  # 並行環境數量
    env_spacing=2.5,  # 環境之間的間距
    ground=sim_utils.GroundPlaneCfg(),  # 添加地面
    light=sim_utils.DomeLightCfg(intensity=1000.0, color=(1.0, 1.0, 1.0)),  # 添加燈光
    robot=TURTLEBOT3_WAFFLE_CFG  # 添加 Turtlebot3 Waffle 機器人
)

# 定義動作配置
from isaaclab.envs import ActionManager, ActionTermCfg

action_manager = ActionManager(
    action_terms=[
        ActionTermCfg(
            asset_name="robot",
            joint_names=["left_wheel", "right_wheel"],
            scale=1.0,
            mode="velocity"  # 使用速度控制模式
        )
    ]
)

# 定義觀測配置
from isaaclab.envs import ObservationManager, ObservationTermCfg

observation_manager = ObservationManager(
    observation_groups={
        "policy": [
            ObservationTermCfg(term="root_pos", relative=True),  # 機器人根部位置
            ObservationTermCfg(term="root_quat", relative=True),  # 機器人根部四元數
            ObservationTermCfg(term="joint_vel", relative=True)   # 關節速度
        ]
    }
)

# 定義事件配置
from isaaclab.envs import EventManager, EventTermCfg

event_manager = EventManager(
    event_terms=[
        EventTermCfg(
            mode="reset",
            func="reset_root_state",
            params={"pos_range": ((-1.0, 1.0), (-1.0, 1.0), (0.1, 0.1))}  # 重置時隨機位置範圍
        )
    ]
)

# 定義環境配置類別
class Turtlebot3WaffleEnvCfg(ManagerBasedEnvCfg):
    def __init__(self):
        super().__init__()
        self.scene = scene_cfg
        self.actions = action_manager
        self.observations = observation_manager
        self.events = event_manager
        self.sim = sim_utils.SimulationCfg(dt=0.01, substeps=2)  # 模擬時間步長和子步數

# 定義環境類別
class Turtlebot3WaffleEnv(ManagerBasedEnv):
    def __init__(self, cfg: Turtlebot3WaffleEnvCfg):
        super().__init__(cfg)

    def reset(self):
        # 自定義重置邏輯（可選）
        return super().reset()

    def step(self, action):
        # 自定義步進邏輯（可選）
        return super().step(action)

# 註冊環境（可選）
if __name__ == "__main__":
    from isaaclab.envs import register_env
    register_env("Turtlebot3Waffle", Turtlebot3WaffleEnv, Turtlebot3WaffleEnvCfg)