# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations for the lift task.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab_tasks.manager_based.manipulation.playground_g1.mdp import hand_is_grasping
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def task_done(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    cube_3_cfg: SceneEntityCfg = SceneEntityCfg("cube_3"),
    xy_threshold: float = 0.05,
    height_threshold: float = 0.005,
    height_diff: float = 0.06,
    right_eef_max_x: float = 0.30,
    right_eef_max_y: float = -0.10,
):
    """Check if three cubes are stacked by the specified robot.

    This function determines if cube_1 is stacked on cube_2, and cube_2 is stacked on cube_3,
    by checking:
    1. The xy-distance between consecutive cubes is within the xy_threshold.
    2. The height difference (adjusted by height_diff) is within the height_threshold.
    3. The robot's right hand is in an open state (not grasping), as determined by observation buffer.

    Args:
        env (ManagerBasedRLEnv): The reinforcement learning environment instance.
        robot_cfg (SceneEntityCfg, optional): Configuration for the robot entity.
                                             Defaults to SceneEntityCfg("robot").
        cube_1_cfg (SceneEntityCfg, optional): Configuration for the first cube entity.
                                              Defaults to SceneEntityCfg("cube_1").
        cube_2_cfg (SceneEntityCfg, optional): Configuration for the second cube entity.
                                              Defaults to SceneEntityCfg("cube_2").
        cube_3_cfg (SceneEntityCfg, optional): Configuration for the third cube entity.
                                              Defaults to SceneEntityCfg("cube_3").
        xy_threshold (float, optional): Maximum allowed xy-distance between cubes.
        height_threshold (float, optional): Maximum allowed height difference.
        height_diff (float, optional): Expected height difference between cubes.
        
        right_eef_max_x: Maximum x position of the right eef for task completion.
        right_eef_max_y: Maximum y position of the right eef for task completion.
    
    Returns:
        torch.Tensor: A boolean tensor of shape (num_envs,) indicating whether each environment
                     has the cubes stacked.
    """
    # Get the robot and cube entities from the environment
    robot: Articulation = env.scene[robot_cfg.name]
    cube_1: RigidObject = env.scene[cube_1_cfg.name]
    cube_2: RigidObject = env.scene[cube_2_cfg.name]
    cube_3: RigidObject = env.scene[cube_3_cfg.name]

    # Calculate position differences between consecutive cubes
    pos_diff_c12 = cube_1.data.root_pos_w - cube_2.data.root_pos_w
    pos_diff_c23 = cube_2.data.root_pos_w - cube_3.data.root_pos_w

    # Compute cube position difference in x-y plane
    xy_dist_c12 = torch.norm(pos_diff_c12[:, :2], dim=1)
    xy_dist_c23 = torch.norm(pos_diff_c23[:, :2], dim=1)

    # Compute cube height difference
    h_dist_c12 = torch.norm(pos_diff_c12[:, 2:], dim=1)
    h_dist_c23 = torch.norm(pos_diff_c23[:, 2:], dim=1)

    # Check cube positions for stacking
    stacked = torch.logical_and(xy_dist_c12 < xy_threshold, xy_dist_c23 < xy_threshold)
    #print(f"Stacked C12: {stacked}, xy_dist_c12: {xy_dist_c12} < xy_threshold: {xy_threshold}, xy_dist_c23: {xy_dist_c23} < xy_threshold: {xy_threshold}")
    stacked = torch.logical_and(h_dist_c12 - height_diff < height_threshold, stacked)
    #print(f"Stacked C12: {stacked}, h_dist_c12: {h_dist_c12} - height_diff: {height_diff} < height_threshold: {height_threshold}")
    stacked = torch.logical_and(h_dist_c23 - height_diff < height_threshold, stacked)
    #print(f"Stacked C23: {stacked}, h_dist_c23: {h_dist_c23} - height_diff: {height_diff} < height_threshold: {height_threshold}")

    # Check if the right hand is open (not grasping) using observation buffer
    grasping_status = hand_is_grasping(env)  # Shape: (num_envs, 2), column 1 is right hand
    right_hand_open = (1.0 - grasping_status[:, 1]).bool()  # True if open (not grasping), False if grasping
    
    # Combine stacking condition with right hand open condition
    stacked = torch.logical_and(stacked, right_hand_open)
    #print(f"Stacked with Right Hand Open: {stacked}, Right Hand Open: {right_hand_open}")

    # Get right eef position relative to environment origin
    ee_frame: FrameTransformer = env.scene["ee_frame"]
    right_eef_x = ee_frame.data.target_pos_w[:, 1, 0] - env.scene.env_origins[:, 0]
    right_eef_y = ee_frame.data.target_pos_w[:, 1, 1] - env.scene.env_origins[:, 1]

    done = torch.logical_and(stacked, right_eef_x < right_eef_max_x)
    #print(f"Done: {done}, right_eef_x: {right_eef_x} < right_eef_max_x: {right_eef_max_x}")
    done = torch.logical_and(done, right_eef_y < right_eef_max_y)
    #print(f"Final Done: {done}, right_eef_y: {right_eef_y} < right_eef_max_y: {right_eef_max_y}")

    return done