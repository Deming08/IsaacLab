# Copyright (c) 2025, The Isaac Lab Project Developers.
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

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from termcolor import colored

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

import carb
carb_settings_iface = carb.settings.get_settings()

# z is from bottom
BASKET_LENTH_WIDTH_HEIGHT = (0.14, 0.1, 0.09)

def task_done(
    env: 'ManagerBasedRLEnv',
    right_wrist_max_x: float = 0.10,
    right_wrist_max_y: float = -0.30,
    min_vel: float = 0.1,
    debug: bool = False,
) -> torch.Tensor:
    """
    Determine if the target object (red_can or blue_can) is placed in the corresponding basket.

    This function checks whether the target object (obtained from carb settings) is placed in
    the correct basket (red_basket for red_can, blue_basket for blue_can) by verifying:
    1. Object is within the target basket's x, y, z range (cubic volume).
    2. Object velocity is below threshold.
    3. Right robot wrist is retracted back towards body (past a given x, y pos threshold).

    Args:
        env: The RL environment instance.
        right_wrist_max_x: Maximum x position of the right wrist for task completion.
        right_wrist_max_y: Maximum y position of the right wrist for task completion.
        min_vel: Minimum velocity magnitude of the object for task completion.

    Returns:
        Boolean tensor indicating which environments have completed the task.
    """
    # Get the target object name from carb settings
    target_object = carb_settings_iface.get("/pickplace_env/target_object")
    if target_object not in ["red_can", "blue_can"]:
        raise ValueError(f"Invalid target object: {target_object}. Must be 'red_can' or 'blue_can'.")
    target_destination = "red_basket" if target_object == "red_can" else "blue_basket"

    # Get object entity from the scene based on target_object
    object = env.scene[target_object]
    destination = env.scene[target_destination]

    # Extract object position relative to environment origin
    obj_x = object.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    obj_y = object.data.root_pos_w[:, 1] - env.scene.env_origins[:, 1]
    obj_z = object.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    obj_vel = torch.abs(object.data.root_vel_w)

    # Get right wrist position relative to environment origin
    eef_frame = env.scene["ee_frame"]
    right_wrist_x = eef_frame.data.target_pos_w[:, 1, 0] - env.scene.env_origins[:, 0]
    right_wrist_y = eef_frame.data.target_pos_w[:, 1, 1] - env.scene.env_origins[:, 1]

    # # Define basket ranges
    basket_center_x = destination.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    basket_center_y = destination.data.root_pos_w[:, 1] - env.scene.env_origins[:, 1]
    basket_center_z = destination.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    basket_length, basket_width, basket_height = BASKET_LENTH_WIDTH_HEIGHT

    min_x = basket_center_x - basket_length / 2
    max_x = basket_center_x + basket_length / 2 
    min_y = basket_center_y - basket_width / 2
    max_y = basket_center_y + basket_width / 2
    min_z = basket_center_z
    max_z = basket_center_z + basket_height

    # Check all success conditions and combine with logical AND
    done = torch.logical_and(obj_x < max_x, obj_x > min_x)
    done = torch.logical_and(done, obj_y < max_y)
    done = torch.logical_and(done, obj_y > min_y)
    done = torch.logical_and(done, obj_z < max_z)
    done = torch.logical_and(done, obj_z > min_z)
    done = torch.logical_and(done, right_wrist_x < right_wrist_max_x)
    done = torch.logical_and(done, right_wrist_y < right_wrist_max_y)
    done = torch.logical_and(done, obj_vel[:, 0] < min_vel)  # x velocity
    done = torch.logical_and(done, obj_vel[:, 1] < min_vel)  # y velocity
    done = torch.logical_and(done, obj_vel[:, 2] < min_vel)  # z velocity

    if debug:
        failed_envs = torch.where(~done)[0]
        if len(failed_envs) > 0:
            print(colored("----------------------------------------", "red"))
            print(colored(f"Task failed for envs: {failed_envs}", "red"))
            for env_idx in failed_envs:
                print(colored(f"  Env {env_idx}:", "red"))
                print(colored(f"    obj_x: {obj_x[env_idx].item():.4f} (min: {min_x[env_idx].item():.4f}, max: {max_x[env_idx].item():.4f})", "red"))
                print(colored(f"    obj_y: {obj_y[env_idx].item():.4f} (min: {min_y[env_idx].item():.4f}, max: {max_y[env_idx].item():.4f})", "red"))
                print(colored(f"    obj_z: {obj_z[env_idx].item():.4f} (min: {min_z[env_idx].item():.4f}, max: {max_z[env_idx].item():.4f})", "red"))
                print(colored(f"    right_wrist_x: {right_wrist_x[env_idx].item():.4f} (threshold: {right_wrist_max_x})", "red"))
                print(colored(f"    right_wrist_y: {right_wrist_y[env_idx].item():.4f} (threshold: {right_wrist_max_y})", "red"))
                print(colored(f"    obj_vel_x: {obj_vel[env_idx, 0].item():.4f} (threshold: {min_vel})", "red"))
                print(colored(f"    obj_vel_y: {obj_vel[env_idx, 1].item():.4f} (threshold: {min_vel})", "red"))
                print(colored(f"    obj_vel_z: {obj_vel[env_idx, 2].item():.4f} (threshold: {min_vel})", "red"))
            print(colored("----------------------------------------", "red"))

    return done

def target_object_dropping(
    env: 'ManagerBasedRLEnv',
    minimum_height: float,
) -> torch.Tensor:
    """Terminate when the target object's root height is below the minimum height.

    The target object is dynamically determined by carb_settings_iface.get("/pickplace_env/target_object"),
    which should return "red_can" or "blue_can".

    Note:
        This is currently only supported for flat terrains, i.e. the minimum height is in the world frame.
    """
    # Get the target object name from carb settings
    target_object = carb_settings_iface.get("/pickplace_env/target_object")
    if target_object not in ["red_can", "blue_can"]:
        raise ValueError(f"Invalid target object: {target_object}. Must be 'red_can' or 'blue_can'.")

    # Get object entity from the scene based on target_object
    asset: RigidObject = env.scene[target_object]

    # Check if the object's root height is below the minimum height
    return asset.data.root_pos_w[:, 2] < minimum_height
