# Copyright (c) 2024-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


import torch
from collections.abc import Sequence

import isaaclab.utils.math as PoseUtils
from isaaclab.envs import ManagerBasedRLMimicEnv


class CabinetPourG1MimicEnv(ManagerBasedRLMimicEnv):

    def get_robot_eef_pose(self, eef_name: str, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        """
        Get current robot end effector pose. Should be the same frame as used by the robot end-effector controller.

        Args:
            eef_name: Name of the end effector.
            env_ids: Environment indices to get the pose for. If None, all envs are considered.

        Returns:
            A torch.Tensor eef pose matrix. Shape is (len(env_ids), 4, 4)
        """
        if env_ids is None:
            env_ids = slice(None)

        eef_pos_name = f"{eef_name}_eef_pos"
        eef_quat_name = f"{eef_name}_eef_quat"

        target_eef_position = self.obs_buf["policy"][eef_pos_name][env_ids]
        target_eef_rot_mat = PoseUtils.matrix_from_quat(self.obs_buf["policy"][eef_quat_name][env_ids])

        return PoseUtils.make_pose(target_eef_position, target_eef_rot_mat)

    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict,
        gripper_action_dict: dict,
        action_noise_dict: dict | None = None,
        env_id: int = 0,
    ) -> torch.Tensor:
        """
        Takes a target pose and gripper action for the end effector controller and returns an action
        (usually a normalized delta pose action) to try and achieve that target pose.
        Noise is added to the target pose action if specified.

        Args:
            target_eef_pose_dict: Dictionary of 4x4 target eef pose for each end-effector.
            gripper_action_dict: Dictionary of gripper actions for each end-effector.
            noise: Noise to add to the action. If None, no noise is added.
            env_id: Environment index to get the action for.

        Returns:
            An action torch.Tensor that's compatible with env.step().
        """

        # target position and rotation
        target_left_eef_pos, left_target_rot = PoseUtils.unmake_pose(target_eef_pose_dict["left"])
        target_right_eef_pos, right_target_rot = PoseUtils.unmake_pose(target_eef_pose_dict["right"])

        target_left_eef_rot_quat = PoseUtils.quat_from_matrix(left_target_rot)
        target_right_eef_rot_quat = PoseUtils.quat_from_matrix(right_target_rot)

        # gripper actions
        left_gripper_action = gripper_action_dict["left"]
        right_gripper_action = gripper_action_dict["right"]

        if action_noise_dict is not None:
            pos_noise_left = action_noise_dict["left"] * torch.randn_like(target_left_eef_pos)
            pos_noise_right = action_noise_dict["right"] * torch.randn_like(target_right_eef_pos)
            quat_noise_left = action_noise_dict["left"] * torch.randn_like(target_left_eef_rot_quat)
            quat_noise_right = action_noise_dict["right"] * torch.randn_like(target_right_eef_rot_quat)

            target_left_eef_pos += pos_noise_left
            target_right_eef_pos += pos_noise_right
            target_left_eef_rot_quat += quat_noise_left
            target_right_eef_rot_quat += quat_noise_right

        return torch.cat(
            (
                target_left_eef_pos,
                target_left_eef_rot_quat,
                target_right_eef_pos,
                target_right_eef_rot_quat,
                left_gripper_action,
                right_gripper_action,
            ),
            dim=0,
        )

    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Converts action (compatible with env.step) to a target pose for the end effector controller.
        Inverse of @target_eef_pose_to_action. Usually used to infer a sequence of target controller poses
        from a demonstration trajectory using the recorded actions.

        Args:
            action: Environment action. Shape is (num_envs, action_dim).

        Returns:
            A dictionary of eef pose torch.Tensor that @action corresponds to.
        """
        target_poses = {}

        target_left_wrist_position = action[:, 0:3]
        target_left_rot_mat = PoseUtils.matrix_from_quat(action[:, 3:7])
        target_pose_left = PoseUtils.make_pose(target_left_wrist_position, target_left_rot_mat)
        target_poses["left"] = target_pose_left
        target_right_wrist_position = action[:, 7:10]
        target_right_rot_mat = PoseUtils.matrix_from_quat(action[:, 10:14])
        target_pose_right = PoseUtils.make_pose(target_right_wrist_position, target_right_rot_mat)
        target_poses["right"] = target_pose_right

        return target_poses

    def actions_to_gripper_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Extracts the gripper actuation part from a sequence of env actions (compatible with env.step).

        Args:
            actions: environment actions. The shape is (num_envs, num steps in a demo, action_dim).

        Returns:
            A dictionary of torch.Tensor gripper actions. Key to each dict is an eef_name.
        """

        return {"left": actions[:, 14:21], "right": actions[:, 21:]}
    
        """# action total 28 dim: first 14 dims are eef poses, last 14 dims are hand joints
        JOINT_INDEX = {
            "left_hand_idx": [14, 15, 16, 20, 21, 22, 26],  # Left hand joints index in actions
            "right_hand_idx": [17, 18, 19, 23, 24, 25, 27],  # Right hand joints index in actions
        }
        return {
            "left": actions[:, JOINT_INDEX["left_hand_idx"]],  # Extract left hand joints
            "right": actions[:, JOINT_INDEX["right_hand_idx"]]  # Extract right hand joints
        }"""

        

    def get_subtask_term_signals(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """
        Gets a dictionary of termination signal flags for each subtask in a task. The flag is 1
        when the subtask has been completed and 0 otherwise. The implementation of this method is
        required if intending to enable automatic subtask term signal annotation when running the
        dataset annotation tool. This method can be kept unimplemented if intending to use manual
        subtask term signal annotation.

        Args:
            env_ids: Environment indices to get the termination signals for. If None, all envs are considered.

        Returns:
            A dictionary termination signal flags (False or True) for each subtask.
        """
        if env_ids is None:
            env_ids = slice(None)

        signals = dict()
        subtask_terms = self.obs_buf["subtask_terms"]
        signals["drawer_opened"] = subtask_terms["drawer_opened"][env_ids]
        signals["mug_grasped"] = subtask_terms["mug_grasped"][env_ids]
        signals["mug_placed"] = subtask_terms["mug_placed"][env_ids]
        signals["drawer_closed"] = subtask_terms["drawer_closed"][env_ids]
        signals["bottle_grasped"] = subtask_terms["bottle_grasped"][env_ids]
        signals["pouring"] = subtask_terms["pouring"][env_ids]
        
        # final subtask signal is not needed
        return signals
