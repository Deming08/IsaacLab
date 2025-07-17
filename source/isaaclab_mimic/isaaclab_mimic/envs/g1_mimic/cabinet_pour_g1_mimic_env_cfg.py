# Copyright (c) 2024-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig, SubTaskConstraintConfig, SubTaskConstraintType
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.playground_g1.cabinet_pour_g1_env_cfg import CabinetPourG1EnvCfg


@configclass
class CubeStackG1MimicEnvCfg(CabinetPourG1EnvCfg, MimicEnvCfg):

    def __post_init__(self):
        # Calling post init of parents
        super().__post_init__()

        # Override the existing values
        self.datagen_config.name = "cabinet_pour_g1_task"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = False
        self.datagen_config.generation_num_trials = 1000
        self.datagen_config.generation_select_src_per_subtask = True
        self.datagen_config.generation_select_src_per_arm = True
        self.datagen_config.generation_relative = False
        self.datagen_config.generation_transform_first_robot_pose = False
        self.datagen_config.generation_interpolate_from_last_target_pose = True
        self.datagen_config.max_num_failures = 25
        self.datagen_config.seed = 1


        subtask_configs_right = []
        subtask_configs_right.append(
            SubTaskConfig(
                object_ref="cabinet",
                subtask_term_signal="drawer_opened",
                first_subtask_start_offset_range=(0, 0),
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 1},
                action_noise=0.0,
                num_interpolation_steps=0,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Open the top drawer",
                next_subtask_description="Grasp the mug in drawer (left hand)",
            )
        )
        subtask_configs_right.append(
            SubTaskConfig(
                object_ref="cabinet",
                subtask_term_signal="drawer_closed",
                first_subtask_start_offset_range=(0, 0),
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 1},
                action_noise=0.0,
                num_interpolation_steps=0,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Close the top drawer",
                next_subtask_description="Grasp the bottle",
            )
        )
        subtask_configs_right.append(
            SubTaskConfig(
                object_ref="bottle",
                subtask_term_signal="bottle_grasped",
                first_subtask_start_offset_range=(0, 0),
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 1},
                action_noise=0.0,
                num_interpolation_steps=0,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Grasp the bottle on the table",
                next_subtask_description="Pour into the mug",
            )
        )
        subtask_configs_right.append(
            SubTaskConfig(
                object_ref="mug",
                subtask_term_signal="pouring",
                first_subtask_start_offset_range=(0, 0),
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 1},
                action_noise=0.0,
                num_interpolation_steps=0,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Pour into the mug",
                next_subtask_description="Put the bottle back",
            )
        )
        subtask_configs_right.append(
            SubTaskConfig(
                object_ref="bottle",
                subtask_term_signal=None,
                first_subtask_start_offset_range=(0, 0),
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 1},
                action_noise=0.0,
                num_interpolation_steps=0,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Put the bottle back on the table",
                next_subtask_description="Task completed",
            )
        )
        self.subtask_configs["right"] = subtask_configs_right


        subtask_configs_left = []
        subtask_configs_left.append(
            SubTaskConfig(
                object_ref="mug",
                subtask_term_signal="mug_grasped",
                first_subtask_start_offset_range=(0, 0),
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 1},
                action_noise=0.0,
                num_interpolation_steps=0,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Grasp the mug in drawer",
                next_subtask_description="Place the mug on the mug mat",
            )
        )
        subtask_configs_left.append(
            SubTaskConfig(
                object_ref="mug_mat",
                subtask_term_signal="mug_placed",
                first_subtask_start_offset_range=(0, 0),
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 1},
                action_noise=0.0,
                num_interpolation_steps=0,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Place the mug on the mug mat",
                next_subtask_description="Close the top drawer (right hand)",
            )
        )
        self.subtask_configs["left"] = subtask_configs_left



        self.task_constraint_configs = [
            SubTaskConstraintConfig(
                eef_subtask_constraint_tuple=[("right", 0), ("left", 0)],
                constraint_type=SubTaskConstraintType.SEQUENTIAL,
                sequential_min_time_diff=-1,
            ),
            SubTaskConstraintConfig(
                eef_subtask_constraint_tuple=[("left", 1), ("right", 1)],
                constraint_type=SubTaskConstraintType.SEQUENTIAL,
                sequential_min_time_diff=-1,
            ),
            SubTaskConstraintConfig(
                eef_subtask_constraint_tuple=[("right", 1), ("right", 2)],
                constraint_type=SubTaskConstraintType.SEQUENTIAL,
                sequential_min_time_diff=-1,
            ),
            SubTaskConstraintConfig(
                eef_subtask_constraint_tuple=[("left", 1), ("right", 3)],
                constraint_type=SubTaskConstraintType.SEQUENTIAL,
                sequential_min_time_diff=-1,
            ),
            SubTaskConstraintConfig(
                eef_subtask_constraint_tuple=[("right", 2), ("right", 3)],
                constraint_type=SubTaskConstraintType.SEQUENTIAL,
                sequential_min_time_diff=-1,
            ),
            SubTaskConstraintConfig(
                eef_subtask_constraint_tuple=[("right", 3), ("right", 4)],
                constraint_type=SubTaskConstraintType.SEQUENTIAL,
                sequential_min_time_diff=-1,
            ),
        ]