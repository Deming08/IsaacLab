# Copyright (c) 2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.devices.openxr import XrCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import isaaclab.envs.mdp as base_mdp

from isaaclab_tasks.manager_based.manipulation.playground_g1.base_g1_env_cfg import BaseG1EnvCfg, G1BaseSceneCfg, G1BaseObservationsCfg

from isaaclab_tasks.manager_based.manipulation.playground_g1.task_scenes.can_sorting import mdp as can_sorting_mdp
from isaaclab_tasks.manager_based.manipulation.playground_g1.task_scenes.cube_stack import mdp as cube_stack_mdp
from isaaclab_tasks.manager_based.manipulation.playground_g1.task_scenes.cabinet_pour import mdp as cabinet_pour_mdp

from isaaclab_tasks.manager_based.manipulation.playground_g1.task_scenes.can_sorting.pickplace_g1_env_cfg import ObjectTableSceneCfg as CanSortingSceneCfg
from isaaclab_tasks.manager_based.manipulation.playground_g1.task_scenes.cube_stack.stack_g1_env_cfg import ObjectTableSceneCfg as CubeStackSceneCfg
from isaaclab_tasks.manager_based.manipulation.playground_g1.task_scenes.cabinet_pour.cabinet_pour_g1_env_cfg import ObjectTableSceneCfg as CabinetPourSceneCfg


import carb
carb_settings_iface = carb.settings.get_settings()

SCENE_OFFSET = 1.2

##
# Scene definition
##
@configclass
class _CanSortingSceneCfg(CanSortingSceneCfg):
    red_cube = None
    green_cube = None
    yellow_cube = None

@configclass
class _CubeStackSceneCfg(CubeStackSceneCfg):
    pass
    
@configclass
class _CabinetPourSceneCfg(CabinetPourSceneCfg):
    pass


@configclass
class MixtureSceneCfg(_CanSortingSceneCfg, _CubeStackSceneCfg, _CabinetPourSceneCfg):
    pass


##
# MDP settings
##
@configclass
class ObservationsCfg(G1BaseObservationsCfg):
    """Observation specifications for the MDP."""
    pass


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)

    any_object_dropping = DoneTerm(
        func=cabinet_pour_mdp.rigid_object_dropping, params={"minimum_height": 0.65}
    )

    #cabinet_pour_success = DoneTerm(func=cabinet_pour_mdp.task_done)
    #cube_stack_success = DoneTerm(func=cube_stack_mdp.task_done)
    #can_sorting_success = DoneTerm(func=can_sorting_mdp.task_done)


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=base_mdp.reset_scene_to_default, mode="reset")
    
    set_robot_to_scene = EventTerm(
        func=cabinet_pour_mdp.reset_robot_state_to_scenes,
        mode="reset",
        params={
            "offset_x_dict": {"CabinetPour": 0.0, 
                              "CanSorting": SCENE_OFFSET, 
                              "CubeStack": SCENE_OFFSET*2,
                              },
            "pose_range": {},
        },
    )

    # --- cabinet pour event ---
    reset_bottle = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": [-0.02, 0.00], "y": [-0.03, 0.03], "z": [0.0, 0.0]},  # {"x": [-0.01, 0.01], "y": [-0.03, 0.03], "z": [0.0, 0.0]},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("bottle"),
        },
    )

    reset_mug = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": [-0.01, 0.01], "y": [-0.03, 0.03], "z": [0.0, 0.0]},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("mug"),
        },
    )

    reset_mug_mat = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": [-0.04, -0.00], "y": [-0.05, 0.01], "z": [0.0, 0.0]},   # {"x": [-0.01, 0.01], "y": [-0.03, 0.03], "z": [0.0, 0.0]} -> Hard to reach
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("mug_mat"),
        },
    )

    robot_physics_material = EventTerm(
        func=base_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (1.5, 1.5),
            "dynamic_friction_range": (1.5, 1.5),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )

    cabinet_physics_material = EventTerm(
        func=base_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("cabinet", body_names="drawer_handle_top"),
            "static_friction_range": (2.0, 2.0),
            "dynamic_friction_range": (2.0, 2.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )

    mug_physics_material = EventTerm(
        func=base_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("mug", body_names=".*"),
            "static_friction_range": (1.0, 1.0),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )

    # --- cube stack event ---
    randomize_cube1_positions = EventTerm(
        func=cube_stack_mdp.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": {"x": (0.32+SCENE_OFFSET*2, 0.35+SCENE_OFFSET*2), "y": (-0.05, -0.02), "z": (0.85, 0.85), "yaw": (0.0, 1.0)}, # yaw = -1 will bend the arm
            "asset_cfgs": [SceneEntityCfg("cube_1")],
        },
    )

    randomize_cube_positions = EventTerm(
        func=cube_stack_mdp.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": {"x": (0.18+SCENE_OFFSET*2, 0.33+SCENE_OFFSET*2), "y": (-0.32, -0.18), "z": (0.85, 0.85), "yaw": (-0.5, 0.4)}, # yaw = -1 will bend the arm
            "min_separation": 0.15,
            "asset_cfgs": [SceneEntityCfg("cube_2"), SceneEntityCfg("cube_3")],
        },
    )

    # --- can sorting event ---
    respawn_object = EventTerm(
        func=can_sorting_mdp.reset_random_choose_object,
        mode="reset",
        params={
            "target_pose": [0.25+SCENE_OFFSET, -0.1, 0.89],
            "idle_pose": [0.6+SCENE_OFFSET, 0.4, 0.88],
            "pose_range": {
                "x": [-0.03, 0.02],
                "y": [-0.1, 0.05]
            },
            "velocity_range": {},
            "asset_cfg_list": [SceneEntityCfg("red_can"), SceneEntityCfg("blue_can")],
        },
    )

    
@configclass
class PlaygroundG1EnvCfg(BaseG1EnvCfg):
    """Configuration for the Unitree G1 playground environment."""

    # Scene settings
    scene: MixtureSceneCfg = MixtureSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()

    # MDP settings
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    # Unused managers
    commands = None
    rewards = None
    curriculum = None

    # Position of the XR anchor in the world frame
    xr: XrCfg = XrCfg(
        anchor_pos=(0.0, 0.0, 0.0),
        anchor_rot=(0.7071068, 0, 0, -0.7071068),
    )


    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 60.0
        # simulation settings
        self.sim.dt = 1 / 60  # 60Hz
        self.sim.render_interval = 2
        

        # adjust table position
        self.scene.work_table.init_state = AssetBaseCfg.InitialStateCfg(
            pos=(0.45+SCENE_OFFSET, 0.0, -0.01), rot=(0.7071, 0, 0, -0.7071)
            )
        self.scene.table.init_state = AssetBaseCfg.InitialStateCfg(
            pos=(0.5+SCENE_OFFSET*2, 0, 0), rot=(0.707, 0, 0, 0.707)
            )