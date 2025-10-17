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
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

import isaaclab.envs.mdp as base_mdp
from . import mdp
from isaaclab_tasks.manager_based.manipulation.playground_g1.base_g1_env_cfg import BaseG1EnvCfg, G1BaseSceneCfg, G1BaseObservationsCfg


import carb
carb_settings_iface = carb.settings.get_settings()

CUBE_SIZE = (0.06, 0.06, 0.06)
CUBE_MASS = 0.1

##
# Scene definition
##
@configclass
class ObjectTableSceneCfg(G1BaseSceneCfg):

    # Object 1: Red Cube
    cube_1 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube_1",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0.05, 0.85), rot=(1, 0, 0, 0)),
        spawn=sim_utils.CuboidCfg(
            size=CUBE_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=CUBE_MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=1.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="max",
                restitution_combine_mode="min",
                static_friction=0.9,
                dynamic_friction=0.9,
                restitution=0.1,
            ),
            semantic_tags=[("class", "cube_1")],
        ),
    )

    # Object 2: Green Cube
    cube_2 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube_2",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.22, -0.22, 0.85), rot=(1, 0, 0, 0)),
        spawn=sim_utils.CuboidCfg(
            size=CUBE_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=CUBE_MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=1.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="max",
                restitution_combine_mode="min",
                static_friction=0.9,
                dynamic_friction=0.9,
                restitution=0.1,
            ),
            semantic_tags=[("class", "cube_2")],
        ),
    )

    # Object 3: Blue Cube
    cube_3 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube_3",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.23, -0.32, 0.85), rot=(1, 0, 0, 0)),
        spawn=sim_utils.CuboidCfg(
            size=CUBE_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=CUBE_MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0), metallic=1.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="max",
                restitution_combine_mode="min",
                static_friction=0.9,
                dynamic_friction=0.9,
                restitution=0.1,
            ),
            semantic_tags=[("class", "cube_3")],
        ),
    )

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0, 0), rot=(0.707, 0, 0, 0.707)),
        spawn=UsdFileCfg(
            usd_path="required_usd/table.usd",
            ),
    )


##
# MDP settings
##
@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        processed_actions = ObsTerm(
            func=mdp.get_processed_action, 
            params={"action_name": "pink_ik_cfg"}
            )
        
        robot_joint_pos = ObsTerm(
            func=base_mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        robot_root_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("robot")})
        robot_root_quat = ObsTerm(func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("robot")})

        left_eef_pos = ObsTerm(func=mdp.get_left_eef_pos)
        left_eef_quat = ObsTerm(func=mdp.get_left_eef_quat)
        right_eef_pos = ObsTerm(func=mdp.get_right_eef_pos)
        right_eef_quat = ObsTerm(func=mdp.get_right_eef_quat)
        hand_joint_state = ObsTerm(func=mdp.get_hand_state)
        
        hand_is_grasping = ObsTerm(func=mdp.hand_is_grasping)
        object_obs = ObsTerm(func=mdp.object_obs)
        cube_positions = ObsTerm(func=mdp.cube_positions_in_world_frame)
        cube_orientations = ObsTerm(func=mdp.cube_orientations_in_world_frame)

        if carb_settings_iface.get("/isaaclab/cameras_enabled"):
            rgb_image = ObsTerm(
                func=base_mdp.image, 
                params={
                    "sensor_cfg": SceneEntityCfg("rgb_image"),
                    "data_type": "rgb",
                    "normalize": False,
                    }
            )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class SubtaskCfg(ObsGroup):
        """Observations for subtask group."""

        grasp_1 = ObsTerm(
            func=mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame", body_ids=[1]),
                "object_cfg": SceneEntityCfg("cube_2"),
                "diff_threshold": 0.085,
            },
        )
        stack_1 = ObsTerm(
            func=mdp.object_stacked,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "upper_object_cfg": SceneEntityCfg("cube_2"),
                "lower_object_cfg": SceneEntityCfg("cube_1"),
            },
        )
        grasp_2 = ObsTerm(
            func=mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame", body_ids=[1]),
                "object_cfg": SceneEntityCfg("cube_3"),
                "diff_threshold": 0.085,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    cube_1_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": 0.8, "asset_cfg": SceneEntityCfg("cube_1")}
    )

    cube_2_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": 0.8, "asset_cfg": SceneEntityCfg("cube_2")}
    )

    cube_3_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": 0.8, "asset_cfg": SceneEntityCfg("cube_3")}
    )

    success = DoneTerm(func=mdp.task_done)


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    randomize_cube1_positions = EventTerm(
        func=mdp.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": {"x": (0.32, 0.35), "y": (-0.05, -0.02), "z": (0.85, 0.85), "yaw": (0.0, 1.0)}, # yaw = -1 will bend the arm
            "asset_cfgs": [SceneEntityCfg("cube_1")],
        },
    )

    randomize_cube_positions = EventTerm(
        func=mdp.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": {"x": (0.18, 0.33), "y": (-0.32, -0.18), "z": (0.85, 0.85), "yaw": (-0.5, 0.4)}, # yaw = -1 will bend the arm
            "min_separation": 0.15,
            "asset_cfgs": [SceneEntityCfg("cube_2"), SceneEntityCfg("cube_3")],
        },
    )

    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (1.5, 1.5),
            "dynamic_friction_range": (1.5, 1.5),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )

@configclass
class CubeStackG1EnvCfg(BaseG1EnvCfg):
    """Configuration for the Unitree G1 pick-and-place environment."""

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)
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
        self.episode_length_s = 45.0    # 1300 steps = 43.33 seconds per episode

        # Add semantics to robot
        self.scene.robot.spawn.semantic_tags = [("class", "robot")]
        # Add semantics to table
        self.scene.table.spawn.semantic_tags = [("class", "table")]
        # Add semantics to ground
        self.scene.ground.spawn.semantic_tags = [("class", "ground")]
