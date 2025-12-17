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

import isaaclab.envs.mdp as base_mdp
from . import mdp
from isaaclab_tasks.manager_based.manipulation.playground_g1.base_g1_env_cfg import BaseG1EnvCfg, G1BaseSceneCfg, G1BaseObservationsCfg


import carb
carb_settings_iface = carb.settings.get_settings()

CUBE_SIZE = (0.06, 0.06, 0.06)  # Size of the cubes in meters
CUBE_MASS = 0.02  # Mass of the cubes in kg

##
# Scene definition
##
@configclass
class ObjectTableSceneCfg(G1BaseSceneCfg):

    # Object 1: Red Cube
    red_cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/CubeRed",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.3, -0.7, 0.85), rot=(1, 0, 0, 0)),
        spawn=sim_utils.CuboidCfg(
            size=CUBE_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=CUBE_MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=1.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="max",
                restitution_combine_mode="min",
                static_friction=0.9,
                dynamic_friction=0.9,
                restitution=0.0,
            ),
        ),
    )
    # Object 2: Green Cube
    green_cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/CubeGreen",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.3, -0.9, 0.85), rot=(1, 0, 0, 0)),
        spawn=sim_utils.CuboidCfg(
            size=CUBE_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=CUBE_MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=1.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="max",
                restitution_combine_mode="min",
                static_friction=0.9,
                dynamic_friction=0.9,
                restitution=0.0,
            ),
        ),
    )

    # Object 3: Yellow Cube
    yellow_cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/CubeYellow",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, -0.9, 0.85), rot=(1, 0, 0, 0)),
        spawn=sim_utils.CuboidCfg(
            size=CUBE_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=CUBE_MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0), metallic=1.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="max",
                restitution_combine_mode="min",
                static_friction=0.9,
                dynamic_friction=0.9,
                restitution=0.0,
            ),
        ),
    )

    red_can = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/RedCan",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0.3, 0.88), rot=(1, 0, 0, 0)),
        spawn=sim_utils.CylinderCfg(
            radius=0.025,
            height=0.125,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=1.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="max",
                restitution_combine_mode="min",
                static_friction=0.9,
                dynamic_friction=0.9,
                restitution=0.0,
            ),
        ),
    )
    blue_can = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/BlueCan",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.3, 0.3, 0.88), rot=(1, 0, 0, 0)),
        spawn=sim_utils.CylinderCfg(
            radius=0.025,
            height=0.125,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0), metallic=1.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="max",
                restitution_combine_mode="min",
                static_friction=0.9,
                dynamic_friction=0.9,
                restitution=0.0,
            ),
        ),
    )

    work_table = AssetBaseCfg(
        prim_path="/World/envs/env_.*/WorkTable",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.45, 0.0, -0.01), rot=(0.7071, 0, 0, -0.7071)),
        spawn=UsdFileCfg(
            usd_path="local_models/table_with_basket.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        ),
    )

##
# MDP settings
##
@configclass
class ObservationsCfg(G1BaseObservationsCfg):
    """Observation specifications for the MDP."""
    # Inherited from the base robot observation group

    @configclass
    class SceneObsCfg(ObsGroup):
        """Observation of objects in the scene."""

        target_object_pose = ObsTerm(func=mdp.target_object_obs)
        task_completion = ObsTerm(func=mdp.task_completion)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    scene_obs: SceneObsCfg = SceneObsCfg()


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.target_object_dropping, params={"minimum_height": 0.8}
    )

    success = DoneTerm(func=mdp.task_done)


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    respawn_object = EventTerm(
        func=mdp.reset_random_choose_object,
        mode="reset",
        params={
            "target_pose": [0.25, -0.1, 0.89],
            "pose_range": {
                "x": [-0.03, 0.02],
                "y": [-0.1, 0.05]
            },
            "velocity_range": {},
            "asset_cfg_list": [SceneEntityCfg("red_can"), SceneEntityCfg("blue_can")],
        },
    )

@configclass
class CanSortingG1EnvCfg(BaseG1EnvCfg):
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
        self.episode_length_s = 15.0


@configclass
class BlockStackG1EnvCfg(CanSortingG1EnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # change robot init pos
        self.scene.robot.init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, -0.8, 0.82),
            rot=(1, 0, 0, 0),
            joint_pos={
                # Right Arm is slightly raised toward right side as the intial pose
                "right_shoulder_pitch_joint": 0.70,  # 0.65
                "right_shoulder_roll_joint": -0.2,
                "right_shoulder_yaw_joint": 0.0,
                "right_elbow_joint": -0.70, # -0.65
                "right_wrist_yaw_joint": -0.25, # -0.5
                "right_wrist_roll_joint": 0.0,
                "right_wrist_pitch_joint": 0.0,
            }
        )

@configclass
class ObjectPlacementG1EnvCfg(CanSortingG1EnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # change robot init pos
        self.scene.robot.init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.8, 0.82),
            rot=(1, 0, 0, 0),
            joint_pos={
                # Right Arm is slightly raised toward right side as the intial pose
                "right_shoulder_pitch_joint": 0.70,  # 0.65
                "right_shoulder_roll_joint": -0.2,
                "right_shoulder_yaw_joint": 0.0,
                "right_elbow_joint": -0.70, # -0.65
                "right_wrist_yaw_joint": -0.25, # -0.5
                "right_wrist_roll_joint": 0.0,
                "right_wrist_pitch_joint": 0.0,
            }
        )