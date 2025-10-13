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
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg

import isaaclab.envs.mdp as base_mdp
from . import mdp

from isaaclab.sensors import CameraCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip

from isaaclab_tasks.manager_based.manipulation.playground_g1.base_g1_env_cfg import BaseG1EnvCfg, G1BaseSceneCfg, G1BaseObservationsCfg

MARKER_CFG = FRAME_MARKER_CFG.copy()
MARKER_CFG.markers["frame"].scale = (0.04, 0.04, 0.04)

import carb
carb_settings_iface = carb.settings.get_settings()

DEBUG_VIS = False

##
# Scene definition
##
@configclass
class ObjectTableSceneCfg(G1BaseSceneCfg):

    # Object: Bottle
    bottle = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Bottle",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.38, -0.25, 0.9), rot=(0, 0, 0, 1)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Beaker/beaker_500ml.usd",
            scale=(0.5, 0.5, 1.0),
            
        ),
    )

    # Object: Mug
    mug = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Mug", #0.72
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0.1, 0.72), rot=(0.92388, 0, 0, -0.38268)),  # (0.4, 0.1, 0.72) in drawer; (0.4, 0.1, 0.81) on mug mat
        spawn=sim_utils.UsdFileCfg(
            usd_path="required_usd/SM_Mug_A2_rigid.usd",
        ),
    )

    # Mug mat
    mug_mat = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/MugMat",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0.1, 0.805), rot=(1, 0, 0, 0)),
        spawn=sim_utils.CylinderCfg(
            radius=0.0475,  # Enlarge from 0.045 to 0.0475
            height=0.005,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.21, 0.04, 0.01), metallic=1.0),
        ),
    )

    # Cabinet
    cabinet = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Cabinet",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Sektion_Cabinet/sektion_cabinet_instanceable.usd",
            scale=(1.0, 1.2, 1.0)
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.62, -0.15, 0.4),
            rot=(0.0, 0.0, 0.0, 1.0),
            joint_pos={
                "door_left_joint": 0.0,
                "door_right_joint": 0.0,
                "drawer_bottom_joint": 0.0,
                "drawer_top_joint": 0.0,
            },
        ),
        actuators={
            "drawers": ImplicitActuatorCfg(
                joint_names_expr=["drawer_top_joint", "drawer_bottom_joint"],
                effort_limit=87.0,
                velocity_limit=100.0,
                stiffness=10.0,
                damping=1.0,
            ),
            "doors": ImplicitActuatorCfg(
                joint_names_expr=["door_left_joint", "door_right_joint"],
                effort_limit=87.0,
                velocity_limit=100.0,
                stiffness=10.0,
                damping=2.5,
            ),
        },
    )

    bottle_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Bottle/beaker",
        debug_vis=DEBUG_VIS,
        visualizer_cfg=MARKER_CFG.replace(prim_path="/Visuals/BottleFrameTransformer"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Bottle/beaker",
                name="bottle_notch",
                offset=OffsetCfg(
                    pos=(0.0, -0.045, 0.055),
                    rot=(0.70711, 0.0, 0.0, -0.70711),
                ),
            ),
        ],
    )

    # Frame definitions for the cabinet.
    cabinet_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Cabinet/sektion",
        debug_vis=DEBUG_VIS,
        visualizer_cfg=MARKER_CFG.replace(prim_path="/Visuals/CabinetFrameTransformer"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Cabinet/drawer_handle_top",
                name="drawer_handle_top",
                offset=OffsetCfg(
                    pos=(0.305, 0.0, 0.01),
                    rot=(0.0, 0.0, 0.0, 1.0),
                ),
            ),
        ],
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
        
        cabinet_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("cabinet", joint_names=["drawer_top_joint"])},
        )

        drawer_pose = ObsTerm(func=mdp.get_drawer_pose)
        
        bottle_pose = ObsTerm(
            func=mdp.get_object_pose,
            params={"object_cfg": SceneEntityCfg("bottle")},
        )
        mug_pose = ObsTerm(
            func=mdp.get_object_pose,
            params={"object_cfg": SceneEntityCfg("mug")},
        )
        mug_mat_pose = ObsTerm(
            func=mdp.get_object_pose,
            params={"object_cfg": SceneEntityCfg("mug_mat")},
        )
        

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

        # 1. right-hand open the drawer.

        # 2. left-hand grasp the mug in drawer.
        # 3. pick&place the mug on the mug mat (and left-hand back).
        
        # 4. right-hand close the drawer.
        # 5. right-hand grasp the bottle on the table.
        # 6. pick the bottle to pouring into the mug.
        # 7. put the bottle back (and right-hand back).

        drawer_opened = ObsTerm(
            func=mdp.drawer_opened,
            params={
                "drawer_cfg": SceneEntityCfg("cabinet", joint_names=["drawer_top_joint"]),
            },
        )

        mug_grasped = ObsTerm(
            func=mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "hand_frame_cfg": SceneEntityCfg("hand_frame", body_ids=[0]),
                "object_cfg": SceneEntityCfg("mug"),
                "diff_threshold": 0.135,
            },
        )

        mug_placed = ObsTerm(
            func=mdp.object_placed,
            params={
                "hand_frame_cfg": SceneEntityCfg("hand_frame"),
                "object_cfg": SceneEntityCfg("mug"),
                "target_cfg": SceneEntityCfg("mug_mat"),
            },
        )

        drawer_closed = ObsTerm(
            func=mdp.drawer_closed,
            params={
                "drawer_cfg": SceneEntityCfg("cabinet", joint_names=["drawer_top_joint"]),
            },
        )

        bottle_grasped = ObsTerm(
            func=mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "hand_frame_cfg": SceneEntityCfg("hand_frame", body_ids=[1]),
                "object_cfg": SceneEntityCfg("bottle"),
                "diff_threshold": 0.08,
            },
        )

        pouring = ObsTerm(
            func=mdp.is_poured,
            params={
                "hand_frame_cfg": SceneEntityCfg("hand_frame", body_ids=[1]),
                "bottle_frame_cfg": SceneEntityCfg("bottle_frame"),
                "target_cfg": SceneEntityCfg("mug"),
                "tilt_angle": 40,
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

    bottle_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": 0.8, "asset_cfg": SceneEntityCfg("bottle")}
    )

    mug_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": 0.65, "asset_cfg": SceneEntityCfg("mug")}
    )

    success = DoneTerm(func=mdp.task_done)

@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_bottle = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": [-0.02, 0.00], "y": [-0.03, 0.03], "z": [0.0, 0.0]},  # {"x": [-0.01, 0.01], "y": [-0.03, 0.03], "z": [0.0, 0.0]},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("bottle"),
        },
    )

    reset_mug = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": [-0.01, 0.01], "y": [-0.03, 0.03], "z": [0.0, 0.0]},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("mug"),
        },
    )

    reset_mug_mat = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": [-0.04, -0.00], "y": [-0.05, 0.01], "z": [0.0, 0.0]},   # {"x": [-0.01, 0.01], "y": [-0.03, 0.03], "z": [0.0, 0.0]} -> Hard to reach
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("mug_mat"),
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

    cabinet_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
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
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("mug", body_names=".*"),
            "static_friction_range": (1.0, 1.0),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )


@configclass
class CabinetPourG1EnvCfg(BaseG1EnvCfg):
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
        # general settings
        self.decimation = 2
        self.episode_length_s = 60.0    # 2000 steps = 66.33 seconds per episode
        # simulation settings
        self.sim.dt = 1 / 60  # 60Hz
        self.sim.render_interval = 2

        # Add semantics to robot
        self.scene.robot.spawn.semantic_tags = [("class", "robot")]
        # Add semantics to ground
        self.scene.ground.spawn.semantic_tags = [("class", "ground")]