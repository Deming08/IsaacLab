# Copyright (c) 2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import tempfile
import torch

from pink.tasks import FrameTask, PostureTask, DampingTask

import isaaclab.controllers.utils as ControllerUtils
import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers.pink_ik_cfg import PinkIKControllerCfg
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
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg

from . import mdp
from .mdp.actions.pink_actions_cfg import PinkInverseKinematicsActionCfg
from .mdp.actions.actions_cfg import JointPositionActionCfg

from isaaclab_assets.robots.unitree import G1_WITH_HAND_CFG  # isort: skip
from isaaclab.sensors import CameraCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip

from isaaclab_tasks.manager_based.manipulation.pick_place_g1 import mdp as pick_place_mdp
from isaaclab_tasks.manager_based.manipulation.stack_g1 import mdp as stack_mdp

MARKER_CFG = FRAME_MARKER_CFG.copy()
MARKER_CFG.markers["frame"].scale = (0.04, 0.04, 0.04)

import carb
carb_settings_iface = carb.settings.get_settings()

CUBE_SIZE = (0.06, 0.06, 0.06)
CUBE_MASS = 0.1
DEBUG_VIS = False

SCENE_OFFSET = 1.2

##
# Scene definition
##
@configclass
class CanSortingSceneCfg(InteractiveSceneCfg):

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
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.45+SCENE_OFFSET, 0.0, -0.01), rot=(0.7071, 0, 0, -0.7071)),
        spawn=UsdFileCfg(
            usd_path="required_usd/table_with_basket.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        ),
    )

@configclass
class CubeStackSceneCfg(InteractiveSceneCfg):

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
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5+SCENE_OFFSET*2, 0, 0), rot=(0.707, 0, 0, 0.707)),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/PackingTable/props/SM_HeavyDutyPackingTable_C02_01/SM_HeavyDutyPackingTable_C02_01_physics.usd",
            scale=(0.005, 0.01, 0.008),
            ),
    )
    
@configclass
class CabinetPourSceneCfg(InteractiveSceneCfg):

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

    # Frame definitions for the bottle notch.
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

    # Frame definitions for the cabinet top handle.
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


@configclass
class MixtureSceneCfg(CabinetPourSceneCfg, CanSortingSceneCfg, CubeStackSceneCfg):

    # Humanoid robot (Unitree G1 with hand)
    robot: ArticulationCfg = G1_WITH_HAND_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0, 0, 0.82),
            rot=(1, 0, 0, 0),
            joint_pos={
                # right-arm modified to match the offset frame in FrameTransformerCfg
                # 'right_arm_eef':
                #       action : [0.0640, -0.24,  0.9645, 0.9828103  -0.10791296 -0.01653928 -0.14887986]
                #       obs: : [ 0.16063991, -0.26858264,  0.97017527,  0.98314404, -0.10782921, -0.01493526, -0.14689295]
                'right_shoulder_pitch_joint': 0.72099626, 
                'right_shoulder_roll_joint': -0.40671825, 
                'right_shoulder_yaw_joint': -0.21167009, 
                'right_elbow_joint': -0.5924336, 
                'right_wrist_yaw_joint': -0.19227865, 
                'right_wrist_roll_joint': 0.2959846, 
                'right_wrist_pitch_joint': -0.09693236,
                
                # left-arm
                "left_shoulder_pitch_joint": 0.0,
                "left_shoulder_roll_joint": 0.4,    # Avoid the thumb poking the torso
                "left_shoulder_yaw_joint": 0.0,
                "left_elbow_joint": 1.57,
                "left_wrist_yaw_joint": 0.0,
                "left_wrist_roll_joint": 0.0,
                "left_wrist_pitch_joint": 0.0,
                # legs
                ".*_hip_pitch_joint": -0.0,
                ".*_hip_roll_joint": 0.0,
                ".*_hip_yaw_joint": 0.0,
                ".*_knee_joint": 0.0,
                ".*_ankle_pitch_joint": -0.0,
                ".*_ankle_roll_joint": 0.0,
                # waist
                "waist_yaw_joint": 0.0,
                "waist_roll_joint": 0.0,
                "waist_pitch_joint": 0.0,
                # hands
                "left_hand_index_0_joint": 0.0,
                "left_hand_index_1_joint": 0.0,
                "left_hand_middle_0_joint": 0.0,
                "left_hand_middle_1_joint": 0.0,
                "left_hand_thumb_0_joint": 0.0,
                "left_hand_thumb_1_joint": 1.0,
                "left_hand_thumb_2_joint": 0.0,
                "right_hand_index_0_joint": 0.0,
                "right_hand_index_1_joint": 0.0,
                "right_hand_middle_0_joint": 0.0,
                "right_hand_middle_1_joint": 0.0,
                "right_hand_thumb_0_joint": 0.0,
                "right_hand_thumb_1_joint": 0.0,
                "right_hand_thumb_2_joint": 0.0,
            },
            joint_vel={".*": 0.0},
        ),
    )

    # Listens to the required transforms
    ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/pelvis",
        debug_vis=DEBUG_VIS,
        visualizer_cfg=MARKER_CFG.replace(prim_path="/Visuals/eefFrameTransformer"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/left_wrist_yaw_link",
                name="left_end_effector",
                offset=OffsetCfg(
                    pos=(0.0, 0.0, 0.0),
                ),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/right_wrist_yaw_link",
                name="right_end_effector",
                offset=OffsetCfg(
                    pos=(0.0, 0.0, 0.0),
                ),
            ),
        ],
    )

    hand_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/pelvis",
        debug_vis=DEBUG_VIS,
        visualizer_cfg=MARKER_CFG.replace(prim_path="/Visuals/HandFrameTransformer"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/left_wrist_yaw_link",
                name="left_hand_palm",
                offset=OffsetCfg(
                    pos=(0.1, 0.0, 0.0),
                ),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/right_wrist_yaw_link",
                name="right_hand_palm",
                offset=OffsetCfg(
                    pos=(0.1, 0.0, 0.0),
                ),
            ),
        ],
    )

    # Sensors
    if carb_settings_iface.get("/isaaclab/cameras_enabled"):
        camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso_link/head_camera",
            update_period=0.1,
            height=480,
            width=640,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=10.0, focus_distance=3.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
            ),
            offset=CameraCfg.OffsetCfg(pos=(0.05, 0.0, 0.47), rot=(0.68301, 0.18301, -0.18301, -0.68301), convention="opengl"),
        )
    
    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=GroundPlaneCfg(),
    )

    # Lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

##
# MDP settings
##
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    
    pink_ik_cfg = PinkInverseKinematicsActionCfg(
        pink_controlled_joint_names=[
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_yaw_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_yaw_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
        ],
        # Joints to be locked in URDF
        ik_urdf_fixed_joint_names=[
            'left_hip_pitch_joint', 
            'right_hip_pitch_joint', 
            'waist_yaw_joint', 
            'left_hip_roll_joint', 
            'right_hip_roll_joint', 
            'waist_roll_joint', 
            'left_hip_yaw_joint', 
            'right_hip_yaw_joint', 
            'waist_pitch_joint', 
            'left_knee_joint', 
            'right_knee_joint', 
            'left_ankle_pitch_joint', 
            'right_ankle_pitch_joint', 
            'left_ankle_roll_joint', 
            'right_ankle_roll_joint', 
            'left_hand_index_0_joint', 
            'left_hand_middle_0_joint', 
            'left_hand_thumb_0_joint', 
            'right_hand_index_0_joint', 
            'right_hand_middle_0_joint', 
            'right_hand_thumb_0_joint', 
            'left_hand_index_1_joint', 
            'left_hand_middle_1_joint', 
            'left_hand_thumb_1_joint', 
            'right_hand_index_1_joint', 
            'right_hand_middle_1_joint', 
            'right_hand_thumb_1_joint', 
            'left_hand_thumb_2_joint', 
            'right_hand_thumb_2_joint',
        ],
        hand_joint_names=[
            "left_hand_index_0_joint",
            "left_hand_middle_0_joint",
            "left_hand_thumb_0_joint",
            "right_hand_index_0_joint",
            "right_hand_middle_0_joint",
            "right_hand_thumb_0_joint",
            "left_hand_index_1_joint",
            "left_hand_middle_1_joint",
            "left_hand_thumb_1_joint",
            "right_hand_index_1_joint",
            "right_hand_middle_1_joint",
            "right_hand_thumb_1_joint",
            "left_hand_thumb_2_joint",
            "right_hand_thumb_2_joint",
        ],
        # the robot in the sim scene we are controlling
        asset_name="robot",
        # Configuration for the IK controller
        controller=PinkIKControllerCfg(
            articulation_name="robot",
            base_link_name="pelvis",
            num_hand_joints=14,
            show_ik_warnings=True,
            variable_input_tasks=[
                FrameTask(
                    "g1_29dof_with_hand_rev_1_0_left_wrist_yaw_link",
                    position_cost=1.0,  # [cost] / [m]
                    orientation_cost=1.0,  # [cost] / [rad]
                    lm_damping=10,  # dampening for solver for step jumps
                    gain=0.5,
                ),
                FrameTask(
                    "g1_29dof_with_hand_rev_1_0_right_wrist_yaw_link",
                    position_cost=1.0,  # [cost] / [m]
                    orientation_cost=1.0,  # [cost] / [rad]
                    lm_damping=10,  # dampening for solver for step jumps
                    gain=0.5,
                ),
            ],
            fixed_input_tasks=[  # type: ignore
                # PostureTask: biases entire robot toward default configuration
                # Ensure default q0 has slight elbow flexion to avoid straight-arm singularity
                PostureTask(
                    cost=1e-2,
                ),
                # DampingTask to regularize velocities in nullspace
                DampingTask(
                    cost=1e-2,
                ),
            ],
        ),
    )


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
        

        if carb_settings_iface.get("/isaaclab/cameras_enabled"):
            rgb_image = ObsTerm(
                func=base_mdp.image, 
                params={
                    "sensor_cfg": SceneEntityCfg("camera"),
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

    any_object_dropping = DoneTerm(
        func=mdp.rigid_object_dropping, params={"minimum_height": 0.65}
    )

    cabinet_pour_success = DoneTerm(func=mdp.task_done)

    #cube_stack_success = DoneTerm(func=stack_mdp.task_done)

    can_sorting_success = DoneTerm(func=pick_place_mdp.task_done)

@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    
    set_robot_to_scene = EventTerm(
        func=mdp.reset_robot_state_to_scenes,
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

    # --- cube stack event ---
    randomize_cube1_positions = EventTerm(
        func=stack_mdp.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": {"x": (0.32+SCENE_OFFSET*2, 0.35+SCENE_OFFSET*2), "y": (-0.05, -0.02), "z": (0.85, 0.85), "yaw": (0.0, 1.0)}, # yaw = -1 will bend the arm
            "asset_cfgs": [SceneEntityCfg("cube_1")],
        },
    )

    randomize_cube_positions = EventTerm(
        func=stack_mdp.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": {"x": (0.18+SCENE_OFFSET*2, 0.33+SCENE_OFFSET*2), "y": (-0.32, -0.18), "z": (0.85, 0.85), "yaw": (-0.5, 0.4)}, # yaw = -1 will bend the arm
            "min_separation": 0.15,
            "asset_cfgs": [SceneEntityCfg("cube_2"), SceneEntityCfg("cube_3")],
        },
    )

    # --- can sorting event ---
    respawn_object = EventTerm(
        func=pick_place_mdp.reset_random_choose_object,
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
class PlaygroundG1EnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the Unitree G1 playground environment."""

    # Scene settings
    scene: MixtureSceneCfg = MixtureSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    terminations: TerminationsCfg = TerminationsCfg()
    events = EventCfg()

    # Unused managers
    commands = None
    rewards = None
    curriculum = None

    # Position of the XR anchor in the world frame
    xr: XrCfg = XrCfg(
        anchor_pos=(0.0, 0.0, 0.0),
        anchor_rot=(1.0, 0.0, 0.0, 0.0),
    )

    # Temporary directory for URDF files
    temp_urdf_dir = tempfile.gettempdir()


    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 60.0
        # simulation settings
        self.sim.dt = 1 / 60  # 60Hz
        self.sim.render_interval = 2


        if not carb_settings_iface.get("/gr00t/use_joint_space"): # Use pink_ik_cfg as usual
            # Convert USD to URDF and change revolute joints to fixed
            temp_urdf_output_path, temp_urdf_meshes_output_path = ControllerUtils.convert_usd_to_urdf(
                self.scene.robot.spawn.usd_path, self.temp_urdf_dir, force_conversion=True
            )
            ControllerUtils.change_revolute_to_fixed(
                temp_urdf_output_path, self.actions.pink_ik_cfg.ik_urdf_fixed_joint_names
            )

            # Set the URDF and mesh paths for the IK controller
            self.actions.pink_ik_cfg.controller.urdf_path = temp_urdf_output_path
            self.actions.pink_ik_cfg.controller.mesh_path = temp_urdf_meshes_output_path

        else:
            """Force replace the ActionCfg with joint space for gr00t inference"""
            self.actions.pink_ik_cfg = JointPositionActionCfg(
                asset_name="robot", 
                joint_names=[
                    'left_shoulder_pitch_joint', 
                    'right_shoulder_pitch_joint', 
                    'left_shoulder_roll_joint', 
                    'right_shoulder_roll_joint', 
                    'left_shoulder_yaw_joint', 
                    'right_shoulder_yaw_joint', 
                    'left_elbow_joint', 
                    'right_elbow_joint', 
                    'left_wrist_roll_joint', 
                    'right_wrist_roll_joint', 
                    'left_wrist_pitch_joint', 
                    'right_wrist_pitch_joint', 
                    'left_wrist_yaw_joint', 
                    'right_wrist_yaw_joint', 
                    'left_hand_index_0_joint', 
                    'left_hand_middle_0_joint', 
                    'left_hand_thumb_0_joint', 
                    'right_hand_index_0_joint', 
                    'right_hand_middle_0_joint', 
                    'right_hand_thumb_0_joint', 
                    'left_hand_index_1_joint', 
                    'left_hand_middle_1_joint', 
                    'left_hand_thumb_1_joint', 
                    'right_hand_index_1_joint', 
                    'right_hand_middle_1_joint', 
                    'right_hand_thumb_1_joint', 
                    'left_hand_thumb_2_joint', 
                    'right_hand_thumb_2_joint',
                ], 
                scale=1.0, 
                use_default_offset=False
                )
