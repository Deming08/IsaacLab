# Copyright (c) 2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import tempfile

from pink.tasks import FrameTask, PostureTask, DampingTask

import isaaclab.controllers.utils as ControllerUtils
import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.controllers.pink_ik import PinkIKControllerCfg, NullSpacePostureTask
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.openxr import ManusViveCfg, OpenXRDeviceCfg, XrCfg
from isaaclab.devices.openxr.retargeters.humanoid.unitree_g1.g1_retargeter import G1RetargeterCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils import configclass

from . import mdp
from isaaclab.envs.mdp.actions.pink_actions_cfg import PinkInverseKinematicsActionCfg
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg

from isaaclab_assets.robots.unitree import G1_INSPIRE_FTP_CFG  # isort: skip
from isaaclab.sensors import CameraCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip


MARKER_CFG = FRAME_MARKER_CFG.copy()
MARKER_CFG.markers["frame"].scale = (0.04, 0.04, 0.04)

import carb
carb_settings_iface = carb.settings.get_settings()

DEBUG_VIS = True

##
# Scene definition
##
@configclass
class G1BaseSceneCfg(InteractiveSceneCfg):

    # Humanoid robot (Unitree G1 with hand)
    robot: ArticulationCfg = G1_INSPIRE_FTP_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0, 0, 0.82),
            rot=(1, 0, 0, 0),
            joint_pos={
                # right-arm
                'right_shoulder_pitch_joint': 0.72099626, 
                'right_shoulder_roll_joint': -0.40671825, 
                'right_shoulder_yaw_joint': -0.21167009, 
                'right_elbow_joint': -0.5924336, 
                'right_wrist_yaw_joint': -0.19227865, 
                'right_wrist_roll_joint': 0.2959846, 
                'right_wrist_pitch_joint': -0.09693236,
                # left-arm
                "left_shoulder_pitch_joint": 0.0,
                "left_shoulder_roll_joint": 0.2,
                "left_shoulder_yaw_joint": 0.0,
                "left_elbow_joint": 1.57,
                "left_wrist_yaw_joint": 0.0,
                "left_wrist_roll_joint": 0.0,
                "left_wrist_pitch_joint": 0.0,
                # --
                "waist_.*": 0.0,
                ".*_hip_.*": 0.0,
                ".*_knee_.*": 0.0,
                ".*_ankle_.*": 0.0,
                "R_.*": 0.0,
                "L_.*": 0.0,
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
                    pos=(0.13, 0.0, 0.0),
                ),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/right_wrist_yaw_link",
                name="right_hand_palm",
                offset=OffsetCfg(
                    pos=(0.13, 0.0, 0.0),
                ),
            ),
        ],
    )
    
    # Sensors
    if carb_settings_iface.get("/isaaclab/cameras_enabled"):
        rgb_image = CameraCfg(
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
            "L_index_proximal_joint",
            "L_middle_proximal_joint",
            "L_pinky_proximal_joint",
            "L_ring_proximal_joint",
            "L_thumb_proximal_yaw_joint",
            "R_index_proximal_joint",
            "R_middle_proximal_joint",
            "R_pinky_proximal_joint",
            "R_ring_proximal_joint",
            "R_thumb_proximal_yaw_joint",
            "L_index_intermediate_joint",
            "L_middle_intermediate_joint",
            "L_pinky_intermediate_joint",
            "L_ring_intermediate_joint",
            "L_thumb_proximal_pitch_joint",
            "R_index_intermediate_joint",
            "R_middle_intermediate_joint",
            "R_pinky_intermediate_joint",
            "R_ring_intermediate_joint",
            "R_thumb_proximal_pitch_joint",
            "L_thumb_intermediate_joint",
            "R_thumb_intermediate_joint",
            "L_thumb_distal_joint",
            "R_thumb_distal_joint",
        ],
        hand_joint_names=[
            "L_index_proximal_joint",
            "L_middle_proximal_joint",
            "L_pinky_proximal_joint",
            "L_ring_proximal_joint",
            "L_thumb_proximal_yaw_joint",
            "R_index_proximal_joint",
            "R_middle_proximal_joint",
            "R_pinky_proximal_joint",
            "R_ring_proximal_joint",
            "R_thumb_proximal_yaw_joint",
            "L_index_intermediate_joint",
            "L_middle_intermediate_joint",
            "L_pinky_intermediate_joint",
            "L_ring_intermediate_joint",
            "L_thumb_proximal_pitch_joint",
            "R_index_intermediate_joint",
            "R_middle_intermediate_joint",
            "R_pinky_intermediate_joint",
            "R_ring_intermediate_joint",
            "R_thumb_proximal_pitch_joint",
            "L_thumb_intermediate_joint",
            "R_thumb_intermediate_joint",
            "L_thumb_distal_joint",
            "R_thumb_distal_joint",
        ],
        target_eef_link_names={
            "left_wrist": "left_wrist_yaw_link",
            "right_wrist": "right_wrist_yaw_link",
        },
        # the robot in the sim scene we are controlling
        asset_name="robot",
        controller=PinkIKControllerCfg(
            articulation_name="robot",
            base_link_name="pelvis",
            num_hand_joints=24,
            show_ik_warnings=False,
            fail_on_joint_limit_violation=False,
            variable_input_tasks=[
                FrameTask(
                    "g1_29dof_rev_1_0_left_wrist_yaw_link",
                    position_cost=8.0,  # [cost] / [m]
                    orientation_cost=2.0,  # [cost] / [rad]
                    lm_damping=10,  # dampening for solver for step jumps
                    gain=0.5,
                ),
                FrameTask(
                    "g1_29dof_rev_1_0_right_wrist_yaw_link",
                    position_cost=8.0,  # [cost] / [m]
                    orientation_cost=2.0,  # [cost] / [rad]
                    lm_damping=10,  # dampening for solver for step jumps
                    gain=0.5,
                ),
                NullSpacePostureTask(
                    cost=0.5,
                    lm_damping=1,
                    controlled_frames=[
                        "g1_29dof_rev_1_0_left_wrist_yaw_link",
                        "g1_29dof_rev_1_0_right_wrist_yaw_link",
                    ],
                    controlled_joints=[
                        "left_shoulder_pitch_joint",
                        "left_shoulder_roll_joint",
                        "left_shoulder_yaw_joint",
                        "right_shoulder_pitch_joint",
                        "right_shoulder_roll_joint",
                        "right_shoulder_yaw_joint",
                        "waist_yaw_joint",
                        "waist_pitch_joint",
                        "waist_roll_joint",
                    ],
                    gain=0.3,
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
            xr_enabled=bool(carb.settings.get_settings().get("/app/xr/enabled")),
        ),
    )


@configclass
class G1BaseObservationsCfg:
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
                    "sensor_cfg": SceneEntityCfg("rgb_image"),
                    "data_type": "rgb",
                    "normalize": False,
                    }
            )
        

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

@configclass
class EventCfg:
    """Configuration for events."""
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")


@configclass
class BaseG1EnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the Unitree G1 playground environment."""

    # Scene settings
    scene: G1BaseSceneCfg = G1BaseSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)
    # Basic settings
    observations: G1BaseObservationsCfg = G1BaseObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
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

    # OpenXR hand tracking has 26 joints per hand
    NUM_OPENXR_HAND_JOINTS = 26
    
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
                    "L_index_proximal_joint",
                    "L_middle_proximal_joint",
                    "L_pinky_proximal_joint",
                    "L_ring_proximal_joint",
                    "L_thumb_proximal_yaw_joint",
                    "R_index_proximal_joint",
                    "R_middle_proximal_joint",
                    "R_pinky_proximal_joint",
                    "R_ring_proximal_joint",
                    "R_thumb_proximal_yaw_joint",
                    "L_index_intermediate_joint",
                    "L_middle_intermediate_joint",
                    "L_pinky_intermediate_joint",
                    "L_ring_intermediate_joint",
                    "L_thumb_proximal_pitch_joint",
                    "R_index_intermediate_joint",
                    "R_middle_intermediate_joint",
                    "R_pinky_intermediate_joint",
                    "R_ring_intermediate_joint",
                    "R_thumb_proximal_pitch_joint",
                    "L_thumb_intermediate_joint",
                    "R_thumb_intermediate_joint",
                    "L_thumb_distal_joint",
                    "R_thumb_distal_joint",
                ], 
                scale=1.0, 
                use_default_offset=False
                )

        self.teleop_devices = DevicesCfg(
            devices={
                "handtracking": OpenXRDeviceCfg(
                    retargeters=[
                        G1RetargeterCfg(
                            enable_visualization=True,
                            # number of joints in both hands
                            num_open_xr_hand_joints=2 * self.NUM_OPENXR_HAND_JOINTS,
                            sim_device=self.sim.device,
                            hand_joint_names=self.actions.pink_ik_cfg.hand_joint_names,
                        ),
                    ],
                    sim_device=self.sim.device,
                    xr_cfg=self.xr,
                ),
                "manusvive": ManusViveCfg(
                    retargeters=[
                        G1RetargeterCfg(
                            enable_visualization=True,
                            num_open_xr_hand_joints=2 * self.NUM_OPENXR_HAND_JOINTS,
                            sim_device=self.sim.device,
                            hand_joint_names=self.actions.pink_ik_cfg.hand_joint_names,
                        ),
                    ],
                    sim_device=self.sim.device,
                    xr_cfg=self.xr,
                ),
            }
        )