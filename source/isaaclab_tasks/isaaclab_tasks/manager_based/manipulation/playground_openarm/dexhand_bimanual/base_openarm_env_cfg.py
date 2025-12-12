# Copyright (c) 2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import tempfile

import isaaclab.controllers.utils as ControllerUtils
import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.openxr import ManusViveCfg, OpenXRDeviceCfg, XrCfg
from isaaclab.devices.openxr.retargeters.humanoid.unitree.inspire.g1_upper_body_retargeter import UnitreeG1RetargeterCfg
from isaaclab.devices.openxr.retargeters.humanoid.unitree.trihand.g1_upper_body_retargeter import G1TriHandUpperBodyRetargeterCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, retrieve_file_path

from . import mdp
from isaaclab.sensors import CameraCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip

from isaaclab_tasks.manager_based.manipulation.playground_openarm.dexhand_bimanual.config.openarm_robot_cfg import (  # isort: skip
    OPEN_ARM_ONLY_CFG,
    OPEN_ARM_ONLY_IK_ACTION_CFG,
    OPEN_ARM_ONLY_JOINT_ACTION_CFG,
)

MARKER_CFG = FRAME_MARKER_CFG.copy()
MARKER_CFG.markers["frame"].scale = (0.04, 0.04, 0.04)

import carb
carb_settings_iface = carb.settings.get_settings()

DEBUG_VIS = True

##
# Scene definition
##
@configclass
class OpenArmBaseSceneCfg(InteractiveSceneCfg):

    # Humanoid robot (OpenArm bimanual with hand)
    robot: ArticulationCfg = OPEN_ARM_ONLY_CFG

    # Listens to the required transforms
    ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/openarm_body_link",
        debug_vis=DEBUG_VIS,
        visualizer_cfg=MARKER_CFG.replace(prim_path="/Visuals/eefFrameTransformer"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/openarm_left_link7",
                name="left_end_effector",
                offset=OffsetCfg(
                    pos=(0.0, 0.0, 0.0),
                ),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/openarm_right_link7",
                name="right_end_effector",
                offset=OffsetCfg(
                    pos=(0.0, 0.0, 0.0),
                ),
            ),
        ],
    )

    hand_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/openarm_body_link",
        debug_vis=DEBUG_VIS,
        visualizer_cfg=MARKER_CFG.replace(prim_path="/Visuals/HandFrameTransformer"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/openarm_left_hand",
                name="left_hand_palm",
                offset=OffsetCfg(
                    pos=(0.0, 0.0, 0.03),
                ),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/openarm_right_hand",
                name="right_hand_palm",
                offset=OffsetCfg(
                    pos=(0.0, 0.0, 0.03),
                ),
            ),
        ],
    )
    
    # Sensors
    if carb_settings_iface.get("/isaaclab/cameras_enabled"):
        rgb_image = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/openarm_body_link/head_camera",
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
    
    # arm_action_cfg = OPEN_ARM_ONLY_IK_ACTION_CFG
    arm_action_cfg = OPEN_ARM_ONLY_JOINT_ACTION_CFG
    

@configclass
class OpenArmBaseObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class RobotObsCfg(ObsGroup):
        """Observations for policy group with state values."""

        processed_actions = ObsTerm(
            func=mdp.get_processed_action, 
            params={"action_name": "arm_action_cfg"}
            )
        
        robot_joint_pos = ObsTerm(
            func=base_mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot", 
                                                joint_names=OPEN_ARM_ONLY_JOINT_ACTION_CFG.joint_names, 
                                                preserve_order=True
                                                )},
        )
        robot_root_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("robot")})
        robot_root_quat = ObsTerm(func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("robot")})

        left_eef_pos = ObsTerm(func=mdp.get_left_eef_pos)
        left_eef_quat = ObsTerm(func=mdp.get_left_eef_quat)
        right_eef_pos = ObsTerm(func=mdp.get_right_eef_pos)
        right_eef_quat = ObsTerm(func=mdp.get_right_eef_quat)
        # hand_joint_state = ObsTerm(func=mdp.get_hand_state)
        
        
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
    robot_obs: RobotObsCfg = RobotObsCfg()


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

@configclass
class EventCfg:
    """Configuration for events."""
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")


@configclass
class BaseOpenArmEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the Unitree G1 playground environment."""

    # Scene settings
    scene: OpenArmBaseSceneCfg = OpenArmBaseSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)
    # Basic settings
    observations: OpenArmBaseObservationsCfg = OpenArmBaseObservationsCfg()
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
        self.decimation = 4
        self.episode_length_s = 60.0
        # simulation settings
        self.sim.dt = 1 / 120  # 120Hz
        self.sim.render_interval = 4


        # if carb_settings_iface.get("/gr00t/use_joint_space"):
        #     """Force replace the ActionCfg with joint space for gr00t inference"""
        #     self.actions.arm_action_cfg = OPEN_ARM_ONLY_JOINT_ACTION_CFG
        # else:
        #     """Use pink_ik_cfg as usual"""
        #     self.actions.arm_action_cfg = OPEN_ARM_ONLY_IK_ACTION_CFG

        #     # Convert USD to URDF
        #     temp_urdf_output_path, temp_urdf_meshes_output_path = ControllerUtils.convert_usd_to_urdf(
        #         self.scene.robot.spawn.usd_path, self.temp_urdf_dir, force_conversion=True
        #     )

        #     # Set the URDF and mesh paths for the IK controller
        #     self.actions.arm_action_cfg.controller.urdf_path = temp_urdf_output_path
        #     self.actions.arm_action_cfg.controller.mesh_path = temp_urdf_meshes_output_path
