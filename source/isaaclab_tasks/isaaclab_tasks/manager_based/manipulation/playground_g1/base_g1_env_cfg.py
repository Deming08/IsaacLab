# Copyright (c) 2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import tempfile
import shutil
from pink.tasks import FrameTask, PostureTask, DampingTask

import isaaclab.controllers.utils as ControllerUtils
import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.controllers.pink_ik import PinkIKControllerCfg, NullSpacePostureTask
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.openxr import ManusViveCfg, OpenXRDeviceCfg, XrCfg
from isaaclab.devices.openxr.retargeters.humanoid.unitree_g1.inspire.g1_retargeter import G1InspireHandRetargeterCfg
from isaaclab.devices.openxr.retargeters.humanoid.unitree_g1.trihand.g1_retargeter import G1TriHandRetargeterCfg
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

from isaaclab_tasks.manager_based.manipulation.playground_g1.config.g1_robot_cfg import (  # isort: skip
    G1_WITH_TRIHAND_ROBOT_CFG,
    G1_WITH_TRIHAND_IK_ACTION_CFG,
    G1_WITH_TRIHAND_JOINT_ACTION_CFG,
    G1_WITH_INSPIRE_ROBOT_CFG,
    G1_WITH_INSPIRE_IK_ACTION_CFG,
    G1_WITH_INSPIRE_JOINT_ACTION_CFG
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
class G1BaseSceneCfg(InteractiveSceneCfg):

    # Humanoid robot (Unitree G1 with hand)
    robot: ArticulationCfg = G1_WITH_INSPIRE_ROBOT_CFG

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
    
    pink_ik_cfg = G1_WITH_INSPIRE_IK_ACTION_CFG


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
        self.decimation = 4
        self.episode_length_s = 60.0
        # simulation settings
        self.sim.dt = 1 / 120  # 120Hz
        self.sim.render_interval = 4

        g1_hand_type = carb_settings_iface.get("/unitree_g1_env/hand_type")

        if g1_hand_type == "trihand":
            self.scene.robot = G1_WITH_TRIHAND_ROBOT_CFG
            ik_action_cfg = G1_WITH_TRIHAND_IK_ACTION_CFG
            joint_action_cfg = G1_WITH_TRIHAND_JOINT_ACTION_CFG

            #! SHORT-TERM Use pre-convert file from isaaclab 2.1.0 ver.(current ver. can't convert normally)
            temp_urdf_output_path = "robot_models/urdf/g1_29dof_with_hand_rev_1_0.urdf"
            temp_urdf_meshes_output_path = self.temp_urdf_dir + "/meshes"

            # Copy pre-convert meshes file to /tmp so that .urdf can load it with the same path on differnt host.
            shutil.copytree(src="robot_models/meshes", dst=temp_urdf_meshes_output_path, dirs_exist_ok=True)

            G1RetargeterCfg = G1TriHandRetargeterCfg

        else: # elif g1_hand_type == "inspire":
            self.scene.robot = G1_WITH_INSPIRE_ROBOT_CFG
            ik_action_cfg = G1_WITH_INSPIRE_IK_ACTION_CFG
            joint_action_cfg = G1_WITH_INSPIRE_JOINT_ACTION_CFG

            # Convert USD to URDF
            temp_urdf_output_path, temp_urdf_meshes_output_path = ControllerUtils.convert_usd_to_urdf(
                self.scene.robot.spawn.usd_path, self.temp_urdf_dir, force_conversion=True
            )
            G1RetargeterCfg = G1InspireHandRetargeterCfg

        if carb_settings_iface.get("/gr00t/use_joint_space"):
            """Force replace the ActionCfg with joint space for gr00t inference"""
            self.actions.pink_ik_cfg = joint_action_cfg
        else:
            """Use pink_ik_cfg as usual"""
            self.actions.pink_ik_cfg = ik_action_cfg

            # Convert USD to URDF and change revolute joints to fixed
            ControllerUtils.change_revolute_to_fixed(
                temp_urdf_output_path, self.actions.pink_ik_cfg.ik_urdf_fixed_joint_names
            )

            # Set the URDF and mesh paths for the IK controller
            self.actions.pink_ik_cfg.controller.urdf_path = temp_urdf_output_path
            self.actions.pink_ik_cfg.controller.mesh_path = temp_urdf_meshes_output_path

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
                }
            )