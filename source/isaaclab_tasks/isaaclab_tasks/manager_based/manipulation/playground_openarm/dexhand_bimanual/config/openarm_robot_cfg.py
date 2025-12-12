
"""Base configuration for the G1 with different hand.

This configuration sets up the pink IK controller for the G1 humanoid robot with
left and right wrist control tasks. The controller is designed for upper body
manipulation tasks.
"""

from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from pink.tasks import FrameTask, PostureTask, DampingTask
from isaaclab.controllers.pink_ik import PinkIKControllerCfg, NullSpacePostureTask
from isaaclab.envs.mdp.actions.pink_actions_cfg import PinkInverseKinematicsActionCfg
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab_tasks.manager_based.manipulation.playground_openarm.assets.openarm_bimanual import OPEN_ARM_HIGH_PD_CFG  # isort: skip
import carb

# robot config
OPEN_ARM_ONLY_CFG: ArticulationCfg = OPEN_ARM_HIGH_PD_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=ArticulationCfg.InitialStateCfg(
                joint_pos={
                    "openarm_left_joint1": 0.0,
                    "openarm_left_joint2": 0.0,
                    "openarm_left_joint3": 0.0,
                    "openarm_left_joint4": 0.0,
                    "openarm_left_joint5": 0.0,
                    "openarm_left_joint6": 0.0,
                    "openarm_left_joint7": 0.0,
                    "openarm_right_joint1": 0.0,
                    "openarm_right_joint2": 0.0,
                    "openarm_right_joint3": 0.0,
                    "openarm_right_joint4": 0.0,
                    "openarm_right_joint5": 0.0,
                    "openarm_right_joint6": 0.0,
                    "openarm_right_joint7": 0.0,
                    "openarm_left_finger_joint.*": 0.0,
                    "openarm_right_finger_joint.*": 0.0,
                },  # Close the gripper
            ),
        )

# pink controller config
OPEN_ARM_ONLY_IK_ACTION_CFG = PinkInverseKinematicsActionCfg(
    pink_controlled_joint_names=[
        "openarm_left_joint1",
        "openarm_left_joint2",
        "openarm_left_joint3",
        "openarm_left_joint4",
        "openarm_left_joint5",
        "openarm_left_joint6",
        "openarm_left_joint7",
        "openarm_right_joint1",
        "openarm_right_joint2",
        "openarm_right_joint3",
        "openarm_right_joint4",
        "openarm_right_joint5",
        "openarm_right_joint6",
        "openarm_right_joint7",
    ],
    hand_joint_names=[
        # All the drive and mimic joints, total 24 joints
    ],
    target_eef_link_names={
        "left_wrist": "openarm_left_hand",
        "right_wrist": "openarm_right_hand",
    },
    # the robot in the sim scene we are controlling
    asset_name="robot",
    controller=PinkIKControllerCfg(
        articulation_name="robot",
        base_link_name="openarm_body_link",
        num_hand_joints=0,
        show_ik_warnings=False,
        fail_on_joint_limit_violation=False,
        variable_input_tasks=[
            FrameTask(
                "openarm_left_hand",
                position_cost=8.0,  # [cost] / [m]
                orientation_cost=2.0,  # [cost] / [rad]
                lm_damping=10,  # dampening for solver for step jumps
                gain=0.5,
            ),
            FrameTask(
                "openarm_right_hand",
                position_cost=8.0,  # [cost] / [m]
                orientation_cost=2.0,  # [cost] / [rad]
                lm_damping=10,  # dampening for solver for step jumps
                gain=0.5,
            ),
        ],
        fixed_input_tasks=[],
        xr_enabled=bool(carb.settings.get_settings().get("/app/xr/enabled")),
    ),
    enable_gravity_compensation=False,
)

# joint action config
OPEN_ARM_ONLY_JOINT_ACTION_CFG = JointPositionActionCfg(
    asset_name="robot",
    joint_names=[
        "openarm_left_joint1",
        "openarm_left_joint2",
        "openarm_left_joint3",
        "openarm_left_joint4",
        "openarm_left_joint5",
        "openarm_left_joint6",
        "openarm_left_joint7",
        "openarm_right_joint1",
        "openarm_right_joint2",
        "openarm_right_joint3",
        "openarm_right_joint4",
        "openarm_right_joint5",
        "openarm_right_joint6",
        "openarm_right_joint7",
    ],
    scale=1.0,
    use_default_offset=True,
    preserve_order=True,
)