
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

from isaaclab_tasks.manager_based.manipulation.playground_openarm.assets.openarm_leaphand import OPENARM_LEAPHAND_CFG  # isort: skip
import carb

# robot config
OPENARM_ROBOT_CFG: ArticulationCfg = OPENARM_LEAPHAND_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(-0.05, 0, 0.45),
                rot=(1, 0, 0, 0),
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
                },  # Close the gripper
            ),
        )

# pink controller config
OPENARM_IK_ACTION_CFG = PinkInverseKinematicsActionCfg(
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
        # All the drive and mimic joints, total 16 joints
        'index_mcp_forward', 
        'middle_mcp_forward', 
        'ring_mcp_forward', 
        'thumb_mcp_side', 
        'index_mcp_side', 
        'middle_mcp_side', 
        'ring_mcp_side', 
        'thumb_mcp_forward', 
        'index_pip', 
        'middle_pip', 
        'ring_pip', 
        'thumb_pip_joint', 
        'index_dip', 
        'middle_dip', 
        'ring_dip', 
        'thumb_dip_joint',
    ],
    target_eef_link_names={
        "left_wrist": "openarm_left_link7",
        "right_wrist": "openarm_right_link7",
    },
    # the robot in the sim scene we are controlling
    asset_name="robot",
    controller=PinkIKControllerCfg(
        articulation_name="robot",
        base_link_name="openarm_body_link0",
        num_hand_joints=16,
        show_ik_warnings=False,
        fail_on_joint_limit_violation=False,
        variable_input_tasks=[
            FrameTask(
                "openarm_left_link7",
                position_cost=8.0,  # [cost] / [m]
                orientation_cost=2.0,  # [cost] / [rad]
                lm_damping=10,  # dampening for solver for step jumps
                gain=0.5,
            ),
            FrameTask(
                "openarm_right_link7",
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
OPENARM_JOINT_ACTION_CFG = JointPositionActionCfg(
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
        'index_mcp_forward', 
        'middle_mcp_forward', 
        'ring_mcp_forward', 
        'thumb_mcp_side', 
        'index_mcp_side', 
        'middle_mcp_side', 
        'ring_mcp_side', 
        'thumb_mcp_forward', 
        'index_pip', 
        'middle_pip', 
        'ring_pip', 
        'thumb_pip_joint', 
        'index_dip', 
        'middle_dip', 
        'ring_dip', 
        'thumb_dip_joint',
    ],
    scale=1.0,
    use_default_offset=False,
    preserve_order=True,
)