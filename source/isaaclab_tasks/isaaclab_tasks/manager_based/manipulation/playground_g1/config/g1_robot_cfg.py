
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
from isaaclab_assets.robots.unitree import G1_INSPIRE_FTP_CFG, G1_29DOF_CFG  # isort: skip
import carb

# robot config
G1_WITH_TRIHAND_ROBOT_CFG: ArticulationCfg = G1_29DOF_CFG.replace(
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
            # --
            "waist_.*": 0.0,
            ".*_hip_.*": 0.0,
            ".*_knee_.*": 0.0,
            ".*_ankle_.*": 0.0,
            # hands
            ".*_hand_.*": 0.0,
            
        },
        joint_vel={".*": 0.0},
    ),
)
G1_WITH_TRIHAND_ROBOT_CFG.spawn.activate_contact_sensors = True
G1_WITH_TRIHAND_ROBOT_CFG.spawn.rigid_props.solver_position_iteration_count = 16
G1_WITH_TRIHAND_ROBOT_CFG.spawn.rigid_props.solver_velocity_iteration_count = 4
G1_WITH_TRIHAND_ROBOT_CFG.spawn.rigid_props.max_depenetration_velocity = 5.0
G1_WITH_TRIHAND_ROBOT_CFG.spawn.articulation_props.fix_root_link = True
G1_WITH_TRIHAND_ROBOT_CFG.spawn.articulation_props.enabled_self_collisions = True

# pink controller config
G1_WITH_TRIHAND_IK_ACTION_CFG = PinkInverseKinematicsActionCfg(
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
    target_eef_link_names={
        "left_wrist": "left_wrist_yaw_link",
        "right_wrist": "right_wrist_yaw_link",
    },
    # the robot in the sim scene we are controlling
    asset_name="robot",
    # Configuration for the IK controller
    controller=PinkIKControllerCfg(
        articulation_name="robot",
        base_link_name="pelvis",
        num_hand_joints=14,
        show_ik_warnings=True,
        fail_on_joint_limit_violation=False,
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
            NullSpacePostureTask(
                cost=0.5,
                lm_damping=1,
                controlled_frames=[
                    "g1_29dof_with_hand_rev_1_0_left_wrist_yaw_link",
                    "g1_29dof_with_hand_rev_1_0_right_wrist_yaw_link",
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
    ),
)

# joint action config
G1_WITH_TRIHAND_JOINT_ACTION_CFG = JointPositionActionCfg(
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

# robot config
G1_WITH_INSPIRE_ROBOT_CFG: ArticulationCfg = G1_INSPIRE_FTP_CFG.replace(
    prim_path="/World/envs/env_.*/Robot",
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0, 0, 0.82),
        rot=(1, 0, 0, 0),
        joint_pos={
            # right-arm
            'right_shoulder_pitch_joint': 0.0,
            'right_shoulder_roll_joint': -0.2,
            'right_shoulder_yaw_joint': 0.0,
            'right_elbow_joint': 1.57, 
            'right_wrist_yaw_joint': 0.0,
            'right_wrist_roll_joint': 0.0,
            'right_wrist_pitch_joint': 0.0,
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
G1_WITH_INSPIRE_ROBOT_CFG.spawn.rigid_props.solver_position_iteration_count = 16
G1_WITH_INSPIRE_ROBOT_CFG.spawn.rigid_props.solver_velocity_iteration_count = 4
G1_WITH_INSPIRE_ROBOT_CFG.spawn.rigid_props.max_depenetration_velocity = 5.0
G1_WITH_INSPIRE_ROBOT_CFG.spawn.articulation_props.fix_root_link = True
G1_WITH_INSPIRE_ROBOT_CFG.spawn.articulation_props.enabled_self_collisions = True
G1_WITH_INSPIRE_ROBOT_CFG.actuators["arms"].damping = 10.0

# pink controller config
G1_WITH_INSPIRE_IK_ACTION_CFG = PinkInverseKinematicsActionCfg(
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
    hand_joint_names=[
        # All the drive and mimic joints, total 24 joints
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
    enable_gravity_compensation=False,
)

# joint action config
G1_WITH_INSPIRE_JOINT_ACTION_CFG = JointPositionActionCfg(
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