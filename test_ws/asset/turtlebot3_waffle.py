import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

TURTLEBOT3_WAFFLE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="D:/IsaacLab/test_ws/usd/turtlebot3/turtlebot3_waffle.usd",  # 替換為你的 USD 文件路徑
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=10.0,
            max_angular_velocity=10.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.1),  # 初始位置
        joint_pos={"left_wheel": 0.0, "right_wheel": 0.0}  # 輪子初始狀態
    ),
    actuators={
        "left_wheel_actuator": ImplicitActuatorCfg(
            joint_names_expr=["left_wheel"],
            effort_limit=10.0,
            velocity_limit=5.0,
            stiffness=0.0,
            damping=1.0,
        ),
        "right_wheel_actuator": ImplicitActuatorCfg(
            joint_names_expr=["right_wheel"],
            effort_limit=10.0,
            velocity_limit=5.0,
            stiffness=0.0,
            damping=1.0,
        ),
    },
)