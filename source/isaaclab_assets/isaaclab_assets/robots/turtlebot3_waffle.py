import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, ISAAC_NUCLEUS_DIR

TURTLEBOT3_WAFFLE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="D:/IsaacLab/test_ws/usd/turtlebot3/turtlebot3_waffle_pi.usd",  # 替換為你的 USD 文件路徑
        #usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Turtlebot/turtlebot3_burger.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=0.5,
            max_angular_velocity=162.72, #deg/s!
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=255, #range[1~255]
            solver_velocity_iteration_count=0, #useless
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.1),  # 初始位置
        joint_pos={"wheel_left_joint": 0.0, "wheel_right_joint": 0.0}  # 輪子初始狀態
    ),
    actuators={
        "left_wheel_actuator": ImplicitActuatorCfg(
            joint_names_expr=["wheel_left_joint"],
            #effort_limit_sim=4.1,
            #velocity_limit_sim=6.67,
            stiffness=0.0,
            damping=10000000.0,
        ),
        "right_wheel_actuator": ImplicitActuatorCfg(
            joint_names_expr=["wheel_right_joint"],
            #effort_limit_sim=4.1,
            #velocity_limit_sim=6.67,
            stiffness=0.0,
            damping=10000000.0,
        ),
    },
    collision_group=0,
)