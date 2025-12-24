# Copyright 2025 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

OPENARM_LEAPHAND_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="local_models/openarm_leaphand/openarm_leaphand.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
            solver_position_iteration_count = 16,
            solver_velocity_iteration_count = 4,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
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
            'index_mcp_forward': 0.0, 
            'middle_mcp_forward': 0.0, 
            'ring_mcp_forward': 0.0, 
            'thumb_mcp_side': 0.0, 
            'index_mcp_side': 0.0, 
            'middle_mcp_side': 0.0, 
            'ring_mcp_side': 0.0, 
            'thumb_mcp_forward': 0.0, 
            'index_pip': 0.0, 
            'middle_pip': 0.0, 
            'ring_pip': 0.0, 
            'thumb_pip_joint': 0.0, 
            'index_dip': 0.0, 
            'middle_dip': 0.0, 
            'ring_dip': 0.0, 
            'thumb_dip_joint': 0.0,
        },
    ),
    actuators={
        "openarm_arm": ImplicitActuatorCfg(
            joint_names_expr=[
                "openarm_left_joint[1-7]",
                "openarm_right_joint[1-7]",
            ],
            velocity_limit_sim={
                "openarm_left_joint[1-2]": 2.175,
                "openarm_right_joint[1-2]": 2.175,
                "openarm_left_joint[3-4]": 2.175,
                "openarm_right_joint[3-4]": 2.175,
                "openarm_left_joint[5-7]": 2.61,
                "openarm_right_joint[5-7]": 2.61,
            },
            effort_limit_sim={
                "openarm_left_joint[1-2]": 40.0,
                "openarm_right_joint[1-2]": 40.0,
                "openarm_left_joint[3-4]": 27.0,
                "openarm_right_joint[3-4]": 27.0,
                "openarm_left_joint[5-7]": 7.0,
                "openarm_right_joint[5-7]": 7.0,
            },
            stiffness={
                "openarm_left_joint[1-3]": 2500.0,
                "openarm_left_joint[4]": 4000.0,
                "openarm_left_joint[5-7]": 5000.0,
                "openarm_right_joint[1-3]": 2500.0,
                "openarm_right_joint[4]": 4000.0,
                "openarm_right_joint[5-7]": 5000.0,
            },
            damping=100.0,
        ),
        "leaphand_right": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_mcp_.*",
                ".*_pip.*",
                ".*_dip.*",
            ],
            velocity_limit_sim=100.0,
            effort_limit_sim=0.5,
            stiffness=3.0,
            damping=0.1,
            friction=0.01,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)