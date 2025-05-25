# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to spawn a Turtlebot3 Waffle and interact with it."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on spawning and interacting with Turtlebot3 Waffle.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

##
# Pre-defined configs
##
from isaaclab_assets import TURTLEBOT3_WAFFLE_CFG  # 請替換為您的 Turtlebot3 配置模組

def design_scene() -> tuple[dict, list[list[float]]]:
    """Designs the scene."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Create separate groups called "Origin1", "Origin2"
    #origins = [[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]
    origins = [[0.0, 0.0, 0.0]]
    prim_utils.create_prim("/World/Origin1", "Xform", translation=origins[0])
    #prim_utils.create_prim("/World/Origin2", "Xform", translation=origins[1])
    # Articulation
    turtlebot_cfg = TURTLEBOT3_WAFFLE_CFG.copy()
    turtlebot_cfg.prim_path = "/World/Origin.*/Robot"
    turtlebot = Articulation(cfg=turtlebot_cfg)

    scene_entities = {"turtlebot": turtlebot}
    return scene_entities, origins

def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, Articulation], origins: torch.Tensor):
    """Runs the simulation loop."""
    turtlebot = entities["turtlebot"]
    sim_dt = sim.get_physics_dt()
    count = 0
    print("Joint names:", turtlebot.joint_names)
    #print("Body masses:", turtlebot.data.body_mass)
    print("Physics dt:", sim_dt)

    left_idx = turtlebot.joint_names.index("wheel_left_joint")
    right_idx = turtlebot.joint_names.index("wheel_right_joint")

    while simulation_app.is_running():
        if count % 500 == 0:
            count = 0
            offset = torch.tensor([0.0, 0.0, 0.0], device=sim.device).unsqueeze(0).expand_as(origins)
            root_state = turtlebot.data.default_root_state.clone()
            root_state[:, :3] = origins + offset
            turtlebot.write_root_pose_to_sim(root_state[:, :7])
            turtlebot.write_root_velocity_to_sim(root_state[:, 7:])
            joint_pos = turtlebot.data.default_joint_pos.clone()
            joint_vel = turtlebot.data.default_joint_vel.clone()
            turtlebot.write_joint_state_to_sim(joint_pos, joint_vel)
            turtlebot.reset()
            print("[INFO]: Resetting Turtlebot3 state...")
        
        # Apply higher wheel velocities
        wheel_velocities = torch.zeros_like(turtlebot.data.joint_vel)
        wheel_velocities[:, left_idx] = -10.0  # 左輪速度
        wheel_velocities[:, right_idx] = -10.0  # 右輪速度
        
        turtlebot.set_joint_velocity_target(wheel_velocities)
        turtlebot.write_data_to_sim()
        
        #print("Position before step:", turtlebot.data.root_state_w[:, :3])
        #print("Joint velocities before step:", turtlebot.data.joint_vel)
        sim.step()
        turtlebot.update(sim_dt)
        #print("Position after step:", turtlebot.data.root_state_w[:, :3])
        #print("Joint velocities after step:", turtlebot.data.joint_vel)
        
        count += 1

def main():
    """Main function."""
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])
    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)
    sim.reset()
    print("[INFO]: Setup complete...")
    run_simulator(sim, scene_entities, scene_origins)

if __name__ == "__main__":
    main()
    simulation_app.close()