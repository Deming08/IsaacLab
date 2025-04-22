# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to use the interactive scene interface to setup a scene with multiple TurtleBot3 robots.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/02_scene/create_turtlebot_scene.py --num_envs 32

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the interactive scene interface with TurtleBot3.")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass

##
# Pre-defined configs
##
from isaaclab_assets import TURTLEBOT3_WAFFLE_CFG


@configclass
class TurtlebotSceneCfg(InteractiveSceneCfg):
    """Configuration for a TurtleBot3 scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",spawn=sim_utils.GroundPlaneCfg(),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    # articulation
    turtlebot: ArticulationCfg = TURTLEBOT3_WAFFLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    # Extract scene entities
    robot = scene["turtlebot"]
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    # Get joint indices
    left_idx = robot.joint_names.index("wheel_left_joint")
    right_idx = robot.joint_names.index("wheel_right_joint")

    # Simulation loop
    while simulation_app.is_running():
        # Reset
        if count % 500 == 0:
            count = 0
            # Reset root state with environment origins
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])
            # Reset joint states
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            # Clear internal buffers
            scene.reset()
            print("[INFO]: Resetting TurtleBot3 state...")
            print("Root pose after reset:", robot.data.root_state_w[:, :7])

        # Apply wheel velocities
        wheel_velocities = torch.zeros_like(robot.data.joint_vel)
        wheel_velocities[:, left_idx] = 20.0  # Left wheel speed
        wheel_velocities[:, right_idx] = 20.0  # Right wheel speed
        robot.set_joint_velocity_target(wheel_velocities)
        
        # Write data to sim
        scene.write_data_to_sim()
        
        # Debug output
        #print("Position before step:", robot.data.root_state_w[:, :3])
        #print("Joint velocities before step:", robot.data.joint_vel)
        
        # Perform step
        sim.step()
        
        # Update buffers
        scene.update(sim_dt)
        
        # Debug output
        #print("Position after step:", robot.data.root_state_w[:, :3])
        #print("Joint velocities after step:", robot.data.joint_vel)
        
        count += 1


def main():
    """Main function."""
    # Load simulation context
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, dt=0.005)  # Smaller dt for better physics
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])
    # Design scene
    scene_cfg = TurtlebotSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()