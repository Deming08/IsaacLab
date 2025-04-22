# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to run the RL environment for the cartpole balancing task.

.. code-block:: bash

    ./isaaclab.sh -p scripts/tutorials/03_envs/run_cartpole_rl_env.py --num_envs 32

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on running the cartpole RL environment.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

from isaaclab.envs import ManagerBasedRLEnv

from isaaclab_tasks.manager_based.custom.turtlebot3.turtlebot3_env_cfg import TurtlebotEnvCfg

#######################################################


#######################################################

def main():
    """Main function."""
    # create environment configuration
    env_cfg = TurtlebotEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    # setup RL environment
    env = ManagerBasedRLEnv(cfg=env_cfg)

    # 初始化動作緩衝區
    wheel_velocities = torch.zeros_like(env.action_manager.action)
    alpha = 0.1  # 平滑因子，值越小越平滑

    # simulate physics
    count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            """
            # reset
            if count % 300 == 0:
                count = 0
                env.reset()
                print("-" * 80)
                print("[INFO]: Resetting environment...")
            """
            # 生成隨機目標速度並平滑
            target_velocities = torch.rand_like(env.action_manager.action)
            wheel_velocities = (1 - alpha) * wheel_velocities + alpha * target_velocities
            obs, rew, terminated, truncated, info = env.step(wheel_velocities)
            
            """
            # sample random actions
            joint_efforts = torch.randn_like(env.action_manager.action)
            # step the environment
            obs, rew, terminated, truncated, info = env.step(joint_efforts)
            """
            # print current orientation of pole
            #print("[Env 0]: Pole joint: ", obs["policy"][0][1].item())
            # update counter
            count += 1

    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()