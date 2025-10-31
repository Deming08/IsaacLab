# Guidance of Dataset Collection for GR00T Project

This document provides a guide on how to operate and tune the trajectories for collecting G1-task dataset, such as `Isaac-Cabinet-Pour-G1-Abs-v0`. The process begins with generating raw waypoints via teleoperation, followed by a tuning process that involves fine-tuning poses and defining sub-task criteria.

## Workflow Overview

The overall workflow is a two-stage process:

1. __Waypoint Generation__: Use the teleoperation script to manually guide the robot through the task and record the key poses. These raw waypoints are saved to `open_drawer_waypoints_inspire.yaml`.
2. __Trajectory Tuning__: Use the raw waypoints as a reference to define and fine-tune precise poses (both absolute and relative) in `constants.py`. The `data_collect_agent.py` then uses these tuned constants to generate and execute smooth, repeatable trajectories.

---

### Stage 1: Waypoint Generation via Teleoperation

The [`teleop_se3_agent_g1_2hands.py`](teleop_se3_agent_g1_2hands.py) script is used to generate the initial set of waypoints by launching a teleoperation script:

  ```bash
  ./isaaclab.sh -p scripts/gr00t_script/teleop_se3_agent_g1_2hands.py
  ```

### Stage 2: Trajectory Tuning Process

After generating the raw waypoints, the next step is to tune the trajectories for better performance. The tuning process involves the following steps:

1. __Record Key Waypoints__: The first step is to record the key waypoints for the task in [`open_drawer_waypoints_inspire.yaml`](scripts/gr00t_script/configs/open_drawer_waypoints_inspire.yaml). These waypoints define the overall trajectory of the robot.

2. __Fine-Tune Poses__: Next, you need to fine-tune the relative/absolute poses in [`constants.py`](scripts/gr00t_script/utils/constants.py) with respect to the target object. This is an iterative process of adjusting the poses and observing the robot's behavior in the simulation.

3. __Derive Relative Poses__: You can use the [`convert.py`](scripts/gr00t_script/utils/convert.py) script to derive the relative pose (position and orientation) between the hand and the target object. This can be useful for fine-tuning the poses in [`constants.py`](scripts/gr00t_script/utils/constants.py).

4. __Playback Trajectory__: Once you have fine-tuned the poses, you can use the [`data_collect_agent.py`](scripts/gr00t_script/data_collect_agent.py) script to playback the trajectory and observe the robot's behavior.

5. __Define Sub-Task Criteria__: The criterion for each sub-task is defined in [`terminations.py`](source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/playground_g1/task_scenes/cabinet_pour/mdp/terminations.py) and [`observation.py`](source/isaaclab_tasks/isaaclab_tasks/manager_based/navigation/mdp/observation.py). You may need to adjust these criteria to match the desired behavior of the robot.

By following these steps, you can tune the trajectories for the GR00T project and achieve the desired robot behavior.
