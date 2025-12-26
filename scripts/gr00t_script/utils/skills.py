# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This module contains modular skill definitions based on the SkillMimicGen paradigm."""

from typing import Optional, Dict, Union
import logging
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from . import constants as C
from .quaternion_utils import quat_xyzw_to_wxyz, quat_wxyz_to_xyzw
from .trajectory_player import TrajectoryPlayer

# Set up module-level logger for debug printing
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Default to WARNING; change to DEBUG to enable debug prints

# Add console handler if not already configured
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(name)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def format_pose(pose, digits = 5):
    """Recursively format numeric values in nested structures to 5 decimal places."""
    if isinstance(pose, np.ndarray):
        return np.round(pose.astype(float), digits).tolist()
    elif isinstance(pose, list):
        return [round(x, digits) if isinstance(x, (float, np.floating)) else format_pose(x) for x in pose]
    elif isinstance(pose, tuple):
        return tuple(format_pose(p) for p in pose)
    elif isinstance(pose, (float, np.floating)):
        return round(pose, digits)
    else:
        return pose


def create_waypoint(left_eef_pos, left_eef_quat, left_hand_bool, right_eef_pos, right_eef_quat, right_hand_bool):
    """Helper to create a waypoint dictionary."""
    wp = {
        "left_eef": np.concatenate([left_eef_pos, left_eef_quat]),
        "right_eef": np.concatenate([right_eef_pos, right_eef_quat]),
        "left_hand_bool": int(left_hand_bool), "right_hand_bool": int(right_hand_bool)
    }
    return wp


def generate_transit_or_transfer_motion(obs: Dict, initial_poses: Optional[dict] = None, target_poses: Optional[Union[dict, list]] = None) -> tuple[list, dict]:
    """A generic skill to move one or both arms to a target pose or a series of target poses."""
    # (current_left_eef_pos, current_left_eef_quat, current_right_eef_pos, current_right_eef_quat, *_) = TrajectoryPlayer.extract_essential_obs_data(obs)
    
    # if initial_poses:
    #     start_right_pos, start_right_quat = initial_poses["right_eef_pos"], initial_poses["right_eef_quat"]
    #     start_left_pos, start_left_quat = initial_poses["left_eef_pos"], initial_poses["left_eef_quat"]
    #     start_right_hand, start_left_hand = initial_poses.get("right_hand_closed", False), initial_poses.get("left_hand_closed", False)
    # else:
    #     start_right_pos, start_right_quat = current_right_eef_pos, current_right_eef_quat
    #     start_left_pos, start_left_quat = current_left_eef_pos, current_left_eef_quat
    #     start_right_hand, start_left_hand = False, False
    # init_waypoints = [create_waypoint(start_left_pos, start_left_quat, start_left_hand, start_right_pos, start_right_quat, start_right_hand)]

    waypoints = []
    if target_poses:
        if not isinstance(target_poses, list):
            target_poses = [target_poses]
        
        for poses in target_poses:
            waypoints.append(create_waypoint(poses["left_eef_pos"], poses["left_eef_quat"], poses["left_hand_closed"],
                                             poses["right_eef_pos"], poses["right_eef_quat"], poses["right_hand_closed"]))
    
    if not waypoints:
        return [], {}

    final_wp = waypoints[-1]
    final_poses = {
        "left_eef_pos": final_wp["left_eef"][:3], "left_eef_quat": final_wp["left_eef"][3:7],
        "right_eef_pos": final_wp["right_eef"][:3], "right_eef_quat": final_wp["right_eef"][3:7],
        "left_hand_closed": bool(final_wp["left_hand_bool"]), "right_hand_closed": bool(final_wp["right_hand_bool"]),
    }
    return waypoints, final_poses


def generate_retract_trajectory(obs: Dict, initial_poses: Optional[dict] = None) -> tuple[list, dict]:
    """Generates a multi-point trajectory to retract arms to specific intermediate poses."""
    retract_targets = [
        {
            "left_eef_pos": C.RETRACT_WAYPOINTS["left_retract_pos"], "left_eef_quat": C.RETRACT_WAYPOINTS["left_retract_quat"], "left_hand_closed": False,
            "right_eef_pos": C.RETRACT_WAYPOINTS["right_retract_pos"], "right_eef_quat": C.RETRACT_WAYPOINTS["right_retract_quat"], "right_hand_closed": False,
        },
        {
            "left_eef_pos": C.RETRACT_WAYPOINTS["left_retract_pos"], "left_eef_quat": C.RETRACT_WAYPOINTS["left_retract_quat"], "left_hand_closed": False,
            "right_eef_pos": C.RETRACT_WAYPOINTS["right_restore_pos"], "right_eef_quat": C.RETRACT_WAYPOINTS["right_restore_quat"], "right_hand_closed": False,
        }
    ]
    return generate_transit_or_transfer_motion(obs, initial_poses=initial_poses, target_poses=retract_targets)


class Skill:
    """Base class for a single robotic skill, divided into phases."""

    def __init__(self, obs: Dict, initial_poses: Optional[dict] = None):
        self.obs = obs
        self.initial_poses = initial_poses
        self.init_waypoints, self.motion_waypoints, self.terminal_waypoints = [], [], []
        self.waypoints = []

    def log_skill_poses(self, skill_name: str, approach_pose, action_pose, object_pose):
        """
        Modular method to log skill pose information for debugging.

        Args:
            skill_name: Name of the skill (e.g., "Grasp Can", "Place Can In Basket")
            approach_pose: Pose for approaching the object (pos, quat)
            action_pose: Pose for the main action (grasp/place) (pos, quat)
            object_pose: Original object pose (pos, quat)
        """
        logger.debug("=" * 60)
        logger.debug(f"{skill_name} Skill Poses:")
        logger.debug(f"Approach Pose: {format_pose(approach_pose)}")
        logger.debug(f"Action Pose: {format_pose(action_pose)}")
        logger.debug(f"Object Pose: {format_pose(object_pose)}")

    def init_phase(self):
        """ Move the robot's end-effector(s) to an approach pose and an interaction pose. """
        raise NotImplementedError

    def motion_phase(self):
        """ Main action of the skill where the robot interacts (grasp or release) with objects. """
        raise NotImplementedError

    def terminal_phase(self):
        """ leave to a safe or neutral position (offset) for the following transition or transfer. """
        raise NotImplementedError

    def get_skill_trajectory(self) -> tuple[list, dict]:
        """Executes all phases and returns the complete waypoint trajectory."""
        self.init_phase()
        self.motion_phase()
        self.terminal_phase()
        self.waypoints = self.init_waypoints + self.motion_waypoints + self.terminal_waypoints
        
        if not self.waypoints:
            return [], {}

        final_wp = self.waypoints[-1]
        final_poses = {
            "left_eef_pos": final_wp["left_eef"][:3], "left_eef_quat": final_wp["left_eef"][3:7],
            "right_eef_pos": final_wp["right_eef"][:3], "right_eef_quat": final_wp["right_eef"][3:7],
            "left_hand_closed": bool(final_wp["left_hand_bool"]), "right_hand_closed": bool(final_wp["right_hand_bool"]),
        }
        return self.waypoints, final_poses


class SubTask:
    """A higher-level task combining a pre-transit motion and a skill."""

    def __init__(self, obs: Dict, skill: Skill, initial_poses: Optional[dict] = None):
        self.obs = obs
        if initial_poses is None:
            (current_left_eef_pos, current_left_eef_quat, current_right_eef_pos, current_right_eef_quat, *_) = TrajectoryPlayer.extract_essential_obs_data(obs)
            self.initial_poses = {
                "left_eef_pos": current_left_eef_pos, "left_eef_quat": current_left_eef_quat,
                "right_eef_pos": current_right_eef_pos, "right_eef_quat": current_right_eef_quat,
                "right_hand_closed": False, "left_hand_closed": False
            }
        else:
            self.initial_poses = initial_poses
        
        # Skill starts from the approached pose, moves to the pick/place pose, grasps/releases the object, and ends at its own terminal (lift) pose.
        self.skill = skill

    def get_full_trajectory(self) -> tuple[list, dict]:
        """Generates the full trajectory for the sub-task including the pre-transit motion and the skill."""
        # Generate transit or transfer motion
        # transit_waypoints, transit_final_poses = generate_transit_or_transfer_motion(self.obs, initial_poses=self.initial_poses, target_poses=self.transit_target_pose)

        skill_waypoints, skill_final_poses = self.skill.get_skill_trajectory()

        # # Generate skill trajectory
        # if self.skill:
        #     # self.skill.initial_poses = transit_final_poses
        #     skill_waypoints, skill_final_poses = self.skill.get_skill_trajectory()
        #     # Combine trajectories
        #     waypoints = transit_waypoints + skill_waypoints[1:]
        
        # return transit_waypoints, transit_final_poses
        return skill_waypoints, skill_final_poses


class GraspCanSkill(Skill):
    """Skill to grasp a can."""

    def __init__(self, obs: Dict, initial_poses: Optional[dict] = None):
        super().__init__(obs, initial_poses)
        # Extract the object - can
        (_, _, _, _, _, _, _, _, _, _, obj_pos, obj_quat, can_color_id, *_) = TrajectoryPlayer.extract_essential_obs_data(obs)

        # Define picking/placing, approach, and leaving the object poses
        R_world_can = Rotation.from_quat(quat_wxyz_to_xyzw(obj_quat))
        self.act_quat = quat_xyzw_to_wxyz((R_world_can * Rotation.from_euler('xyz', C.CAN_GRASP_QUAT, degrees=True)).as_quat())

        self.act_pos = obj_pos + R_world_can.apply(C.CAN_GRASP_POS)
        self.approach_pos = self.act_pos + C.CAN_APPROACH_OFFSET_POS  # object_approach_pos = object_act_pos + R_world_can.apply(C.CAN_APPROACH_OFFSET_POS)
        self.leave_pos = self.act_pos + C.CAN_LEAVE_OFFSET_POS  # object_leave_pos = object_act_pos + R_world_can.apply(C.CAN_LEAVE_OFFSET_POS)

        # Log skill poses for debugging
        self.log_skill_poses(
            skill_name=f"{self.__class__.__name__} -  {'Red' if can_color_id==0 else 'Blue'} Can:",
            approach_pose = (self.approach_pos, self.act_quat),
            action_pose = (self.act_pos, self.act_quat),
            object_pose = (obj_pos.tolist(), obj_quat.tolist()),
        )

    def init_phase(self):
        """ Approaching the can """
        self.init_waypoints.append(create_waypoint(C.HOME_POSES["left_eef_pos"], C.HOME_POSES["left_eef_quat"], C.HOME_POSES["left_hand_closed"], 
                                                   self.approach_pos, self.act_quat, False))
        self.init_waypoints.append(create_waypoint(C.HOME_POSES["left_eef_pos"], C.HOME_POSES["left_eef_quat"], C.HOME_POSES["left_hand_closed"], 
                                                   self.act_pos, self.act_quat, False))
    
    def motion_phase(self):
        """ Definition: Grasping the can """
        self.motion_waypoints.append(create_waypoint(C.HOME_POSES["left_eef_pos"], C.HOME_POSES["left_eef_quat"], C.HOME_POSES["left_hand_closed"], 
                                                     self.act_pos, self.act_quat, True))

    def terminal_phase(self):
        """Lifting the can """
        self.terminal_waypoints.append(create_waypoint(C.HOME_POSES["left_eef_pos"], C.HOME_POSES["left_eef_quat"], C.HOME_POSES["left_hand_closed"], 
                                                       self.leave_pos, self.act_quat, True))


class PlaceCanInBasketSkill(Skill):
    """Skill to place a can in a basket."""
    
    def __init__(self, obs: Dict, initial_poses: Optional[dict] = None):
        super().__init__(obs, initial_poses)
        # Extract the object - basket
        (_, _, _, _, _, _, _, _, _, _, _, _, can_color_id, obj_pos, obj_quat, *_) = TrajectoryPlayer.extract_essential_obs_data(obs)

        # Define poses of approaching, releasing, and leaving the object
        R_world_object = Rotation.from_quat(quat_wxyz_to_xyzw(np.array([1.0, 0.0, 0.0, 0.0])))  # TODO: Replace with obj_quat
        self.act_quat = quat_xyzw_to_wxyz((R_world_object * Rotation.from_euler('xyz', C.BASKET_PLACE_QUAT, degrees=True)).as_quat())

        self.act_pos = obj_pos + R_world_object.apply(C.BASKET_PLACE_POS)
        self.approach_pos = self.act_pos + C.BASKET_APPROACH_OFFSET_POS  # approach_pos = act_pos + R_world_object.apply(C.BASKET_APPROACH_OFFSET_POS)
        self.leave_pos = self.act_pos + C.BASKET_LEAVE_OFFSET_POS  # leave_pos = act_pos + R_world_can.apply(C.CAN_LEAVE_OFFSET_POS)

        # Log skill poses for debugging
        self.log_skill_poses(
            skill_name=f"{self.__class__.__name__} -  {'Red' if can_color_id==0 else 'Blue'} Can:",
            approach_pose = (self.approach_pos, self.act_quat),
            action_pose = (self.act_pos, self.act_quat),
            object_pose = (obj_pos.tolist(), obj_quat.tolist()),
        )
    
    def init_phase(self):
        """ Approaching the basket """
        self.init_waypoints.append(create_waypoint(C.HOME_POSES["left_eef_pos"], C.HOME_POSES["left_eef_quat"], C.HOME_POSES["left_hand_closed"], 
                                                   self.approach_pos, self.act_quat, True))
        self.init_waypoints.append(create_waypoint(C.HOME_POSES["left_eef_pos"], C.HOME_POSES["left_eef_quat"], C.HOME_POSES["left_hand_closed"], 
                                                   self.act_pos, self.act_quat, True))

    def motion_phase(self):
        """ Releasing the can """
        self.motion_waypoints.append(create_waypoint(C.HOME_POSES["left_eef_pos"], C.HOME_POSES["left_eef_quat"], C.HOME_POSES["left_hand_closed"], 
                                                     self.act_pos, self.act_quat, False))

    def terminal_phase(self):
        """ Leaving away the can """
        self.terminal_waypoints.append(create_waypoint(C.HOME_POSES["left_eef_pos"], C.HOME_POSES["left_eef_quat"], C.HOME_POSES["left_hand_closed"], 
                                                       self.leave_pos, self.act_quat, False))


class OpenDrawerSkill(Skill):
    """Skill to open a drawer."""

    def __init__(self, obs: Dict, initial_poses: Optional[dict] = None):
        super().__init__(obs, initial_poses)
        # Extract the object - drawer handle
        (*_, obj_pos, obj_quat, _, _, _, _, _, _) = TrajectoryPlayer.extract_essential_obs_data(self.obs)
        
        # Define poses of approaching, acting(grasping), and leaving(drawing) the object
        R_world_object = Rotation.from_quat(quat_wxyz_to_xyzw(obj_quat))
        self.act_quat = quat_xyzw_to_wxyz((R_world_object * Rotation.from_euler('xyz', C.DRAWER_HANDLE_GRASP_QUAT, degrees=True)).as_quat())
        
        self.act_pos = obj_pos + R_world_object.apply(C.DRAWER_HANDLE_GRASP_POS)
        self.approach_pos = self.act_pos + C.DRAWER_HANDLE_APPROACH_OFFSET_POS
        self.leave_pos = self.act_pos + C.DRAWER_HANDLE_LEAVE_OFFSET_POS
        
        # Log skill poses for debugging
        self.log_skill_poses(
            skill_name=f"{self.__class__.__name__}:",
            approach_pose = (self.approach_pos, self.act_quat),
            action_pose = (self.act_pos, self.act_quat),
            object_pose = (obj_pos.tolist(), obj_quat.tolist()),
        )

    def init_phase(self):
        """ Approaching the drawer handle. """
        self.init_waypoints.append(create_waypoint(C.HOME_POSES["left_eef_pos"], C.HOME_POSES["left_eef_quat"], False, 
                                                   self.approach_pos, self.act_quat, False))
        self.init_waypoints.append(create_waypoint(C.HOME_POSES["left_eef_pos"], C.HOME_POSES["left_eef_quat"], False, 
                                                   self.act_pos, self.act_quat, False))

    def motion_phase(self):
        """ Grasping the drawer. """
        self.motion_waypoints.append(create_waypoint(C.HOME_POSES["left_eef_pos"], C.HOME_POSES["left_eef_quat"], False, 
                                                     self.act_pos, self.act_quat, True))

    def terminal_phase(self):
        """ Pulling out the drawer. """
        self.terminal_waypoints.append(create_waypoint(C.HOME_POSES["left_eef_pos"], C.HOME_POSES["left_eef_quat"], False, 
                                                       self.act_pos, self.act_quat, True))


class PickMugFromDrawerSkill(Skill):
    """Skill to pick a mug from inside a drawer."""

    def __init__(self, obs: Dict, initial_poses: Optional[dict] = None):
        super().__init__(obs, initial_poses)
        # Extract the object - mug
        (self.current_left_eef_pos, self.current_left_eef_quat, self.current_right_eef_pos, self.current_right_eef_quat,
         *_, obj_pos, obj_quat, _, _) = TrajectoryPlayer.extract_essential_obs_data(self.obs)
        
        # Define poses of approaching, acting(grasping), and leaving(lifting) the object
        R_world_object = Rotation.from_quat(quat_wxyz_to_xyzw(obj_quat))
        self.act_quat = quat_xyzw_to_wxyz((R_world_object * Rotation.from_euler('xyz', C.MUG_GRASP_QUAT, degrees=True)).as_quat())
        
        self.act_pos = obj_pos + C.MUG_GRASP_POS
        self.approach_pos = self.act_pos + C.MUG_APPROACH_OFFSET_POS
        self.leave_pos = self.act_pos + C.MUG_LEAVE_OFFSET_POS
        
        # Log skill poses for debugging
        self.log_skill_poses(
            skill_name=f"{self.__class__.__name__}:",
            approach_pose = (self.approach_pos, self.act_quat),
            action_pose = (self.act_pos, self.act_quat),
            object_pose = (obj_pos.tolist(), obj_quat.tolist()),
        )

    def init_phase(self):
        """ Approaching the mug. """
        self.init_waypoints.append(create_waypoint(self.approach_pos, self.act_quat, False,
                                                   self.current_right_eef_pos, self.current_right_eef_quat, True))
        self.init_waypoints.append(create_waypoint(self.act_pos, self.act_quat, False, 
                                                   self.current_right_eef_pos, self.current_right_eef_quat, True))

    def motion_phase(self):
        """ Grasping the mug. """
        self.motion_waypoints.append(create_waypoint(self.act_pos, self.act_quat, True,
                                                     self.current_right_eef_pos, self.current_right_eef_quat, True))

    def terminal_phase(self):
        """ Lifting the mug off the drawer. """
        self.terminal_waypoints.append(create_waypoint(self.leave_pos, self.act_quat, True,
                                                       self.current_right_eef_pos, self.current_right_eef_quat, True))


class PlaceMugOnMatSkill(Skill):
    """Skill to place a held mug onto a mat."""

    def __init__(self, obs: Dict, initial_poses: Optional[dict] = None):
        super().__init__(obs, initial_poses)
        # Extract the object - mat
        (self.current_left_eef_pos, self.current_left_eef_quat, self.current_right_eef_pos, self.current_right_eef_quat,
         *_, _, _, _, _, _, _, obj_pos, obj_quat) = TrajectoryPlayer.extract_essential_obs_data(self.obs)
        
        # Lower the mug onto the mat
        obj_yaw = Rotation.from_quat(quat_wxyz_to_xyzw(obj_quat)).as_euler('zyx', degrees=True)[0]  # TODO: Extract yaw only from mat orientation (No needed?)
        self.act_quat = quat_xyzw_to_wxyz((Rotation.from_euler('z', obj_yaw, degrees=True) * Rotation.from_euler('xyz', C.MAT_PLACE_ABS_QUAT, degrees=True)).as_quat())
        
        self.act_pos = obj_pos + C.MAT_PLACE_POS
        self.approach_pos = self.act_pos + C.MAT_APPROACH_OFFSET_POS
        self.leave_pos = self.act_pos + C.MAT_LEAVE_OFFSET_POS

        # Log skill poses for debugging
        self.log_skill_poses(
            skill_name=f"{self.__class__.__name__}:",
            approach_pose = (self.approach_pos, self.act_quat),
            action_pose = (self.act_pos, self.act_quat),
            object_pose = (obj_pos.tolist(), obj_quat.tolist()),
        )

    def init_phase(self):
        """ Lowering the mug on the mat. """
        self.init_waypoints.append(create_waypoint(self.approach_pos, self.act_quat, True, 
                                                   self.current_right_eef_pos, self.current_right_eef_quat, True))
        self.init_waypoints.append(create_waypoint(self.act_pos, self.act_quat, True, 
                                                   self.current_right_eef_pos, self.current_right_eef_quat, True))

    def motion_phase(self):
        """ Releasing the mug. """
        self.init_waypoints.append(create_waypoint(self.act_pos, self.act_quat, False, 
                                                   self.current_right_eef_pos, self.current_right_eef_quat, True))

    def terminal_phase(self):
        """ Lifting the hand and pushing the drawer. """
        # Push back the opened drawer (right EEF), and lift the left EEF away from the mug
        push_approach_pos = self.current_right_eef_pos + C.DRAWER_PUSH_DIRECTION_OFFSET
        self.terminal_waypoints.append(create_waypoint(self.leave_pos, self.act_quat, False, 
                                                       push_approach_pos, self.current_right_eef_quat, True))


class GraspBottleSkill(Skill):
    """Skill to grasp a bottle."""

    def __init__(self, obs: Dict, initial_poses: Optional[dict] = None):
        super().__init__(obs, initial_poses)
        # Extract the object - bottle
        (self.current_left_eef_pos, self.current_left_eef_quat, self.current_right_eef_pos, self.current_right_eef_quat,
         *_, obj_pos, obj_quat, _, _, _, _) = TrajectoryPlayer.extract_essential_obs_data(self.obs)
        
        # Grasp the bottle
        obj_yaw = Rotation.from_quat(quat_wxyz_to_xyzw(obj_quat)).as_euler('zyx', degrees=True)[0]  # TODO: Extract yaw only from mat orientation (No needed?)
        self.act_quat = quat_xyzw_to_wxyz((Rotation.from_euler('z', obj_yaw, degrees=True) * Rotation.from_euler('xyz', C.BOTTLE_GRASP_QUAT, degrees=True)).as_quat())
        
        self.act_pos = obj_pos + C.BOTTLE_GRASP_POS
        self.approach_pos = self.act_pos + C.BOTTLE_APPROACH_OFFSET_POS
        self.leave_pos = self.act_pos + C.BOTTLE_LEAVE_OFFSET_POS

        # Log skill poses for debugging
        self.log_skill_poses(
            skill_name=f"{self.__class__.__name__}:",
            approach_pose = (self.approach_pos, self.act_quat),
            action_pose = (self.act_pos, self.act_quat),
            object_pose = (obj_pos.tolist(), obj_quat.tolist()),
        )

    def init_phase(self):
        """ Approaching the bottle. """
        self.init_waypoints.append(create_waypoint(C.HOME_POSES["left_eef_pos"], C.HOME_POSES["left_eef_quat"], False, 
                                                   self.approach_pos, self.act_quat, False))
        self.init_waypoints.append(create_waypoint(C.HOME_POSES["left_eef_pos"], C.HOME_POSES["left_eef_quat"], False, 
                                                   self.act_pos, self.act_quat, False))

    def motion_phase(self):
        """ Grasping the bottle. """
        self.motion_waypoints.append(create_waypoint(C.HOME_POSES["left_eef_pos"], C.HOME_POSES["left_eef_quat"], False, 
                                                     self.act_pos, self.act_quat, True))

    def terminal_phase(self):
        """ Lifting the bottle. """
        self.terminal_waypoints.append(create_waypoint(C.HOME_POSES["left_eef_pos"], C.HOME_POSES["left_eef_quat"], False, 
                                                       self.leave_pos, self.act_quat, True))

class PourBottleSkill(Skill):
    """Skill to pour a held bottle."""

    def __init__(self, obs: Dict, initial_poses: Optional[dict] = None):
        super().__init__(obs, initial_poses)
        # Extract the object - mat
        (self.current_left_eef_pos, self.current_left_eef_quat, self.current_right_eef_pos, self.current_right_eef_quat,
         *_, _, _, obj_pos, obj_quat, _, _) = TrajectoryPlayer.extract_essential_obs_data(self.obs)
        
        # Pour the bottle
        self.act_quat = quat_xyzw_to_wxyz(Rotation.from_euler('xyz', C.MAT_POURING_QUAT, degrees=True).as_quat())  # absolute pouring orientation
        
        self.act_pos = obj_pos + C.MAT_POURING_POS # relative to mat instead of the original bottle !!!
        self.approach_pos = self.act_pos + C.MAT_POURING_APPROACH_OFFSET_POS
        self.leave_pos = self.act_pos + C.MAT_POURING_LEAVE_OFFSET_POS

        # Log skill poses for debugging
        self.log_skill_poses(
            skill_name=f"{self.__class__.__name__}:",
            approach_pose = (self.approach_pos, self.current_right_eef_quat),
            action_pose = (self.act_pos, self.act_quat),
            object_pose = (obj_pos.tolist(), obj_quat.tolist()),
        )

    def init_phase(self):
        """ Pouring (inclining) the bottle. """
        self.init_waypoints.append(create_waypoint(C.HOME_POSES["left_eef_pos"], C.HOME_POSES["left_eef_quat"], False, 
                                                   self.approach_pos, self.current_right_eef_quat, True))
        self.init_waypoints.append(create_waypoint(C.HOME_POSES["left_eef_pos"], C.HOME_POSES["left_eef_quat"], False, 
                                                   self.act_pos, self.act_quat, True))

    def motion_phase(self):
        """ No action for the hand needed during pouring. """
        pass
    
    def terminal_phase(self):
        """
        Definition: Rotating the bottle back to the vertical pose.
        """
        # Restore the bottle to the vertical pose
        self.terminal_waypoints.append(create_waypoint(C.HOME_POSES["left_eef_pos"], C.HOME_POSES["left_eef_quat"], False, 
                                                       self.leave_pos, self.current_right_eef_quat, True))


class ReturnBottleSkill(Skill):
    """Skill to return a held bottle to a target location."""

    def __init__(self, obs: Dict, initial_poses: Optional[dict] = None):
        super().__init__(obs, initial_poses)
        # TODO: Extract the object - drawer (?)
        (self.current_left_eef_pos, self.current_left_eef_quat, self.current_right_eef_pos, self.current_right_eef_quat,
         *_, obj_pos, obj_quat, _, _, _, _, _, _) = TrajectoryPlayer.extract_essential_obs_data(self.obs)
        
        # Return the bottle
        self.act_quat = quat_xyzw_to_wxyz(Rotation.from_euler('xyz', C.DRAWER_RETURN_QUAT, degrees=True).as_quat())  # absolute returning orientation
        
        self.act_pos = obj_pos + C.DRAWER_RETURN_POS # relative to mat instead of the original bottle !!!
        self.approach_pos = self.act_pos + C.DRAWER_RETURN_APPROACH_OFFSET_POS
        self.leave_pos = self.act_pos + C.DRAWER_RETURN_LEAVE_OFFSET_POS
        
        # Log skill poses for debugging
        self.log_skill_poses(
            skill_name=f"{self.__class__.__name__}:",
            approach_pose = (self.approach_pos, self.act_quat),
            action_pose = (self.act_pos, self.act_quat),
            object_pose = (obj_pos.tolist(), obj_quat.tolist()),
        )

    def init_phase(self):
        """ Approaching the table (lowering the bottle). """
        self.init_waypoints.append(create_waypoint(C.HOME_POSES["left_eef_pos"], C.HOME_POSES["left_eef_quat"], False, 
                                                   self.approach_pos, self.act_quat, True))
        self.init_waypoints.append(create_waypoint(C.HOME_POSES["left_eef_pos"], C.HOME_POSES["left_eef_quat"], False, 
                                                   self.act_pos, self.act_quat, True))

    def motion_phase(self):
        """ Release the bottle. """
        self.motion_waypoints.append(create_waypoint(C.HOME_POSES["left_eef_pos"], C.HOME_POSES["left_eef_quat"], False, 
                                                     self.act_pos, self.act_quat, False))

    def terminal_phase(self):
        """ Leaving away the bottle. """
        self.terminal_waypoints.append(create_waypoint(C.HOME_POSES["left_eef_pos"], C.HOME_POSES["left_eef_quat"], False, 
                                                       self.leave_pos, self.act_quat, False))
