#!/usr/bin/env python3
"""
Script to calculate relative poses for can grasping task.
Uses pose_difference() to compute CAN_GRASP_POS and CAN_GRASP_QUAT
relative to the can's reference frame.
"""

import numpy as np
import sys
import os

# Add the parent directory to the path to import convert functions
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.convert import pose_difference

def calculate_can_grasp():
    """
    Calculate relative poses for can grasping using the pose_difference function.

    Input data:
    - Target Can Pose: [0.25272524, -0.08587377, 0.89195764], [1.0, 0.00001174, 0.00001437, 0.00016935]
    - Grasp Pose from YAML: [0.11500076, -0.20500796, 0.92999208], [0.99984902, 0.00004765, -0.00002402, -0.00003580]
    """

    print("=" * 60)
    print("Calculating Relative Poses for Can Grasping Task")
    print("=" * 60)
    
    # Target can pose in world frame
    can_pos = np.array([0.19640331, -0.22273035, 0.89195764])
    can_quat = [0.55653214, -0.00001343, 0.0000129, -0.83082604]  # [w, x, y, z]
    
    # Hand grasp pose from YAML (step 1.2) in world frame adjusted for holding the can toward the inside
    # right_arm_eef: [0.11500076, -0.20500796, 0.92999208, 0.99984902, 0.00004765, -0.00002402, -0.00003580]
    hand_pos = np.array([0.06000590, -0.31500667, 0.92999375])
    hand_quat = [0.99984908, 0.00004850, -0.00003252, -0.00003294]  # [w, x, y, z]    
    
    print(f"Target Can Position (world): {can_pos}")
    print(f"Target Can Quaternion (world, wxyz): {can_quat}")
    print()
    print(f"Hand Grasp Position (world): {hand_pos}")
    print(f"Hand Grasp Quaternion (world, wxyz): {hand_quat}")
    print()
    
    # Calculate relative poses using pose_difference function
    # This gives hand pose relative to can frame
    relative_pos, relative_euler = pose_difference(
        hand_pos, hand_quat, can_pos, can_quat, 
        order='xyz', degrees=True
    )
    
    print("Calculating relative poses using pose_difference()...")
    print()
    
    print(f"CAN_GRASP_POS (relative to can frame):")
    print(f"  np.array([{relative_pos[0]:.8f}, {relative_pos[1]:.8f}, {relative_pos[2]:.8f}])")
    print()
    
    print(f"CAN_GRASP_QUAT (relative to can frame, Euler XYZ degrees):")
    print(f"  np.array([{relative_euler[0]:.8f}, {relative_euler[1]:.8f}, {relative_euler[2]:.8f}])")
    print()
    
    # Also provide the constants in the exact format for copying
    print("=" * 60)
    print("COPY-PASTE READY CONSTANTS:")
    print("=" * 60)
    print("Replace the following in scripts/gr00t_script/utils/constants/openarm_leaphand.py:")
    print()
    print(f"CAN_GRASP_POS       = np.array([{relative_pos[0]:.8f}, {relative_pos[1]:.8f}, {relative_pos[2]:.8f}])")
    print(f"CAN_GRASP_QUAT      = np.array([{relative_euler[0]:.3f}, {relative_euler[1]:.3f}, {relative_euler[2]:.3f}])  # degrees, relative to can's orientation")
    print()
    
    # For reference, show the existing approach and leave offsets
    print("Existing approach and leave offsets (unchanged):")
    print("CAN_APPROACH_POS    = CAN_GRASP_POS + np.array([-0.05, -0.00, 0.00])")
    print("CAN_LEAVE_POS       = CAN_GRASP_POS + np.array([ 0.00,  0.00, 0.10])")
    print()
    
    print("=" * 60)
    print("Grasping Calculation Complete!")
    print("=" * 60)
    
    return relative_pos, relative_euler



def calculate_basket_placement():
    """
    Calculate relative poses for can placement in red basket using the pose_difference function.

    Input data:
    - Red Basket Pose: [0.25, -0.05, 0.81], [1.0, 0.0, 0.0, 0.0]
    - Hand Place Pose from YAML (step 1.5): [0.23995677, -0.15493181, 0.94485849], [0.99984890, -0.00008668, 0.00049764, -0.00023014]
    """

    print("=" * 60)
    print("Calculating Relative Poses for Can Placement in Red Basket")
    print("=" * 60)

    # Red basket pose in world frame
    # basket_pos = np.array([0.4, -0.05, 0.81])
    basket_pos = np.array([0.4,  -0.05,   0.81])  # Updated based on actual environment setup
    basket_quat = [1.0, 0.0, 0.0, 0.0]  # [w, x, y, z]

    # Hand place pose from YAML (step 1.5) in world frame
    # right_arm_eef: [0.23995677, -0.15493181, 0.94485849, 0.99984890, -0.00008668, 0.00049764, -0.00023014]
    hand_pos = np.array([0.23995677, -0.15493181, 0.94485849])
    hand_quat = [0.99984890, -0.00008668, 0.00049764, -0.00023014]  # [w, x, y, z]

    print(f"Red Basket Position (world): {basket_pos}")
    print(f"Red Basket Quaternion (world, wxyz): {basket_quat}")
    print()
    print(f"Hand Place Position (world): {hand_pos}")
    print(f"Hand Place Quaternion (world, wxyz): {hand_quat}")
    print()

    # Calculate relative poses using pose_difference function
    # This gives hand pose relative to basket frame
    relative_pos, relative_euler = pose_difference(
        hand_pos, hand_quat, basket_pos, basket_quat,
        order='xyz', degrees=True
    )

    print("Calculating relative poses using pose_difference()...")
    print()

    print(f"CAN_PLACE_POS (relative to basket frame):")
    print(f"  np.array([{relative_pos[0]:.8f}, {relative_pos[1]:.8f}, {relative_pos[2]:.8f}])")
    print()

    print(f"CAN_PLACE_QUAT (relative to basket frame, Euler XYZ degrees):")
    print(f"  np.array([{relative_euler[0]:.8f}, {relative_euler[1]:.8f}, {relative_euler[2]:.8f}])")
    print()

    # Also provide the constants in the exact format for copying
    print("=" * 60)
    print("COPY-PASTE READY CONSTANTS:")
    print("=" * 60)
    print("Add the following to scripts/gr00t_script/utils/constants/openarm_leaphand.py:")
    print()
    print(f"CAN_PLACE_POS       = np.array([{relative_pos[0]:.8f}, {relative_pos[1]:.8f}, {relative_pos[2]:.8f}])")
    print(f"CAN_PLACE_QUAT      = np.array([{relative_euler[0]:.3f}, {relative_euler[1]:.3f}, {relative_euler[2]:.3f}])  # degrees, relative to basket's orientation")
    print()

    # For reference, show the approach and leave offsets
    print("Suggested approach and leave offsets:")
    print("CAN_PLACE_APPROACH_POS = CAN_PLACE_POS + np.array([0.00, 0.00, 0.05])  # Approach from above")
    print("CAN_PLACE_LEAVE_POS    = CAN_PLACE_POS + np.array([0.00, 0.00, 0.10])  # Leave upward")
    print()

    print("=" * 60)
    print("Basket Placement Calculation Complete!")
    print("=" * 60)

    return relative_pos, relative_euler


def main():
    calculate_can_grasp()
    
    print("\n\n")
    calculate_basket_placement()

if __name__ == "__main__":
    main()
