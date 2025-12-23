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

def main():
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
    can_pos = np.array([0.25272524, -0.08587377, 0.89195764])
    can_quat = [1.0, 0.00001174, 0.00001437, 0.00016935]  # [w, x, y, z]
    
    # Hand grasp pose from YAML (step 1.2) in world frame
    # right_arm_eef: [0.11500076, -0.20500796, 0.92999208, 0.99984902, 0.00004765, -0.00002402, -0.00003580]
    hand_pos = np.array([0.11500076, -0.20500796, 0.92999208])
    hand_quat = [0.99984902, 0.00004765, -0.00002402, -0.00003580]  # [w, x, y, z]
    
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
    print("Calculation Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
