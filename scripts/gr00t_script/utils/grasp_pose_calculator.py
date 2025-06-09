import numpy as np
from scipy.spatial.transform import Rotation as ScipyRotation
from typing import Optional

from .quaternion_utils import quat_wxyz_to_xyzw, quat_xyzw_to_wxyz

class GraspPoseCalculator:
    """
    Calculates the target end-effector (EE) pose for grasping an object,
    given an example of a successful grasp (cube pose and EE pose).
    Assumes poses are in the world frame and quaternions are in (w, x, y, z) format.
    """

    # Define default values at the class level or directly in the __init__ signature
    #  DEFAULT_CUBE_POS = np.array([0.20, -0.215, 0.87])
    # DEFAULT_CUBE_QUAT_WXYZ = np.array([1.0, 0.0, 0.0, 0.0])
    # DEFAULT_EE_POS = np.array([0.07789905, -0.17129795,  0.8736194])
    # DEFAULT_EE_QUAT_WXYZ = np.array([0.94551858, 0.00000000, 0.00000000, -0.32556815])
    DEFAULT_CUBE_POS = np.array([0.20, -0.10, 0.87])
    DEFAULT_CUBE_QUAT_WXYZ = np.array([1.0, 0.0, 0.0, 0.0])
    DEFAULT_EE_POS = np.array([0.07789905, -0.17129795,  0.8736194])
    DEFAULT_EE_QUAT_WXYZ = np.array([0.99619470, 0.00000000, 0.00000000, 0.08715574])

    def __init__(self, example_cube_pos_w: Optional[np.ndarray] = None,
                 example_cube_quat_wxyz_w: Optional[np.ndarray] = None,
                 example_ee_pos_w: Optional[np.ndarray] = None,
                 example_ee_quat_wxyz_w: Optional[np.ndarray] = None):
        """
        Initializes the calculator with an example grasp.

        Args:
            example_cube_pos_w and example_cube_quat_wxyz_w: Position (x, y, z) and orientation (w, x, y, z) of the cube in the world frame for the example grasp.
            exaple_ee_pos_w and example_ee_quat_wxyz_w: Position (x, y, z) and orientation (w, x, y, z) of the end-effector (EE) in the world frame for the example grasp.
        """
        if example_cube_pos_w is None: example_cube_pos_w = GraspPoseCalculator.DEFAULT_CUBE_POS
        self.example_cube_pos_w = np.asarray(example_cube_pos_w)

        if example_cube_quat_wxyz_w is None: example_cube_quat_wxyz_w = GraspPoseCalculator.DEFAULT_CUBE_QUAT_WXYZ
        self.example_cube_quat_wxyz_w = np.asarray(example_cube_quat_wxyz_w)

        if example_ee_pos_w is None: example_ee_pos_w = GraspPoseCalculator.DEFAULT_EE_POS
        self.example_ee_pos_w = np.asarray(example_ee_pos_w)

        if example_ee_quat_wxyz_w is None: example_ee_quat_wxyz_w = GraspPoseCalculator.DEFAULT_EE_QUAT_WXYZ
        self.example_ee_quat_wxyz_w = np.asarray(example_ee_quat_wxyz_w)

        self._compute_relative_grasp_transform()

    def _compute_relative_grasp_transform(self):
        """
        Computes the transformation from the cube's frame to the EE's frame at the moment of grasp.
        """
        # Convert example world poses to scipy Rotation objects
        # R_w_cube_ex: Transforms vectors from example cube frame to world frame
        R_w_cube_ex = ScipyRotation.from_quat(quat_wxyz_to_xyzw(self.example_cube_quat_wxyz_w))
        # R_w_ee_ex: Transforms vectors from example EE frame to world frame
        R_w_ee_ex = ScipyRotation.from_quat(quat_wxyz_to_xyzw(self.example_ee_quat_wxyz_w))

        # Inverse of R_w_cube_ex: Transforms vectors from world frame to example cube frame
        R_cube_w_ex = R_w_cube_ex.inv()

        # Position of the EE origin relative to the cube origin, expressed in the cube's frame
        # t_ee_in_cube_frame = R_cube_w_ex * (world_ee_pos - world_cube_pos)
        self.t_ee_in_cube_frame = R_cube_w_ex.apply(self.example_ee_pos_w - self.example_cube_pos_w)

        # Orientation of the EE frame relative to the cube frame
        # R_ee_in_cube_frame = R_cube_w_ex * R_w_ee_ex
        self.R_ee_in_cube_frame = R_cube_w_ex * R_w_ee_ex

    def calculate_target_ee_pose(self, new_cube_pos_w: np.ndarray, new_cube_quat_wxyz_w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculates the target EE pose in the world frame for grasping a cube at a new pose.

        Returns:
            A tuple (target_ee_pos_w, target_ee_quat_wxyz_w):
        """
        new_cube_pos_w = np.asarray(new_cube_pos_w)
        new_cube_quat_wxyz_w = np.asarray(new_cube_quat_wxyz_w)

        # R_w_cube_new: Transforms vectors from the new cube's frame to the world frame
        R_w_cube_new = ScipyRotation.from_quat(quat_wxyz_to_xyzw(new_cube_quat_wxyz_w))

        # Calculate target EE position in the world frame:
        # target_pos = new_cube_pos_w + R_w_cube_new * t_ee_in_cube_frame
        target_ee_pos_w = new_cube_pos_w + R_w_cube_new.apply(self.t_ee_in_cube_frame)

        # Calculate target EE orientation in the world frame:
        # R_w_ee_new = R_w_cube_new * R_ee_in_cube_frame
        R_w_ee_new = R_w_cube_new * self.R_ee_in_cube_frame
        
        target_ee_quat_xyzw_w = R_w_ee_new.as_quat()
        target_ee_quat_wxyz_w = quat_xyzw_to_wxyz(target_ee_quat_xyzw_w)

        return target_ee_pos_w, target_ee_quat_wxyz_w

# --- Example Check ---
if __name__ == "__main__":
    # Initialize the calculator using default grasp values
    grasp_calculator = GraspPoseCalculator()
    print("--- Initialized GraspPoseCalculator with default values ---")
    print(f"Default Cube Pos: {grasp_calculator.example_cube_pos_w}")
    print(f"Default EE Pos: {grasp_calculator.example_ee_pos_w}\n")

    # 1. Test with the original cube pose (should return the original EE pose)
    print("--- Test with original cube pose ---")
    target_pos, target_quat = grasp_calculator.calculate_target_ee_pose(
        grasp_calculator.DEFAULT_CUBE_POS, grasp_calculator.DEFAULT_CUBE_QUAT_WXYZ
    )
    print(f"Calculated EE Pos: {target_pos}, Original EE Pos: {grasp_calculator.DEFAULT_EE_POS}")
    print(f"Calculated EE Quat (wxyz): {target_quat}, Original EE Quat (wxyz): {grasp_calculator.DEFAULT_EE_QUAT_WXYZ}")
    assert np.allclose(target_pos, grasp_calculator.DEFAULT_EE_POS), "Position mismatch for original pose test"
    assert np.allclose(target_quat, grasp_calculator.DEFAULT_EE_QUAT_WXYZ), "Orientation mismatch for original pose test"
    print("Original pose test PASSED.\n")

    # 2. An example: Define a new arbitrary pose for the cube
    print("--- Test with a new cube pose ---")
    new_cube_pos = np.array([0.25, 0.5, 1.0])
    # Example: new cube rotated 45 degrees around its Z-axis
    rotation_z_45_deg = ScipyRotation.from_euler('z', 45, degrees=True)
    new_cube_quat_xyzw = rotation_z_45_deg.as_quat()
    new_cube_quat_wxyz = quat_xyzw_to_wxyz(new_cube_quat_xyzw) # Helper to convert

    print(f"New Cube Pos: {new_cube_pos}")
    print(f"New Cube Quat (wxyz): {new_cube_quat_wxyz}")

    # Calculate the target EE pose for this new cube pose
    target_ee_pos_new, target_ee_quat_wxyz_new = grasp_calculator.calculate_target_ee_pose(
        new_cube_pos_w=new_cube_pos,
        new_cube_quat_wxyz_w=new_cube_quat_wxyz
    )

    print(f"\nCalculated Target EE Pos for new cube: {target_ee_pos_new}")
    print(f"Calculated Target EE Quat (wxyz) for new cube: {target_ee_quat_wxyz_new}")
