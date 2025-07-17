import numpy as np
from scipy.spatial.transform import Rotation as R

def euler_to_quat(euler_angles, order='xyz', degrees=True):
    """Convert Euler angles to quaternion [w, x, y, z]"""
    r = R.from_euler(order, euler_angles, degrees=degrees)
    qx, qy, qz, qw = r.as_quat()  # scipy returns [x, y, z, w]
    return [qw, qx, qy, qz]

def quat_to_euler(q, order='xyz', degrees=True):
    """Convert quaternion [w, x, y, z] to Euler angles"""
    qw, qx, qy, qz = q
    r = R.from_quat([qx, qy, qz, qw])  # scipy expects [x, y, z, w]
    return r.as_euler(order, degrees=degrees)

def pose_difference(hand_pos, hand_quat, drawer_pos, drawer_quat, order='xyz', degrees=True):
    # Position offset
    pos_offset = hand_pos - drawer_pos

    # Quaternion to rotation
    # Assuming input quaternions are [w, x, y, z]
    hand_r = R.from_quat([hand_quat[1], hand_quat[2], hand_quat[3], hand_quat[0]])
    drawer_r = R.from_quat([drawer_quat[1], drawer_quat[2], drawer_quat[3], drawer_quat[0]])
    # Relative rotation: hand in drawer frame
    rel_r = hand_r * drawer_r.inv()
    euler_offset = rel_r.as_euler(order, degrees=degrees)
    return pos_offset, euler_offset

def main():
    # Example 1: Euler to Quaternion
    euler_angles = [180, 0, 0]  # degrees, order 'xyz'
    quat = euler_to_quat(euler_angles)
    print(f"3. Euler angles: {euler_angles} -> quat(w,x,y,z) = [{quat[0]:.7f}, {quat[1]:.7f}, {quat[2]:.7f}, {quat[3]:.7f}]")

    # Example 2: Quaternion to Euler
    q = [-0.0000183,  -0.00000098, -0.00000065,  1.        ]
    # Extract quaternion part (assuming [w, x, y, z] is the last 4 elements)
    euler = quat_to_euler(q)
    print(f"2. quat(w,x,y,z) = [{q[0]:.7f}, {q[1]:.7f}, {q[2]:.7f}, {q[3]:.7f}] -> Euler angles (deg): {euler}")

    # Example 3: Pose difference
    hand_pos = np.array([0.18171435,  0.20596554, 0.88])
    hand_quat = [0.4849 ,  0.6280 , -0.3420, -0.5130]  # [w, x, y, z]
    object_pos = np.array([0.39819676, 0.08400775, 0.805])
    object_quat = [0.92615926, -0.00062712, -0.0118468, -0.37694612]  # [w, x, y, z]

    pos_offset, euler_offset = pose_difference(hand_pos, hand_quat, object_pos, object_quat)
    print("3. Position offset:", pos_offset, "Euler offset (deg):", euler_offset)

if __name__ == "__main__":
    main()