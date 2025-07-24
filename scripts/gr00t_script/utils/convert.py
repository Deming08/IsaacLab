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
    euler_angles = [-50, -10, -18]  # degrees, order 'xyz'
    quat = euler_to_quat(euler_angles)
    print(f"1. Euler angles: {euler_angles} -> quat(w,x,y,z) = [{quat[0]:.7f}, {quat[1]:.7f}, {quat[2]:.7f}, {quat[3]:.7f}]")

    # Example 2: Quaternion to Euler
    q = [0.8859813, -0.4281835, -0.0121569, -0.1776184]
    # Extract quaternion part (assuming [w, x, y, z] is the last 4 elements)
    euler = quat_to_euler(q)
    print(f"2. quat(w,x,y,z) = [{q[0]:.7f}, {q[1]:.7f}, {q[2]:.7f}, {q[3]:.7f}]" \
          f" -> Euler angles (deg): [{(', '.join('{:.8f}'.format(x) for x in euler))}]")

    # Example 3: Pose difference
    hand_pos = np.array([0.29, -0.01, 0.93])
    hand_quat = [0.9848, 0.0, 0.0, -0.17363]  # [w, x, y, z]
    object_pos = np.array([0.4, 0.1, 0.80718344])
    object_quat = [0.9238198,  -0.00037755, -0.01167601, -0.3826494]  # [w, x, y, z]

    pos_offset, euler_offset = pose_difference(hand_pos, hand_quat, object_pos, object_quat)
    print(f"3. Position offset: [{(', '.join('{:.8f}'.format(x) for x in pos_offset))}]", \
          f"; Euler offset (deg):[{(', '.join('{:.8f}'.format(x) for x in euler_offset))}]")

if __name__ == "__main__":
    main()
  

# data_collect:
# [INFO] Mug Position: [0.40936592 0.08435203 0.85      ], Quat: [ 0.92388  0.       0.      -0.38268]
# [INFO] Mug Mat Position: [0.3947057  0.09577163 0.805     ], Quat: [1. 0. 0. 0.]
# [INFO] Bottle Position: [ 0.382561   -0.23695832  0.9       ], Quat: [0. 0. 0. 1.]

# teleoperation:
#     Bottle Pose: [0.39220732 0.01346831 0.93462497], [-0.06396611  0.01772782 -0.00482274  0.9977829 ]
#     Mug Pose: [0.39995837 0.10002603 0.80718344], [ 0.9238198  -0.00037755 -0.01167601 -0.3826494 ]
#     Mug Mat Pose: [0.4   0.1   0.805], [1. 0. 0. 0.]