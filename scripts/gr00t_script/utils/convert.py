import numpy as np
from scipy.spatial.transform import Rotation as R
from quaternion_utils import quat_wxyz_to_xyzw

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

def pose_difference(hand_pos, hand_quat, object_pos, object_quat, order='xyz', degrees=True):
    # Position offset
    pos_offset = hand_pos - object_pos

    # Quaternion to rotation. Assuming input quaternions are [w, x, y, z]
    hand_r = R.from_quat(quat_wxyz_to_xyzw(hand_quat))
    object_r = R.from_quat(quat_wxyz_to_xyzw(object_quat))
    # Relative rotation: hand in object frame
    rel_r = object_r.inv() * hand_r
    euler_offset = rel_r.as_euler(order, degrees=degrees)
    return pos_offset, euler_offset

def main():
    # Example 1: Euler to Quaternion
    euler_angles = [90, 20, -80]  # degrees, order 'xyz'
    quat = euler_to_quat(euler_angles)
    print(f"1. Euler angles: {euler_angles} -> quat(w,x,y,z) = [{quat[0]:.7f}, {quat[1]:.7f}, {quat[2]:.7f}, {quat[3]:.7f}]")

    # Example 2: Quaternion to Euler
    q = [0.707,  0.,     0.,    -0.707]
    # Extract quaternion part (assuming [w, x, y, z] is the last 4 elements)
    euler = quat_to_euler(q)
    print(f"2. quat(w,x,y,z) = [{q[0]:.7f}, {q[1]:.7f}, {q[2]:.7f}, {q[3]:.7f}]" \
          f" -> Euler angles (deg): [{(', '.join('{:.8f}'.format(x) for x in euler))}]")

    # Example 3: Pose difference
    hand_pos = np.array([0.21, -0.08, 0.9 ])
    hand_quat = [0.99719107, -0.07012144, -0.00358765, 0.02607828]  # [w, x, y, z]
    object_pos = np.array([0.4,  0.1,  0.805])
    object_quat = [0.00000004, -0.00000133, -0.00000094, 1.        ]  # [w, x, y, z]

    pos_offset, euler_offset = pose_difference(hand_pos, hand_quat, object_pos, object_quat)
    print(f"3. Position offset: [{(', '.join('{:.8f}'.format(x) for x in pos_offset))}]", \
          f"; Euler offset (deg):[{(', '.join('{:.8f}'.format(x) for x in euler_offset))}]")

    # # 1. Convert to Rotation objects (which use xyzw format)
    # R_desired_hand = R.from_quat(quat_wxyz_to_xyzw(hand_quat))
    # R_mug = R.from_quat(quat_wxyz_to_xyzw(object_quat))
    
    # # 2. Isolate the mug's yaw rotation
    # mug_yaw_degrees = R_mug.as_euler('zyx', degrees=True)[0]
    # R_mug_yaw = R.from_euler('z', mug_yaw_degrees, degrees=True)

    # # 3. Calculate the required offset rotation
    # R_grasp_offset = R_mug_yaw.inv() * R_desired_hand

    # # 4. Convert the offset to Euler angles (xyz order)
    # MUG_GRASP_QUAT_euler = R_grasp_offset.as_euler('xyz', degrees=True)

    # print(f"Set MUG_GRASP_QUAT in constants.py to: {MUG_GRASP_QUAT_euler}")


if __name__ == "__main__":
    main()
  

# data_collect:
# [INFO] Drawer Position: [ 0.26349986 -0.14999999  0.7271999 ], Quat: [-1.0000001 -0.         0.         0.       ]
# [INFO] Mug Position: [0.2183024  0.09513299 0.67665535], Quat: [ 0.92388  0.       0.      -0.38268]
# [INFO] Mug Mat Position: [0.4   0.1   0.805], Quat: [1. 0. 0. 0.]
# [INFO] Bottle Position: [ 0.37549874 -0.24149555  0.9       ], Quat: [0. 0. 0. 1.]

# teleoperation:
#     Bottle Pose: [0.39220732 0.01346831 0.93462497], [-0.06396611  0.01772782 -0.00482274  0.9977829 ]
#     Mug Pose: [0.39995837 0.10002603 0.80718344], [ 0.9238198  -0.00037755 -0.01167601 -0.3826494 ]
#     Mug Mat Pose: [0.4   0.1   0.805], [1. 0. 0. 0.]