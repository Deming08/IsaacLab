
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

def main():
    # Example 1: Euler to Quaternion
    euler_angles = [-100, 0, 0]  # degrees, order 'xyz'
    quat = euler_to_quat(euler_angles)
    print(f"Euler angles: {euler_angles} -> quat(w,x,y,z) = [{quat[0]:.7f}, {quat[1]:.7f}, {quat[2]:.7f}, {quat[3]:.7f}]")

    # Example 2: Quaternion to Euler
    right_arm_eef = [0.30660954, -0.04, 0.98, 0.86796325, -0.48300108, 0.04342201, 0.10707159]
    # Extract quaternion part (assuming [w, x, y, z] is the last 4 elements)
    q = right_arm_eef[3:]
    euler = quat_to_euler(q)
    print(f"quat(w,x,y,z) = [{q[0]:.7f}, {q[1]:.7f}, {q[2]:.7f}, {q[3]:.7f}] -> Euler angles (deg): {euler}")

if __name__ == "__main__":
    main()