import numpy as np
from numpy import sin,cos
from math import atan2

# Robot properties
a1 = 108 
a2 = 142
a3 = 158.8
r = 0.0507 # offset from mobile base to manipulator base

# Function to wrap angle to [-pi,pi]
def wrap_angle(angle):
    return (angle + ( 2.0 * np.pi * np.floor( ( np.pi - angle ) / ( 2.0 * np.pi ) ) ) )

# Transformation from mobile base footprint to manipulator base
deg90 = np.pi/2
mobile_to_manip = np.array([[cos(-deg90), -sin(-deg90), 0, r],
                            [sin(-deg90), cos(-deg90), 0, 0],
                            [0,0,1,-0.198],
                            [0,0,0,1]])

# Function to compute transformation matrix from a given pose
def transformMatrix(x,y,z,yaw):
    T = np.array([[cos(yaw), -sin(yaw), 0, x],
                  [sin(yaw), cos(yaw), 0, y],
                  [0,0,1,z],
                  [0,0,0,1]])
    return T

# Kinematics of EE w.r.t. manipulator base
def kinematics_manipulator(q):
    # Extract joint angles
    theta1 = q[0]
    theta2 = q[1]
    theta3 = q[2]
    theta4 = q[3]

    # Compute EE pose w.r.t manipulator base
    x_m = cos(theta1)*(13.2 - a2*sin(theta2) + a3*cos(theta3) +56.5) / 1000 # working with [m]
    y_m = sin(theta1)*(13.2 - a2*sin(theta2) + a3*cos(theta3) +56.5) / 1000 # working with [m]
    z_m = -(a1 + a2*cos(theta2) + a3 *sin(theta3) -72.2) / 1000
    yaw_m = wrap_angle(theta1 + theta4)

    return x_m, y_m, z_m, yaw_m

# Kinematics of EE w.r.t. world
def kinematics(q, base_pos, base_ori):
    '''
        Function to compute the end-effector position and orientation through kinematics.

        Arguments:
        q (Numpy array): position of the manipulator joints
        base_pos (Numpy array): position (x and y) of the mobile base
        base_ori (float): orientation (yaw) of the mobile base

        Returns:
        pos (Numpy array): end-effector position
        ori (Numpy array): end-effector orientation (yaw)
    '''

    # Make sure the arguments passed are 1-dimensional
    assert q.shape == (4,)
    assert base_pos.shape == (2,)

    # Compute EE pose w.r.t manipulator base
    x_m, y_m, z_m, yaw_m = kinematics_manipulator(q)

    # convert to transformation matrix
    manip_to_ee = transformMatrix(x_m, y_m, z_m, yaw_m)

    # compute transformation from world to mobile base (from odom)
    world_to_mobile = transformMatrix(base_pos[0],base_pos[1],0,base_ori)

    # compute transformation from world to EE (by matrix multiplication)
    world_to_ee = world_to_mobile @ mobile_to_manip @ manip_to_ee
    
    # Extract EE pose
    x = world_to_ee[0,-1]
    y = world_to_ee[1,-1]
    z = world_to_ee[2,-1]
    yaw = atan2(world_to_ee[1,0],world_to_ee[0,0])

    # Convert to Numpy arrays
    pos = np.array([x,y,z]).reshape((3,1))
    ori = np.array([[wrap_angle(yaw)]])

    return pos, ori

# Jacobian of the manipulator (without the mobile base)
def jacobian_manipulator(q):
    '''
        Function to compute the Jacobian of the Swiftpro uArm manipulator.

        Arguments:
        q (Numpy array): position of the manipulator joints

        Returns:
        (Numpy array): Jacobian of the end-effector
    '''

    # Make sure the argument passed is 1-dimensional
    assert q.shape == (4,)

    # Extract joint angles
    theta1 = q[0]
    theta2 = q[1]
    theta3 = q[2]
    theta4 = q[3]

    L1 = 13.2 - a2*sin(theta2) + a3*cos(theta3) + 56.5

    # Construct the Jacobian
    J = np.zeros((6,4))

    J[0] = [-sin(theta1)*L1/1000, -a2*cos(theta1)*cos(theta2)/1000, -a3*cos(theta1)*sin(theta3)/1000, 0]
    J[1] = [cos(theta1)*L1/1000, -a2*sin(theta1)*cos(theta2)/1000, -a3*sin(theta1)*sin(theta3)/1000, 0]
    J[2] = [0, a2*sin(theta2)/1000, -a3*cos(theta3)/1000, 0]

    J[-1] = [1,0,0,1]

    return J

# Full Jacobian of the mobile manipulator
def jacobian(q,base_pos,base_ori):
    '''
        Function to compute the Jacobian of the mobile manipulator (mobile base + manipulator)

        Arguments:
        q (Numpy array): position of the manipulator joints
        base_pos (Numpy array): position (x and y) of the mobile base
        base_ori (float): orientation (yaw) of the mobile base

        Returns:
        (Numpy array): Jacobian of the mobile base
    '''

    # Make sure the arguments passed are 1-dimensional
    assert q.shape == (4,)
    assert base_pos.shape == (2,)

    # Extract joint angles
    theta1 = q[0]
    theta2 = q[1]
    theta3 = q[2]
    theta4 = q[3]

    # Extract mobile base pos
    xm = base_pos[0]
    ym = base_pos[1]

    # EE position w.r.t. manipulator base
    x_m, y_m, _, _ = kinematics_manipulator(q)

    # Construct Jacobian
    J = np.zeros((6,6))

    Jm = jacobian_manipulator(q) # Jacobian of the EE w.r.t. manipulator base

    J[0,0] = -x_m * sin(base_ori-np.pi/2) - (y_m + r) * cos(base_ori-np.pi/2)
    J[0,1] = -sin(base_ori-np.pi/2)

    J[1,0] = x_m * cos(base_ori-np.pi/2) - (y_m + r) * sin(base_ori-np.pi/2)
    J[1,1] = cos(base_ori-np.pi/2)

    J[-1,0] = 1.0

    for n in range(2,6):
        J[0,n] = Jm[0,n-2] * cos(base_ori-np.pi/2) + Jm[1,n-2] * sin(base_ori-np.pi/2)
        J[1,n] = Jm[0,n-2] * sin(base_ori-np.pi/2) + Jm[1,n-2] * cos(base_ori-np.pi/2)
        J[2,n] = Jm[2,n-2]
    
    J[-1,2:6] = Jm[-1]

    return J

# Damped Least-Squares
def DLS(A, damping):
    '''
        Function computes the damped least-squares (DLS) solution to the matrix inverse problem.

        Arguments:
        A (Numpy array): matrix to be inverted
        damping (double): damping factor

        Returns:
        (Numpy array): inversion of the input matrix
    '''
    return A.T @ np.linalg.inv(A@A.T + damping**2*np.eye(A.shape[0]))

# Weighted DLS
def WeightedDLS(A, damping, W):
    '''
        Function computes the weighted damped least-squares (DLS) solution to the matrix inverse problem.

        Arguments:
        A (Numpy array): matrix to be inverted
        damping (double): damping factor
        W (Numpy array): weighting matrix

        Returns:
        (Numpy array): inversion of the input matrix
    '''
    
    return np.linalg.inv(W) @ A.T @ np.linalg.inv(A@np.linalg.inv(W)@A.T + damping**2*np.eye(A.shape[0]))