import numpy as np
import matplotlib.pyplot as plt
import os

# Define data file location
data_loc = os.path.expanduser('~/catkin_ws/src/ho_intervention_project/data')

# Load data
joint_position_data = np.load(os.path.join(data_loc, 'joint_position_data.npy'))
ee_position_data = np.load(os.path.join(data_loc, 'ee_position_data.npy'))
base_position_data = np.load(os.path.join(data_loc, 'base_position_data.npy'))
joint_vel_data = np.load(os.path.join(data_loc, 'joint_vel_data.npy'))

# Plot the joint positions
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.set_title('Joint Positions Evolution')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Angle [rad]')
# ax1.set_aspect('equal')
ax1.grid()

ax1.plot(joint_position_data[-1],joint_position_data[0], label= "Joint 1")
ax1.plot(joint_position_data[-1],joint_position_data[1], label= "Joint 2")
ax1.plot(joint_position_data[-1],joint_position_data[2], label= "Joint 3")
ax1.plot(joint_position_data[-1],joint_position_data[3], label= "Joint 4")

ax1.legend()

# Plot the base position and EE position on the same figure
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.set_title('Position Evolution')
ax2.set_xlabel('y [m]')
ax2.set_ylabel('x [m]')
ax2.set_aspect('equal')
ax2.grid()

ax2.plot(base_position_data[1],base_position_data[0], label= "Base Position")
ax2.plot(ee_position_data[1],ee_position_data[0], label="End-effector Position")

ax2.legend()

# Plot the joint velocities
fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
ax3.set_title('Quasi-Velocities Evolution')
ax3.set_xlabel('Time [s]')
ax3.set_ylabel('Velocity [rad/s or m/s]')
# ax1.set_aspect('equal')
ax3.grid()

for i in range(len(joint_vel_data)-1):
    ax3.plot(joint_vel_data[-1],joint_vel_data[i])

labels3 = ["Mobile base - angular",
           "Mobile base - linear",
           "Joint 1",
           "Joint 2",
           "Joint 3",
           "Joint 4"]

ax3.legend(labels3)

plt.show()