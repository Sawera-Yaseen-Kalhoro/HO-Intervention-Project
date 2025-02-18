import numpy as np
# import pickle
import os
import matplotlib.pyplot as plt

# Define data file location
data_loc = os.path.expanduser('~/catkin_ws/src/ho_intervention_project/data')

# Define subtrees
move_to_object = ['Rest Position start_error.npy',
                  'Move robot to pick up position_error.npy',
                  'Move manipulator to position_error.npy']

pick_up_object = ['Move EE Down pos1_error.npy',
                  'Move EE High waypoint 1_error.npy',
                  'Move EE to top of robot_error.npy']

move_to_dropoff = ['Move robot to drop position_error.npy',
                   'Move EE to front_error.npy',
                   'Move EE High waypoint 2_error.npy']

drop_object = ['Move EE Down pos2_error.npy',
               'Rest Position final_error.npy']

### Plot the error evol data

# Sub-tree 1: Move to Object
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.set_title('Error Evolution - Move to Object Sequence')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Error')
ax1.grid()

separate_time = []

for data in move_to_object:
    error_evol_data = np.load(os.path.join(data_loc, data))

    for i in range(len(error_evol_data)-1):
        ax1.plot(error_evol_data[-1],error_evol_data[i])
    
    if not (data == move_to_object[-1]):
        separate_time.append(error_evol_data[-1][-1])

# give a vertical line to separate behaviors
for x in separate_time:
    ax1.axvline(x = x, color = 'r', linestyle = '--')

# For "move to object" sequence
labels1 = ["Rest Position Start - Joint 1 Position",
          "Rest Position Start - Joint 2 Position", 
          "Rest Position Start - Joint 3 Position", 
          "Rest Position Start - Joint 4 Position",
          "Move Robot to Pickup Position - Base Orientation",
          "Move Robot to Pickup Position - Base Position",
          "Move Manipulator to Pickup Position - EE Position"]

ax1.legend(labels1)

##########################################

# Sub-tree 2: Pick Up Object
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.set_title('Error Evolution - Pick Up Object Sequence')
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('Error')
ax2.grid()

separate_time = []

for data in pick_up_object:
    error_evol_data = np.load(os.path.join(data_loc, data))

    for i in range(len(error_evol_data)-1):
        ax2.plot(error_evol_data[-1],error_evol_data[i])
    
    if not (data == pick_up_object[-1]):
        separate_time.append(error_evol_data[-1][-1])

# give a vertical line to separate behaviors
for x in separate_time:
    ax2.axvline(x = x, color = 'r', linestyle = '--')

# For "move to object" sequence
labels2 = ["Move EE Down 1 - EE Position",
          "Move EE High 1 - EE Position", 
          "Move EE to Top - Joint 1 Position"]

ax2.legend(labels2)

##########################################

# Sub-tree 3: Move to Drop-Off
fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
ax3.set_title('Error Evolution - Move to Drop-Off Sequence')
ax3.set_xlabel('Time [s]')
ax3.set_ylabel('Error')
ax3.grid()

separate_time = []

for data in move_to_dropoff:
    error_evol_data = np.load(os.path.join(data_loc, data))

    for i in range(len(error_evol_data)-1):
        ax3.plot(error_evol_data[-1],error_evol_data[i])
    
    if not (data == move_to_dropoff[-1]):
        separate_time.append(error_evol_data[-1][-1])

# give a vertical line to separate behaviors
for x in separate_time:
    ax3.axvline(x = x, color = 'r', linestyle = '--')

# For "move to object" sequence
labels3 = ["Move Robot to Drop-Off Position - Base Orientation",
          "Move Robot to Drop-Off Position - Base Position",
          "Move EE to Front - Joint 1 Position",
          "Move EE High 2 - EE Position"]

ax3.legend(labels3)

##########################################

# Sub-tree 4: Drop Object
fig4 = plt.figure()
ax4 = fig4.add_subplot(111)
ax4.set_title('Error Evolution - Drop Object Sequence')
ax4.set_xlabel('Time [s]')
ax4.set_ylabel('Error')
ax4.grid()

separate_time = []

for data in drop_object:
    error_evol_data = np.load(os.path.join(data_loc, data))

    for i in range(len(error_evol_data)-1):
        ax4.plot(error_evol_data[-1],error_evol_data[i])
    
    if not (data == drop_object[-1]):
        separate_time.append(error_evol_data[-1][-1])

# give a vertical line to separate behaviors
for x in separate_time:
    ax4.axvline(x = x, color = 'r', linestyle = '--')

# For "move to object" sequence
labels4 = ["Move EE Down 2 - EE Position",
           "Rest Position Final - Joint 1 Position",
          "Rest Position Final - Joint 2 Position", 
          "Rest Position Final - Joint 3 Position", 
          "Rest Position Final - Joint 4 Position"]

ax4.legend(labels4)

plt.show()