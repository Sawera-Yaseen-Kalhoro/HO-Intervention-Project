#!/usr/bin/env python3

import rospy
import numpy as np
import tf

from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
from std_srvs.srv import Trigger, TriggerResponse

from ho_intervention_project.msg import TaskError
from ho_intervention_project.srv import TaskTrigger, TaskTriggerResponse

from utils.tasks import *
from utils.functions import DLS, WeightedDLS

import signal
import sys
import os

class TP_Controller:
    def __init__(self) -> None:
        # Initialize attributes
        self.q = np.zeros((4,1)) # manipulator joint positions
        self.base_pos = np.zeros((2,1)) # mobile base position
        self.base_orientation = 0 # mobile base orientation (yaw)

        # Initialize attribute storing EE goal position or configuration (for visualization in Rviz)
        self.ee_goal = np.zeros((4,1))

        # Initialize attribute storing mobile base goal position (for visualization in Rviz)
        self.base_goal = np.zeros((2,1))

        # Velocity limits
        self.dq_max = np.array([0.5,0.1,0.2,0.2,0.2,0.2]).reshape((6,1))

        # Starting time
        self.start_time = rospy.Time.now()

        # Lists to store data for plotting
        self.joint_position_data = [[],[],[],[],[]] # a list for each joint + 1 list of timestamps
        self.ee_position_data = [[],[]] # x and y coordinates
        self.base_position_data = [[],[]] # x and y coordinates
        self.joint_vel_data = [[],[],[],[],[],[],[]] # a list for each quasivelocity + 1 list of timestamps

        # Task hierarchy
        self.tasks = [JointLimits("Joint 1 limits", np.array([-np.pi/2,np.pi/2]),1),
                      JointLimits("Joint 2 limits", np.array([-np.pi/2,0.05]),2),
                      JointLimits("Joint 3 limits", np.array([-np.pi/2,0.05]),3),
                      JointLimits("Joint 4 limits", np.array([-np.pi/2,np.pi/2]),4),
                      EEPosition("End-effector position", np.zeros((3,1))),
                      BaseOrientation("Base orientation", np.array([[0.0]])),
                      BasePosition("Base position", np.zeros((2,1))),
                      JointPosition("Joint 1 position",np.array([[0.0]]),1),
                      JointPosition("Joint 2 position",np.array([[0.0]]),2),
                      JointPosition("Joint 3 position",np.array([[0.0]]),3),
                      JointPosition("Joint 4 position",np.array([[0.0]]),4)]
        
        # TF broadcaster
        self.broadcaster = tf.TransformBroadcaster()
        
        # Publishers
        self.error_pub = rospy.Publisher('~control_error',TaskError,queue_size=1)
        self.joint_vel_pub = rospy.Publisher('/turtlebot/swiftpro/joint_velocity_controller/command',Float64MultiArray,queue_size=1)
        self.base_vel_pub = rospy.Publisher('/cmd_vel',Twist,queue_size=1)

        # Subscribers
        self.joint_states_sub = rospy.Subscriber('/turtlebot/joint_states',JointState, self.joint_states_cb)
        self.odom_sub = rospy.Subscriber('/odom',Odometry, self.odom_cb)

        # Services
        self.ee_position_srv = rospy.Service('~ee_position', TaskTrigger, self.ee_position_handle)
        self.base_orientation_srv = rospy.Service('~base_orientation', TaskTrigger, self.base_orientation_handle)
        self.base_position_srv = rospy.Service('~base_position', TaskTrigger, self.base_position_handle)
        self.joint1_position_srv = rospy.Service('~joint1_position', TaskTrigger, self.joint1_position_handle)
        self.joint2_position_srv = rospy.Service('~joint2_position', TaskTrigger, self.joint2_position_handle)
        self.joint3_position_srv = rospy.Service('~joint3_position', TaskTrigger, self.joint3_position_handle)
        self.joint4_position_srv = rospy.Service('~joint4_position', TaskTrigger, self.joint4_position_handle)

        self.stop_srv = rospy.Service('~stop_all', Trigger, self.stop_handle)

        # Timer
        self.timer = rospy.Timer(rospy.Duration(0.1), self.compute_commands)

        # Register signal handler for SIGINT (Ctrl+C) to save data for plotting
        signal.signal(signal.SIGINT, self.signal_handler)

    # Subscriber callbacks
    def joint_states_cb(self,joints_msg):
        # save the current joint states
        if joints_msg.name == ['turtlebot/swiftpro/joint1', 'turtlebot/swiftpro/joint2', 'turtlebot/swiftpro/joint3', 'turtlebot/swiftpro/joint4']:
            self.q = np.array(joints_msg.position).reshape((4,1))

            # store joint position data
            self.joint_position_data[0].append(self.q[0,0])
            self.joint_position_data[1].append(self.q[1,0])
            self.joint_position_data[2].append(self.q[2,0])
            self.joint_position_data[3].append(self.q[3,0])
            self.joint_position_data[4].append((rospy.Time.now() - self.start_time).to_sec())

            # compute EE pose
            ee_pos, ee_ori = kinematics(self.q.flatten(),self.base_pos.flatten(),self.base_orientation)

            # publish tf
            x = ee_pos[0,0]
            y = ee_pos[1,0]
            z = ee_pos[2,0]
            yaw = ee_ori[0,0]
            
            self.broadcaster.sendTransform((x, y, z),
                        tf.transformations.quaternion_from_euler(0, 0, yaw),
                        rospy.Time.now(),
                        "end-effector_frame", # Child frame
                        "world_ned")  # Parent frame
            
            # store EE position data
            self.ee_position_data[0].append(x)
            self.ee_position_data[1].append(y)
    
    def odom_cb(self,odom_msg):
        # save the pose of mobile base
        self.base_pos = np.array([[odom_msg.pose.pose.position.x], 
                                  [odom_msg.pose.pose.position.y]])
        _,_,self.base_orientation = tf.transformations.euler_from_quaternion([odom_msg.pose.pose.orientation.x,
                                                                        odom_msg.pose.pose.orientation.y,
                                                                        odom_msg.pose.pose.orientation.z,
                                                                        odom_msg.pose.pose.orientation.w])
        
        # store base position data
        self.base_position_data[0].append(self.base_pos[0,0])
        self.base_position_data[1].append(self.base_pos[1,0])

    # Service handles
    def ee_position_handle(self,ee_position_req):
        # find the instance in self.tasks
        for i in range(len(self.tasks)):
            if isinstance(self.tasks[i],EEPosition):
                task_index = i
                break

        # enable/disable the task based on the request
        self.tasks[task_index].setStatus(ee_position_req.enable)

        # also set the desired state if it is enabled
        if ee_position_req.enable:
            desired = np.array(ee_position_req.desired).reshape((3,1))
            self.tasks[task_index].setDesired(desired)
            
            # store the goal EE position (for visualization in Rviz)
            self.ee_goal[:3] = desired
        else:
            self.ee_goal[:3,0] = [None,None,None]

        # send success report to service client
        return TaskTriggerResponse(True)

    def base_orientation_handle(self,base_orientation_req):
        # find the instance in self.tasks
        for i in range(len(self.tasks)):
            if isinstance(self.tasks[i],BaseOrientation):
                task_index = i
                break

        # enable/disable the task based on the request
        self.tasks[task_index].setStatus(base_orientation_req.enable)

        # also set the desired state if it is enabled
        if base_orientation_req.enable:
            desired = np.array(base_orientation_req.desired).reshape((1,1))
            self.tasks[task_index].setDesired(desired)

        # send success report to service client
        return TaskTriggerResponse(True)
    
    def base_position_handle(self,base_position_req):
        # find the instance in self.tasks
        for i in range(len(self.tasks)):
            if isinstance(self.tasks[i],BasePosition):
                task_index = i
                break

        # enable/disable the task based on the request
        self.tasks[task_index].setStatus(base_position_req.enable)

        # also set the desired state if it is enabled
        if base_position_req.enable:
            desired = np.array(base_position_req.desired).reshape((2,1))
            self.tasks[task_index].setDesired(desired)

            # store the goal base position (for visualization in Rviz)
            self.base_goal = desired
        else:
            self.base_goal[:,0] = [None,None]

        # send success report to service client
        return TaskTriggerResponse(True)

    def joint1_position_handle(self,joint1_position_req):
        # find the instance in self.tasks
        for i in range(len(self.tasks)):
            task = self.tasks[i]
            if isinstance(task,JointPosition) and task.name == "Joint 1 position":
                task_index = i
                break

        # enable/disable the task based on the request
        self.tasks[task_index].setStatus(joint1_position_req.enable)

        # also set the desired state if it is enabled
        if joint1_position_req.enable:
            desired = np.array(joint1_position_req.desired).reshape((1,1))
            self.tasks[task_index].setDesired(desired)

        # send success report to service client
        return TaskTriggerResponse(True)
    
    def joint2_position_handle(self,joint2_position_req):
        # find the instance in self.tasks
        for i in range(len(self.tasks)):
            task = self.tasks[i]
            if isinstance(task,JointPosition) and task.name == "Joint 2 position":
                task_index = i
                break

        # enable/disable the task based on the request
        self.tasks[task_index].setStatus(joint2_position_req.enable)

        # also set the desired state if it is enabled
        if joint2_position_req.enable:
            desired = np.array(joint2_position_req.desired).reshape((1,1))
            self.tasks[task_index].setDesired(desired)

        # send success report to service client
        return TaskTriggerResponse(True)
    
    def joint3_position_handle(self,joint3_position_req):
        # find the instance in self.tasks
        for i in range(len(self.tasks)):
            task = self.tasks[i]
            if isinstance(task,JointPosition) and task.name == "Joint 3 position":
                task_index = i
                break

        # enable/disable the task based on the request
        self.tasks[task_index].setStatus(joint3_position_req.enable)

        # also set the desired state if it is enabled
        if joint3_position_req.enable:
            desired = np.array(joint3_position_req.desired).reshape((1,1))
            self.tasks[task_index].setDesired(desired)

        # send success report to service client
        return TaskTriggerResponse(True)
    
    def joint4_position_handle(self,joint4_position_req):
        # find the instance in self.tasks
        for i in range(len(self.tasks)):
            task = self.tasks[i]
            if isinstance(task,JointPosition) and task.name == "Joint 4 position":
                task_index = i
                break

        # enable/disable the task based on the request
        self.tasks[task_index].setStatus(joint4_position_req.enable)

        # also set the desired state if it is enabled
        if joint4_position_req.enable:
            desired = np.array(joint4_position_req.desired).reshape((1,1))
            self.tasks[task_index].setDesired(desired)

        # send success report to service client
        return TaskTriggerResponse(True)
    
    def stop_handle(self,stop_req):
        for task in self.tasks:
            if not isinstance(task,JointLimits):
                task.setStatus(False)

        # also delete all stored goal values for visualization
        self.ee_goal = np.zeros((4,1))
        self.base_goal = np.zeros((2,1))

        return TriggerResponse(success=True, message="All tasks stopped.")

    # Timer action
    def compute_commands(self,event):
        # Visualize goal (EE position or configuration or base position) in Rviz
        if self.ee_goal[:3].any():
            if self.ee_goal[-1,0] is None:
                goal_yaw = 0 # visualize the desired orientation as 0 if there's no desired orientation
            else:
                goal_yaw = self.ee_goal[-1,0]
            
            # Send tf
            self.broadcaster.sendTransform((self.ee_goal[0,0],self.ee_goal[1,0],self.ee_goal[2,0]),
                                       tf.transformations.quaternion_from_euler(0, 0, goal_yaw),
                                        rospy.Time.now(),
                                        "goal_frame", # Child frame
                                        "world_ned")
        elif self.base_goal.any():
            # Send tf
            self.broadcaster.sendTransform((self.base_goal[0,0],self.base_goal[1,0],0.0),
                                        (0,0,0,1),
                                        rospy.Time.now(),
                                        "goal_frame", # Child frame
                                        "world_ned")
            
        # Prepare error message
        error_name = []
        error_value = []

        ## Run the TP algorithm
        # Initialize null-space projector
        Pi_1 = np.eye(6)

        # Initialize output vector (quasi-velocities)
        dqi_1 = np.zeros((6, 1))
        dq = np.zeros((6,1))

        # Loop over tasks
        for task in self.tasks:
            # Skip if the task is not enabled
            if not task.getStatus():
                continue

            # Update task state
            task.update(self.q, self.base_pos, self.base_orientation)
            Ji = task.getJacobian()
            erri = task.getError()

            # Publish error to ROS topic
            error_name.append(task.name)
            error_value.append(np.linalg.norm(erri))

            # Skip if the task is not active
            if not task.isActive(self.q):
                continue

            # Get feedforward velocity and gain matrix
            feedforward_velocity = task.getFeedforwardVelocity()
            K = task.getGainMatrix()

            xdoti = feedforward_velocity + K @ erri

            # Compute augmented Jacobian
            Ji_bar = Ji @ Pi_1

            # Compute task velocity with feedforward term and K matrix
            dqi = DLS(Ji_bar,0.1) @ (xdoti - Ji @ dqi_1)
            # dqi = WeightedDLS(Ji_bar, 0.1, W) @ (xdoti - Ji @ dqi_1)
            dq = dqi_1 + dqi
            
            # Update null-space projector
            Pi = Pi_1 - np.linalg.pinv(Ji_bar) @ Ji_bar

            # Store the current P and dq for the next iteration
            Pi_1 = Pi
            dqi_1 = dq

        # Scale velocity if required
        if dq.any():
            s = np.max(abs(dq)/self.dq_max)
            if s > 1:
                dq /= s

        # Store joint velocity data
        for i in range(dq.shape[0]):
            self.joint_vel_data[i].append(dq[i,0])
        self.joint_vel_data[-1].append((rospy.Time.now() - self.start_time).to_sec())

        # Send error message
        error_msg = TaskError()
        error_msg.tasks = error_name
        error_msg.error = error_value
        self.error_pub.publish(error_msg)

        # Send quasi-velocities to the corresponding topics
        self.send_velocity(dq)

    def send_velocity(self,dq):
        # send mobile base vel
        base_vel = Twist()
        base_vel.linear.x = dq[1,0]
        base_vel.angular.z = -dq[0,0]
        self.base_vel_pub.publish(base_vel)

        # send joint vel
        joint_vel = Float64MultiArray()
        joint_vel.data = list(dq[2:6].flatten())
        self.joint_vel_pub.publish(joint_vel)

    def signal_handler(self, sig, frame):
        rospy.loginfo("Shutdown signal received.")
        
        # Define the directory to save the npy files
        save_dir = os.path.expanduser('~/catkin_ws/src/ho_intervention_project/data')
        
        # Ensure the directory exists
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Save the attributes to .npy files
        np.save(os.path.join(save_dir, 'joint_position_data.npy'), self.joint_position_data)
        np.save(os.path.join(save_dir, 'ee_position_data.npy'),self.ee_position_data)
        np.save(os.path.join(save_dir, 'base_position_data.npy'), self.base_position_data)
        np.save(os.path.join(save_dir, 'joint_vel_data.npy'), self.joint_vel_data)

        sys.exit(0)


if __name__=='__main__':
    rospy.init_node('tp_controller_node') # initialize the node
    node = TP_Controller() # a newly-created class

    rospy.spin()