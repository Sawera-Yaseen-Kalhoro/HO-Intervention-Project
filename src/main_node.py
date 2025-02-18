#!/usr/bin/env python3

import py_trees
import time
import math
import rospy
import py_trees
import numpy as np
from nav_msgs.msg import Odometry
from std_srvs.srv import SetBool
from geometry_msgs.msg import PoseStamped, PointStamped
from ho_intervention_project.srv import TaskTrigger, TaskTriggerRequest
from ho_intervention_project.msg import TaskError
from std_srvs.srv import Trigger, TriggerRequest
import os


class Detect_Aruco(py_trees.behaviour.Behaviour):
    def __init__(self, name):
        # super(Detect_Aruco).__init__(name)
        super(Detect_Aruco, self).__init__(name)
        self.blackboard = self.attach_blackboard_client(name="Blackboard")
        self.blackboard.register_key("Goal_pos", access=py_trees.common.Access.WRITE)

        # Subscribes to Aruco node
        self.aruco_subscriber = rospy.Subscriber("/target_position", PointStamped, self.aruco_callback)
        self.aruco_detected = False

    def setup(self):
        self.logger.debug("  %s [Detect_Aruco::setup()]" % self.name)

    def initialise(self):
        self.logger.debug(" %s [Detect_Aruco::initialise()]" % self.name)

    def update(self):
        print("Aruco detected")
        if self.aruco_detected:
            print("Aruco marker is not deteced...")
            rospy.logdebug("Aruco marker detected!")
            self.blackboard.Goal_pos = self.aruco_pose
            return py_trees.common.Status.SUCCESS
        else:
            return py_trees.common.Status.RUNNING
        
    def aruco_callback(self, aruco_pose_msg):
        # print("Checking if the callback is working or not!!!")
        self.aruco_pose = [aruco_pose_msg.point.x, aruco_pose_msg.point.y, aruco_pose_msg.point.z]
        self.aruco_detected = True
    
    def terminate(self, new_status):
        self.logger.debug(" %s [Detect_Aruco::terminate().terminate()] [%s->%s]" % 
                          (self.name, self.status, new_status))


class Rest_pos(py_trees.behaviour.Behaviour):
    def __init__(self, name):
        super(Rest_pos, self).__init__(name)
        self.rest_joint_positions = [np.pi/2,-np.pi/4,-np.pi/4,np.pi/2]  # Random rest joint positions, need correct values
        self.all_responses = []
        self.error = [float('inf')] * 4
        self.task_active = False  # Track task activation

        self.error_subscriber = rospy.Subscriber('/tp_controller_node/control_error', TaskError, self.error_callback)
        self.threshold = np.deg2rad(5)  # Error threshold in radians

    def setup(self):
        self.logger.debug("  %s [Rest_pos::setup()]" % self.name)

        rospy.wait_for_service('/tp_controller_node/joint1_position')
        rospy.wait_for_service('/tp_controller_node/joint2_position')
        rospy.wait_for_service('/tp_controller_node/joint3_position')
        rospy.wait_for_service('/tp_controller_node/joint4_position')
        rospy.wait_for_service('/tp_controller_node/stop_all')
        try:
            self.server1 = rospy.ServiceProxy('/tp_controller_node/joint1_position', TaskTrigger)
            self.server2 = rospy.ServiceProxy('/tp_controller_node/joint2_position', TaskTrigger)
            self.server3 = rospy.ServiceProxy('/tp_controller_node/joint3_position', TaskTrigger)
            self.server4 = rospy.ServiceProxy('/tp_controller_node/joint4_position', TaskTrigger)
            self.server_stop = rospy.ServiceProxy('/tp_controller_node/stop_all', Trigger)
            self.logger.debug(" %s [Rest_pos::setup() Server Connected!]" % self.name)

            # setup error evol data
            self.error_evol_data = [[],[],[],[],[]] # 4 joints + timestamp
            self.start_time = rospy.Time.now()
        except rospy.ServiceException as e:
            self.logger.debug(" %s [Rest_pos::setup() ERROR!]" % self.name)

    def initialise(self):
        self.logger.debug(" %s [Rest_pos::initialise()]" % self.name)
        self.task_active = False  # Reset task active flag
        self.error = [float('inf')] * 4  # Reset error list

    def error_callback(self, error_msg):
        if not self.task_active:
            return

        task_names = error_msg.tasks
        task_errors = error_msg.error

        if "Joint 1 position" in task_names:
            start_index = task_names.index("Joint 1 position")
            self.error = task_errors[start_index:start_index+4]

            # store error evol data
            for i in range(len(self.error)):
                self.error_evol_data[i].append(self.error[i])
            self.error_evol_data[-1].append((rospy.Time.now() - self.start_time).to_sec())
        else:
            rospy.logerr("Task 'Joint 1 position' not found in error message")
            rospy.loginfo("Received tasks: %s", task_names)
            rospy.loginfo("Corresponding errors: %s", task_errors)

    def update(self):
        print("Rest position is working")
        if not self.task_active:
            try:
                response1 = self.server1(enable=True, desired=[self.rest_joint_positions[0]])
                response2 = self.server2(enable=True, desired=[self.rest_joint_positions[1]])
                response3 = self.server3(enable=True, desired=[self.rest_joint_positions[2]])
                response4 = self.server4(enable=True, desired=[self.rest_joint_positions[3]])
                rospy.loginfo("Attempting to set rest position")

                response1_status = response1.status
                response2_status = response2.status
                response3_status = response3.status
                response4_status = response4.status

                self.all_responses = [response1_status, response2_status, response3_status, response4_status]
                
                if all(self.all_responses):
                    rospy.loginfo("Rest position set successfully")
                    self.task_active = True  # Mark task as active
                else:
                    rospy.logwarn("Failed to set rest position")
                    return py_trees.common.Status.FAILURE
            except rospy.ServiceException as e:
                rospy.logerr("Service call failed: %s" % str(e))
                return py_trees.common.Status.FAILURE

        # Check the task completion
        if all([error <= self.threshold for error in self.error]):
            try:
                stop_resp = self.server_stop(TriggerRequest())
                if stop_resp.success:
                    self.task_active = False  # Deactivate task
                    return py_trees.common.Status.SUCCESS
            except rospy.ServiceException as e:
                rospy.logerr("Failed to stop controller: %s" % str(e))
                return py_trees.common.Status.FAILURE
        else:
            return py_trees.common.Status.RUNNING
        
    def terminate(self, new_status):
        self.logger.debug(" %s [Rest_pos::terminate().terminate()] [%s->%s]" % 
                          (self.name, self.status, new_status))
        
        # Save error evol data to npy
        # Define the directory to save the npy files
        save_dir = os.path.expanduser('~/catkin_ws/src/ho_intervention_project/data')
        
        # Ensure the directory exists
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Save the attributes to .npy files
        np.save(os.path.join(save_dir, self.name + '_error.npy'), self.error_evol_data)

        
class Track_Aruco(py_trees.behaviour.Behaviour):
    def __init__(self, name):
        # super(Detect_Aruco).__init__(name)
        super(Track_Aruco, self).__init__(name)
        self.blackboard = self.attach_blackboard_client(name="Blackboard")
        self.blackboard.register_key("Goal_pos", access=py_trees.common.Access.WRITE)

        # Subscribes to Aruco node
        self.aruco_subscriber = rospy.Subscriber("/target_position", PointStamped, self.aruco_callback)
        self.aruco_detected = False

    def setup(self):
        self.logger.debug("  %s [Detect_Aruco::setup()]" % self.name)

    def initialise(self):
        self.logger.debug(" %s [Detect_Aruco::initialise()]" % self.name)

    def update(self):
        if self.aruco_detected:
            rospy.logdebug("Aruco marker detected!")
            self.blackboard.Goal_pos = self.aruco_pose
            print("Goal value in blackboard = ", self.blackboard.Goal_pos)
        return py_trees.common.Status.RUNNING
        
    def aruco_callback(self, aruco_pose_msg):
        self.aruco_pose = [aruco_pose_msg.point.x, aruco_pose_msg.point.y, aruco_pose_msg.point.z]
        self.aruco_detected = True
    
    def terminate(self, new_status):
        self.logger.debug(" %s [Detect_Aruco::terminate().terminate()] [%s->%s]" % 
                          (self.name, self.status, new_status))
        

class Move_robot(py_trees.behaviour.Behaviour):
    def __init__(self, name, distance_threshold = 0.5, desired_orientation=0.0): # in m, distance between object and manipulator base
        super(Move_robot, self).__init__(name)
        # self.manipulator_base_pos = None
        self.ee_position_enable = False
        self.goal_pos = None
        self.response = None
        self.goal_ori = desired_orientation
        self.dis_threshold = distance_threshold # m between manipulator base and goal position
        self.blackboard = self.attach_blackboard_client(name = "Blackboard")
        # self.blackboard.register_key("Aruco_pos", access=py_trees.common.Access.READ)
        self.blackboard.register_key("Goal_pos", access=py_trees.common.Access.READ)
        
        # Subscribe to error node
        self.error_sub = rospy.Subscriber('/tp_controller_node/control_error',TaskError, self.error_callback)
        self.task_active = False
        self.error = None


    def setup(self):
        self.logger.debug("  %s [Move_robot::setup()]" % self.name)
        rospy.wait_for_service('/tp_controller_node/base_position')
        rospy.wait_for_service('/tp_controller_node/base_orientation')
        rospy.wait_for_service('/tp_controller_node/stop_all')
        try:
            self.server_pos = rospy.ServiceProxy('/tp_controller_node/base_position', TaskTrigger)
            self.server_ori = rospy.ServiceProxy('/tp_controller_node/base_orientation', TaskTrigger)
            self.server_stop = rospy.ServiceProxy('/tp_controller_node/stop_all', Trigger)
            self.logger.debug(" %s [Move_robot::setup() Server Connected!]" % self.name)

            # setup error evol data
            self.error_evol_data = [[],[],[]] # base orientation + base position + timestamp
            self.start_time = rospy.Time.now()
        except rospy.ServiceException as e:
            self.logger.debug(" %s [Move_robot::setup() ERROR!]" % self.name)


    def initialise(self):
        self.logger.debug(" %s [Move_robot::initialise()]" % self.name)
        self.task_active = False
        self.error = float('inf')
  
    def update(self):
        # Send request/update goal to TP controller
        try:
            self.logger.debug(
                "  {}: call service /tp_controller_node/base_position".format(self.name))
            self.goal_pos = self.blackboard.Goal_pos[:2] # only take the x and y position
            # activate base_position service
            response_pos = self.server_pos(enable=True, desired=self.goal_pos)

            self.logger.debug(
                "  {}: call service /tp_controller_node/base_orientation".format(self.name))
            # activate base_orientation service
            response_ori = self.server_ori(enable=True, desired=[self.goal_ori])

            self.response = [response_pos.status, response_ori.status]
            
            if all(self.response):
                rospy.loginfo("Base position and orientation set successfully")
                self.task_active = True
            else:
                rospy.logwarn("Failed to set base position and orientation")
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s" % str(e))
            return py_trees.common.Status.FAILURE

        if self.error is None:
            return py_trees.common.Status.RUNNING

        # Check task completion
        if abs(self.error) > self.dis_threshold:
            print("current distance from goal: ", self.error)
            return py_trees.common.Status.RUNNING
        else:
            print('distance threshold ', self.dis_threshold, ' reached')
            # deactivate the task
            try:
                stop_resp = self.server_stop(TriggerRequest())
                if stop_resp.success:
                    return py_trees.common.Status.SUCCESS
            except rospy.ServiceException as e:
                rospy.logerr("Failed to stop controller: %s" % str(e))
                return py_trees.common.Status.FAILURE
    
    def error_callback(self, error_msg):
        if not self.task_active:
            return

        task_names = error_msg.tasks
        task_errors = error_msg.error

        if "Base position" in task_names:
            start_index = task_names.index("Base position")
            self.error = task_errors[start_index]

        # store error evol data
        if "Base orientation" in task_names:
            ori_index = task_names.index("Base orientation")
            self.error_evol_data[0].append(task_errors[ori_index])

            self.error_evol_data[1].append(self.error)
            self.error_evol_data[-1].append((rospy.Time.now() - self.start_time).to_sec())

    def terminate(self, new_status):
        self.logger.debug(" %s [Move_robot::terminate().terminate()] [%s->%s]" % 
                          (self.name, self.status, new_status))
        
        # Save error evol data to npy
        # Define the directory to save the npy files
        save_dir = os.path.expanduser('~/catkin_ws/src/ho_intervention_project/data')
        
        # Ensure the directory exists
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Save the attributes to .npy files
        np.save(os.path.join(save_dir, self.name + '_error.npy'), self.error_evol_data)


# Move the cup on over the object with certain z distance to its top face
class Move_mani(py_trees.behaviour.Behaviour):
    def __init__(self, name, distance_threshold = 0.02): # in m
        # super(Move_mani).__init__(name)
        super(Move_mani, self).__init__(name)
        # self.ee_position_enable = False
        self.goal_pos = None
        self.response = None
        self.z_offset = 0.1 # in m, distance between Ef and top of object
        self.dis_threshold = distance_threshold # m between EF and goal position
        self.ee_position_enable = False
        self.desired_ee_position = None  # set the position
        self.blackboard = self.attach_blackboard_client(name = "Blackboard")
        self.blackboard.register_key("Goal_pos", access=py_trees.common.Access.READ)
        
        # Subscribe to error node
        self.error_sub = rospy.Subscriber('/tp_controller_node/control_error',TaskError, self.error_cb)
        self.task_active = False
        self.error = None

    def setup(self):
        self.logger.debug("  %s [Move_mani::setup()]" % self.name)
        rospy.wait_for_service('/tp_controller_node/ee_position')
        rospy.wait_for_service('/tp_controller_node/stop_all')
        try:
            self.server = rospy.ServiceProxy('/tp_controller_node/ee_position', TaskTrigger)
            self.server_stop = rospy.ServiceProxy('/tp_controller_node/stop_all', Trigger)
            self.logger.debug(" %s [Move_robot::setup() Server Connected!]" % self.name)

            # setup error evol data
            self.error_evol_data = [[],[]] # EE position + timestamp
            self.start_time = rospy.Time.now()
        except rospy.ServiceException as e:
            self.logger.debug(" %s [Move_robot::setup() ERROR!]" % self.name)

    def initialise(self):
        self.logger.debug(" %s [Move_mani::initialise()]" % self.name)
        self.task_active = False
        self.error = float('inf')

    def update(self):
        # Enable the EE position task
        try:
            self.logger.debug(
                "  {}: call service /tp_controller_node/ee_position".format(self.name))
            self.goal_pos = self.blackboard.Goal_pos
            # considering that Z axis is down, we add the offset with a negative sign
            # self.goal_pos[2] -= self.z_offset 
            goal_high = [self.goal_pos[0],self.goal_pos[1],self.goal_pos[2] - self.z_offset] # to prevent changing blackboard value
            # activate base_position service
            response = self.server(enable=True, desired=goal_high)
            self.response = response.status
            
            if self.response:
                rospy.loginfo("Move mani set successfully")
                self.task_active = True
            else:
                rospy.logwarn("Failed to set Manipulator position")
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s" % str(e))
            return py_trees.common.Status.FAILURE
            
        if self.error is None:
            return py_trees.common.Status.RUNNING
        
        # Check task completion
        if abs(self.error) > self.dis_threshold:
            print("current distance from goal: ", self.error)
            return py_trees.common.Status.RUNNING
        else:
            print('distance threshold ', self.dis_threshold, ' reached')
            # deactivate the task
            try:
                stop_resp = self.server_stop(TriggerRequest())
                # stop_resp = self.server(enable=False, desired=[0.0,0.0,0.0])
                # if stop_resp.status:
                if stop_resp.success:
                    self.task_active = False  # Deactivate task
                    return py_trees.common.Status.SUCCESS
            except rospy.ServiceException as e:
                rospy.logerr("Failed to stop controller: %s" % str(e))
                return py_trees.common.Status.FAILURE

    def error_cb(self, error_msg):
        if not self.task_active:
            return

        task_names = error_msg.tasks
        task_errors = error_msg.error

        if "End-effector position" in task_names:
            start_index = task_names.index("End-effector position")
            self.error = task_errors[start_index]

        # store error evol data
        self.error_evol_data[0].append(self.error)
        self.error_evol_data[-1].append((rospy.Time.now() - self.start_time).to_sec())

    def terminate(self, new_status):
        self.logger.debug(" %s [Move_mani::terminate().terminate()] [%s->%s]" % 
                          (self.name, self.status, new_status))
        
        # Save error evol data to npy
        # Define the directory to save the npy files
        save_dir = os.path.expanduser('~/catkin_ws/src/ho_intervention_project/data')
        
        # Ensure the directory exists
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Save the attributes to .npy files
        np.save(os.path.join(save_dir, self.name + '_error.npy'), self.error_evol_data)


class Move_EEdown(py_trees.behaviour.Behaviour):
    def __init__(self, name, distance_threshold = 0.005): # in m
        super(Move_EEdown,self).__init__(name)
        self.ee_position_enable = False
        self.goal_pos = None
        self.response = None
        self.dis_threshold = distance_threshold # m between EF and goal position
        self.ee_position_enable = False
        self.desired_ee_position = None  # set the position
        self.blackboard = self.attach_blackboard_client(name = "Blackboard")
        # self.blackboard.register_key("Aruco_pos", access=py_trees.common.Access.READ)
        self.blackboard.register_key("Goal_pos", access=py_trees.common.Access.READ)
        
        # Subscribe to error node
        self.error_sub = rospy.Subscriber('/tp_controller_node/control_error',TaskError, self.error_cb)
        self.task_active = False
        self.error = None

    def setup(self):
        self.logger.debug("  %s [Move_EEdown::setup()]" % self.name)
        rospy.wait_for_service('/tp_controller_node/ee_position')
        rospy.wait_for_service('/tp_controller_node/stop_all')
        try:
            self.server = rospy.ServiceProxy('/tp_controller_node/ee_position', TaskTrigger)
            self.server_stop = rospy.ServiceProxy('/tp_controller_node/stop_all', Trigger)
            self.logger.debug(" %s [Move_EEdown::setup() Server Connected!]" % self.name)

            # setup error evol data
            self.error_evol_data = [[],[]] # EE position + timestamp
            self.start_time = rospy.Time.now()
        except rospy.ServiceException as e:
            self.logger.debug(" %s [Move_EEdown::setup() ERROR!]" % self.name)

    def initialise(self):
        self.logger.debug(" %s [Move_EEdown::initialise()]" % self.name)
        self.task_active = False
        self.error = float('inf')

    def update(self):
        if not self.task_active:
            # Enable the EE position task
            try:
                self.logger.debug(
                    "  {}: call service /tp_controller_node/ee_position".format(self.name))
                self.goal_pos = self.blackboard.Goal_pos
                print("Goal = ", self.goal_pos)
                # activate base_position service
                response = self.server(enable=True, desired=self.goal_pos)
                self.response = response.status
                if self.response:
                    rospy.loginfo("Move EEdown set successfully")
                    self.task_active = True
                else:
                    rospy.logwarn("Failed to set Manipulator position")
            except rospy.ServiceException as e:
                rospy.logerr("Service call failed: %s" % str(e))
                return py_trees.common.Status.FAILURE
        
        if self.error is None:
            return py_trees.common.Status.RUNNING
        
        # Check task completion
        if abs(self.error) > self.dis_threshold:
            print("current distance from goal: ", self.error)
            return py_trees.common.Status.RUNNING
        else:
            print('distance threshold ', self.dis_threshold, ' reached')
            # deactivate the task
            try:
                stop_resp = self.server_stop(TriggerRequest())
                # stop_resp = self.server(enable=False, desired=[0.0,0.0,0.0])
                # if stop_resp.status:
                if stop_resp.success:
                    self.task_active = False  # Deactivate task
                    return py_trees.common.Status.SUCCESS
            except rospy.ServiceException as e:
                rospy.logerr("Failed to stop controller: %s" % str(e))
                return py_trees.common.Status.FAILURE
    
    def error_cb(self, error_msg):
        if not self.task_active:
            return

        task_names = error_msg.tasks
        task_errors = error_msg.error

        if "End-effector position" in task_names:
            start_index = task_names.index("End-effector position")
            self.error = task_errors[start_index]
        else:
            rospy.logerr("Task 'End-effector position' not found in error message")
            rospy.loginfo("Received tasks: %s", task_names)
            rospy.loginfo("Corresponding errors: %s", task_errors)

        # store error evol data
        self.error_evol_data[0].append(self.error)
        self.error_evol_data[-1].append((rospy.Time.now() - self.start_time).to_sec())

    def terminate(self, new_status):
        self.logger.debug(" %s [Move_EEdown::terminate().terminate()] [%s->%s]" % 
                          (self.name, self.status, new_status))
        
        # Save error evol data to npy
        # Define the directory to save the npy files
        save_dir = os.path.expanduser('~/catkin_ws/src/ho_intervention_project/data')
        
        # Ensure the directory exists
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Save the attributes to .npy files
        np.save(os.path.join(save_dir, self.name + '_error.npy'), self.error_evol_data)


class Activate_Suction(py_trees.behaviour.Behaviour):
    def __init__(self, name):
        super(Activate_Suction,self).__init__(name)
        self.service_proxy = rospy.ServiceProxy("/turtlebot/swiftpro/vacuum_gripper/set_pump", SetBool)
        self.suction_activated = False
       
    def setup(self):
        self.logger.debug("  %s [Activate_Suction::setup()]" % self.name)

    def initialise(self):
        self.logger.debug(" %s [Activate_Suction::initialise()]" % self.name)

    def update(self):
        if not self.suction_activated:
            try:
                self.service_proxy(True)
                self.suction_activated = True
                return py_trees.common.Status.SUCCESS
            except rospy.ServiceException as e:
                self.logger.error("Failed to call service: %s" % str(e))
                return py_trees.common.Status.FAILURE
        else:
            return py_trees.common.Status.SUCCESS

    
    def terminate(self, new_status):
        self.logger.debug(" %s [Activate_Suction::terminate().terminate()] [%s->%s]" % 
                          (self.name, self.status, new_status))


class Move_EEhigh(py_trees.behaviour.Behaviour):
    def __init__(self, name, distance_threshold = 0.01): # in m
        super(Move_EEhigh,self).__init__(name)
        self.ee_position_enable = False
        self.goal_pos = None
        self.response = None
        self.dis_threshold = distance_threshold # m between manipulator base and goal position
        self.blackboard = self.attach_blackboard_client(name = "Blackboard")
        # self.blackboard.register_key("Aruco_pos", access=py_trees.common.Access.READ)
        self.blackboard.register_key("Goal_pos", access=py_trees.common.Access.READ)
        
        # Subscribe to error node
        self.error_sub = rospy.Subscriber('/tp_controller_node/control_error',TaskError, self.error_cb)
        self.task_active = False
        self.error = None

    def setup(self):
        self.logger.debug("  %s [Move_EEhigh::setup()]" % self.name)
        rospy.wait_for_service('/tp_controller_node/ee_position')
        rospy.wait_for_service('/tp_controller_node/stop_all')
        try:
            self.server = rospy.ServiceProxy('/tp_controller_node/ee_position', TaskTrigger)
            self.server_stop = rospy.ServiceProxy('/tp_controller_node/stop_all', Trigger)
            self.logger.debug(" %s [Move_EEhigh::setup() Server Connected!]" % self.name)

            # setup error evol data
            self.error_evol_data = [[],[]] # EE position + timestamp
            self.start_time = rospy.Time.now()
        except rospy.ServiceException as e:
            self.logger.debug(" %s [Move_EEhigh::setup() ERROR!]" % self.name)

    def initialise(self):
        self.logger.debug(" %s [Move_EEhigh::initialise()]" % self.name)
        self.task_active = False
        self.error = float('inf')

    def update(self):
        if not self.task_active:
            # Enable the EE position task
            try:
                self.logger.debug(
                    "  {}: call service /tp_controller_node/ee_position".format(self.name))
                self.goal_pos = self.blackboard.Goal_pos
                # set to a high position above the robot, play changing the value of the 3 cm
                # kobuki height + height of object + 2 cm
                # self.goal_pos[2] = -(0.198 + 0.15 + 0.02)
                goal_high = [self.goal_pos[0],self.goal_pos[1],-(0.198 + 0.15 + 0.02)] # to prevent from changing blackboard value
                # activate base_position service
                response = self.server(enable=True, desired=goal_high)
                
                self.response = response.status
                
                if self.response:
                    rospy.loginfo("Move EEhigh set successfully")
                    self.task_active = True
                else:
                    rospy.logwarn("Failed to set Manipulator position")
            except rospy.ServiceException as e:
                rospy.logerr("Service call failed: %s" % str(e))
                return py_trees.common.Status.FAILURE
        
        if self.error is None:
            return py_trees.common.Status.RUNNING
        
        # Check task completion
        if abs(self.error) > self.dis_threshold:
            print("current distance from goal: ", self.error)
            return py_trees.common.Status.RUNNING
        else:
            print('distance threshold ', self.dis_threshold, ' reached')
            # deactivate the task
            try:
                stop_resp = self.server_stop(TriggerRequest())
                # stop_resp = self.server(enable=False, desired=[0.0,0.0,0.0])
                # if stop_resp.status:
                if stop_resp.success:
                    self.task_active = False  # Deactivate task
                    return py_trees.common.Status.SUCCESS
            except rospy.ServiceException as e:
                rospy.logerr("Failed to stop controller: %s" % str(e))
                return py_trees.common.Status.FAILURE

    def error_cb(self, error_msg):
        if not self.task_active:
            return

        task_names = error_msg.tasks
        task_errors = error_msg.error

        if "End-effector position" in task_names:
            start_index = task_names.index("End-effector position")
            self.error = task_errors[start_index]
        else:
            rospy.logerr("Task 'End-effector position' not found in error message")
            rospy.loginfo("Received tasks: %s", task_names)
            rospy.loginfo("Corresponding errors: %s", task_errors)

        # store error evol data
        self.error_evol_data[0].append(self.error)
        self.error_evol_data[-1].append((rospy.Time.now() - self.start_time).to_sec())

    def terminate(self, new_status):
        self.logger.debug(" %s [Move_EEhigh::terminate().terminate()] [%s->%s]" % 
                          (self.name, self.status, new_status))
        
        # Save error evol data to npy
        # Define the directory to save the npy files
        save_dir = os.path.expanduser('~/catkin_ws/src/ho_intervention_project/data')
        
        # Ensure the directory exists
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Save the attributes to .npy files
        np.save(os.path.join(save_dir, self.name + '_error.npy'), self.error_evol_data)


class Move_EEtop(py_trees.behaviour.Behaviour): # move EF to the top of the robot
    def __init__(self, name, desired):
        super(Move_EEtop,self).__init__(name)
        self.desired_joint1_position = desired # Joint1 position where the object will be on top of robot base
        self.all_responses = []
        self.error = float('inf')
        self.task_active = False  # Track task activation

        self.error_subscriber = rospy.Subscriber('/tp_controller_node/control_error', TaskError, self.error_callback)
        self.threshold = np.deg2rad(5)  # Error threshold in radians

    def setup(self):
        self.logger.debug("  %s [Move_EEtop::setup()]" % self.name)

        rospy.wait_for_service('/tp_controller_node/joint1_position')
        try:
            self.server1 = rospy.ServiceProxy('/tp_controller_node/joint1_position', TaskTrigger)
            self.server_stop = rospy.ServiceProxy('/tp_controller_node/stop_all', Trigger) 
            self.logger.debug(" %s [Move_EEtop::setup() Server Connected!]" % self.name)

            # setup error evol data
            self.error_evol_data = [[],[]] # joint 1 position + timestamp
            self.start_time = rospy.Time.now()
        except rospy.ServiceException as e:
            self.logger.debug(" %s [Move_EEtop::setup() ERROR!]" % self.name)

    def initialise(self):
        self.logger.debug(" %s [Move_EEtop::initialise()]" % self.name)
        self.task_active = False  # Reset task active flag
        self.error = float('inf')  # Reset error list

    def error_callback(self, error_msg):
        if not self.task_active:
            return None

        task_names = error_msg.tasks
        task_errors = error_msg.error

        if "Joint 1 position" in task_names:
            start_index = task_names.index("Joint 1 position")
            self.error = task_errors[start_index]
        else:
            rospy.logerr("Task 'Joint 1 position' not found in error message")
            rospy.loginfo("Received tasks: %s", task_names)
            rospy.loginfo("Corresponding errors: %s", task_errors)

        # store error evol data
        self.error_evol_data[0].append(self.error)
        self.error_evol_data[-1].append((rospy.Time.now() - self.start_time).to_sec())

    def update(self):
        if not self.task_active:    
            try:
                # Send a request to set the joint positions
                response1 = self.server1(enable=True, desired=[self.desired_joint1_position])
                response_status = response1.status
                
                if response_status:
                    rospy.loginfo("EE top position reached successfully")
                    self.task_active = True  # Mark task as active
                else:
                    rospy.logwarn("Failed to reach the EE top position")
                    return py_trees.common.Status.FAILURE
            except rospy.ServiceException as e:
                rospy.logerr("Service call failed: %s" % str(e))
                return py_trees.common.Status.FAILURE
        
        # Check the task completion
        if self.error <= self.threshold:
            try:
                stop_resp = self.server_stop(TriggerRequest())
                if stop_resp.success:
                    self.task_active = False  # Deactivate task
                    return py_trees.common.Status.SUCCESS
            except rospy.ServiceException as e:
                rospy.logerr("Failed to stop controller: %s" % str(e))
                return py_trees.common.Status.FAILURE
        else:
            print("Angle error = ", self.error)
            return py_trees.common.Status.RUNNING
    
    def terminate(self, new_status):
        self.logger.debug(" %s [Move_EEtop::terminate().terminate()] [%s->%s]" % 
                          (self.name, self.status, new_status))
        
        # Save error evol data to npy
        # Define the directory to save the npy files
        save_dir = os.path.expanduser('~/catkin_ws/src/ho_intervention_project/data')
        
        # Ensure the directory exists
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Save the attributes to .npy files
        np.save(os.path.join(save_dir, self.name + '_error.npy'), self.error_evol_data)  
        

class Deactive_suction(py_trees.behaviour.Behaviour):
    def __init__(self, name):
        super(Deactive_suction,self).__init__(name)
        self.service_proxy = rospy.ServiceProxy("/turtlebot/swiftpro/vacuum_gripper/set_pump", SetBool)
        self.suction_deactivated = False  # Initially activation is on

    def setup(self):
        self.logger.debug("  %s [Deactive_suction::setup()]" % self.name)

    def initialise(self):
        self.logger.debug(" %s [Deactive_suction::initialise()]" % self.name)

    def update(self):
        if not self.suction_deactivated:
            try:
                self.service_proxy(False)
                self.suction_deactivated = True
                return py_trees.common.Status.SUCCESS
            except rospy.ServiceException as e:
                self.logger.error("Failed to call service: %s" % str(e))
                return py_trees.common.Status.FAILURE
        else:
            return py_trees.common.Status.SUCCESS
        
    
    def terminate(self, new_status):
        self.logger.debug(" %s [Deactive_suction::terminate().terminate()] [%s->%s]" % 
                          (self.name, self.status, new_status))
        

def create_tree():

    # Aruco detection Behavior
    Detect_aruco = Detect_Aruco("Detect Aruco Pos") # Set threshold distance
    Track_aruco = Track_Aruco("Track ArUco Pos")


    # Rest position Behavior
    Rest_position1 = Rest_pos("Rest Position start")
    Rest_position2 = Rest_pos("Rest Position final")

    # Move robot Behavior
    Move_robot_pos1 = Move_robot("Move robot to pick up position")
    Move_robot_pos2 = Move_robot("Move robot to drop position",desired_orientation=np.deg2rad(60))

    # Move Manipulator behavior
    Move_manipulator = Move_mani("Move manipulator to position")
    Move_EE_High2 = Move_mani("Move EE High waypoint 2")

    # Move EE down behavior
    Move_EE_Down1 = Move_EEdown("Move EE Down pos1")
    Move_EE_Down2 = Move_EEdown("Move EE Down pos2",distance_threshold=0.03)

    # Activate Suction Behavior
    activate_Suction = Activate_Suction("Activate Suction")

    # Move EE high behavior
    Move_EE_High1 = Move_EEhigh("Move EE High waypoint 1")

    # Move EE to the top of robot behavior
    Move_EE_Top = Move_EEtop("Move EE to top of robot",-np.pi/2)
    Move_EE_Front = Move_EEtop("Move EE to front", np.pi/2)

    # Activate Suction Behavior
    deactivate_Suction = Deactive_suction("Deactivate Suction")

    # Set the value of the goal pos in the BlackBoard where patryk wants the robot to drop it, in this case to the origin
    set_goal_pos = py_trees.behaviours.SetBlackboardVariable(
        name="Set Drop position", 
        variable_name="Goal_pos", 
        variable_value= [3.0,1.0,-0.15],
        overwrite=True)
    
    # Sequence: move robot and move manipulator
    approach_object = py_trees.composites.Sequence(
        name = "Approach object",
        memory = True,
        children = [Move_robot_pos1,Move_manipulator]
    )

    # Parallel aruco detection and approach object
    parallel = py_trees.composites.Parallel(
        name = "AruCo detection and approach object",  
        policy = py_trees.common.ParallelPolicy.SuccessOnOne(),
        children = [Track_aruco, approach_object]
    )

    # Sequence for Move to object position
    Move_to_object = py_trees.composites.Sequence(
        name = "Move to Object",
        memory = True,
        children = [Rest_position1, parallel]
    )

    # Sequence for Picking the object
    Pick_up_object = py_trees.composites.Sequence(
        name = "Pick up Object",
        memory = True,
        children = [Move_EE_Down1, activate_Suction, Move_EE_High1, Move_EE_Top]
    )

    # Sequence for Moving to object drop off position
    Move_to_dropoff = py_trees.composites.Sequence(
        name = "Move to drop off",
        memory = True,
        children = [Move_robot_pos2, Move_EE_Front, Move_EE_High2]
    )

    # Sequence for Dropping the objects
    Drop_object = py_trees.composites.Sequence(
        name = "Drop Object",
        memory = True,
        children = [Move_EE_Down2, deactivate_Suction, Rest_position2]
    )

    # Sequence of all actions (Root Node)
    root = py_trees.composites.Sequence(
        name = "Intervention Task",
        memory = True,
        children = [Detect_aruco, Move_to_object, Pick_up_object, set_goal_pos, Move_to_dropoff, Drop_object]
    )

    return root

# Function to run the behavior tree
def run(it=1000):
    root = create_tree()

    try:
        print("Call setup for all tree children")
        root.setup_with_descendants() 
        print("Setup done!\n\n")
        py_trees.display.ascii_tree(root)
        
        for _ in range(it):
            root.tick_once()
            time.sleep(1)
            if root.status == py_trees.common.Status.SUCCESS:
                break
    except KeyboardInterrupt:
        exit()


if __name__ == "__main__":
    py_trees.logging.level = py_trees.logging.Level.DEBUG
    rospy.init_node("behavior_trees")

    # Create the behavior tree
    root = create_tree()
    
    # Display the behavior tree
    # py_trees.display.render_dot_tree(root, target_directory="catkin_ws/src/ho_intervention_project/img") # generates a figure for the 'root' tree.

    # Run the behavior tree
    run()