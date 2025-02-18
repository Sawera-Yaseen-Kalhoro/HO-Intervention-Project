#!/usr/bin/env python3

import rospy
import numpy as np
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, PoseArray, PointStamped
import tf

class ArUcoDetector:
    def __init__(self) -> None:
        self.tf_listener = tf.TransformListener()

        self.image = np.zeros((1,1))
        self.camera_intrinsics = np.zeros((3,3))
        self.camera_distortions = np.zeros((5,))
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
        self.aruco_marker_length = 0.05 #m 
        self.aruco_pose_list = []
        self.pose_array_msg = PoseArray()
        self.marker_ids = None

        # Subscribers
        self.image_sub = rospy.Subscriber("/turtlebot/kobuki/realsense/color/image_color", Image, self.image_callback)
        self.intrinsics_sub = rospy.Subscriber("/turtlebot/kobuki/realsense/color/camera_info", CameraInfo, self.intrinsics_callback)
        
        # Publishers
        # self.pose_pub = rospy.Publisher("/aruco_pose", PoseStamped, queue_size=1)
        self.box_pos_pub = rospy.Publisher("/target_position", PointStamped, queue_size=1)
        self.pose_array_pub = rospy.Publisher("/aruco_pose_array", PoseArray, queue_size=1)  # New publisher for PoseArray

        #Timers
        self.timer = rospy.Timer(rospy.Duration(0.01), self.send_messages)


    def image_callback(self, image_msg):
        # save timestamp
        self.current_time = image_msg.header.stamp

        # convert image msg into array
        self.encoding = image_msg.encoding
        bridge = CvBridge()
        self.image = bridge.imgmsg_to_cv2(image_msg, self.encoding)

        # detect ArUco markers
        marker_corners, self.marker_ids = self.detect_aruco()

        # ArUco marker pose estimation
        if self.marker_ids is not None:
            self.aruco_pose_list = self.estimate_aruco_pose(marker_corners, self.marker_ids)

            # get position of the top of the box based on the detected ArUco markers
            marker_to_box = np.array([0,0.15/2,-0.055/2,1.0]).reshape((4,1))
            box_top_pos_list = []
            for marker_transform in self.aruco_pose_list:
                box_top_pos = marker_transform @ marker_to_box
                # convert from homogenous to cartesian coordinate
                box_top_pos = np.array([[box_top_pos[0,0]/box_top_pos[-1,0]],
                                        [box_top_pos[1,0]/box_top_pos[-1,0]],
                                        [box_top_pos[2,0]/box_top_pos[-1,0]]])
                box_top_pos_list.append(box_top_pos)

            # take average in case more than 1 markers are detected
            self.box_top_position = np.zeros((3,1))
            self.box_top_position[0,0] = np.mean([pos[0,0] for pos in box_top_pos_list])
            self.box_top_position[1,0] = np.mean([pos[1,0] for pos in box_top_pos_list])
            self.box_top_position[2,0] = np.mean([pos[2,0] for pos in box_top_pos_list])


    def detect_aruco(self):
        aruco_params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(self.aruco_dict, aruco_params)
        marker_corners, marker_ids, _ = detector.detectMarkers(self.image)
        return marker_corners, marker_ids
    
    
    def estimate_aruco_pose(self, marker_corners, marker_ids):
        object_points = np.zeros((4,1,3))
        object_points[0] = np.array((-self.aruco_marker_length/2, self.aruco_marker_length/2, 0)).reshape((1,3))
        object_points[1] = np.array((self.aruco_marker_length/2, self.aruco_marker_length/2, 0)).reshape((1,3))
        object_points[2] = np.array((self.aruco_marker_length/2, -self.aruco_marker_length/2, 0)).reshape((1,3))
        object_points[3] = np.array((-self.aruco_marker_length/2, -self.aruco_marker_length/2, 0)).reshape((1,3))

        marker_pose_list_world_ned = []

        for i in range(len(marker_ids)):
            _, rvec, tvec = cv2.solvePnP(object_points, marker_corners[i], self.camera_intrinsics, self.camera_distortions)
            
            R, _ = cv2.Rodrigues(rvec)
            
            T = np.zeros((4,4))
            T[:3, :3] = R
            T[:3, 3] = tvec.flatten()
            T[3,3] = 1

            try:
                (trans, rot) = self.tf_listener.lookupTransform('world_ned', 'turtlebot/kobuki/realsense_color', rospy.Time(0))
                camera_to_world_translation = np.array(trans)
                camera_to_world_rotation = tf.transformations.quaternion_matrix(rot)
                camera_to_world_transform = np.eye(4)
                camera_to_world_transform[:3, :3] = camera_to_world_rotation[:3, :3]
                camera_to_world_transform[:3, 3] = camera_to_world_translation

                marker_pose_world_ned = np.dot(camera_to_world_transform, T)

                marker_pose_list_world_ned.append(marker_pose_world_ned)
                    
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                rospy.logwarn("Failed to transform marker pose to world_ned frame.")

        return marker_pose_list_world_ned

        

    def intrinsics_callback(self, intrinsics_msg):
        if not self.camera_intrinsics.any():
            self.camera_intrinsics = np.array(intrinsics_msg.K).reshape((3,3))
            self.camera_distortions = np.array(intrinsics_msg.D)

    
    def send_messages(self, event):
        if self.aruco_pose_list:
            # publish visualization of detected markers
            self.publish_markers_vis()
            # publish position of box
            self.publish_box_pos()

    def publish_markers_vis(self):
        self.pose_array_msg.poses = []

        for i, marker_pose_world_ned in enumerate(self.aruco_pose_list):
            # Create PoseStamped message
            pose_msg = PoseStamped()
            pose_msg.header.stamp = self.current_time
            pose_msg.header.frame_id = "world_ned" 
            pose_msg.pose.position.x = marker_pose_world_ned[0, 3]
            pose_msg.pose.position.y = marker_pose_world_ned[1, 3]
            pose_msg.pose.position.z = marker_pose_world_ned[2, 3]

            rotation_matrix = np.zeros((4,4))
            rotation_matrix[:3,:3] = marker_pose_world_ned[:3, :3]
            rotation_matrix[-1,-1] = 1
            quaternion = tf.transformations.quaternion_from_matrix(rotation_matrix)
            pose_msg.pose.orientation.x = quaternion[0]
            pose_msg.pose.orientation.y = quaternion[1]
            pose_msg.pose.orientation.z = quaternion[2]
            pose_msg.pose.orientation.w = quaternion[3]

            # self.pose_pub.publish(pose_msg)

            # Append PoseStamped message to PoseArray
            self.pose_array_msg.poses.append(pose_msg.pose)

        # Publish the PoseArray
        self.pose_array_msg.header.stamp = self.current_time
        self.pose_array_msg.header.frame_id = "world_ned"
        self.pose_array_pub.publish(self.pose_array_msg)

    def publish_box_pos(self):
        box_pos_msg = PointStamped()
        box_pos_msg.header.frame_id = "world_ned"
        box_pos_msg.header.stamp = self.current_time

        box_pos_msg.point.x = self.box_top_position[0,0]
        box_pos_msg.point.y = self.box_top_position[1,0]
        box_pos_msg.point.z = self.box_top_position[2,0]

        self.box_pos_pub.publish(box_pos_msg)


if __name__=='__main__':
    rospy.init_node('aruco_pose_to_range', anonymous=False) # initialize the node
    node = ArUcoDetector()

    rospy.spin()