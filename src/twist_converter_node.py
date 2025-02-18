#!/usr/bin/env python3

'''
    Node to convert velocity command in Twist into wheel velocities.
    This node is needed because in Stonefish, the wheel velocities are controlled by a Float64MultiArray message.
    Format of message: [vl, vr]

    With this node running, the turtlebot can be controlled by teleop or rqt_robot_steering.
'''

import rospy
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Twist

class Controller:
    def __init__(self) -> None:
        # initialize robot velocity
        self.v = 0
        self.w = 0

        # initialize wheel velocities
        self.wl = 0 # left wheel
        self.wr = 0 # right wheel

        # robot properties
        self.wheel_radius = 0.035 #m
        self.wheelbase = 0.235 #m

        # Publisher
        self.wheel_vel_pub = rospy.Publisher('/turtlebot/kobuki/commands/wheel_velocities',Float64MultiArray,queue_size=10)

        # Subscriber
        self.twist_sub = rospy.Subscriber('/cmd_vel', Twist, self.twist_callback)

    def twist_callback(self,twist_msg):
        # extract robot velocity
        self.v = twist_msg.linear.x
        self.w = twist_msg.angular.z
        
        # convert to wheel angular velocities
        self.wr = (self.v + (self.w * self.wheelbase)/2)/self.wheel_radius
        self.wl = (self.v - (self.w * self.wheelbase)/2)/self.wheel_radius

        # create wheel velocity message
        wheel_msg = Float64MultiArray()
        wheel_msg.data = [self.wl,self.wr]
        
        # publish wheel velocities
        self.wheel_vel_pub.publish(wheel_msg)

    

if __name__=='__main__':
    rospy.init_node('twist_to_wheel_vel', anonymous=True) # initialize the node
    node = Controller()
    

    rospy.spin()