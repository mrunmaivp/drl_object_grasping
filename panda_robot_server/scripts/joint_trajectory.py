#! /usr/bin/env python

import rospy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import random

def publish_joint_states():
    rospy.init_node('panda_publish_joint_states')

    joint_state_publisher = rospy.Publisher('/effort_joint_trajectory_controller/command', JointTrajectory, queue_size=10)
    while not rospy.is_shutdown():
	
        msg = JointTrajectory()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = ''
        msg.joint_names = ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7']

        point = JointTrajectoryPoint()
        j1 = random.random()
        j2 = random.random()
        j3 = random.random()
        j4 = random.random()
        j5 = random.random()
        j6 = random.random()
        j7 = random.random()

        # point.positions = [j1, j2, j3, j4, j5, j6, j7]
        point.positions = [1.5708, 0.0, 0.0, -1.9708, 0.0, 0.0, 0.0,]
        point.accelerations = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
        point.effort = []

        point.time_from_start = rospy.Duration(5)

        msg.points.append(point)

        joint_state_publisher.publish(msg)

        rospy.loginfo(msg)

	
if __name__ == '__main__':
    publish_joint_states()