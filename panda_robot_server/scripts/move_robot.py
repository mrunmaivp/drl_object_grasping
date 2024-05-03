#! /usr/bin/env python

import sys
import copy
import rospy
import numpy as np
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from moveit_commander.conversions import pose_to_list
import tf.transformations as tf_trans
from moveit_msgs.srv import GetCartesianPathRequest
from moveit_msgs.srv import GetCartesianPath
from geometry_msgs.msg import Pose, Quaternion
from moveit_msgs.msg import MoveGroupAction

moveit_commander.roscpp_initialize(sys.argv)
rospy.init_node('move_panda_arm_moveit', anonymous=True)

robot = moveit_commander.RobotCommander()
scene = moveit_commander.PlanningSceneInterface()

group_name = "panda_arm"
group = moveit_commander.MoveGroupCommander(group_name)

hand_group = moveit_commander.MoveGroupCommander("panda_hand")

display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
                                               moveit_msgs.msg.DisplayTrajectory,
                                               queue_size=20)

planning_frame = group.get_planning_frame()

eef_link = group.get_end_effector_link()
print("============ eef_link:", eef_link)

group_names = robot.get_group_names()
print("============ group_names:", group_names)

print("============ Printing Robot Frame:")
print(robot.get_current_state())

print("================ Current Joint states", group.get_current_joint_values())

print("=================== Current POse" , group.get_current_pose().pose)

print("================ HAND Current Joint states", hand_group.get_current_joint_values())

current_pose = group.get_current_pose().pose
next_pose = copy.deepcopy(current_pose)
next_pose.position.x = 0.3
next_pose.position.y = 0.1
next_pose.position.z = 0.5

