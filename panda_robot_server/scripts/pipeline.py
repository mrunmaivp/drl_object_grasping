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
from sensor_msgs.msg import JointState
from franka_gripper.msg import GraspActionGoal
import actionlib
from franka_gripper.msg import MoveAction, MoveGoal, GraspAction, GraspGoal


moveit_commander.roscpp_initialize(sys.argv)
rospy.init_node('move_panda_arm_moveit', anonymous=True)

robot = moveit_commander.RobotCommander()
scene = moveit_commander.PlanningSceneInterface()

group_name = "panda_arm"
group = moveit_commander.MoveGroupCommander(group_name)

hand_group = moveit_commander.MoveGroupCommander("panda_hand")

planning_frame = group.get_planning_frame()

move_client = actionlib.SimpleActionClient('/franka_gripper/move', MoveAction)
move_client.wait_for_server()
move_goal = MoveGoal()
move_goal.width = 0.08
move_goal.speed = 0.1
timeout = rospy.Duration(10)
move_client.send_goal(move_goal)
move_client.wait_for_result(timeout=timeout)

hand_joint_values = hand_group.get_current_joint_values()
gripper_width = hand_joint_values[0] + hand_joint_values[1]
print("Hand joint values", hand_joint_values[0], hand_joint_values[1])
# rospy.logdebug("Current hand joint values", hand_joint_values[0], hand_joint_values[1])
print("gripper_width", gripper_width)

pose_goal = geometry_msgs.msg.Pose()

pose_goal.position.x = 0.5082890456013688
pose_goal.position.y = -0.0018683320816446241
pose_goal.position.z = 0.1852563388462632
pose_goal.orientation.x = -0.9239135044717155
pose_goal.orientation.y =  0.38259947445596704
pose_goal.orientation.z = -0.0011979978744919744
pose_goal.orientation.w = 0.00020785067692016053

waypoints = [pose_goal]
(plan, fraction) = group.compute_cartesian_path(waypoints, 0.001, 0.0)

execute_success = group.execute(plan, wait=True)
print("execution success", execute_success)

pose_goal = geometry_msgs.msg.Pose()

pose_goal.position.x = 0.5084322661344377
pose_goal.position.y = -0.0019603369081262825
pose_goal.position.z = 0.13539654278851604
pose_goal.orientation.x = -0.9239135044717155
pose_goal.orientation.y =  0.38259947445596704
pose_goal.orientation.z = -0.0011979978744919744
pose_goal.orientation.w = 0.00020785067692016053

waypoints = [pose_goal]
(plan, fraction) = group.compute_cartesian_path(waypoints, 0.001, 0.0)

execute_success = group.execute(plan, wait=True)
print("execution success", execute_success)

move_goal = MoveGoal()
move_goal.width = 0.00
move_goal.speed = 0.1
timeout = rospy.Duration(10)
move_client.send_goal(move_goal)
move_client.wait_for_result(timeout=timeout)

hand_joint_values = hand_group.get_current_joint_values()
gripper_width = hand_joint_values[0] + hand_joint_values[1]
# rospy.logdebug("Current hand joint values", hand_joint_values[0], hand_joint_values[1])
print("Hand joint values", hand_joint_values[0], hand_joint_values[1])
print("gripper_width", gripper_width)

grasp_client = actionlib.SimpleActionClient('/franka_gripper/grasp', GraspAction)
grasp_client.wait_for_server()

grasp_goal = GraspGoal()
grasp_goal.width = gripper_width
grasp_goal.force = 10
grasp_goal.speed = 0.1
grasp_goal.epsilon.inner = 0.005
grasp_goal.epsilon.outer = 0.005

grasp_client.send_goal(grasp_goal) 
grasp_client.wait_for_result(timeout=timeout)


