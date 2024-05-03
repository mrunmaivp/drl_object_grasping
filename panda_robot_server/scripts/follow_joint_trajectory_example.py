#! /usr/bin/env python

import rospy
import actionlib
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

rospy.init_node('follow_joint_trajectory_panda')

client = actionlib.SimpleActionClient('/effort_joint_trajectory_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
client.wait_for_server()

joint_names = ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7']
# initial_joint_positions = [-0.0001914530812952009, -0.7856326455955855, -0.2635022375917515e-05, -2.355965542375113, 6.172411132432387e-06, 1.571755571494223, 0.7853925609812951]
# initial_joint_positions = [0.2920332555729077, 0.061510866826177235, -0.2963577346808357, -2.293474076394883, 0.025595937147190106, 2.354275229181834, 0.7632559123980425]
initial_joint_positions = [-0.0032674590170405082, 0.32823800587027563, -0.002367261320395997, -2.4649363984696544, 0.0031544200104445252, 2.7953975051930815, 0.777078248909441]

goal = FollowJointTrajectoryGoal()

goal.trajectory.joint_names = joint_names

point = JointTrajectoryPoint()
point.positions = [1.5708, -0.785398163, 0.0, -2.4243337162902208, 0.0, 1.57079632679, 0.785398163397]
point.velocities = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
point.time_from_start = rospy.Duration(5.0)

goal.trajectory.points.append(point)

client.send_goal(goal)

client.wait_for_result()

if client.get_state() == actionlib.GoalStatus.SUCCEEDED:
    rospy.loginfo("Trajectory execution succeeded")
else:
    rospy.logwarn("Trajectory execution failed")