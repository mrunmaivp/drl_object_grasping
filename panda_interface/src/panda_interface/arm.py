#! /usr/bin/env python

import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from moveit_commander.conversions import pose_to_list

INTERPOLATION_STEP = 0.001
THRESHOLD = 0.0
INITIAL_JOINT_POSITIONS = [-0.0001914530812952009, -0.7856326455955855, -1.2635022375917515e-05, -2.355965542375113, 6.172411132432387e-06, 1.571755571494223, 0.7853925609812951]
JOINT_NAMES = ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7']
FINGER_JOINTS = ['panda_finger_joint1', 'panda_finger_joint2']

class ArmInterface(object):

    def __init__(self):
        self.arm_group = "panda_arm"
        self.hand_group = "hand"

        self.joint_names = JOINT_NAMES
        self.finger_joints = FINGER_JOINTS

        self.arm_interface = moveit_commander.MoveGroupCommander(self.arm_group)
        self.hand_interface = moveit_commander.MoveGroupCommander(self.hand_group)

        self.set_model_state_service = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.get_model_state_service = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)

    
    def get_current_gripper_width(self):
        hand_joint_values = self.hand_interface.get_current_joint_values()
        gripper_width = hand_joint_values[0] + hand_joint_values[1]
        rospy.logdebug("Current hand joint values", hand_joint_values[0], hand_joint_values[1])
        return gripper_width

    def move_to_pose(self, x, y, z, ex, ey, ez):
        pose_goal = geometry_msgs.msg.Pose()

        pose_goal.position.x = x
        pose_goal.position.y = y
        pose_goal.position.z = z

        orientation = tf.transformations.quaternion_from_euler(ex, ey, ez)

        pose_goal.orientation.x = orientation[0]
        pose_goal.orientation.y = orientation[1]
        pose_goal.orientation.z = orientation[2]
        pose_goal.orientation.w = orientation[3]

        waypoints = [pose]

        (plan, fraction) = self.arm_interface.compute_cartesian_path(waypoints, INTERPOLATION_STEP, THRESHOLD)

        return plan

    def get_joint_values(group):
        joint_angles = group.get_current_joint_angles()
        return joint_angles

    def move_to_joint_pose(joint_positions):

        msg = JointTrajectory()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = ''
        msg.joint_names = JOINT_NAMES
        
        point = JointTrajectoryPoint()
        point.positions = joint_positions
        point.velocities = []
        point.accelerations = []
        point.effort = []

        point.time_from_start = rospy.Duration(1)
        msg.points.append(point)

        rospy.loginfo(msg)

    def execute_plan(self, plan, group):
        success = group.plan(plan) == moveit_msgs.msg.MoveItErrorCodes.success
        rospy.loginfo("Motion Plan for End-Effector:", "SUCCESSED" if success else "FAILED")

        if success:
            group.execute(plan)
        







    