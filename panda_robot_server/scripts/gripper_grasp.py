#!/usr/bin/env python

import rospy
import actionlib
from franka_gripper.msg import MoveAction, MoveGoal, GraspAction, GraspGoal

def main():
    rospy.init_node('franka_gripper_grasp_example')

    move_client = actionlib.SimpleActionClient('/franka_gripper/move', MoveAction)
    grasp_client = actionlib.SimpleActionClient('/franka_gripper/grasp', GraspAction)
    
    rospy.loginfo("Waiting for /franka_gripper/move action server...")
    move_client.wait_for_server()
    rospy.loginfo("Waiting for /franka_gripper/grasp action server...")
    grasp_client.wait_for_server()

    move_goal = MoveGoal()
    move_goal.width = 0.08  # Adjust the width as needed for your object
    move_goal.speed = 0.1 
    rospy.loginfo("Sending /franka_gripper/move action goal to open the gripper...")
    move_client.send_goal(move_goal)
    
    move_client.wait_for_result()
    
    if move_client.get_state() == actionlib.GoalStatus.SUCCEEDED:
        rospy.loginfo("Gripper is slightly open. Proceeding to grasp...")
        
        grasp_goal = GraspGoal()
        grasp_goal.width = 0.00# Adjust the width to fully close the gripper around the object
        grasp_goal.force = 5
        grasp_goal.speed = 0.1
        grasp_goal.epsilon.inner = 0.005
        grasp_goal.epsilon.outer = 0.005
        rospy.loginfo("Sending /franka_gripper/grasp action goal to grasp the object...")
        grasp_client.send_goal(grasp_goal)

        grasp_client.wait_for_result()
        
        if grasp_client.get_state() == actionlib.GoalStatus.SUCCEEDED:
            rospy.loginfo("Object successfully grasped.")
        else:
            rospy.logerr("Failed to grasp the object.")
    else:
        rospy.logerr("Failed to open the gripper for grasping.")

if __name__ == '__main__':
    main()
