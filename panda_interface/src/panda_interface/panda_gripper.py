#! /usr/bin/env python

import rospy
from sensor_msgs.msg import JointState
from franka_gripper.msg import GraspActionGoal
import actionlib
from franka_gripper.msg import MoveAction, MoveGoal, GraspAction, GraspGoal
from std_msgs.msg import Float64

class GripperForceListener():
    def __init__(self):
        pass

    def ft_reward_callback(self, msg):
        self.ft_reward = msg.data

    def get_ft_reward(self):
        return self.ft_reward

    def grasp_state_callback(self, msg):
        joint_names = msg.name
        joint_positions = msg.position
        joint_velocities = msg.velocity
        joint_forces = msg.effort

        self.gripper_joint_forces = tuple(round(joint_force, 4) for joint_force in joint_forces)

    def get_current_joint_forces(self):
        gripper_subsriber = rospy.Subscriber('/franka_gripper/joint_states', JointState, self.grasp_state_callback)
        return self.gripper_joint_forces

    def execute_open_gripper(self):
        move_client = actionlib.SimpleActionClient('/franka_gripper/move', MoveAction)
        move_client.wait_for_server()
        move_goal = MoveGoal()
        move_goal.width = 0.07
        move_goal.speed = 0.1
        timeout = rospy.Duration(1)
        move_client.send_goal(move_goal)
        move_client.wait_for_result(timeout=timeout)

    def execute_close_gripper(self):
        move_client = actionlib.SimpleActionClient('/franka_gripper/move', MoveAction)
        move_client.wait_for_server()
        move_goal = MoveGoal()
        move_goal.width = 0.00
        move_goal.speed = 0.1
        timeout = rospy.Duration(2)
        move_client.send_goal(move_goal)
        move_client.wait_for_result(timeout=timeout)

    def execute_gripper_action(self, force, width, object_name):
        if object_name == "cube" or object_name == "cylinder":
            gripper_width = 0.05
        else:
            gripper_width = 0.032
        grasp_client = actionlib.SimpleActionClient('/franka_gripper/grasp', GraspAction)
        grasp_client.wait_for_server()
        grasp_goal = GraspGoal()
        grasp_goal.width = gripper_width
        grasp_goal.force = force
        grasp_goal.speed = 0.1
        grasp_goal.epsilon.inner = 0.005
        grasp_goal.epsilon.outer = 0.005
        timeout = rospy.Duration(3)
        
        rospy.loginfo("Sending /franka_gripper/grasp action goal to grasp the object...")
                
        grasp_client.send_goal(grasp_goal) 
        grasp_client.wait_for_result(timeout=timeout)
                