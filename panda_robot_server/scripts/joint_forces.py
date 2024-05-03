#! /usr/bin/env python

import rospy
from sensor_msgs.msg import JointState
import actionlib
from franka_gripper.msg import GraspActionGoal, MoveActionGoal, GraspAction

def joint_state_callback(msg):
    joint_names = msg.name
    joint_positions = msg.position
    joint_velocities = msg.velocity
    joint_forces = msg.effort

    rounded_joint_forces = tuple(round(joint_force, 4) for joint_force in joint_forces)

    print("Joint Forces:" , rounded_joint_forces, type(rounded_joint_forces))
    
    # rospy.signal_shutdown("Recieved Forces, Shutting Down Subscriber")

class GripperMovement():
    def __init__(self):
        self.gripper_move_publisher = rospy.Publisher('/franka_gripper/move/goal', MoveActionGoal, queue_size = 10)
        self.gripper_grasp_publisher = rospy.Publisher('/franka_gripper/grasp/goal', GraspActionGoal, queue_size = 10)

    def open_gripper(self):
        move_action_goal = MoveActionGoal()
        move_action_goal.goal.speed = 0.1
        move_action_goal.goal.width = 0.08

        self.gripper_move_publisher.publish(move_action_goal)

    def grasp(self, width, force):
        grasp_action_goal = GraspActionGoal()
        grasp_action_goal.goal.epsilon.inner = 0.005
        grasp_action_goal.goal.epsilon.outer = 0.005
        grasp_action_goal.goal.force = force
        grasp_action_goal.goal.speed = 0.1
        grasp_action_goal.goal.width = width

        self.gripper_grasp_publisher.publish(grasp_action_goal)


if __name__ == "__main__":
    rospy.init_node('gripper_force_listener')

    gripper = GripperMovement()
    gripper.open_gripper()
    rospy.spin()
