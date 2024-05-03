#! /usr/bin/env python

import rospy
import numpy as np

from panda_interface import ObjectPosition
from panda_interface import MovePanda

def move_robot():
    panda_robot = MovePanda()
    obj_position = ObjectPosition()

    plan = panda_robot.move_to_pose(0.30697425309106013, -5.334090225032015e-05, 0.5906692483699203, 1, 1, 1)
    execution_status = panda_robot.execute_plan(plan, panda_robot.arm_interface)

    obj_position.set_obj_state()

    open_gripper_status = panda_robot.open_gripper()
    # panda_robot.close_gripper()

    move_plan_success , move_to_new_pose_plan = panda_robot.move_in_small_steps(0.30, 0.2, 0.2, 1, 1, 1)
    execution_status = panda_robot.execute_plan(move_to_new_pose_plan, panda_robot.arm_interface)

    move2_to_new_pose_plan = panda_robot.move_to_pose(0.30, 0.4, 0.2, 1, 1, 1)
    execution_status = panda_robot.execute_plan(move2_to_new_pose_plan, panda_robot.arm_interface)

    move3_to_new_pose_plan = panda_robot.move_to_pose(0.30, 0.4, 0.4, 1, 1, 1)
    execution_status = panda_robot.execute_plan(move3_to_new_pose_plan, panda_robot.arm_interface)

    panda_robot.close_gripper()

    observation = get_new_observation(panda_robot, obj_position)
    print("Current Observation", observation)
    
def get_new_observation(panda_robot, obj_position):
        ee_current_pose = panda_robot.get_ee_pose()
        ee_current_position = np.array([ee_current_pose.position.x, ee_current_pose.position.y, ee_current_pose.position.z])
        ee_current_orientation = np.array([ee_current_pose.orientation.x, ee_current_pose.orientation.y, ee_current_pose.orientation.z, ee_current_pose.orientation.w])
        
        current_gripper_width = np.array(panda_robot.get_current_gripper_width())
        # current_obj_position = obj_position.obj_get_state()
        current_gripper_force = np.array(panda_robot.get_current_joint_forces()) 

        observation = [ee_current_position, ee_current_orientation, current_gripper_width, current_gripper_force]
        # print("OBSERVATION", observation)
        return observation

if __name__ == "__main__":
    rospy.init_node('move_panda')

    move_robot()

    rospy.spin()
