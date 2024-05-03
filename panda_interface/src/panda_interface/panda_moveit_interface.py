#! /usr/bin/env python

import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import trajectory_msgs.msg
import sys
import franka_gripper.msg
import actionlib
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import tf.transformations as tf_trans
from geometry_msgs.msg import Pose, Quaternion
from franka_gripper.msg import GraspActionGoal, GraspAction, MoveActionGoal
from std_msgs.msg import Duration
from sensor_msgs.msg import JointState
import copy
import numpy as np
from moveit_msgs.msg import MoveGroupActionFeedback
import tf2_ros
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import WrenchStamped
from tf.transformations import quaternion_inverse, quaternion_multiply, euler_from_quaternion, quaternion_from_euler

sys.path.append('/home/ros2/panda_grasping_ws/src/panda_grasping')
from panda_interface.srv import PlanAndExecute, PlanAndExecuteResponse

INTERPOLATION_STEP = 0.01
THRESHOLD = 0.0
EPSILON = 0.005
GRIPPER_FORCE = 30
GRIPPER_SPEED = 0.1
GRIPPER_WIDTH_OPEN = 0.08
LOWER_LIMIT_X = 0.40
UPPPER_LIMIT_X = 0.60
LOWER_LIMIT_Y = -0.15
UPPER_LIMIT_Y = 0.15
LOWER_LIMIT_Z = 0.1
UPPER_LIMIT_Z = 0.35

class MovePandaInterface():
    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        
        self.arm_group = "panda_arm"
        self.hand_group = "panda_hand"

        self.arm_interface = moveit_commander.MoveGroupCommander(self.arm_group)
        self.hand_interface = moveit_commander.MoveGroupCommander(self.hand_group)
        
        self.service = rospy.Service('plan_and_execute', PlanAndExecute, self.move_panda)
        self.object_lift = rospy.Service('ligt_object', PlanAndExecute, self.move_to_pose)
        self.randomize_robot_pose = rospy.Service('robot_pose_randomizer', PlanAndExecute, self.move_to_pose)
        
    def move_to_pose(self, request):
        current_ee_pose = self.arm_interface.get_current_pose().pose
        pose_goal = copy.deepcopy(current_ee_pose)
        # pose_goal = geometry_msgs.msg.Pose()

        pose_goal.position.x = request.x
        pose_goal.position.y = request.y
        pose_goal.position.z = request.z

        # orientation = quaternion_from_euler(roll, pitch, yaw)

        # pose_goal.orientation.x = orientation[0]
        # pose_goal.orientation.y = orientation[1]
        # pose_goal.orientation.z = orientation[2]
        # pose_goal.orientation.w = orientation[3]

        pose_goal.orientation.x = -0.9239135044717155
        pose_goal.orientation.y =  0.38259947445596704
        pose_goal.orientation.z = -0.0011979978744919744
        pose_goal.orientation.w = 0.00020785067692016053

        waypoints = [pose_goal]

        (plan, fraction) = self.arm_interface.compute_cartesian_path(waypoints, INTERPOLATION_STEP, THRESHOLD)
        
        if plan != None:
            out_of_workspace = False
            execute_success = self.arm_interface.execute(plan)
        else:
            out_of_workspace = True 
            execute_success = False
        
        response = PlanAndExecuteResponse(execute_success, out_of_workspace)
            
        return response
        
    def move_in_small_steps(self, request):
        current_ee_pose = self.arm_interface.get_current_pose().pose
        next_ee_pose = copy.deepcopy(current_ee_pose)
        next_ee_pose.position.x += request.x
        next_ee_pose.position.y += request.y
        next_ee_pose.position.z += request.z
        out_of_workspace = next_ee_pose.position.x < LOWER_LIMIT_X or next_ee_pose.position.x > UPPPER_LIMIT_X or next_ee_pose.position.y < LOWER_LIMIT_Y or next_ee_pose.position.y > UPPER_LIMIT_Y or next_ee_pose.position.z < LOWER_LIMIT_Z or next_ee_pose.position.z > UPPER_LIMIT_Z 
        print("out of workspace", out_of_workspace)
        if out_of_workspace:
            return None

        current_ee_pose_orientation = np.array([current_ee_pose.orientation.x, current_ee_pose.orientation.y, current_ee_pose.orientation.z, current_ee_pose.orientation.w])
        current_ee_orientation = tf_trans.quaternion_from_matrix(tf_trans.quaternion_matrix(current_ee_pose_orientation))

        rotation = tf_trans.quaternion_from_euler(request.roll, request.pitch, request.yaw)


        new_ee_orientation = tf_trans.quaternion_multiply(rotation, current_ee_orientation)
        
        normalized_new_ee_orientation = new_ee_orientation / np.linalg.norm(new_ee_orientation)

        new_ee_quaternion = Quaternion()
        new_ee_quaternion.x = normalized_new_ee_orientation[0]
        new_ee_quaternion.y = normalized_new_ee_orientation[1]
        new_ee_quaternion.z = normalized_new_ee_orientation[2]
        new_ee_quaternion.w = normalized_new_ee_orientation[3]

        next_ee_pose.orientation = new_ee_quaternion

        self.arm_interface.set_max_velocity_scaling_factor(0.5)

        waypoints = [next_ee_pose]

        (plan, fraction) = self.arm_interface.compute_cartesian_path(waypoints, INTERPOLATION_STEP, THRESHOLD)
        return plan

    def move_panda(self, request):
        plan = self.move_in_small_steps(request)
        
        if plan != None:
            out_of_workspace = False
            execute_success = self.arm_interface.execute(plan)
        else:
            out_of_workspace = True 
            execute_success = False
        
        response = PlanAndExecuteResponse(execute_success, out_of_workspace)
            
        return response 
        
    def execute_plan(self, plan, group):
        execute_success = group.execute(plan)
        return execute_success
        
def main():
    rospy.init_node('moveit_manager')
    move_panda = MovePandaInterface()
    rospy.spin()
        
        
if __name__ == '__main__':
    main()