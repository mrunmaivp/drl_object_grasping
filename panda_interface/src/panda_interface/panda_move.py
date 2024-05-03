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


INTERPOLATION_STEP = 0.01
THRESHOLD = 0.0
EPSILON = 0.005
GRIPPER_FORCE = 30
GRIPPER_SPEED = 0.1
GRIPPER_WIDTH_OPEN = 0.08
LOWER_LIMIT_X = 0.45
UPPPER_LIMIT_X = 0.55
LOWER_LIMIT_Y = -0.08
UPPER_LIMIT_Y = 0.08
LOWER_LIMIT_Z = 0.1
UPPER_LIMIT_Z = 0.25

JOINT_NAMES = ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7']

class MovePanda():

    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('move_panda_arm_moveit', anonymous=True)
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()

        self.arm_group = "panda_arm"
        self.hand_group = "panda_hand"

        self.arm_interface = moveit_commander.MoveGroupCommander(self.arm_group)
        self.hand_interface = moveit_commander.MoveGroupCommander(self.hand_group)

        self.gripper_move_publisher = rospy.Publisher('/franka_gripper/move/goal', MoveActionGoal, queue_size = 10)
        self.gripper_grasp_publisher = rospy.Publisher('/franka_gripper/grasp/goal', GraspActionGoal, queue_size = 10)

        self.joint_state_publisher = rospy.Publisher('/effort_joint_trajectory_controller/command', JointTrajectory, queue_size=10)

        self.tfBuffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tfBuffer)

        self.gripper_joint_forces = None
        self.gripper_subsriber = rospy.Subscriber('/franka_gripper/joint_states', JointState, self.grasp_state_callback)

        self.tcp_ft = None
        self.tcp_ft_buffer = []
        self.tcp_buffer_size = 20
        self.tcp_ft_subscriber = rospy.Subscriber('franka_state_controller/F_ext', WrenchStamped, self.tcp_ft_callback)


    def start_node(self):
        rospy.spin()

    def wait_time(self, waiting_time):
        rate = rospy.Rate(waiting_time)
        rate.sleep()
    
    def get_current_gripper_width(self):
        hand_joint_values = self.hand_interface.get_current_joint_values()
        gripper_width = hand_joint_values[0] + hand_joint_values[1]
        return gripper_width

    def grasp_state_callback(self, msg):
        joint_names = msg.name
        joint_positions = msg.position
        joint_velocities = msg.velocity
        joint_forces = msg.effort
        self.gripper_joint_forces = tuple(round(joint_force, 4) for joint_force in joint_forces)

    def get_current_joint_forces(self):
        return self.gripper_joint_forces

    def tcp_ft_callback(self, msg):
        fx = msg.wrench.force.x
        fy = msg.wrench.force.y
        fz = msg.wrench.force.z
        mx = msg.wrench.torque.x
        my = msg.wrench.torque.y
        mz = msg.wrench.torque.z

        self.tcp_ft_buffer.append([fx, fy, fz, mx, my, mz])
        if len(self.tcp_ft_buffer) > self.tcp_buffer_size:
            self.tcp_ft_buffer.pop(0)

        self.tcp_ft = np.mean(self.tcp_ft_buffer, axis=0)

    def get_tcp_ft(self):
        return np.array(self.tcp_ft)

    def move_to_pose(self, x, y, z, roll, pitch, yaw):
        pose_goal = geometry_msgs.msg.Pose()

        pose_goal.position.x = x
        pose_goal.position.y = y
        pose_goal.position.z = z

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

        return plan

    def move_to_next_pose(self, x, y, z):
        pose_goal = geometry_msgs.msg.Pose()

        pose_goal.position.x = x
        pose_goal.position.y = y
        pose_goal.position.z = z

        # orientation = tf_trans.quaternion_from_euler(roll, pitch, yaw)

        # pose_goal.orientation.x = orientation[0]
        # pose_goal.orientation.y = orientation[1]
        # pose_goal.orientation.z = orientation[2]
        # pose_goal.orientation.w = orientation[3]

        waypoints = [pose_goal]

        (plan, fraction) = self.arm_interface.compute_cartesian_path(waypoints, INTERPOLATION_STEP, THRESHOLD)

        return plan

    def get_joint_values(self, group):
        joint_angles = group.get_current_joint_angles()
        return joint_angles

    def move_to_joint_pose(self, joint_positions):
        client = actionlib.SimpleActionClient('/effort_joint_trajectory_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
        client.wait_for_server()

        joint_names = ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7']
        initial_joint_positions = [-0.0001914530812952009, -0.7856326455955855, -1.2635022375917515e-05, -2.355965542375113, 6.172411132432387e-06, 1.571755571494223, 0.7853925609812951]

        goal = FollowJointTrajectoryGoal()

        goal.trajectory.joint_names = joint_names

        point = JointTrajectoryPoint()
        point.positions = joint_positions
        point.velocities = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        point.time_from_start = rospy.Duration(5.0)

        goal.trajectory.points.append(point)

        client.send_goal(goal)

        client.wait_for_result()

        if client.get_state() == actionlib.GoalStatus.SUCCEEDED:
            rospy.loginfo("Trajectory execution succeeded")
        else:
            rospy.logwarn("Trajectory execution failed")

    def get_rotation_radian(self, rotation_degree):
        return rotation_degree / 360 * 2* math.pi

    def convert_quaternion_to_euler(self, quaternion):
        euler = tf_trans.euler_from_quaternion(quaternion)
        return np.array([euler[2], euler[1], euler[0]])

    def move_in_small_steps(self, x, y, z, roll, pitch, yaw):
        current_ee_pose = self.arm_interface.get_current_pose().pose
        next_ee_pose = copy.deepcopy(current_ee_pose)
        next_ee_pose.position.x += x
        next_ee_pose.position.y += y
        next_ee_pose.position.z += z
        out_of_workspace = next_ee_pose.position.x < LOWER_LIMIT_X or next_ee_pose.position.x > UPPPER_LIMIT_X or next_ee_pose.position.y < LOWER_LIMIT_Y or next_ee_pose.position.y > UPPER_LIMIT_Y or next_ee_pose.position.z < LOWER_LIMIT_Z or next_ee_pose.position.z > UPPER_LIMIT_Z 
        if out_of_workspace:
            return None

        current_ee_pose_orientation = np.array([current_ee_pose.orientation.x, current_ee_pose.orientation.y, current_ee_pose.orientation.z, current_ee_pose.orientation.w])
        current_ee_orientation = tf_trans.quaternion_from_matrix(tf_trans.quaternion_matrix(current_ee_pose_orientation))

        rotation = tf_trans.quaternion_from_euler(roll, pitch, yaw)


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

    
    def execute_plan(self, plan, group):
        execute_success = group.execute(plan)
        return execute_success

    def execution_feedback(self):
        pass

    def compute_orientation_difference(self, gripper_orientation, object_orientation):
        gripper_quat = quaternion_multiply(gripper_orientation, quaternion_inverse(object_orientation))
        gripper_roll, gripper_pitch, gripper_yaw = euler_from_quaternion(gripper_quat)
        return gripper_roll, gripper_pitch, gripper_yaw

    def get_ee_pose_tf(self):
        left_finger_pose = self.tfBuffer.lookup_transform('world', 'panda_leftfinger', rospy.Time(0), rospy.Duration(1.0))
        right_finger_pose = self.tfBuffer.lookup_transform('world', 'panda_rightfinger', rospy.Time(0), rospy.Duration(1.0))
        left_finger_position = [left_finger_pose.transform.translation.x, left_finger_pose.transform.translation.y, left_finger_pose.transform.translation.z]
        right_finger_position = [right_finger_pose.transform.translation.x, right_finger_pose.transform.translation.y, right_finger_pose.transform.translation.z]
        gripper_midpoint_pose = np.array([(left_finger_position[0] + right_finger_position[0])/2, (left_finger_position[1]+right_finger_position[1])/2, (left_finger_position[2]+right_finger_position[2])/2])
        gripper_orientation = np.array([left_finger_pose.transform.rotation.x, left_finger_pose.transform.rotation.y, left_finger_pose.transform.rotation.z, left_finger_pose.transform.rotation.w])
        return gripper_midpoint_pose, gripper_orientation

    def get_gripper_finger_orientation(self):
        left_finger = self.tfBuffer.lookup_transform('world', 'panda_leftfinger', rospy.Time(0), rospy.Duration(1.0))
        right_finger = self.tfBuffer.lookup_transform('world', 'panda_rightfinger', rospy.Time(0), rospy.Duration(1.0))
        left_finger_orientation = np.array([left_finger.transform.rotation.x, left_finger.transform.rotation.y, left_finger.transform.rotation.z, left_finger.transform.rotation.w])
        right_finger_orientation = np.array([right_finger.transform.rotation.x, right_finger.transform.rotation.y, right_finger.transform.rotation.z, right_finger.transform.rotation.w])
        return left_finger_orientation, right_finger_orientation

    def quaternion_multiply(self, q0, q1):
        x0 = q0[0]
        y0 = q0[1]
        z0 = q0[2]
        w0 = q0[3]

        x1 = q1[0]
        y1 = q1[1]
        z1 = q1[2]
        w1 = q1[3]

        q0q1_w = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1
        q0q1_x = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1
        q0q1_y = w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1
        q0q1_z = w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1

        final_quaternion = np.array([q0q1_w, q0q1_x, q0q1_y, q0q1_z])

        return final_quaternion

    def get_ee_pose(self):
        return self.arm_interface.get_current_pose().pose

    def open_gripper(self):
        move_action_goal = MoveActionGoal()
        move_action_goal.goal.speed = GRIPPER_SPEED
        move_action_goal.goal.width = 0.08

        self.gripper_move_publisher.publish(move_action_goal)

        current_gripper_width = self.get_current_gripper_width()
        return True


    def grasp(self, width, force):
        grasp_action_goal = GraspActionGoal()
        grasp_action_goal.goal.epsilon.inner = EPSILON
        grasp_action_goal.goal.epsilon.outer = EPSILON
        grasp_action_goal.goal.force = force
        grasp_action_goal.goal.speed = GRIPPER_SPEED
        grasp_action_goal.goal.width = width

        self.gripper_grasp_publisher.publish(grasp_action_goal)


    def close_gripper(self):
        move_action_goal = MoveActionGoal()
        move_action_goal.goal.speed = GRIPPER_SPEED
        move_action_goal.goal.width = 0.00

        self.gripper_move_publisher.publish(move_action_goal)
