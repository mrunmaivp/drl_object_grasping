import gymnasium as gym
from gymnasium import spaces
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
import rospy
import sys
import subprocess
import random
import os
sys.path.append('/home/ros2/panda_grasping_ws/src/panda_grasping')

from panda_interface.src.panda_interface import ObjectPosition
from panda_interface.src.panda_interface import MovePanda
from panda_interface.src.panda_interface import GripperForceListener
import math
from tf.transformations import quaternion_inverse, quaternion_multiply, euler_from_quaternion,quaternion_from_euler																													

INITIAL_JOINT_POSITIONS = [0.028321078208972672, 0.25952951166771854, -0.007253497323791436, -2.4243337162902208, 0.0028962027517787092, 2.685928661651886, 0.8039933070393666]
RANDOM_JOINT_OFFSET = []
FORCE_REWARD = 100
POSITION_DIFFERENCE_FACTOR = 0.06
TRANSLATION_SCALE_FACTOR = 0.025
DISTANCE_THRESHOLD = 0.1
ROTATION_SCALE_FACTOR = 0.2
OBJECT_HEIGHT = 0.09
FORCE_THRESHOLD = 0.15
OBJECT_SDF_PATH = "/home/ros2/panda_grasping_ws/src/panda_grasping/load_rl_env/urdf"

POSITION_DIFFERENCE_FACTOR = 0.06
ORIENTATION_DIFFERENCE_FACTOR = 0.04

class PandaGraspingEnv(gym.Env):

    def __init__(self):

        self.panda_robot = MovePanda()
        self.obj_position = ObjectPosition()
        self.panda_gripper = GripperForceListener()

        self.initial_pose = INITIAL_JOINT_POSITIONS

        self.observation_space = self._get_observation_space()
        self.action_space = self._get_action_space()

        self.execution_success = False

        self.spawned_object = ""


    def _get_observation_space(self):
        lower_relative_position = np.array([-1, -1, -1])
        upper_relative_position = np.array([1, 1, 1])

        lower_relative_orientation = np.array([-(math.pi)/4, -(math.pi/4), -(math.pi)])
        upper_relative_orientaiton = np.array([math.pi/4, math.pi/4, math.pi])

        lower_obj_distance = np.full((1), 0)
        upper_obj_distance = np.full((1), 1.0)

        #lower_obj_orientation = np.full((4), -1.0)
        #upper_obj_orientation = np.full((4), 1.0)

        lower_gripper_force = np.array([5.0, 5.0])
        upper_gripper_force = np.array([15.0, 15.0])

        lower_tcp_ft = np.full((6), -20)
        upper_tcp_ft = np.full((6), 20)

        lower_obj_grasped = np.array([-1])
        upper_obj_grasped = np.array([1])

        lower_limits = np.concatenate((lower_relative_position, lower_relative_orientation, lower_obj_distance , lower_gripper_force, lower_tcp_ft, lower_obj_grasped))
        upper_limits = np.concatenate((upper_relative_position, upper_relative_orientaiton, upper_obj_distance, upper_gripper_force, upper_tcp_ft, upper_obj_grasped))

        return spaces.Box(low=lower_limits, high=upper_limits, shape=(16,), dtype=np.float32)

    def _get_action_space(self):
        lower_relative_position = np.array([-1, -1, -1])
        upper_relative_position = np.array([1, 1, 1])

        lower_relative_orientation = np.array([-(math.pi)/4, -(math.pi/4), -(math.pi)])
        upper_relative_orientaiton = np.array([math.pi/4, math.pi/4, math.pi])

        lower_gripper_force = np.array([10.0])
        upper_gripper_force = np.array([30.0])

        return spaces.Box(low=np.concatenate((lower_relative_position, lower_relative_orientation, lower_gripper_force)), 
                            high=np.concatenate((upper_relative_position, upper_relative_orientaiton, upper_gripper_force)), shape=(7,), dtype=np.float32)

    def _set_panda_initial_pose(self, initial_joint_state=INITIAL_JOINT_POSITIONS):
        self.panda_robot.move_to_joint_pose(initial_joint_state)

    def _set_obj_initial_pose(self):
        self.obj_position.set_obj_state()
    
    def spawn_random_object(self):
        if self.spawned_object:
            self.obj_position.delete_object(self.spawned_object)
        object_path_list = ["cube/model.sdf", "cylinder/model.sdf"]
        random_selected_object = random.choice(object_path_list)
        object_name = random_selected_object.split("/")[0]
        random_object_sdf_path = os.path.join(OBJECT_SDF_PATH, random_selected_object)
        subprocess.call(["rosrun", "gazebo_ros", "spawn_model", "-sdf", "-model", object_name, "-file", random_object_sdf_path])
        return object_name 

    def reset(self, seed=None, options=None):
        self.spawned_object = self.spawn_random_object()
        self.obj_position.initialize_world()
        self.panda_gripper.execute_open_gripper()
        self.panda_robot.move_to_joint_pose(INITIAL_JOINT_POSITIONS)
        self.obj_position.randomize_object_position(self.spawned_object)
        
        observation = self._get_new_observation()
        obj_grasped = np.array([-1])
        observation = [observation, obj_grasped]
        observation = np.concatenate(observation)
        return observation, {}

    def _compute_relative_orientation(self, obj_orientation, gripper_orientation):
        relative_orientation = quaternion_multiply(obj_orientation, quaternion_inverse(gripper_orientation))
        relative_euler = euler_from_quaternion(relative_orientation)
        return relative_euler

    def _get_new_observation(self):
        gripper_position, gripper_orientation = self.panda_robot.get_ee_pose_tf()
        gripper_euler = self.panda_robot.convert_quaternion_to_euler(gripper_orientation)
        
        current_obj_position, current_obj_orientation = self.obj_position.obj_get_state(self.spawned_object)
        relative_position = current_obj_position - gripper_position
        relative_orientation = self._compute_relative_orientation(current_obj_orientation, gripper_orientation)
        current_distance_between_ee_object = np.array([self.calc_dist(current_obj_position, gripper_position)])

        current_gripper_width = np.array([self.panda_robot.get_current_gripper_width()])
        
        current_gripper_force = np.array(self.panda_robot.get_current_joint_forces()) 
        tcp_ft = self.panda_robot.get_tcp_ft()
        observation = [relative_position, relative_orientation, current_distance_between_ee_object, current_gripper_force, tcp_ft]
        observation = np.concatenate(observation)
        return observation

    def calc_dist(self,p1,p2):
        x_d = math.pow(p1[0] - p2[0],2)
        y_d = math.pow(p1[1] - p2[1],2)
        z_d = math.pow(p1[2] - p2[2],2)
        d = math.sqrt(x_d + y_d + z_d)

        return d

    def step(self, action):
        x = TRANSLATION_SCALE_FACTOR * float(action[0])
        y = TRANSLATION_SCALE_FACTOR * float(action[1])
        z = TRANSLATION_SCALE_FACTOR * float(action[2])
        er = ROTATION_SCALE_FACTOR * float(action[3])
        ep = ROTATION_SCALE_FACTOR * float(action[4])
        ey = ROTATION_SCALE_FACTOR * float(action[5])
        gripper_force = action[6]

        done = False
        truncated = False
        out_of_workspace_flag = False
        execution_status = False
        is_optimal_grasp = False
        is_object_lifted = False

        self.panda_gripper.execute_open_gripper()
        plan = self.panda_robot.move_in_small_steps(x, y, z, er, ep, ey)
        if plan == None:
            out_of_workspace_flag = True
        else:
            execution_status = self.panda_robot.execute_plan(plan, self.panda_robot.arm_interface)

        if out_of_workspace_flag == False and execution_status == False:
            done = True
            truncated = True

        is_optimal_grasp, check_obj_position = self.grasp_object(action)

        next_observation = self._get_new_observation()

        angle_difference = self.check_approach_angle()
        orientation_difference_value = self.check_angle_difference(angle_difference)

        if is_optimal_grasp:
            is_object_lifted = self._lifting_object()
            done = True
            truncated = True

        if check_obj_position:
            done = True
            truncated = True

        obj_grasped = np.array([1]) if is_object_lifted else np.array([-1])
        next_observation = [next_observation, obj_grasped]
        next_observation = np.concatenate(next_observation)

        reward = self.calculate_reward(done, is_optimal_grasp, is_object_lifted, next_observation, check_obj_position, execution_status, orientation_difference_value, out_of_workspace_flag)
        return next_observation, reward, done, truncated, {}

    def grasp_object(self, action):
        is_object_grasp = False
        is_balanced = False
        check_obj_position = self.obj_position.check_obj_position_change(self.spawned_object)
        if check_obj_position:
            return is_object_grasp, check_obj_position
        else:
            gripper_position, gripper_orientation = self.panda_robot.get_ee_pose_tf()
            object_position = self.obj_position.object_initial_pose[:3]
            is_gripper_x_close = math.isclose(gripper_position[0], object_position[0], abs_tol=0.015)
            is_gripper_y_close = math.isclose(gripper_position[1], object_position[1], abs_tol=0.015)
            gripper_width = 0.05
            if is_gripper_x_close and is_gripper_y_close and gripper_position[2] < OBJECT_HEIGHT:
                ft_values_before = self.panda_robot.get_tcp_ft()
                self.panda_gripper.execute_gripper_action(action[-1], gripper_width, self.spawned_object)
                ft_values = self.panda_robot.get_tcp_ft()
                after_gripper_width = self.panda_robot.get_current_gripper_width()
                after_gripper_forces = self.panda_robot.get_current_joint_forces()
                is_object_grasp = False if any(abs(ft_values[3:]) > FORCE_THRESHOLD) else True

            return is_object_grasp, False

    def _grasping_object(self, action):
        optimal_grasp = False
        gripper_position, gripper_orientation = self.panda_robot.get_ee_pose_tf()
        object_position = self.obj_position.object_initial_pose[:3]
        is_gripper_x_close = math.isclose(gripper_position[0], object_position[0], abs_tol=0.015)
        is_gripper_y_close = math.isclose(gripper_position[1], object_position[1], abs_tol=0.015)
        object_height = 0.05
        gripper_forces = self.panda_robot.get_current_joint_forces()
        if is_gripper_x_close and is_gripper_y_close and gripper_position[2] < OBJECT_HEIGHT:
            self.panda_gripper.execute_close_gripper()
            gripper_width = self.panda_robot.get_current_gripper_width()
            print("---------------------------GRASPING THE OBJECT--------------------------", gripper_width)
            self.panda_gripper.execute_gripper_action(action[-1], gripper_width, self.spawned_object)
            ft_values = self.panda_robot.get_tcp_ft()
            after_gripper_forces = self.panda_robot.get_current_joint_forces()
            optimal_grasp = False if any(abs(ft_values[3:]) > FORCE_THRESHOLD) else True
           

        return optimal_grasp


    def check_angle_difference(self, orientation_difference):
        scaling_factor = 1.0
        desired_differences = [0, np.pi, 2*np.pi] 
        closest_desired_difference = min(desired_differences, key=lambda x: abs(orientation_difference - x))

        if np.isclose(orientation_difference, closest_desired_difference, atol=0.01):
            angular_difference_value = scaling_factor
        else:
            angular_difference_value = scaling_factor / (1 + abs(closest_desired_difference - orientation_difference))
        return angular_difference_value

    def check_approach_angle(self):
        left_finger_orientation, right_finger_orientation = self.panda_robot.get_gripper_finger_orientation()
        current_obj_position, current_obj_orientation = self.obj_position.obj_get_state(self.spawned_object)

        quat_difference = np.dot(left_finger_orientation, np.conjugate(current_obj_orientation))

        angle_difference = 2 * np.arccos(abs(quat_difference))
        return angle_difference

    def _lifting_object(self):
        is_optimal_grasp = False
        object_position = self.obj_position.object_initial_pose[:3]
        gripper_position, gripper_orientation = self.panda_robot.get_ee_pose_tf()
        self.lift_object(gripper_position)
        current_obj_position, current_obj_orientation = self.obj_position.obj_get_state(self.spawned_object)

        if round(current_obj_position[2], 2) > round(object_position[2], 2):
            print("Lifted Object Observation", round(current_obj_position[2], 2), round(object_position[2], 2))
            is_optimal_grasp = True
            return is_optimal_grasp

    def lift_object(self, next_state):
        x = next_state[0]
        y = next_state[1]
        z = 0.2
        er = 0
        ep = 0
        ey = 0
        plan = self.panda_robot.move_to_pose(x, y, z, er, ep, ey)
        print("=========LIFTING THE OBJECT===========")
        execution_success = self.panda_robot.execute_plan(plan, self.panda_robot.arm_interface)

    def calculate_reward(self, done, is_optimal_grasp, is_object_lifted, next_observation, check_obj_position, execution_status, orientation_difference_value, out_of_workspace_flag):
        reward = 0
        next_ee_object_distance = next_observation[6]
        gripper_position, gripper_orientation = self.panda_robot.get_ee_pose_tf()
        object_position = self.obj_position.object_initial_pose[:3]
        is_gripper_x_close = math.isclose(gripper_position[0], object_position[0], abs_tol=0.015)
        is_gripper_y_close = math.isclose(gripper_position[1], object_position[1], abs_tol=0.015)

        if not done:

            if out_of_workspace_flag:
                print("OUT OF WORKPSACE")
                reward = -0.1

            elif is_gripper_x_close and is_gripper_y_close and gripper_position[2] < OBJECT_HEIGHT:
                print("-------------------- VERY CLOSE TO THE OBJECT --------------------")
                reward = 20

            else:
                reward = POSITION_DIFFERENCE_FACTOR * (1/next_ee_object_distance) + ORIENTATION_DIFFERENCE_FACTOR * orientation_difference_value
                print("APPROACHING OBJECT", reward)

        else:
            if is_object_lifted:
                print("OBJECT GRASPED OPTIMALLY")
                reward = 100

            elif is_optimal_grasp:
                print("FT TRUE")
                reward = 50

            elif check_obj_position:
                print(" ....... OBJECT MOVED MISTAKENLY .........")
                reward = 0

            elif out_of_workspace_flag == False and execution_status == False:
                print("ABORTED Execution")
                reward = -20

            else:
                print("NOT OPTIMAL GRASP")
                reward = 0

        return reward

    def compute_force(self, next_gripper_force, input_force):
        gripper_force_difference = abs(next_gripper_force[0] - next_gripper_force[1])
        d = input_force - gripper_force_difference
        force_fraction = d / input_force
        return force_fraction
