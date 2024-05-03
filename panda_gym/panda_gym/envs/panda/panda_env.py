import gym
from gym import spaces
from gym.utils import seeding
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
import rospy
import sys
import subprocess
import os
import random
import time

sys.path.append('/home/ros2/mt_panda_grasping_ws/src/panda_grasping')

from panda_interface.srv import PlanAndExecute, PlanAndExecuteRequest, PlanAndExecuteResponse

from panda_interface.src.panda_interface import ObjectPosition
from panda_interface.src.panda_interface import PandaUtils
from panda_interface.src.panda_interface import GripperForceListener
from panda_interface.src.panda_interface import Randomizer
from panda_interface.src.panda_interface import GazeboConnection
import math
from tf.transformations import quaternion_inverse, quaternion_multiply, euler_from_quaternion, quaternion_from_euler


INITIAL_JOINT_POSITIONS = [0.028321078208972672, 0.25952951166771854, -0.007253497323791436, -2.4243337162902208, 0.0028962027517787092, 2.685928661651886, 0.8039933070393666]
RANDOM_JOINT_OFFSET = []
FORCE_REWARD = 100
INITIAL_GRIPPER_ORIENTATION = np.around(np.array([np.pi, 0, np.pi]),2)
POSITION_DIFFERENCE_FACTOR = 0.06
ORIENTATION_DIFFERENCE_FACTOR = 0.04

DISTANCE_THRESHOLD = 0.1
POSITION_SCALE_FACTOR = 0.02
ROTATION_SCALE_FACTOR = 0.2
OBJECT_HEIGHT = 0.09
CUBE_HEIGHT = 0.05
FORCE_THRESHOLD = 1.0
OBJECT_SDF_PATH = "/home/ros2/mt_panda_grasping_ws/src/panda_grasping/load_rl_env/urdf"

class PandaEnv(gym.Env):

    def __init__(self):
        rospy.init_node('panda_env')
        rospy.wait_for_service('plan_and_execute')
        self.plan_and_execute_service = rospy.ServiceProxy('plan_and_execute', PlanAndExecute)
        self.lift_object_service = rospy.ServiceProxy('ligt_object', PlanAndExecute)
        self.robot_pose_randomizer_service = rospy.ServiceProxy('robot_pose_randomizer', PlanAndExecute)

        self.panda_robot = PandaUtils()
        self.obj_position = ObjectPosition()
        self.panda_gripper = GripperForceListener()
        self.randomizer = Randomizer() 

        self.initial_pose = INITIAL_JOINT_POSITIONS

        self.observation_space = self._get_observation_space()
        self.action_space = self._get_action_space()

        # self._seed()

        self.execution_success = False

        self.spawned_object = ""


    def _get_observation_space(self):
        lower_ee_position = np.array([-1, -1, -1])
        upper_ee_position = np.array([1, 1, 1])

        lower_ee_orientation = np.array([-(math.pi)/4, -(math.pi/4), -(math.pi)])
        upper_ee_orientaiton = np.array([math.pi/4, math.pi/4, math.pi])
        lower_obj_position = np.full((3), 0.0)
        upper_obj_position = np.full((3), 1.0)

        lower_obj_distance = np.full((1), 0)
        upper_obj_distance = np.full((1), 1.0)

        lower_gripper_width  = np.array([0.0])
        upper_gripper_width  = np.array([1.0])

        lower_gripper_force = np.array([5.0, 5.0])
        upper_gripper_force = np.array([30.0, 30.0])

        lower_tcp_ft = np.full((6), -20)
        upper_tcp_ft = np.full((6), 20)

        lower_limits = np.concatenate((lower_ee_position, lower_ee_orientation, lower_obj_position, lower_obj_distance, lower_gripper_width, lower_gripper_force ))
        upper_limits = np.concatenate((upper_ee_position, upper_ee_orientaiton, upper_obj_position, upper_obj_distance, upper_gripper_width, upper_gripper_force))

        return spaces.Box(low=lower_limits, high=upper_limits, shape=(13,), dtype=np.float32)

    def _get_action_space(self):
        lower_ee_position = np.array([-0.02, -0.02, -0.02])
        upper_ee_position = np.array([0.02, 0.02, 0.02])

        lower_orientation = np.array([-(math.pi)/4, -(math.pi/4), -(math.pi)])
        upper_orientation = np.array([math.pi/4, math.pi/4, math.pi])

        lower_gripper_width = np.array([0.0])
        upper_gripper_width = np.array([1.0])

        lower_gripper_force = np.array([5.0])
        upper_gripper_force = np.array([30.0])

        return spaces.Box(low=np.concatenate((lower_ee_position, lower_orientation, lower_gripper_width, lower_gripper_force)), 
                            high=np.concatenate((upper_ee_position, upper_orientation, upper_gripper_width, upper_gripper_force)), shape=(8,), dtype=np.float32)

    
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def reset_gazebo(self):
        gazebo_connection = GazeboConnection()
        # gazebo_connection.pause_physics()
        # gazebo_connection.reset_simulation()
        # gazebo_connection.unpause_physics()
        # self.panda_robot.reset_tf_tree()
        gazebo_connection.stop_gazebo()
        gazebo_connection.start_gazebo()
    
    def _set_panda_initial_pose(self, initial_joint_state=INITIAL_JOINT_POSITIONS):
        self.panda_robot.move_to_joint_pose(initial_joint_state)

    def _set_obj_initial_pose(self):
        self.obj_position.set_obj_state()
    
    def spawn_random_object(self):
        if self.spawned_object:
            self.obj_position.delete_object(self.spawned_object)
        object_path_list = ["cube/model.sdf", "cylinder/model.sdf", "stone/model.sdf"]
        random_selected_object = random.choice(object_path_list)
        object_name = random_selected_object.split("/")[0]
        random_object_sdf_path = os.path.join(OBJECT_SDF_PATH, random_selected_object)
        subprocess.call(["rosrun", "gazebo_ros", "spawn_model", "-sdf", "-model", object_name, "-file", random_object_sdf_path])
        return object_name 

    def reset(self):
        self.spawned_object = self.spawn_random_object()
        self.obj_position.initialize_world()
        self.panda_gripper.execute_open_gripper()
        self.panda_robot.move_to_joint_pose(INITIAL_JOINT_POSITIONS)
        initial_pose = self.randomizer.randomize_robot_initial_pose()
        request = PlanAndExecuteRequest()
        request.x = initial_pose[0]
        request.y = initial_pose[1]
        request.z = initial_pose[2]
        request.roll = 0
        request.pitch = 0
        request.yaw = 0
        response = self.robot_pose_randomizer_service(request)
        self.obj_position.randomize_object_position(self.spawned_object)
        
        observation = self._get_new_observation()
        return observation 

    def _get_new_observation(self):
        
        gripper_position, gripper_orientation = self.panda_robot.get_ee_pose_tf()
        current_gripper_width = np.array([self.panda_robot.get_current_gripper_width()])
        current_obj_position, current_obj_orientation = self.obj_position.obj_get_state(self.spawned_object)
        current_gripper_force = np.array(self.panda_robot.get_current_joint_forces()) 
        current_distance_between_ee_object = np.array([self.calc_dist(current_obj_position, gripper_position)])
        gripper_euler = self.panda_robot.convert_quaternion_to_euler(gripper_orientation)
        tcp_ft = self.panda_robot.get_tcp_ft()
        observation = [gripper_position, gripper_euler, current_obj_position, current_distance_between_ee_object, current_gripper_width, current_gripper_force]
        observation = np.concatenate(observation)
        return observation

    def calc_dist(self,p1,p2):
        x_d = math.pow(p1[0] - p2[0],2)
        y_d = math.pow(p1[1] - p2[1],2)
        z_d = math.pow(p1[2] - p2[2],2)
        d = math.sqrt(x_d + y_d + z_d)

        return d

    def step(self, action):
        request = PlanAndExecuteRequest()

        request.x = POSITION_SCALE_FACTOR * float(action[0])
        request.y = POSITION_SCALE_FACTOR * float(action[1])
        request.z = POSITION_SCALE_FACTOR * float(action[2])
        request.roll = ROTATION_SCALE_FACTOR * float(action[3])
        request.pitch = ROTATION_SCALE_FACTOR * float(action[4])
        request.yaw = ROTATION_SCALE_FACTOR * float(action[5])
        gripper = action[6]
        gripper_force = action[7]
        done = False
        out_of_workspace = False
        success = False
        is_optimal_grasp = False

        self.panda_gripper.execute_open_gripper()

        response = self.plan_and_execute_service(request)
        
        if response.out_of_workspace == False and response.success == False:
            done = True

        next_observation = self._get_new_observation()
        orientation_difference_value = self.check_angle_difference()


        check_obj_position = self.obj_position.check_obj_position_change(self.spawned_object)

        if check_obj_position and next_observation[-4] > 0.04:
            done = True
        
        is_optimal_grasp = self.grasp_object(next_observation, action)
        
        if is_optimal_grasp:
            done = True
  
        reward = self.calculate_reward(done, is_optimal_grasp, next_observation, check_obj_position, response.success, response.out_of_workspace, [next_observation[-1], next_observation[-2]], next_observation[-4], orientation_difference_value, gripper_force)
        return next_observation, reward, done

    def check_angle_difference(self):
        left_finger_orientation, right_finger_orientation = self.panda_robot.get_gripper_finger_orientation()
        gripper_euler = euler_from_quaternion(left_finger_orientation)
        current_obj_position, current_obj_orientation = self.obj_position.obj_get_state(self.spawned_object)
        object_euler = euler_from_quaternion(current_obj_orientation)
        roll_difference = np.pi - abs(abs(object_euler[0]) - abs(gripper_euler[0]))
        pitch_difference = abs(abs(object_euler[1]) - abs(gripper_euler[1]))
        yaw_difference = abs(abs(object_euler[2]) - abs(gripper_euler[2]))

        scaling_factor = 1.0
        desired_differences = [0, np.pi, 2*np.pi] 
        closest_desired_difference = min(desired_differences, key=lambda x: abs(yaw_difference - x))
        roll_angular_difference = self._compute_difference_value(roll_difference, 0)
        pitch_angular_difference = self._compute_difference_value(yaw_difference, 0)
        yaw_angular_difference_value = self._compute_difference_value(yaw_difference, closest_desired_difference)
        return yaw_angular_difference_value

    def _compute_difference_value(self, angle_difference, closest_desired_difference):
        scaling_factor = 1.0
        if np.isclose(angle_difference, closest_desired_difference, atol=0.01):
            angular_difference_value = scaling_factor

        else:
            angular_difference_value = scaling_factor / (1 + abs(closest_desired_difference - angle_difference))
        return angular_difference_value

    def grasp_object(self, next_state, action):
        optimal_grasp = False
        is_balanced = False
        gripper_position = next_state[:3]
        object_position = self.obj_position.object_initial_pose[:3]
        is_gripper_x_close = math.isclose(gripper_position[0], object_position[0], abs_tol=0.01)
        is_gripper_y_close = math.isclose(gripper_position[1], object_position[1], abs_tol=0.01)
        gripper_forces = self.panda_robot.get_current_joint_forces()
        object_height = object_position[2] * 2 + 0.035
        is_gripper_z_close = math.isclose(gripper_position[2], object_height, abs_tol=0.02)
        if is_gripper_x_close and is_gripper_y_close and (gripper_position[2] < object_height):
            gripper_width = 0.05
            self.panda_gripper.execute_gripper_action(action[-1], gripper_width, self.spawned_object)
            ft_values = self.panda_robot.get_tcp_ft()
            after_gripper_forces = self.panda_robot.get_current_joint_forces()
            lifted_object_observation = self.lift_object(next_state, action)
            if round(lifted_object_observation[8], 2) > round(object_position[2], 2):
                is_balanced = True
            optimal_grasp = True if is_balanced else False
        return optimal_grasp

    def lift_object(self, next_state, action):

        request = PlanAndExecuteRequest()
        request.x = next_state[0]
        request.y = next_state[1]
        request.z = 0.2
        request.roll = 0
        request.pitch = 0
        request.yaw = 0
        response = self.lift_object_service(request)
        self.panda_robot.wait_time(waiting_time=5)
        observation = self._get_new_observation()
        return observation

    def calculate_reward(self, done, is_optimal_grasp, next_observation, check_obj_position, execution_status, out_of_workspace_flag, next_gripper_force, next_ee_object_distance, orientation_difference_value,  gripper_force):
        reward = 0
        gripper_position = next_observation[:3]
        object_position = self.obj_position.object_initial_pose[:3]
        is_gripper_x_close = math.isclose(gripper_position[0], object_position[0], abs_tol=0.01)
        is_gripper_y_close = math.isclose(gripper_position[1], object_position[1], abs_tol=0.01)
        object_height = object_position[2] * 2 + 0.035
        is_gripper_z_close = math.isclose(gripper_position[2], object_height, abs_tol=0.02)
        
        if not done:

            if out_of_workspace_flag:
                print("OUT OF WORKPSACE")
                reward = -1

            elif is_gripper_x_close and is_gripper_y_close and (gripper_position[2] < object_height):
                print("-------------------- VERY CLOSE TO THE OBJECT --------------------")
                reward = 5

            else:
                reward = POSITION_DIFFERENCE_FACTOR * (1/next_ee_object_distance) + ORIENTATION_DIFFERENCE_FACTOR * orientation_difference_value
                print("APPROACHING OBJECT", reward)

        else:
            if is_optimal_grasp:
                print("OBJECT GRASPED OPTIMALLY")
                reward = reward + 50
            else:
                print("NOT OPTIMAL GRASP")
                reward = 0 
            
            if check_obj_position:
                print(" ....... OBJECT MOVED MISTAKENLY .........")
                reward = reward + 0

            if out_of_workspace_flag == False and execution_status == False:
                print("ABORTED Execution")
                reward = reward - 20

        return reward

    def compute_force(self, next_gripper_force, input_force):
        gripper_force_difference = abs(next_gripper_force[0] - next_gripper_force[1])
        d = input_force - gripper_force_difference
        force_fraction = d / input_force
        return force_fraction