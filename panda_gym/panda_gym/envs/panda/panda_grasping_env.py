import gym
from gym import spaces
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
import rospy
import sys
import subprocess
import os
import random
import time
sys.path.append('/home/ros2/panda_grasping_ws/src/panda_grasping')

from panda_interface.src.panda_interface import ObjectPosition
from panda_interface.src.panda_interface import MovePanda
from panda_interface.src.panda_interface import GripperForceListener
from panda_interface.src.panda_interface import GazeboConnection
from panda_interface.src.panda_interface import Randomizer
import math
from tf.transformations import quaternion_inverse, quaternion_multiply, euler_from_quaternion, quaternion_from_euler

# from robo_gym_server_modules.robot_server.grpc_msgs.python import robot_server_pb2

INITIAL_JOINT_POSITIONS = [-0.0001914530812952009, -0.7856326455955855, -1.2635022375917515e-05, -2.355965542375113, 6.172411132432387e-06, 1.571755571494223, 0.7853925609812951]
# INITIAL_JOINT_POSITIONS = [0.2920332555729077, 0.061510866826177235, -0.2963577346808357, -2.293474076394883, 0.025595937147190106, 2.354275229181834, 0.7632559123980425]
# INITIAL_JOINT_POSITIONS = [-0.0032674590170405082, 0.32823800587027563, -0.002367261320395997, -2.4649363984696544, 0.0031544200104445252, 2.7953975051930815, 0.777078248909441]
# INITIAL_JOINT_POSITIONS = [0.028321078208972672, 0.25952951166771854, -0.007253497323791436, -2.4243337162902208, 0.0028962027517787092, 2.685928661651886, 0.8039933070393666]
RANDOM_JOINT_OFFSET = []
FORCE_REWARD = 100
INITIAL_GRIPPER_ORIENTATION = np.around(np.array([np.pi, 0, np.pi]),2)
POSITION_DIFFERENCE_FACTOR = 0.06
ORIENTATION_DIFFERENCE_FACTOR = 0.04

DISTANCE_THRESHOLD = 0.1
ROTATION_SCALE_FACTOR = 0.2
OBJECT_HEIGHT = 0.09
CUBE_HEIGHT = 0.05
OBJECT_SDF_PATH = "/home/ros2/panda_grasping_ws/src/panda_grasping/load_rl_env/urdf"

class PandaGraspingEnv(gym.Env):

    def __init__(self):
        # rospy.init_node('move_panda_arm_moveit', anonymous=True)

        self.panda_robot = MovePanda()
        self.obj_position = ObjectPosition()
        self.panda_gripper = GripperForceListener()
        self.randomizer = Randomizer()

        self.initial_pose = INITIAL_JOINT_POSITIONS

        self.observation_space = self._get_observation_space()
        self.action_space = self._get_action_space()

        self.execution_success = False

        self.spawned_object = ""

        # obs = self._get_obs()

    def reset_gazebo(self):
        gazebo_connection = GazeboConnection()
        gazebo_connection.pause_physics()
        gazebo_connection.reset_simulation()
        gazebo_connection.unpause_physics()

    def _get_observation_space(self):
        lower_ee_position = np.array([-0.5, -0.5, -0.5])
        upper_ee_position = np.array([0.5, 0.5, 0.5])

        lower_ee_orientation = np.array([-(math.pi)/4, -(math.pi/4), -(math.pi)])
        upper_ee_orientaiton = np.array([math.pi/4, math.pi/4, math.pi])
        #Replace inf with real world scenario
        lower_obj_position = np.full((3), 0.0)
        upper_obj_position = np.full((3), 1.0)

        lower_obj_distance = np.full((1), 0)
        upper_obj_distance = np.full((1), 1.0)

        # lower_obj_orientation = np.full((4), -1.0)
        # upper_obj_orientation = np.full((4), 1.0)

        lower_gripper_width  = np.array([0.0])
        upper_gripper_width  = np.array([1.0])

        lower_gripper_force = np.array([10.0, 10.0])
        upper_gripper_force = np.array([30.0, 30.0])

        lower_tcp_ft = np.full((6), -20)
        upper_tcp_ft = np.full((6), 20)

        lower_limits = np.concatenate((lower_ee_position, lower_ee_orientation, lower_obj_position, lower_obj_distance, lower_gripper_width, lower_gripper_force ))
        upper_limits = np.concatenate((upper_ee_position, upper_ee_orientaiton, upper_obj_position, upper_obj_distance, upper_gripper_width, upper_gripper_force))

        return spaces.Box(low=lower_limits, high=upper_limits, shape=(13,), dtype=np.float32)

    def _get_action_space(self):
        lower_ee_position = np.array([-0.05, -0.05, -0.05])
        upper_ee_position = np.array([0.05, 0.05, 0.05])

        lower_orientation = np.array([-(math.pi)/4, -(math.pi/4), -(math.pi)])
        upper_orientation = np.array([math.pi/4, math.pi/4, math.pi])

        lower_gripper_width = np.array([0.0])
        upper_gripper_width = np.array([1.0])

        lower_gripper_force = np.array([10.0])
        upper_gripper_force = np.array([30.0])

        return spaces.Box(low=np.concatenate((lower_ee_position, lower_orientation, lower_gripper_width, lower_gripper_force)), 
                            high=np.concatenate((upper_ee_position, upper_orientation, upper_gripper_width, upper_gripper_force)), shape=(8,), dtype=np.float32)

    def _set_panda_initial_pose(self, initial_joint_state=INITIAL_JOINT_POSITIONS):
        self.panda_robot.move_to_joint_pose(initial_joint_state)

    def _set_obj_initial_pose(self):
        self.obj_position.set_obj_state()
    
    def spawn_random_object(self):
        if self.spawned_object:
            # print(f"Deleted {self.spawned_object}")
            self.obj_position.delete_object(self.spawned_object)
        object_path_list = ["cube/model.sdf", "cylinder/model.sdf", "stone/model.sdf"]
        random_selected_object = random.choice(object_path_list)
        object_name = random_selected_object.split("/")[0]
        random_object_sdf_path = os.path.join(OBJECT_SDF_PATH, random_selected_object)
        # print("PATH", random_object_sdf_path)
        subprocess.call(["rosrun", "gazebo_ros", "spawn_model", "-sdf", "-model", object_name, "-file", random_object_sdf_path])
        return object_name 

    def reset(self):
        self.spawned_object = self.spawn_random_object()
        self.obj_position.initialize_world()
        self.panda_gripper.execute_open_gripper()
        self.panda_robot.move_to_joint_pose(INITIAL_JOINT_POSITIONS)
        
        initial_pose = self.randomizer.randomize_robot_initial_pose()
        plan = self.panda_robot.move_to_pose(initial_pose[0], initial_pose[1], initial_pose[2], 0, 0, 0)
        execution_success = self.panda_robot.execute_plan(plan, self.panda_robot.arm_interface)
        self.obj_position.randomize_object_position(self.spawned_object)
        observation = self._get_new_observation()
        return observation 

    def _get_new_observation(self):
        # ee_current_pose = self.panda_robot.get_ee_pose()
        # ee_current_position = np.array([ee_current_pose.position.x, ee_current_pose.position.y, ee_current_pose.position.z])
        # ee_current_orientation = np.array([ee_current_pose.orientation.x, ee_current_pose.orientation.y, ee_current_pose.orientation.z, ee_current_pose.orientation.w])
        
        gripper_position, gripper_orientation = self.panda_robot.get_ee_pose_tf()
        current_gripper_width = np.array([self.panda_robot.get_current_gripper_width()])
        current_obj_position, current_obj_orientation = self.obj_position.obj_get_state(self.spawned_object)
        current_gripper_force = np.array(self.panda_robot.get_current_joint_forces()) 
        current_distance_between_ee_object = np.array([self.calc_dist(current_obj_position, gripper_position)])
        gripper_euler = self.panda_robot.convert_quaternion_to_euler(gripper_orientation)
        tcp_ft = self.panda_robot.get_tcp_ft()
        print("TCP", tcp_ft)
        observation = [gripper_position, gripper_euler, current_obj_position, current_distance_between_ee_object, current_gripper_width, current_gripper_force]
        observation = np.concatenate(observation)
        return observation

    def _get_obs(self):

        ee_pose = self.panda_robot.get_ee_pose()
        ee_array_position = [ee_pose.position.x, ee_pose.position.y, ee_pose.position.z]

        object_position, obj_orientation = self.obj_position.obj_get_state()

        # obj_pose = object_position[:3]

        distance = self.calc_dist(object_position, ee_array_position)

        # print("EE POSE ===========", ee_array_position)
        # print("OBJ POSE ==============", object_position)
        # print("DISTANCE ========", distance)

        return distance

    def calc_dist(self,p1,p2):
        """
        d = ((2 - 1)2 + (1 - 1)2 + (2 - 0)2)1/2
        """
        x_d = math.pow(p1[0] - p2[0],2)
        y_d = math.pow(p1[1] - p2[1],2)
        z_d = math.pow(p1[2] - p2[2],2)
        d = math.sqrt(x_d + y_d + z_d)

        return d

    def step(self, action):
        # print("ACTION", action)
        x = float(action[0])
        y = float(action[1])
        z = float(action[2])
        er = ROTATION_SCALE_FACTOR * float(action[3])
        ep = ROTATION_SCALE_FACTOR * float(action[4])
        ey = ROTATION_SCALE_FACTOR * float(action[5])
        gripper = action[6]
        gripper_force = action[7]
        print("INPUT GRIPPER FORCE", gripper_force)
        done = False
        out_of_workspace_flag = False
        execution_status = False
        is_optimal_grasp = False

        previous_pose = self.panda_robot.get_ee_pose()
        self.panda_gripper.execute_open_gripper()
        # move_plan_success , move_to_new_pose_plan = self.panda_robot.move_in_small_steps(x, y, z, er, ep, ey)
        # print("MOVE PLAN SUCCESS", move_plan_success)
        print("Planning started")
        plan = self.panda_robot.move_in_small_steps(x, y, z, er, ep, ey)
        if plan == None:
            out_of_workspace_flag = True
        else:
            execution_status = self.panda_robot.execute_plan(plan, self.panda_robot.arm_interface)

        print("Execution Complete")
        if out_of_workspace_flag == False and execution_status == False:
            done = True
        
        # movement_success = self.panda_gripper.execute_gripper_action(gripper_force)

        next_observation = self._get_new_observation()
        print("DISTANCE", next_observation[-4])

        angle_difference = self.check_approach_angle()
        # print("Normalized orientation difference", normalized_orientation_difference)
        # print("PRE_REWARD", -np.linalg.norm(normalized_orientation_difference, ord=2))
        # max_angle_difference = 0.2  # Define a maximum allowed angle difference (adjust as needed)
        # orientation_difference_value = 1.0 - min(1.0, abs(angle_difference) / max_angle_difference)
        # orientation_difference_value = np.exp(-np.linalg.norm(normalized_orientation_difference, ord=2))
        orientation_difference_value = self.check_angle_difference(angle_difference)
        # orientation_difference_value = scaling_factor / (1 + angle_difference)
        print("orientation_difference_value", orientation_difference_value)

        check_obj_position = self.obj_position.check_obj_position_change(self.spawned_object)
        
        # self.object_grasp()

        if check_obj_position and next_observation[-4] > 0.04:
            # print("Check Obj IF", next_observation[:3])
            done = True
        
        is_optimal_grasp = self.grasp_object(next_observation, action)
        # print("OPTIMAL_GRASP", is_optimal_grasp)

        if is_optimal_grasp:
            done = True
        
        print("Current Position", next_observation[:3])
        reward = self.calculate_reward(done, is_optimal_grasp, next_observation, check_obj_position, execution_status, out_of_workspace_flag, next_observation[-4], orientation_difference_value)
        # print("REWARD", reward)
        return next_observation, reward, done

    def check_angle_difference(self, orientation_difference):
        scaling_factor = 1.0
        desired_differences = [0, np.pi, 2*np.pi] 
        closest_desired_difference = min(desired_differences, key=lambda x: abs(orientation_difference - x))

        if np.isclose(orientation_difference, closest_desired_difference, atol=0.01):
            angular_difference_value = scaling_factor
        else:
            angular_difference_value = scaling_factor / (1 + orientation_difference)
        return angular_difference_value

    def check_approach_angle(self):
        left_finger_orientation, right_finger_orientation = self.panda_robot.get_gripper_finger_orientation()
        print("GRIPPER ORIENTATION", left_finger_orientation)
        current_obj_position, current_obj_orientation = self.obj_position.obj_get_state(self.spawned_object)
        print("OBJECT ORIENTATION", current_obj_orientation)

        quat_difference = np.dot(left_finger_orientation, np.conjugate(current_obj_orientation))
        print("QUAT DIFF", quat_difference)
        # Convert the quaternion difference to an angle difference
        angle_difference = 2 * np.arccos(abs(quat_difference))
        print("ANGLE DIFF", angle_difference)

        # relative_rotation = quaternion_multiply(current_obj_orientation, quaternion_inverse(left_finger_orientation))
        # angle_difference = 2 * euler_from_quaternion(relative_rotation)[2]
        # print("ANGLE DIFFERENCE", angle_difference)
        # orientation_quaternion = self.panda_robot.quaternion_multiply(current_obj_orientation, left_finger_orientation)
        # gripper_roll, gripper_pitch, gripper_yaw = euler_from_quaternion(orientation_quaternion)
        # orientation_difference = np.around(np.array([gripper_roll, gripper_pitch, gripper_yaw]),2)
        # orientation_difference = np.abs(orientation_difference) - np.abs(INITIAL_GRIPPER_ORIENTATION)
        # print("ORIENTATION DIFFERENCE", orientation_difference)
        # normalized_orientation_difference = orientation_difference / np.pi
        # # print(f"Roll: {gripper_roll},  Pitch: {gripper_pitch}, Yaw: {gripper_yaw}")
        return angle_difference

    def object_grasp(self):
        current_obj_position, current_obj_orientation = self.obj_position.obj_get_state(self.spawned_object)
        # print("OBJ_HEIGHT", current_obj_position[-2], CUBE_HEIGHT)
        if current_obj_position[-2] > CUBE_HEIGHT:
            print("OBJECT GRASP LOGIC")
        elif self.obj_position.check_obj_position_change(self.spawned_object):
            print("OBJECT IS MOVED MISTAKENLY")
        else:

            print("ALL OKAY ! AGENT IS LEARNING")

    def grasp_object(self, next_state, action):
        optimal_grasp = False
        is_balanced = False
        gripper_position = next_state[:3]
        object_position = self.obj_position.object_initial_pose[:3]
        print("GRIPPER POSTION", gripper_position)
        print("OBJECT POSITION", object_position)
        is_gripper_x_close = math.isclose(gripper_position[0], object_position[0], abs_tol=0.015)
        is_gripper_y_close = math.isclose(gripper_position[1], object_position[1], abs_tol=0.015)
        object_height = 0.05
        gripper_forces = self.panda_robot.get_current_joint_forces()
        print("Gripper forces before grasping", gripper_forces)
        # is_gripper_z_close = math.isclose(gripper_position[2], object_height, abs_tol=0.025)
        if is_gripper_x_close and is_gripper_y_close and gripper_position[2] < OBJECT_HEIGHT:
            self.panda_gripper.execute_close_gripper()
            gripper_width = self.panda_robot.get_current_gripper_width()
            print("---------------------------GRASPING THE OBJECT--------------------------", gripper_width)
            self.panda_gripper.execute_gripper_action(action[-1], gripper_width, self.spawned_object)
            ft_values = self.panda_robot.get_tcp_ft()
            print("FT_VALUES", ft_values)
            after_gripper_forces = self.panda_robot.get_current_joint_forces()
            print("Gripper forces AFTER grasping", after_gripper_forces)
            lifted_object_observation = self.lift_object(next_state, action)
            if round(lifted_object_observation[8], 2) > round(object_position[2], 2):
                print("Lifted Object Observation", round(lifted_object_observation[8], 2), round(object_position[2], 2))
                is_balanced = True
            optimal_grasp = True if is_balanced else False
        return optimal_grasp

    def grasp_object_old(self, next_state, action):
        # observation = [ee_current_position, ee_current_orientation, current_obj_position, current_distance_between_ee_object, current_gripper_width, current_gripper_force]
        optimal_grasp = False
        is_done = False
        is_balanced = False
        gripper_z_position = next_state[2]
        object_z_position = next_state[8]
        # gripper_joint_forces = next_state[-1]
        balanced_gripper_force = action[-1]/2
        print("Object z position", object_z_position)
        print("GRIPPER", gripper_z_position, next_state[-4])
        #TODO: Check for the alignment of the gripper vector and object vector. Right now, the gripper is closing near to the object
        is_ee_close = math.isclose(next_state[-4], 0.05, abs_tol=0.02)
        if (gripper_z_position < OBJECT_HEIGHT) and is_ee_close:
            # self.panda_gripper.execute_close_gripper()
            # gripper_width = self.panda_robot.get_current_gripper_width()
            # print("GRIPPER WIDTH", gripper_width)
            self.panda_gripper.execute_gripper_action(action[-1])
            print(" !!!!!!!!!!!!!! Gripper is close to object !!!!!!!!!!", gripper_z_position, OBJECT_HEIGHT)
            lifted_object_observation = self.lift_object(next_state, action)
            print("FORCES", lifted_object_observation[-1], lifted_object_observation[-2])
            if round(lifted_object_observation[8], 2) > round(object_z_position, 2):
                print("Lifted Object Observation", round(lifted_object_observation[8], 2), round(object_z_position, 2))
                is_balanced = True
            # is_balanced = math.isclose(lifted_object_observation[-1], balanced_gripper_force, abs_tol=0.01) and math.isclose(lifted_object_observation[-2], balanced_gripper_force, abs_tol=0.01)
            optimal_grasp = True if is_balanced else False
   
        return optimal_grasp

    def lift_object(self, next_state, action):
        x = next_state[0]
        y = next_state[1]
        z = 0.2
        er = next_state[3]
        ep = next_state[4]
        ey = next_state[5]
        plan = self.panda_robot.move_to_pose(x, y, z, er, ep, ey)
        print("=========LIFTING THE OBJECT===========")
        execution_success = self.panda_robot.execute_plan(plan, self.panda_robot.arm_interface)
        self.panda_robot.wait_time(waiting_time=5)
        # time.sleep(3)
        observation = self._get_new_observation()
        return observation

    def calculate_reward(self, done, is_optimal_grasp, next_observation, check_obj_position, execution_status, out_of_workspace_flag, next_ee_object_distance, orientation_difference_value):
        reward = 0
        gripper_position = next_observation[:3]
        object_position = self.obj_position.object_initial_pose[:3]
        is_gripper_x_close = math.isclose(gripper_position[0], object_position[0], abs_tol=0.015)
        is_gripper_y_close = math.isclose(gripper_position[1], object_position[1], abs_tol=0.015)
        object_height = object_position[2] * 2
        is_gripper_z_close = math.isclose(gripper_position[2], 0.05, abs_tol=0.025)
        
        if not done:

            if out_of_workspace_flag:
                print("OUT OF WORKPSACE")
                reward = -1

            elif is_gripper_x_close and is_gripper_y_close and gripper_position[2] < OBJECT_HEIGHT:
                print("-------------------- VERY CLOSE TO THE OBJECT --------------------")
                reward = 5

            # elif(next_ee_object_distance >= 0.05):
            #     print("APPROACHING OBJECT")
            #     reward = 0.05 * (1/next_ee_object_distance)
            else:
                reward = POSITION_DIFFERENCE_FACTOR * (1/next_ee_object_distance) + ORIENTATION_DIFFERENCE_FACTOR * orientation_difference_value
                print("APPROACHING OBJECT", reward)

            # else:
            #     print("REACHED CLOSE TO THE OBJECT")
            #     # force_fraction = self.compute_force(next_gripper_force, gripper_force)
            #     # reward = FORCE_REWARD * force_fraction
            #     reward = 1/ next_ee_object_distance

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
                reward = reward -20

        return reward

    def compute_force(self, next_gripper_force, input_force):
        gripper_force_difference = abs(next_gripper_force[0] - next_gripper_force[1])
        d = input_force - gripper_force_difference
        force_fraction = d / input_force
        return force_fraction

        
    

# gym.register(id="PandaGrasping-v0", entry_point='panda_grasping_env:PandaGraspingEnv')

# if __name__ == "__main__":
#     panda_env = PandaGraspingEnv()
#     print("action sample", panda_env.action_space.sample())