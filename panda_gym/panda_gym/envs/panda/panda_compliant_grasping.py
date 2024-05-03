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
import math
from tf.transformations import quaternion_inverse, quaternion_multiply, euler_from_quaternion, quaternion_from_euler

class ComplianceControlEnv(gym.Env):
    def __init__(self):
        super(ComplianceControlEnv, self).__init__()

        self.panda_robot = MovePanda()
        self.obj_position = ObjectPosition()
        self.panda_gripper = GripperForceListener()

        self.initial_pose = INITIAL_JOINT_POSITIONS

        self.observation_space = self._get_observation_space()
        self.action_space = self._get_action_space()

        self.execution_success = False

        self.spawned_object = ""

        # Compliance control parameters
        self.position_stiffness = 1000.0  # Adjust as needed
        self.orientation_stiffness = 50.0  # Adjust as needed

        # Initialize state variables
        self.current_force_torque = np.zeros(6)
        self.current_pose = np.zeros(6)

    def _get_observation_space(self):
        lower_ee_position = np.array([-0.05, -0.05, -0.05])
        upper_ee_position = np.array([0.05, 0.05, 0.05])

        lower_ee_orientation = np.array([-(math.pi)/4, -(math.pi/4), -(math.pi)])
        upper_ee_orientaiton = np.array([math.pi/4, math.pi/4, math.pi])

        lower_obj_position = np.full((3), 0.0)
        upper_obj_position = np.full((3), 1.0)

        lower_obj_distance = np.full((1), 0)
        upper_obj_distance = np.full((1), 1.0)

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

    def step(self, action):
        # Calculate the state based on the action and update compliance control
        next_state = self.calculate_next_state(action)
        reward = self.calculate_reward(next_state)
        done = self.is_done(next_state)
        return next_state, reward, done, {}

    def reset(self):
        # Reset the environment to an initial state
        self.current_force_torque = np.zeros(6)
        self.current_pose = np.zeros(6)
        return self.current_pose

    def calculate_next_state(self, action):
        # Calculate the next state based on the action and compliance control
        position_error = self.current_force_torque[:3] / self.position_stiffness
        orientation_error = self.current_force_torque[3:] / self.orientation_stiffness

        # Update the state based on the errors and action
        self.current_pose[:3] += position_error + action[:3]
        self.current_pose[3:] += orientation_error + action[3:]

        return self.current_pose

    def calculate_reward(self, state):
        # Define the reward function based on the state and task objectives
        # Reward should encourage compliance and grasping performance
        return -np.linalg.norm(state)  # Example: negative distance to the origin

    def is_done(self, state):
        # Define termination conditions based on the state and task objectives
        return np.linalg.norm(state) < 0.01  # Example: reaching a threshold

    def render(self, mode='human'):
        # Implement visualization if needed
        pass

    def close(self):
        # Implement any cleanup if needed
        pass

if __name__ == '__main__':
    env = ComplianceControlEnv()

    for _ in range(10):
        obs = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()  # Random actions for testing
            obs, reward, done, _ = env.step(action)
            print(f"Position: {obs}, Reward: {reward}")
