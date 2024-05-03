#! /usr/bin/env python

import gym
from gym import spaces
import numpy as np
import copy 
import robo_gym_server_modules.robot_server.client as rs_client
from robo_gym_server_modules.robot_server.grpc_msgs.python import robot_server_pb2

INITIAL_JOINT_POSITIONS = []

class PandaBaseEnv(gym.Env):

    real_robot = False
    max_episode_steps = 300

    def __init__(self, rs_address=None, rs_state_to_info=True):

        self.rs_state_to_info = rs_state_to_info

        self.observation_space = self._get_observation_space()
        self.action_space = self._get_action_space()
        print(self.observation_space)

        self.rs_state = None

        if rs_address:
            self.client = rs_client.Client(rs_address)
        else:
            print("WARNING: No IP and Posrt passesd. Simulation will not be started")
            print("WARNING: Use this only to get environment shape")

    def _get_observation_space(self):
        lower_ee_position = np.full((3), -1.0)
        upper_ee_position = np.full((3), 1.0)

        lower_ee_orientation = np.full((4), -1.0)
        upper_ee_orientaiton = np.full((4), 1.0)
        #Replace inf with real world scenario
        lower_obj_position = np.full((3), -np.inf)
        upper_obj_position = np.full((3), np.inf)

        lower_obj_orientation = np.full((4), -1.0)
        upper_obj_orientation = np.full((4), 1.0)

        lower_gripper_width  = np.array([0.0])
        upper_gripper_width  = np.array([1.0])

        lower_limits = np.concatenate((lower_ee_position, lower_ee_orientation, lower_obj_position, lower_obj_orientation, lower_gripper_width))
        upper_limits = np.concatenate((upper_ee_position, upper_ee_orientaiton, upper_obj_position, upper_obj_orientation, upper_gripper_width))

        return spaces.Box(low=lower_limits, high=upper_limits, shape=(15,), dtype=np.float32)

    def _get_action_space(self):
        lower_pose = np.full((7), -1.0)
        upper_pose = np.full((7), 1.0)

        lower_gripper_width = np.array([0.0])
        upper_gripper_width = np.array([1.0])

        return spaces.Box(low=np.concatenate((lower_pose, lower_gripper_width)), high=np.concatenate((upper_pose, upper_gripper_width)), shape=(8,), dtype=np.float32)

    def _set_initial_robot_server_state(self):
        pass

    
    def reset(self, initial_position):
        pass

    def step(self, action):
        pass

    def render(self):
        pass
    

if __name__ == "__main__":
    panda_env = PandaBaseEnv()
    print("action sample", panda_env. action_space.sample())
        

