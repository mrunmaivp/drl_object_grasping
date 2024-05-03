import gym
from gym import spaces
import numpy as np
from scipy.spatial.transform import Rotation as R
import rospy
import sys
sys.path.append('/home/ros2/mt_panda_grasping_ws/src/panda_grasping')

from panda_interface.src.panda_interface import ObjectPosition
from panda_interface.src.panda_interface import MovePanda
from panda_interface.src.panda_interface import GripperForceListener
import math

# from robo_gym_server_modules.robot_server.grpc_msgs.python import robot_server_pb2

INITIAL_JOINT_POSITIONS = [-0.0001914530812952009, -0.7856326455955855, -1.2635022375917515e-05, -2.355965542375113, 6.172411132432387e-06, 1.571755571494223, 0.7853925609812951]

RANDOM_JOINT_OFFSET = []
FORCE_REWARD = 100

DISTANCE_THRESHOLD = 0.1

class PandaGraspingEnv(gym.Env):

    def __init__(self):
        # rospy.init_node('move_panda_arm_moveit', anonymous=True)

        self.panda_robot = MovePanda()
        self.obj_position = ObjectPosition()
        self.panda_gripper = GripperForceListener()

        self.initial_pose = INITIAL_JOINT_POSITIONS

        self.observation_space = self._get_observation_space()
        self.action_space = self._get_action_space()

        self.execution_success = False

        # obs = self._get_obs()
        
    def start_node(self):
        rospy.spin()

    def _get_observation_space(self):
        lower_ee_position = np.array([0.5, -0.25, 0.1])
        upper_ee_position = np.array([0.35, 0.25, 0.60])

        lower_ee_orientation = np.array([-math.pi/4, -math.pi/4, -(math.pi)])
        upper_ee_orientaiton = np.array([math.pi/4, math.pi/4, math.pi])
        #Replace inf with real world scenario
        lower_obj_position = np.full((3), 0.0)
        upper_obj_position = np.full((3), 1.0)

        # lower_obj_orientation = np.full((4), -1.0)
        # upper_obj_orientation = np.full((4), 1.0)

        lower_gripper_width  = np.array([0.0])
        upper_gripper_width  = np.array([1.0])

        lower_gripper_force = np.array([0.0])
        upper_gripper_force = np.array([10.0])

        lower_limits = np.concatenate((lower_ee_position, lower_ee_orientation, lower_obj_position, lower_gripper_width, lower_gripper_force))
        upper_limits = np.concatenate((upper_ee_position, upper_ee_orientaiton, upper_obj_position, upper_gripper_width, upper_gripper_force))

        return spaces.Box(low=lower_limits, high=upper_limits, shape=(11,), dtype=np.float32)

    def _get_action_space(self):
        lower_ee_position = np.array([-0.1, -0.1, -0.1])
        upper_ee_position = np.array([0.1, 0.01, 0.1])

        lower_orientation = np.array([-(math.pi)/4, -(math.pi/4), -(math.pi)])
        upper_orientation = np.array([math.pi/4, math.pi/4, math.pi])

        lower_gripper_width = np.array([0.0])
        upper_gripper_width = np.array([1.0])

        lower_gripper_force = np.array([0.0])
        upper_gripper_force = np.array([10.0])

        return spaces.Box(low=np.concatenate((lower_ee_position, lower_orientation, lower_gripper_width, lower_gripper_force)), 
                            high=np.concatenate((upper_ee_position, upper_orientation, upper_gripper_width, upper_gripper_force)), shape=(8,), dtype=np.float32)

    def _set_panda_initial_pose(self, initial_joint_state=INITIAL_JOINT_POSITIONS):
        self.panda_robot.move_to_joint_pose(initial_joint_state)

    def _set_obj_initial_pose(self):
        self.obj_position.set_obj_state()

    def reset(self):
        print("INITIAL_JOINT_POSITIONS", INITIAL_JOINT_POSITIONS)
        # self.panda_robot.move_to_joint_pose(INITIAL_JOINT_POSITIONS)
        plan = self.panda_robot.move_to_pose(0.30697425309106013, -5.334090225032015e-05, 0.5906692483699203, 1, 1, 1)
        execution_status = self.panda_robot.execute_plan(plan, self.panda_robot.arm_interface)
        self.obj_position.set_obj_state()

        # open_gripper_status = self.panda_robot.open_gripper()
        # print('gripper_status', open_gripper_status)
        # if open_gripper_status:
        self.panda_robot.close_gripper()
        # self.panda_gripper.execute_gripper_action()

        observation = self._get_new_observation()
        print("Current Observation", observation)
        return observation 

    def _get_new_observation(self):
        ee_current_pose = self.panda_robot.get_ee_pose()
        ee_current_position = np.array([ee_current_pose.position.x, ee_current_pose.position.y, ee_current_pose.position.z])
        ee_current_orientation = np.array([ee_current_pose.orientation.x, ee_current_pose.orientation.y, ee_current_pose.orientation.z, ee_current_pose.orientation.w])
        
        current_gripper_width = np.array(self.panda_robot.get_current_gripper_width())
        current_obj_position = self.obj_position.obj_get_state()
        current_gripper_force = np.array(self.panda_robot.get_current_joint_forces()) 

        observation = [ee_current_position, ee_current_orientation, current_gripper_width, current_obj_position, current_gripper_force]
        # print("OBSERVATION", observation)
        return observation

    def _get_obs(self):

        ee_pose = self.panda_robot.get_ee_pose()
        ee_array_position = [ee_pose.position.x, ee_pose.position.y, ee_pose.position.z]

        object_position = self.obj_position.obj_get_state()

        # obj_pose = object_position[:3]

        distance = self.calc_dist(object_position, ee_array_position)

        print("EE POSE ===========", ee_array_position)
        print("OBJ POSE ==============", object_position)
        print("DISTANCE ========", distance)

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
        x = float(action[0])
        y = float(action[1])
        z = float(action[2])
        er = float(action[3])
        ep = float(action[4])
        ey = float(action[5])
        # qx = action[3]
        # qy = action[4]
        # qz = action[5]
        # qw = action[6]
        gripper = action[6]
        gripper_force = action[7]

        done = False

        previous_pose = self.panda_robot.get_ee_pose()
        print("PREVIOUS POSE==========", previous_pose)

        self.panda_robot.open_gripper()
        # move_plan_success , move_to_new_pose_plan = self.panda_robot.move_in_small_steps(x, y, z, er, ep, ey)
        # print("MOVE PLAN SUCCESS", move_plan_success)

        plan = self.panda_robot.move_in_small_steps(x, y, z, er, ep, ey)
        execution_status = self.panda_robot.execute_plan(plan, self.panda_robot.arm_interface)

        # if not move_plan_success:
        #     done = True
        #     info = {"Planning": "Failed"}

        # print("NEW POSE", move_to_new_pose_plan)
        # if move_plan_success:
        # execution_status = self.panda_robot.execute_plan(move_to_new_pose_plan, self.panda_robot.arm_interface)
        # rospy.loginfo("Execution successful !")

        print("NEW POSE+++++++++++++", self.panda_robot.get_ee_pose())
        self.panda_robot.close_gripper()
        # self.panda_robot.grasp(gripper, gripper_force)
        # self.panda_gripper.execute_gripper_action()

        # observation = self._get_new_observation()
        # print("Current Observation", observation)

        
        ee_next_pose = self.panda_robot.get_ee_pose()
        ee_next_position = np.array([ee_next_pose.position.x, ee_next_pose.position.y, ee_next_pose.position.z])
        ee_next_orientation = np.array([ee_next_pose.orientation.x, ee_next_pose.orientation.y, ee_next_pose.orientation.z, ee_next_pose.orientation.w])

        next_obj_position = self.obj_position.obj_get_state()
        # next_distance = self.calc_dist(ee_next_position, next_obj_position)
        
        next_gripper_width = np.array(self.panda_robot.get_current_gripper_width())
        # next_obj_position = self.obj_position.obj_get_state()
        # next_object_height = next_obj_position[2]
        # next_obj_ee_distance = self._get_obs()
        next_gripper_force = np.array(self.panda_robot.get_current_joint_forces())

        next_state = [ee_next_position, ee_next_orientation, next_gripper_width, next_gripper_force]
        move_plan_success = True 

        reward = self.calculate_reward(done, move_plan_success, next_gripper_force, gripper_force)
        
        return next_state, reward, done

    def calculate_reward(self, done, move_plan_success, next_gripper_force, gripper_force):

        reward = 0

        if not move_plan_success:
            reward = -100

        else:
            force_fraction = self.compute_force(next_gripper_force, gripper_force)
            reward = FORCE_REWARD * force_fraction

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