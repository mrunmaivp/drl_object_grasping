#! /usr/bin/env python

import numpy as np
import rospy
from gazebo_msgs.srv import GetWorldProperties, GetModelState, SetModelState, DeleteModel
from gazebo_msgs.msg import ModelState
from std_srvs.srv import Empty
import sys
import random
import math
from tf.transformations import quaternion_from_euler
from geometry_msgs.msg import Quaternion

class ObjectPosition(object):

    def __init__(self):
        self.model_names = None
        self.get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        self.set_model_state = rospy.ServiceProxy('gazebo/set_model_state', SetModelState)
        self.delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        self.object_initial_pose = None

    def delete_object(self, object_name):
        self.delete_model(object_name)
        
    def initialize_world(self):
        world_specs = rospy.ServiceProxy('/gazebo/get_world_properties', GetWorldProperties)()
        self.model_names = world_specs.model_names

    def _get_object_initial_pose(self, object_name):
        obj_position, obj_orientation = self.obj_get_state(object_name)
        self.object_initial_pose = np.round(np.concatenate((obj_position, obj_orientation)), 2)
    
    def check_obj_position_change(self, object_name): 
        obj_position, obj_orientation = self.obj_get_state(object_name)
        current_obj_position = np.round(np.concatenate((obj_position, obj_orientation)), 2)

        #TODO: Modify this to check if the object is lifted , if yes, it is a good sign. Else, check if the object is on the ground and has changed its position
        change_x = math.isclose(current_obj_position[0], self.object_initial_pose[0], abs_tol=0.01)
        change_y = math.isclose(current_obj_position[1], self.object_initial_pose[1], abs_tol=0.01)
        orientation_change = all([math.isclose(x, y, abs_tol=0.05) for x, y in zip(obj_orientation, self.object_initial_pose[3:])])
        if change_x and change_y:
            return False
        else:
            return True


    def obj_get_state(self, object_name):
        for model_name in self.model_names:
            if model_name == object_name:
                data = self.get_model_state(model_name, "ground_plane")
                obj_position = np.array([data.pose.position.x, data.pose.position.y, data.pose.position.z])
                obj_orientation = np.array([data.pose.orientation.x, data.pose.orientation.y, data.pose.orientation.z, data.pose.orientation.w])
                return obj_position, obj_orientation

    def set_obj_state(self, object_name):
        model_state = ModelState()
        for model_name in self.model_names:
            if model_name == object_name:
                model_state.model_name = object_name
                model_state.pose.position.x = 0.5
                model_state.pose.position.y = 0.0
                model_state.pose.position.z = 0.0
                model_state.pose.orientation.x = 0
                model_state.pose.orientation.y = 0
                model_state.pose.orientation.z = 0
                model_state.pose.orientation.w = 1

                response = self.set_model_state(model_state)
        obj_position = [model_state.pose.position.x, model_state.pose.position.y, model_state.pose.position.z]
        obj_orientation = [model_state.pose.orientation.x, model_state.pose.orientation.y, model_state.pose.orientation.z, model_state.pose.orientation.w]
        self.object_initial_pose = np.round(np.concatenate((obj_position, obj_orientation)), 2)
        

    def randomize_object_position(self, object_name):
        model_state = ModelState()
        random_pose = self.random_pose()
        noise = self.add_noise(mean=0.0, std_dev=0.005)
        quaternion = self._randomize_z_rotation()
        # quaternion = [0,0,0,1]
        for model_name in self.model_names:
            if model_name == object_name:
                model_state.model_name = object_name
                model_state.pose.position.x = random_pose[0] + noise[0]
                model_state.pose.position.y = random_pose[1] + noise[1]
                model_state.pose.position.z = random_pose[2] + noise[2]
                model_state.pose.orientation.x = quaternion[0]
                model_state.pose.orientation.y = quaternion[1]
                model_state.pose.orientation.z = quaternion[2]
                model_state.pose.orientation.w = quaternion[3]

                response = self.set_model_state(model_state)

        
        obj_position = [random_pose[0], random_pose[1], random_pose[2]]
        obj_orientation = [model_state.pose.orientation.x, model_state.pose.orientation.y, model_state.pose.orientation.z, model_state.pose.orientation.w]
        gz_object_position, gz_object_orientation = self.obj_get_state(object_name)
        obj_position[2] = obj_position[2] + gz_object_position[2]
        self.object_initial_pose = np.concatenate((obj_position, obj_orientation))

    def random_pose(self):
        x = random.uniform(0.48, 0.58)
        y = random.uniform(-0.12, 0.12)
        z = random.uniform(0, 0)
        return [x, y, z]

    def add_noise(self, mean, std_dev): 
        noisy_x = np.random.uniform(-0.005, 0.005)
        noisy_y = np.random.uniform(-0.005, 0.005)
        noisy_z = np.random.uniform(0, 0)
        return [noisy_x, noisy_y, noisy_z]

    def _randomize_z_rotation(self):
        roll = 0.0
        pitch = 0.0
        yaw = random.uniform(-0.866, 0.866)
        quaternion = quaternion_from_euler(roll, pitch, yaw)
        return quaternion
        
