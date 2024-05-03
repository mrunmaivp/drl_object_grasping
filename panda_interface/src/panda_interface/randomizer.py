#! /usr/bin/env python

import rospy
import sys
from gazebo_msgs.msg import ModelState

import xml.etree.ElementTree as ET
import random

OBJECT_SDF_PATH = "/home/ros2/mt_panda_grasping_ws/src/panda_grasping/load_rl_env/urdf"


class Randomizer():
    def __init__(self):
        pass

    def randomize_robot_initial_pose(self):
        x = random.uniform(0.45, 0.55)
        y = random.uniform(-0.05, 0.05)
        z = random.uniform(0.2, 0.3)
        return [x, y, z]

    def _randomize_z_rotation(self):
            roll = 0.0
            pitch = 0.0
            yaw = random.uniform(-0.785, 0.785)
            quaternion = quaternion_from_euler(roll, pitch, yaw)
            return quaternion

    def object_position_randomizer(self, randomize_orientation):
        model_state = ModelState()
        model_state.pose.position.x  = random.uniform(0.45, 0.55)
        model_state.pose.position.y = random.uniform(-0.08, 0.08)
        model_state.pose.position.z  = random.uniform(0, 0)
        if randomize_orientation:
            quaternion = self._randomize_z_rotation()
        else:
            quaternion = [0,0,0,1]
        model_state.pose.orientation.x = quaternion[0]
        model_state.pose.orientation.y = quaternion[1]
        model_state.pose.orientation.z = quaternion[2]
        model_state.pose.orientation.w =  quaternion[3]

    def robot_position_randomizer(self):
        pass

    def object_mass_randomizer(self, obj_sdf_path, object_name):
        tree = ET.parse(obj_sdf_path)
        root = tree.getroot()
        link_element = root.find('.//link[@name="cube"]')
        mass_element = link_element.find('.//mass')
        random_mass = random.uniform(0.1, 0.8)
        i_xx, i_yy, i_zz = self._iniertia_calculator(float(random_mass), link_element, object_name)
        root = self._update_inertia_matrix([i_xx, i_yy, i_zz], random_mass, root, obj_sdf_path)
        root = self.randmize_friction_coefficient(root)
        self._update_obj_sdf(obj_sdf_path, root)

    def _update_obj_sdf(self, obj_sdf_path, root):
        tree = ET.ElementTree(root)
        tree.write(obj_sdf_path, encoding="utf-8", xml_declaration=True)
        

        updated_tree = ET.parse(obj_sdf_path)
        updated_root = updated_tree.getroot()

        # Print the content of the updated XML file
        updated_xml_content = ET.tostring(updated_root, encoding="unicode")



    def _update_inertia_matrix(self, inertia_matrix, mass, root, obj_sdf_path):
        inertial_tag = root.find(".//inertial")

        if inertial_tag is not None:
            # Update values for the <mass> tag
            inertial_tag.find(".//mass").text = str(mass)
            inertial_tag.find(".//ixx").text = str(inertia_matrix[0])
            inertial_tag.find(".//ixy").text = str(0)
            inertial_tag.find(".//ixz").text = str(0)
            inertial_tag.find(".//iyy").text = str(inertia_matrix[1])
            inertial_tag.find(".//iyz").text = str(0)
            inertial_tag.find(".//izz").text = str(inertia_matrix[2])

        else:
            print("The <inertia> tag is not found in the XML content.")

        return root

    def randmize_friction_coefficient(self, root):
        friction_tag = root.find(".//friction/ode") 
        if friction_tag is not None:
            friction_value = random.uniform(0.3, 1.0)
            friction_tag.find(".//mu").text = str(friction_value)
            friction_tag.find(".//mu2").text = str(friction_value)

        return root

    def _iniertia_calculator(self, mass, link_element, obj_type):
        if obj_type == "cube":
            size_element = link_element.find('.//size')
            width_list = [float(element) for element in (size_element.text).split()]
            # print("width_list", width_list)
            i_xx = (1/6) * mass * pow(width_list[0], 2)
            i_yy = (1/6) * mass * pow(width_list[1], 2)
            i_zz = (1/6) * mass * pow(width_list[2], 2)

        elif obj_type == "cylinder":
            i_xx =  (1/12) * mass * (3 * R**2 + H**2)
            i_yy = i_xx
            i_zz = (1/2) * mass * R**2

        elif obj_type == "cuboid":
            i_xx = (1/12) * mass * (b^2 + c^2)
            i_yy = (1/12) * mass * (a^2 + c^2)
            i_zz = (1/12) * mass * (a^2 + b^2)

        return i_xx, i_yy, i_zz
    
