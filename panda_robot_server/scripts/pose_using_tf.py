#!/usr/bin/env python

import rospy
import tf2_ros
from geometry_msgs.msg import PoseStamped
from gazebo_msgs.srv import GetWorldProperties, GetModelState, SetModelState, DeleteModel
from gazebo_msgs.msg import ModelState
import numpy as np
from tf.transformations import quaternion_inverse, quaternion_multiply, euler_from_quaternion, quaternion_from_euler
import tf

if __name__ == "__main__":
    rospy.init_node("ee_pose_listener")
    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)
    get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
    world_specs = rospy.ServiceProxy('/gazebo/get_world_properties', GetWorldProperties)()
    model_names = world_specs.model_names
    object_name = "cube"

    # while not rospy.is_shutdown():
    left_trans = tfBuffer.lookup_transform('world', 'panda_leftfinger', rospy.Time(0), rospy.Duration(1.0))
    print("LEFT_TRANS", left_trans)
    trans = tfBuffer.lookup_transform('world', 'panda_EE', rospy.Time(0), rospy.Duration(1.0))
    print("TRANS", trans)
    right_trans = tfBuffer.lookup_transform('world', 'panda_rightfinger', rospy.Time(0), rospy.Duration(1.0))
    left_finger_pose = [left_trans.transform.translation.x + 0.018, left_trans.transform.translation.y + 0.018, left_trans.transform.translation.z + 0.018]
    right_finger_pose = [right_trans.transform.translation.x, right_trans.transform.translation.y, right_trans.transform.translation.z]
    print("LEFT", left_finger_pose)
    print("RIGHT", right_finger_pose)
    gripper_midpoint_pose = [(left_finger_pose[0] + right_finger_pose[0])/2, (left_finger_pose[1]+right_finger_pose[1])/2, (left_finger_pose[2]+right_finger_pose[2])/2]
    print("MIDPOINT", gripper_midpoint_pose)
    left_gripper_orientation = [left_trans.transform.rotation.x, left_trans.transform.rotation.y, left_trans.transform.rotation.z, left_trans.transform.rotation.w]
    right_gripper_orientation = [right_trans.transform.rotation.x, right_trans.transform.rotation.y, right_trans.transform.rotation.z, right_trans.transform.rotation.w]
    
    trans_orientation = [trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z, trans.transform.rotation.w]
    
    print("LEFT_ORIENTATION", left_gripper_orientation)
    print("RIGHT_ORIENTATION", right_gripper_orientation)
    # left_gripper_orientation[0] = left_gripper_orientation[0] - np.pi
    euler_list = list(euler_from_quaternion(left_gripper_orientation))
    print("EULER", euler_list)
    print("QUTERNION", quaternion_from_euler(euler_list[0], euler_list[1], euler_list[2]))

    for model_name in model_names:
            if model_name == object_name:
                data = get_model_state(model_name, "ground_plane")
                print("DATA", data)
                obj_position = np.array([data.pose.position.x, data.pose.position.y, data.pose.position.z])
                obj_orientation = np.array([data.pose.orientation.x, data.pose.orientation.y, data.pose.orientation.z, data.pose.orientation.w])
                print("Obj Position", obj_position)
                print("Obj Orientation", obj_orientation)

    gripper_quat = quaternion_multiply(quaternion_inverse(trans_orientation), obj_orientation)
    print("QUAT", gripper_quat)


    gripper_roll, gripper_pitch, gripper_yaw = euler_from_quaternion(gripper_quat)
    print("GRIPPER ORIENTATION DIFF", gripper_roll, gripper_pitch, gripper_yaw)
    orientation_difference = np.array([gripper_roll, gripper_pitch, gripper_yaw])
    normalized_orientation_difference = orientation_difference / np.pi
    print("NORMALIZED", normalized_orientation_difference)
    reward = -np.linalg.norm(normalized_orientation_difference, ord=2)
    print("REWARD", reward)

    #Frobenius Norm method
    gripper_matrix = tf.transformations.quaternion_matrix(trans_orientation)
    object_matrix = tf.transformations.quaternion_matrix(obj_orientation)

    orientation_difference = np.linalg.norm(gripper_matrix - object_matrix)

    print("DIFFERENCE", orientation_difference)
    reward2 = 1.0 / (1.0 + orientation_difference)
    print("REWARD2", reward2)
    

    

    



