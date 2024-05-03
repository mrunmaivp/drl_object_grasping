#! /usr/bin/env python

import rospy
import tf2_ros
from gazebo_msgs.srv import GetWorldProperties, GetModelState, SetModelState, DeleteModel
from gazebo_msgs.msg import ModelState
import numpy as np
from geometry_msgs.msg import PoseStamped
from tf2_geometry_msgs import do_transform_pose
from tf.transformations import quaternion_inverse, quaternion_multiply, euler_from_quaternion, quaternion_from_euler

INITIAL_GRIPPER_ORIENTATION = np.around(np.array([np.pi, 0, np.pi]),2)

def quaternion_multiply(q0, q1):
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

def compute_orientation_difference():
    rospy.init_node('orientation_difference_calculator')
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)
    get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
    world_specs = rospy.ServiceProxy('/gazebo/get_world_properties', GetWorldProperties)()
    model_names = world_specs.model_names
    object_name = "cube"

    for model_name in model_names:
            if model_name == object_name:
                data = get_model_state(model_name, "ground_plane")
                # print("DATA", data)
                obj_position = np.array([data.pose.position.x, data.pose.position.y, data.pose.position.z])
                obj_orientation = np.array([data.pose.orientation.x, data.pose.orientation.y, data.pose.orientation.z, data.pose.orientation.w])

    object_pose = PoseStamped()
    object_pose.header.frame_id = 'world'
    object_pose.pose.position.x = data.pose.position.x
    object_pose.pose.position.y = data.pose.position.y
    object_pose.pose.position.z = data.pose.position.z
    object_pose.pose.orientation.x = data.pose.orientation.x
    object_pose.pose.orientation.y = data.pose.orientation.y
    object_pose.pose.orientation.z = data.pose.orientation.z
    object_pose.pose.orientation.w = data.pose.orientation.w
    gripper_frame_id = 'panda_EE' 
    
    
    gripper_transform = tf_buffer.lookup_transform('world', gripper_frame_id, rospy.Time(), rospy.Duration(1.0))

    gripper_pose = PoseStamped()
    gripper_pose.header.frame_id = 'world'
    gripper_pose.pose.position.x = gripper_transform.transform.translation.x
    gripper_pose.pose.position.y = gripper_transform.transform.translation.y
    gripper_pose.pose.position.z = gripper_transform.transform.translation.z
    gripper_pose.pose.orientation.x = gripper_transform.transform.rotation.x
    gripper_pose.pose.orientation.y = gripper_transform.transform.rotation.y
    gripper_pose.pose.orientation.z = gripper_transform.transform.rotation.z
    gripper_pose.pose.orientation.w = gripper_transform.transform.rotation.w

    gripper_quaternion = np.array([gripper_transform.transform.rotation.x, gripper_transform.transform.rotation.y, gripper_transform.transform.rotation.z, gripper_transform.transform.rotation.w])
    angular_difference= np.dot(gripper_quaternion, quaternion_inverse(obj_orientation))
    cosine_similarity = gripper_quaternion.dot(obj_orientation)
    print("Angular difference", angular_difference)
    print("cosine_similarity", round(cosine_similarity, 2))
    quaternion_multiplication = quaternion_multiply(gripper_quaternion, obj_orientation)
    print("multiplication", quaternion_multiplication)

    orientation_difference = do_transform_pose(data, gripper_transform)
    orientation_quaternion = np.array([orientation_difference.pose.orientation.x, orientation_difference.pose.orientation.y, orientation_difference.pose.orientation.z, orientation_difference.pose.orientation.w])

    print("Orientation Difference (Quaternion):", orientation_difference)
    gripper_roll, gripper_pitch, gripper_yaw = euler_from_quaternion(orientation_quaternion)
    print("GRIPPER ORIENTATION DIFF", gripper_roll, gripper_pitch, gripper_yaw)
    orientation_difference2 = np.array([gripper_roll, gripper_pitch, gripper_yaw])
    normalized_orientation_difference = orientation_difference2 / np.pi
    print("NORMALIZED", normalized_orientation_difference)
    reward = -np.linalg.norm(normalized_orientation_difference, ord=2)
    print("REWARD", reward)

    
        

if __name__ == '__main__':
    rospy.init_node('orientation_difference_calculator')
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)
    get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
    world_specs = rospy.ServiceProxy('/gazebo/get_world_properties', GetWorldProperties)()
    model_names = world_specs.model_names
    object_name = "cube"

    for model_name in model_names:
            if model_name == object_name:
                data = get_model_state(model_name, "ground_plane")
                obj_position = np.array([data.pose.position.x, data.pose.position.y, data.pose.position.z])
                obj_orientation = np.array([data.pose.orientation.x, data.pose.orientation.y, data.pose.orientation.z, data.pose.orientation.w])

    gripper_transform = tf_buffer.lookup_transform('world', 'panda_leftfinger', rospy.Time(), rospy.Duration(1.0))
    gripper_quaternion_inverse = np.array([gripper_transform.transform.rotation.x, gripper_transform.transform.rotation.y, gripper_transform.transform.rotation.z, -gripper_transform.transform.rotation.w])


    qr = quaternion_multiply(obj_orientation, gripper_quaternion_inverse)
 
    gripper_roll, gripper_pitch, gripper_yaw = euler_from_quaternion(qr)

    orientation_difference = np.around(np.array([gripper_roll, gripper_pitch, gripper_yaw]),2)

    orientation_difference = np.abs(orientation_difference) - np.abs(INITIAL_GRIPPER_ORIENTATION)
    normalized_orientation_difference = orientation_difference / np.pi

    reward = np.exp(-np.linalg.norm(normalized_orientation_difference, ord=2))
