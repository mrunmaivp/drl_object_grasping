#! /usr/bin/env python

import rospy
import tf2_ros
import geometry_msgs.msg

def gripper_frame_callback(frame_id):
    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)

    target_frame = frame_id 
    source_frame = "world"  

    try:

        transform = tfBuffer.lookup_transform(target_frame, source_frame, rospy.Time(0), rospy.Duration(5.0))

        trans = transform.transform.translation
        rot = transform.transform.rotation
        print(f"Gripper Link Frame '{target_frame}' in the '{source_frame}' frame:")
        print(f"Position: x={trans.x}, y={trans.y}, z={trans.z}")
        print(f"Orientation: x={rot.x}, y={rot.y}, z={rot.z}, w={rot.w}")

    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
        print("An error occurred while looking up the transformation.")

def main():
    rospy.init_node('gripper_frame_listener', anonymous=True)
    gripper_frame_callback('panda_link7')
    gripper_frame_callback('panda_link8')

    rospy.spin()

if __name__ == '__main__':
    main()

