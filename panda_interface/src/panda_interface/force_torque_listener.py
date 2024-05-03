#!/usr/bin/env python

import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
import numpy as np
from geometry_msgs.msg import WrenchStamped

# Define your threshold values for force/torque (adjust as needed)
FORCE_THRESHOLD = 1.0  # Example threshold in Newtons

class ForceTorqueMonitorNode:
    def __init__(self):
        rospy.init_node('force_torque_listener')

        self.force_torque_sub = rospy.Subscriber('/franka_state_controller/F_ext', WrenchStamped, self.force_torque_callback)
        self.reward_pub = rospy.Publisher('/rl_agent_reward', Float64, queue_size=10)
        self.tcp_ft = None
        self.tcp_ft_buffer = []
        self.tcp_buffer_size = 20

    def force_torque_callback(self, msg):
        fx = msg.wrench.force.x
        fy = msg.wrench.force.y
        fz = msg.wrench.force.z
        mx = msg.wrench.torque.x
        my = msg.wrench.torque.y
        mz = msg.wrench.torque.z

        self.tcp_ft_buffer.append([fx, fy, fz, mx, my, mz])
        if len(self.tcp_ft_buffer) > self.tcp_buffer_size:
            self.tcp_ft_buffer.pop(0)

        self.tcp_ft = np.mean(self.tcp_ft_buffer, axis=0)

        if any(abs(self.tcp_ft[3:]) > FORCE_THRESHOLD):
            negative_reward = -2.0  
            self.reward_pub.publish(negative_reward)
        else:
            self.reward_pub.publish(0)

if __name__ == '__main__':
    try:
        node = ForceTorqueMonitorNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass