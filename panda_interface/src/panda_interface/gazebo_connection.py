#!/usr/bin/env python

import rospy
import subprocess
import time
from std_srvs.srv import Empty

class GazeboConnection:
    def __init__(self):
        pass

    def is_gazebo_running(self):
        process = subprocess.Popen(["rostopic", "list"], stdout=subprocess.PIPE)
        output, _ = process.communicate()
        return '/gazebo/link_states' in output

    def start_gazebo(self):
        # if not self.is_gazebo_running():
        # Launch Gazebo using roslaunch
        subprocess.Popen(["roslaunch", "panda_moveit_config", "demo_gazebo.launch"])
        # Wait a moment for Gazebo to start
        time.sleep(5)

    def stop_gazebo(self):
        # Terminate Gazebo
        subprocess.Popen(["pkill", "gzserver"])
        subprocess.Popen(["pkill", "gzclient"])

    def pause_physics(self):
        rospy.wait_for_service('/gazebo/pause_physics', timeout=5)
        pause_physics = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        pause_physics()
        rospy.loginfo("Physics Paused")

    def unpause_physics(self):
        rospy.wait_for_service('/gazebo/unpause_physics', timeout=5)
        unpause_physics = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        unpause_physics()
        rospy.loginfo("Physics Unpaused")

    def reset_simulation(self):
        try:
            # Wait for the reset simulation service to be available
            rospy.wait_for_service('/gazebo/reset_simulation', timeout=5)
            reset_sim = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
            reset_sim()  # Call the reset service
            rospy.loginfo("Simulation reset.")
        except rospy.ServiceException as e:
            rospy.logerr("Service call to reset simulation failed: %s" % str(e))

