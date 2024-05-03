#! /usr/bin/env python

import rospy
from geometry_msgs.msg import WrenchStamped
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from scipy.signal import butter, lfilter

timestamps = []
fx_data = []
fy_data = []
fz_data = []
mx_data = []
my_data = []
mz_data = []


fs = 1.0 

b, a = butter(1, cutoff_frequency, btype='low', analog=False)

zi = None

def wrench_callback(data):
    global zi  
    timestamps.append(rospy.get_time())  
    fx_data.append(data.wrench.force.x)
    fy_data.append(data.wrench.force.y)
    fz_data.append(data.wrench.force.z)
    mx_data.append(data.wrench.torque.x)
    my_data.append(data.wrench.torque.y)
    mz_data.append(data.wrench.torque.z)
    if len(timestamps) > 1:
        fx_data_filtered, zi = lfilter(b, a, [fx_data[-1]], zi=zi)
        fy_data_filtered, zi = lfilter(b, a, [fy_data[-1]], zi=zi)
        fz_data_filtered, zi = lfilter(b, a, [fz_data[-1]], zi=zi)
        mx_data_filtered, zi = lfilter(b, a, [mx_data[-1]], zi=zi)
        my_data_filtered, zi = lfilter(b, a, [my_data[-1]], zi=zi)
        mz_data_filtered, zi = lfilter(b, a, [mz_data[-1]], zi=zi)

        fx_data[-1] = fx_data_filtered[0]
        fy_data[-1] = fy_data_filtered[0]
        fz_data[-1] = fz_data_filtered[0]
        mx_data[-1] = mx_data_filtered[0]
        my_data[-1] = my_data_filtered[0]
        mz_data[-1] = mz_data_filtered[0]

def update_plot(frame):
    plt.clf()

    plt.subplot(3, 2, 1)
    plt.plot(timestamps, fx_data, color='b')
    plt.xlabel("Time (seconds)")
    plt.ylabel("Fx (N)")
    plt.title("Wrench Component Fx")
    plt.ylim(-10, 10) 

    plt.subplot(3, 2, 2)
    plt.plot(timestamps, fy_data, color='g')
    plt.xlabel("Time (seconds)")
    plt.ylabel("Fy (N)")
    plt.title("Wrench Component Fy")
    plt.ylim(-10, 10) 

    plt.subplot(3, 2, 3)
    plt.plot(timestamps, fz_data, color='r')
    plt.xlabel("Time (seconds)")
    plt.ylabel("Fz (N)")
    plt.title("Wrench Component Fz")
    plt.ylim(-10, 10) 

    plt.subplot(3, 2, 4)
    plt.plot(timestamps, mx_data, color='c')
    plt.xlabel("Time (seconds)")
    plt.ylabel("Mx (N-m)")
    plt.title("Wrench Component Mx")
    plt.ylim(-10, 10) 

    plt.subplot(3, 2, 5)
    plt.plot(timestamps, my_data, color='m')
    plt.xlabel("Time (seconds)")
    plt.ylabel("My (N-m)")
    plt.title("Wrench Component My")
    plt.ylim(-10, 10)

    plt.subplot(3, 2, 6)
    plt.plot(timestamps, mz_data, color='y')
    plt.xlabel("Time (seconds)")
    plt.ylabel("Mz (N-m)")
    plt.title("Wrench Component Mz")
    plt.ylim(-10, 10) 

    plt.tight_layout()


if __name__ == '__main__':

    rospy.init_node('wrench_plotter', anonymous=True)

    rospy.Subscriber('/franka_state_controller/F_ext', WrenchStamped, wrench_callback)

    plt.figure(figsize=(12, 8))

    ani = FuncAnimation(plt.gcf(), update_plot, interval=1000)

    plt.show()

    rospy.spin()
