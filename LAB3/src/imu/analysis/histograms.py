#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import rosbag
import numpy as np
import tf

def extract_imu_data(filepath):
    lin_acc_x = []
    lin_acc_y = []
    lin_acc_z = []
    ang_vel_x = []
    ang_vel_y = []
    ang_vel_z = []
    roll = []
    pitch = []
    yaw = []

    with rosbag.Bag(filepath) as bag:
        for _, message, _ in bag.read_messages(topics=['/imu']):
            lin_acc_x.append(message.IMU.linear_acceleration.x)
            lin_acc_y.append(message.IMU.linear_acceleration.y)
            lin_acc_z.append(message.IMU.linear_acceleration.z)
            ang_vel_x.append(message.IMU.angular_velocity.x)
            ang_vel_y.append(message.IMU.angular_velocity.y)
            ang_vel_z.append(message.IMU.angular_velocity.z)
            euler = tf.transformations.euler_from_quaternion(
                [message.IMU.orientation.x, message.IMU.orientation.y, message.IMU.orientation.z, message.IMU.orientation.w])
            roll.append(euler[0])
            pitch.append(euler[1])
            yaw.append(euler[2])

    return lin_acc_x, lin_acc_y, lin_acc_z, ang_vel_x, ang_vel_y, ang_vel_z, roll, pitch, yaw

def plot_histogram(data, title, subplot_num, color):
    ax = plt.subplot(2, 2, subplot_num)
    ax.hist(data, bins=50, color=color, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")

    mean_val = np.mean(data)
    std_val = np.std(data)
    
    ax.text(0.05, 0.95, "Mean: {:.6f}".format(mean_val), transform=ax.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.text(0.95, 0.95, "Std Deviation: {:.6f}".format(std_val), transform=ax.transAxes, verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

def plot_combined_histogram(data_x, data_y, data_z, title, subplot_num):
    ax = plt.subplot(2, 2, subplot_num)
    ax.hist(data_x, bins=50, color='r', alpha=0.5, label='X')
    ax.hist(data_y, bins=50, color='g', alpha=0.5, label='Y')
    ax.hist(data_z, bins=50, color='b', alpha=0.5, label='Z')
    ax.set_title(title)
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    ax.legend(loc='upper right')

filepath = 'fifteen_min_2.bag'
lin_acc_x, lin_acc_y, lin_acc_z, ang_vel_x, ang_vel_y, ang_vel_z, roll, pitch, yaw = extract_imu_data(filepath)

# Linear Acceleration Histograms
plt.figure(figsize=(12,10))
plot_histogram(lin_acc_x, "Histogram of Linear Acceleration X", 1, 'r')
plot_histogram(lin_acc_y, "Histogram of Linear Acceleration Y", 2, 'g')
plot_histogram(lin_acc_z, "Histogram of Linear Acceleration Z", 3, 'b')
plot_combined_histogram(lin_acc_x, lin_acc_y, lin_acc_z, "Combined Linear Acceleration", 4)
plt.tight_layout()

# Angular Velocity Histograms
plt.figure(figsize=(12,10))
plot_histogram(ang_vel_x, "Histogram of Angular Velocity X", 1, 'r')
plot_histogram(ang_vel_y, "Histogram of Angular Velocity Y", 2, 'g')
plot_histogram(ang_vel_z, "Histogram of Angular Velocity Z", 3, 'b')
plot_combined_histogram(ang_vel_x, ang_vel_y, ang_vel_z, "Combined Angular Velocity", 4)
plt.tight_layout()

# Roll-Pitch-Yaw Histograms
plt.figure(figsize=(12,10))
plot_histogram(roll, "Histogram of Roll", 1, 'r')
plot_histogram(pitch, "Histogram of Pitch", 2, 'g')
plot_histogram(yaw, "Histogram of Yaw", 3, 'b')
plot_combined_histogram(roll, pitch, yaw, "Combined Roll-Pitch-Yaw", 4)
plt.tight_layout()

plt.show()

