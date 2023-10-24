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
    time_vals = []
    magfield_x = []
    magfield_y = []
    magfield_z = []

    with rosbag.Bag(filepath) as bag:
        for _, message, _ in bag.read_messages(topics=['/imu']):
            lin_acc_x.append(message.IMU.linear_acceleration.x)
            lin_acc_y.append(message.IMU.linear_acceleration.y)
            lin_acc_z.append(message.IMU.linear_acceleration.z)
            ang_vel_x.append(message.IMU.angular_velocity.x)
            ang_vel_y.append(message.IMU.angular_velocity.y)
            ang_vel_z.append(message.IMU.angular_velocity.z)
            euler = tf.transformations.euler_from_quaternion([message.IMU.orientation.x, message.IMU.orientation.y, message.IMU.orientation.z, message.IMU.orientation.w])
            roll.append(euler[0])
            pitch.append(euler[1])
            yaw.append(euler[2])
            time_vals.append(message.header.stamp.secs)
            magfield_x.append(message.MagField.magnetic_field.x)
	    magfield_y.append(message.MagField.magnetic_field.y)
	    magfield_z.append(message.MagField.magnetic_field.z)

    return time_vals, lin_acc_x, lin_acc_y, lin_acc_z, ang_vel_x, ang_vel_y, ang_vel_z, roll, pitch, yaw, magfield_x, magfield_y, magfield_z

def scale_time_to_range(time, original_min, original_max, new_min, new_max):
    """Linearly scales a time value from its original range to a new range."""
    return new_min + (time - original_min) * (new_max - new_min) / (original_max - original_min)

def calculate_rmse(data, ref_value):
    """Calculate the root mean square error of the data with respect to a reference value."""
    data_array = np.array(data)  # Convert the list to a numpy array
    return np.sqrt(np.mean((data_array - ref_value) ** 2))

def plot_individual(time, data, title, ylabel, subplot_num, color):
    ax = plt.subplot(4, 1, subplot_num)
    ax.plot(time, data, color=color)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Time (s)") 
    mean_val = np.mean(data)
    std_val = np.std(data)
    rmse1 = calculate_rmse(data, mean_val)
    rmse2 = calculate_rmse(data, 0)
    rmse3 = calculate_rmse(data, data[0])
    rmse4 = calculate_rmse(data, np.median(data))
    avg_rmse = np.mean([rmse1, rmse2, rmse3, rmse4])
    
    ax.text(0.05, 0.95, "Mean: {:.6f}".format(mean_val), transform=ax.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.text(0.48, 0.95, "Avg RMSE: {:.6f}".format(avg_rmse), transform=ax.transAxes, verticalalignment='top', horizontalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.text(0.95, 0.95, "Std Deviation: {:.6f}".format(std_val), transform=ax.transAxes, verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

def plot_combined(time, data_x, data_y, data_z, title, ylabel, subplot_num):
    ax = plt.subplot(4, 1, subplot_num)  
    ax.plot(time, data_x, color='r', label='X')
    ax.plot(time, data_y, color='g', label='Y')
    ax.plot(time, data_z, color='b', label='Z')
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Time (s)")
    ax.legend(loc='upper right')

filepath = 'fifteen_min_2.bag'
time_vals, lin_acc_x, lin_acc_y, lin_acc_z, ang_vel_x, ang_vel_y, ang_vel_z, roll, pitch, yaw, magfield_x, magfield_y, magfield_z = extract_imu_data(filepath)

# Find the original time range.
original_start_time = min(time_vals)
original_end_time = max(time_vals)

# Map the original time range to 0 to 664 seconds.
scaled_time_vals = [scale_time_to_range(t, original_start_time, original_end_time, 0, 664) for t in time_vals]

# Linear Acceleration Figure
plt.figure(figsize=(12,12))
plot_individual(scaled_time_vals, lin_acc_x, "Linear Acceleration X", "m/s^2", 1, 'r')
plot_individual(scaled_time_vals, lin_acc_y, "Linear Acceleration Y", "m/s^2", 2, 'g')
plot_individual(scaled_time_vals, lin_acc_z, "Linear Acceleration Z", "m/s^2", 3, 'b')
plot_combined(scaled_time_vals, lin_acc_x, lin_acc_y, lin_acc_z, "Combined Linear Acceleration", "m/s^2", 4)
plt.tight_layout()

# Angular Velocity Figure
plt.figure(figsize=(12,12))
plot_individual(scaled_time_vals, ang_vel_x, "Angular Velocity X", "rad/s", 1, 'r')
plot_individual(scaled_time_vals, ang_vel_y, "Angular Velocity Y", "rad/s", 2, 'g')
plot_individual(scaled_time_vals, ang_vel_z, "Angular Velocity Z", "rad/s", 3, 'b')
plot_combined(scaled_time_vals, ang_vel_x, ang_vel_y, ang_vel_z, "Combined Angular Velocity", "rad/s", 4)
plt.tight_layout()

# RPY Figure
plt.figure(figsize=(12,12))
plot_individual(scaled_time_vals, roll, "Roll", "rad", 1, 'r')
plot_individual(scaled_time_vals, pitch, "Pitch", "rad", 2, 'g')
plot_individual(scaled_time_vals, yaw, "Yaw", "rad", 3, 'b')
plot_combined(scaled_time_vals, roll, pitch, yaw, "Combined Roll-Pitch-Yaw", "rad", 4)
plt.tight_layout()

# Magnetic Field Figure
plt.figure(figsize=(12,12))
plot_individual(scaled_time_vals, magfield_x, "Magnetic Field X", "Gauss", 1, 'r')
plot_individual(scaled_time_vals, magfield_y, "Magnetic Field Y", "Gauss", 2, 'g')
plot_individual(scaled_time_vals, magfield_z, "Magnetic Field Z", "Gauss", 3, 'b')
plot_combined(scaled_time_vals, magfield_x, magfield_y, magfield_z, "Combined Magnetic Field", "Gauss", 4)
plt.tight_layout()

plt.show()
