import rosbag
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def read_altitude_data(bag_path, topic_name):
    bag = rosbag.Bag(bag_path)
    times = []
    altitudes = []
    for _, msg, _ in bag.read_messages(topics=[topic_name]):
        times.append(msg.header.stamp.to_sec()) # Convert ROS Time to seconds
        altitudes.append(msg.altitude)
    bag.close()
    return times, altitudes

# Read altitude data from each bag file
times_stationary_occ, altitudes_stationary_occ = read_altitude_data('stationary_data_occ.bag', '/GPS')
times_walking_occ, altitudes_walking_occ = read_altitude_data('walking_data_occ.bag', '/GPS')
times_stationary, altitudes_stationary = read_altitude_data('stationary_data.bag', '/GPS')
times_walking_circle, altitudes_walking_circle = read_altitude_data('walking_data_circle.bag', '/GPS')

# Create subplots with independent x-axes
fig, axs = plt.subplots(4, 1, figsize=(10, 12))

# Plot altitude vs time for each dataset
axs[0].plot(times_stationary_occ, altitudes_stationary_occ, label='Stationary (OCC)', color='b')
axs[0].set_ylabel('Altitude (m)')
axs[0].set_xlabel('Time (s)')
axs[0].set_title('Altitude vs Time for Stationary Data (OCC)')
axs[0].legend()

axs[1].plot(times_walking_occ, altitudes_walking_occ, label='Walking (OCC)', color='g')
axs[1].set_ylabel('Altitude (m)')
axs[1].set_xlabel('Time (s)')
axs[1].set_title('Altitude vs Time for Walking Data (OCC)')
axs[1].legend()

axs[2].plot(times_stationary, altitudes_stationary, label='Stationary', color='r')
axs[2].set_ylabel('Altitude (m)')
axs[2].set_xlabel('Time (s)')
axs[2].set_title('Altitude vs Time for Stationary Data')
axs[2].legend()

axs[3].plot(times_walking_circle, altitudes_walking_circle, label='Walking (Circle)', color='c')
axs[3].set_xlabel('Time (s)')
axs[3].set_ylabel('Altitude (m)')
axs[3].set_title('Altitude vs Time for Walking Data (Circle)')
axs[3].legend()

plt.tight_layout()
plt.show()
