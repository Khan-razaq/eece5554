import rosbag
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import pandas as pd
import statistics

style.use('ggplot')

def read_bag_data(bag_path, topic_name):
    bag = rosbag.Bag(bag_path)
    msgs = []
    for topic, msg, _ in bag.read_messages(topics=[topic_name]):
        msgs.append(msg)
    bag.close()
    return msgs

def calculate_rmse(segment, reference):
    """Calculate RMSE for a data segment against a reference value."""
    return np.sqrt(((segment - reference) ** 2).mean())

# Reading data
msgs_stationary = read_bag_data('stationary_data_occ.bag', '/GPS')
msgs_walking = read_bag_data('walking_data_occ.bag', '/GPS')

# Convert to dataframe
df_stationary = pd.DataFrame([(msg.utm_easting, msg.utm_northing) for msg in msgs_stationary], columns=['easting', 'northing'])
x_stationary = df_stationary['easting']
y_stationary = df_stationary['northing']

df_walking = pd.DataFrame([(msg.utm_easting, msg.utm_northing) for msg in msgs_walking], columns=['easting', 'northing'])
x_walking = df_walking['easting']
y_walking = df_walking['northing']

# Calculate RMSE for walking data
x_mean = x_walking.mean()
y_mean = y_walking.mean()

segment_length = len(x_walking) // 4
rmse_values = []

for i in range(4):
    start_index = i * segment_length
    end_index = (i + 1) * segment_length
    x_segment = x_walking[start_index:end_index]
    y_segment = y_walking[start_index:end_index]

    rmse_x = calculate_rmse(x_segment, x_mean)
    rmse_y = calculate_rmse(y_segment, y_mean)

    rmse_values.append((rmse_x + rmse_y) / 2)

avg_rmse = np.mean(rmse_values)

# Print RMSE values
print("RMSE1: {:.12f}".format(rmse_values[0]))
print("RMSE2: {:.12f}".format(rmse_values[1]))
print("RMSE3: {:.12f}".format(rmse_values[2]))
print("RMSE4: {:.12f}".format(rmse_values[3]))
print("Average RMSE: {:.12f}".format(avg_rmse))

fig, axs = plt.subplots(2, 1, figsize=(10, 10))

# Adjust axes to show values in standard decimal notation
axs[0].ticklabel_format(useOffset=False, style='plain')
axs[1].ticklabel_format(useOffset=False, style='plain')

# Calculate standard deviation for stationary data
std_dev_x_stationary = statistics.stdev(x_stationary)
std_dev_y_stationary = statistics.stdev(y_stationary)

# Plotting for stationary data
axs[0].scatter(x_stationary, y_stationary, linewidths=1)
axs[0].set_title('Stationary Data: UTM Easting vs UTM Northing')
axs[0].set_xlabel('UTM Easting (m)', fontsize=10)
axs[0].set_ylabel('UTM Northing (m)', fontsize=10)

# Add Standard Deviation values to the stationary data plot as text
stats_text_stationary = "\n".join([
    "Std Deviation (Easting): {:.12f}".format(std_dev_x_stationary),
    "Std Deviation (Northing): {:.12f}".format(std_dev_y_stationary)
])
axs[0].text(0.95, 0.95, stats_text_stationary, transform=axs[0].transAxes, fontsize=8, verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle="square,pad=0.3", facecolor="white", edgecolor="black"))

# Plotting for walking data
axs[1].scatter(x_walking, y_walking, linewidths=1)
axs[1].set_title('Walking Data: UTM Easting vs UTM Northing')
axs[1].set_xlabel('UTM Easting (m)', fontsize=10)
axs[1].set_ylabel('UTM Northing (m)', fontsize=10)

# Add RMSE values and Standard Deviation to the walking data plot as text
std_dev_x = statistics.stdev(x_walking)
stats_text = "\n".join([
    "RMSE1: {:.12f}".format(rmse_values[0]),
    "RMSE2: {:.12f}".format(rmse_values[1]),
    "RMSE3: {:.12f}".format(rmse_values[2]),
    "RMSE4: {:.12f}".format(rmse_values[3]),
    "Avg RMSE: {:.12f}".format(avg_rmse),
    "Std Deviation (X): {:.12f}".format(std_dev_x)
])
axs[1].text(0.95, 0.95, stats_text, transform=axs[1].transAxes, fontsize=8, verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle="square,pad=0.3", facecolor="white", edgecolor="black"))

plt.tight_layout()
plt.show()

