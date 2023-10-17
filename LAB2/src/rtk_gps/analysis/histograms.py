import rosbag
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import pandas as pd

style.use('ggplot')

def read_bag_data(bag_path, topic_name):
    bag = rosbag.Bag(bag_path)
    msgs = []
    for _, msg, _ in bag.read_messages(topics=[topic_name]):
        msgs.append(msg)
    bag.close()
    return msgs

def analyze_data(msgs):
    # Convert to dataframe
    df = pd.DataFrame([(msg.utm_easting, msg.utm_northing) for msg in msgs], columns=['easting', 'northing'])
    # Calculate Mean Deviation
    mean_e = df['easting'].mean()
    mean_n = df['northing'].mean()
    mean_dev_e = abs(df['easting'] - mean_e).mean()
    mean_dev_n = abs(df['northing'] - mean_n).mean()
    return df, mean_dev_e, mean_dev_n

# Reading data from the four bag files
msgs_stationary_occ = read_bag_data('stationary_data_occ.bag', '/GPS')
msgs_walking_occ = read_bag_data('walking_data_occ.bag', '/GPS')
msgs_stationary = read_bag_data('stationary_data.bag', '/GPS')
msgs_walking_circle = read_bag_data('walking_data_circle.bag', '/GPS')

df_stationary_occ, mean_dev_e_stationary_occ, mean_dev_n_stationary_occ = analyze_data(msgs_stationary_occ)
df_walking_occ, mean_dev_e_walking_occ, mean_dev_n_walking_occ = analyze_data(msgs_walking_occ)
df_stationary, mean_dev_e_stationary, mean_dev_n_stationary = analyze_data(msgs_stationary)
df_walking_circle, mean_dev_e_walking_circle, mean_dev_n_walking_circle = analyze_data(msgs_walking_circle)

# Plotting for the first set of data (stationary_data_occ & walking_data_occ)
fig1, axs1 = plt.subplots(2, 1, figsize=(10, 10))

# Plot stationary_occ
cax1 = axs1[0].hist2d(df_stationary_occ['easting'], df_stationary_occ['northing'], bins=50, cmap='plasma')
axs1[0].set_title('2D Histogram for Stationary Data (OCC)')
axs1[0].set_xlabel('UTM Easting (m)')
axs1[0].set_ylabel('UTM Northing (m)')
stats_text_stationary_occ = "\n".join([
    "Mean Deviation (Easting): {:.4f}".format(mean_dev_e_stationary_occ),
    "Mean Deviation (Northing): {:.4f}".format(mean_dev_n_stationary_occ)
])
axs1[0].text(0.05, 0.95, stats_text_stationary_occ, transform=axs1[0].transAxes, fontsize=8, verticalalignment='top', bbox=dict(boxstyle="square,pad=0.3", facecolor="white", edgecolor="black"))
fig1.colorbar(cax1[3], ax=axs1[0])

# Plot walking_occ
cax2 = axs1[1].hist2d(df_walking_occ['easting'], df_walking_occ['northing'], bins=50, cmap='plasma')
axs1[1].set_title('2D Histogram for Walking Data (OCC)')
axs1[1].set_xlabel('UTM Easting (m)')
axs1[1].set_ylabel('UTM Northing (m)')
stats_text_walking_occ = "\n".join([
    "Mean Deviation (Easting): {:.4f}".format(mean_dev_e_walking_occ),
    "Mean Deviation (Northing): {:.4f}".format(mean_dev_n_walking_occ)
])
axs1[1].text(0.05, 0.95, stats_text_walking_occ, transform=axs1[1].transAxes, fontsize=8, verticalalignment='top', bbox=dict(boxstyle="square,pad=0.3", facecolor="white", edgecolor="black"))
fig1.colorbar(cax2[3], ax=axs1[1])

# Plotting for the second set of data (stationary_data & walking_data_circle)
fig2, axs2 = plt.subplots(2, 1, figsize=(10, 10))

# Plot stationary
cax3 = axs2[0].hist2d(df_stationary['easting'], df_stationary['northing'], bins=50, cmap='plasma')
axs2[0].set_title('2D Histogram for Stationary Data')
axs2[0].set_xlabel('UTM Easting (m)')
axs2[0].set_ylabel('UTM Northing (m)')
stats_text_stationary = "\n".join([
    "Mean Deviation (Easting): {:.4f}".format(mean_dev_e_stationary),
    "Mean Deviation (Northing): {:.4f}".format(mean_dev_n_stationary)
])
axs2[0].text(0.05, 0.95, stats_text_stationary, transform=axs2[0].transAxes, fontsize=8, verticalalignment='top', bbox=dict(boxstyle="square,pad=0.3", facecolor="white", edgecolor="black"))
fig2.colorbar(cax3[3], ax=axs2[0])

# Plot walking_circle
cax4 = axs2[1].hist2d(df_walking_circle['easting'], df_walking_circle['northing'], bins=50, cmap='plasma')
axs2[1].set_title('2D Histogram for Walking Data (Circle)')
axs2[1].set_xlabel('UTM Easting (m)')
axs2[1].set_ylabel('UTM Northing (m)')
stats_text_walking_circle = "\n".join([
    "Mean Deviation (Easting): {:.4f}".format(mean_dev_e_walking_circle),
    "Mean Deviation (Northing): {:.4f}".format(mean_dev_n_walking_circle)
])
axs2[1].text(0.05, 0.95, stats_text_walking_circle, transform=axs2[1].transAxes, fontsize=8, verticalalignment='top', bbox=dict(boxstyle="square,pad=0.3", facecolor="white", edgecolor="black"))
fig2.colorbar(cax4[3], ax=axs2[1])

plt.tight_layout()
plt.show()

