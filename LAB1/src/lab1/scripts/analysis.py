#!/usr/bin/env python3

import matplotlib.pyplot as plt
import rosbag

if __name__ == '__main__':
    
    def extract_bag_data(filepath):
        east_vals = []
        north_vals = []
        time_vals = []
        alt_vals = []

        with rosbag.Bag(filepath) as bag:
            for _, message, _ in bag.read_messages(topics=['/gps_pub']):
                east_vals.append(message.utm_easting)
                north_vals.append(message.utm_northing)
                time_vals.append(message.header.stamp.secs)
                alt_vals.append(message.altitude)

        return east_vals, north_vals, time_vals, alt_vals

    def compute_vals(values):
        base_val = min(values)
        return [val - base_val for val in values]
    
    stationary_filepath = 'stationary_data.bag'
    walking_filepath = 'walking_data.bag'

    stat_east, stat_north, stat_time, stat_alt = extract_bag_data(stationary_filepath)
    walk_east, walk_north, walk_time, walk_alt = extract_bag_data(walking_filepath)

    stat_east = compute_vals(stat_east)
    stat_north = compute_vals(stat_north)

    walk_east = compute_vals(walk_east)
    walk_north = compute_vals(walk_north)

    # Plotting data
    main_fig = plt.figure(figsize=(12, 8))

    stationary_plot = main_fig.add_subplot(2, 2, 1)
    stationary_plot.plot(stat_east, stat_north, 'bo', label='Data Points')
    stationary_plot.set_title("Stationary Data")
    stationary_plot.set_xlabel("UTM-Easting")
    stationary_plot.set_ylabel("UTM-Northing")
    stationary_plot.legend(loc="upper right")

    altitude_plot = main_fig.add_subplot(2, 2, 2)
    altitude_plot.plot(walk_time, walk_alt, 'g-', label='Altitude')
    altitude_plot.set_title("Walking Data: Altitude vs. Time")
    altitude_plot.set_xlabel("Time (seconds)")
    altitude_plot.set_ylabel("Altitude (meters)")
    altitude_plot.legend(loc="upper right")

    walking_plot = main_fig.add_subplot(2, 2, 3)
    walking_plot.plot(walk_east, walk_north, 'b-', label='Walking Data')
    walking_plot.set_title("Walking Data")
    walking_plot.set_xlabel("UTM-Easting")
    walking_plot.set_ylabel("UTM-Northing")
    walking_plot.legend(loc="upper right")

    plt.tight_layout()
    plt.show()
