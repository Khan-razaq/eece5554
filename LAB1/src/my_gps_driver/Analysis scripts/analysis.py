#!/usr/bin/env python3

import matplotlib.pyplot as plt
import rosbag


def load_bag_data(bag_file_path):
    e_data = []
    n_data = []
    t_data = []
    a_data = []

    bag = rosbag.Bag(bag_file_path)
    for _, msg, _ in bag.read_messages(topics=['/gps']):
        e_data.append(msg.utm_easting)
        n_data.append(msg.utm_northing)

        t_data.append(msg.header.stamp.secs)
        a_data.append(msg.altitude)
    bag.close()
    
    return e_data, n_data, t_data, a_data


def get_relative_data(data):
    base = min(data)
    relative_data = []
    for d in data:
        relative_data.append(d - base)
    return relative_data


if __name__ == '__main__':
    stat_file_path = '/home/shubhankar/catkin_ws/src/my_gps_driver/scripts/stationary_data.bag'

    walk_file_path = '/home/shubhankar/catkin_ws/src/my_gps_driver/scripts/walking_data.bag'

    stat_e, stat_n, stat_time, stat_alt = load_bag_data(stat_file_path)

    walk_e, walk_n, walk_time, walk_alt = load_bag_data(walk_file_path)

    relative_stat_e = get_relative_data(stat_e)
    relative_stat_n = get_relative_data(stat_n)

    relative_walk_e = get_relative_data(walk_e)
    relative_walk_n = get_relative_data(walk_n)

    # Let's plot the data
    plot_figure = plt.figure(figsize=(12, 8))

    stat_plot = plot_figure.add_subplot(2, 2, 1)
    stat_plot.plot(relative_stat_e, relative_stat_n, 'bo', label='Data Points')
    stat_plot.set_title("Stationary Data")

    stat_plot.set_xlabel("UTM-Easting")
    stat_plot.set_ylabel("UTM-Northing")

    stat_plot.legend(loc="upper right")

    alt_plot = plot_figure.add_subplot(2, 2, 2)
    alt_plot.plot(walk_time, walk_alt, 'g-', label='Altitude')

    alt_plot.set_title("Walking Data: Altitude vs. Time")

    alt_plot.set_xlabel("Time (seconds)")
    alt_plot.set_ylabel("Altitude (meters)")

    alt_plot.legend(loc="upper right")

    walk_plot = plot_figure.add_subplot(2, 2, 3)
    walk_plot.plot(relative_walk_e, relative_walk_n, 'b-', label='Walking Data')

    walk_plot.set_title("Walking Data")

    walk_plot.set_xlabel("UTM-Easting")

    walk_plot.set_ylabel("UTM-Northing")

    walk_plot.legend(loc="upper right")

    plt.tight_layout()
    plt.show()
