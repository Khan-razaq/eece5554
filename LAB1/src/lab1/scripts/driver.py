#!/usr/bin/env python3

import rospy
import sys
import serial
import utm
from lab1.msg import gps_msg  

def gpgga_filter(data):
    fields = data.split(',')
    print(fields)
    #example fields = ['$GPGGA', '233404.000', '4220.2832', 'N', '07105.3024', 'W', '1', '08', '1.0', '25.6', 'M', '-33.8', 'M', '', '0000*58']

    if len(fields) < 15 or fields[0] != "$GPGGA":
        return None
    # convert the latitude from NMEA (deg,mins) to decimal format
    latitude_deg = float(fields[2][:2])
    latitude_min = float(fields[2][2:])
    latitude = latitude_deg + latitude_min/60.0
    if fields[3] == 'S':
        latitude = -latitude

    # convert the latitude from NMEA (deg,mins) to decimal format
    longitude_deg = float(fields[4][:3])
    longitude_min = float(fields[4][3:])
    longitude = longitude_deg + longitude_min/60.0
    if fields[5] == 'W':
        longitude = -longitude

    altitude = float(fields[9])
    time_str = fields[1]
    return latitude, longitude, altitude, time_str 

if __name__ == '__main__':
    rospy.init_node('gps_driver')
    gps_publisher = rospy.Publisher('/gps_pub', gps_msg, queue_size=10)
    ser = serial.Serial('/dev/ttyUSB1', 4800, timeout=2.)
    if len(sys.argv)>0:
        rospy.loginfo(sys.argv[1])
        gps_serial_port = rospy.get_param('~port',sys.argv[1])
    elif sys.argv[1] == '':
        gps_serial_port = rospy.get_param('~port', '/dev/ttyUSB0')
    try:
        while not rospy.is_shutdown():
            data = ser.readline().strip()

            gpgga_data = gpgga_filter(data)
            if gpgga_data:
                latitude, longitude, altitude, time_str = gpgga_data

                # Extract time components (hours, minutes, seconds) from the time string
                hours = int(time_str[:2])
                minutes = int(time_str[2:4])
                seconds = float(time_str[4:])

                # Calculate the total seconds since midnight
                total_seconds = hours * 3600 + minutes * 60 + seconds

                # Create a rospy.Time object with the seconds and convert nanoseconds
                timestamp_from_gpgga = rospy.Time(secs=int(total_seconds), nsecs=int((total_seconds % 1) * 1e9))

                # Convert latitude and longitude to UTM
                utm_coords = utm.from_latlon(latitude, longitude)
                utm_easting = utm_coords[0]
                utm_northing = utm_coords[1]
                utm_zone = utm_coords[2]
                utm_letter = utm_coords[3]

                # Create a custom ROS message
                gps_data = gps_msg()  # Create an instance of the gps_msg message
                #gps_data.header = Header()  # Create an instance of Header
                gps_data.header.stamp = timestamp_from_gpgga
                gps_data.header.frame_id = "GPS1_Frame"
                gps_data.latitude = latitude
                gps_data.longitude = longitude
                gps_data.altitude = altitude
                gps_data.utm_easting = utm_easting
                gps_data.utm_northing = utm_northing
                gps_data.zone = utm_zone
                gps_data.letter = utm_letter

                # Publish the custom ROS message
                gps_publisher.publish(gps_data)

    except rospy.ROSInterruptException:
        ser.close()
    except serial.serialutil.SerialException:
        rospy.loginfo("Shutting down puck")

