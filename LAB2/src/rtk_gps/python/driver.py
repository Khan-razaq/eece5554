#!/usr/bin/env python

import rospy
import sys
import serial
import utm
from rtk_gps.msg import gps_msg

def gngga_filter(data):
    fields = data.split(',')
    if len(fields) < 15 or fields[0] != "$GNGGA":
        return None

    latitude_deg = float(fields[2][:2])
    latitude_min = float(fields[2][2:])
    latitude = latitude_deg + latitude_min/60.0
    if fields[3] == 'S':
        latitude = -latitude

    longitude_deg = float(fields[4][:3])
    longitude_min = float(fields[4][3:])
    longitude = longitude_deg + longitude_min/60.0
    if fields[5] == 'W':
        longitude = -longitude

    altitude = float(fields[9])
    time_str = fields[1]
    fix_quality = int(fields[6])  # GNSS fix quality

    return latitude, longitude, altitude, time_str, fix_quality

if __name__ == '__main__':
    rospy.init_node('driver_node')
   
    if not rospy.has_param('~port'):
        rospy.logerr("No port parameter provided.")
        sys.exit(1)

    gps_serial_port = rospy.get_param('~port')
    ser = serial.Serial(gps_serial_port, 4800, timeout=2.)
    
    gps_publisher = rospy.Publisher('/gps_pub', gps_msg, queue_size=10)

    try:
        while not rospy.is_shutdown():
            data = ser.readline().strip()
            gngga_data = gngga_filter(data)
            print(gngga_data)
            if gngga_data:
                latitude, longitude, altitude, time_str, fix_quality = gngga_data
                
                hours = int(time_str[:2])
                minutes = int(time_str[2:4])
                seconds = float(time_str[4:])
                total_seconds = hours * 3600 + minutes * 60 + seconds
                
                timestamp_from_gngga = rospy.Time(secs=int(total_seconds), nsecs=int((total_seconds % 1) * 1e9))

                utm_coords = utm.from_latlon(latitude, longitude)
                utm_easting = utm_coords[0]
                utm_northing = utm_coords[1]
                utm_zone = utm_coords[2]
                utm_letter = utm_coords[3]

                gps_data = gps_msg()
                gps_data.header.stamp = timestamp_from_gngga
                gps_data.header.frame_id = "GPS1_Frame"
                gps_data.latitude = latitude
                gps_data.longitude = longitude
                gps_data.altitude = altitude
                gps_data.utm_easting = utm_easting
                gps_data.utm_northing = utm_northing
                gps_data.zone = utm_zone
                gps_data.letter = utm_letter
                gps_data.fix_quality = fix_quality  # Include fix quality in the message

                gps_publisher.publish(gps_data)

    except rospy.ROSInterruptException:
        ser.close()
    except serial.serialutil.SerialException:
        rospy.loginfo("Shutting down puck")
    except Exception as e:
        rospy.logerr("An error occurred: %s", str(e))

