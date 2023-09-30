#!/usr/bin/env python3

import rospy
import serial
import utm
from std_msgs.msg import Float64, Float32
from my_gps_driver.msg import gps_msg
from rospy import Time
import rosbag
import utm
import math


def initialize_gps_node():
    rospy.init_node("myGPSPuckNode")

    gpsData = gps_msg()
    gpsData.header.frame_id = 'GPS1_Frame'
    
    gpsPublisher = rospy.Publisher("/gps", gps_msg, queue_size=5)
    gps_serial_port = rospy.get_param('~port', '/dev/ttyUSB0')
    gps_baudrate = rospy.get_param('~baudrate', 4800)
    gpsSerialConnection = serial.Serial(gps_serial_port, gps_baudrate, timeout=1)
    
    return gpsData, gpsPublisher, gpsSerialConnection



def parse_time(time_str):
    hours = int(time_str[:2])
    minutes = int(time_str[2:4])
    seconds_float = float(time_str[4:])
    
    seconds_int = int(seconds_float)
    nanoseconds = int((seconds_float - seconds_int) * 1e9)
    
    total_seconds = hours * 3600 + minutes * 60 + seconds_int
    return Time(secs=int(total_seconds), nsecs=int(nanoseconds))

def parse_location_data(line):
    latitude_deg = float(line[2][:2])
    latitude_min = float(line[2][2:])
    latitude = latitude_deg + latitude_min / 60.0
    if line[3] == "S":
        latitude = -latitude

    longitude_deg = float(line[4][:3])
    longitude_min = float(line[4][3:])
    longitude = longitude_deg + longitude_min / 60.0
    if line[5] == "W":
        longitude = -longitude

    altitude = float(line[9]) 
    return latitude, longitude, altitude

def publish_gps_data(gps_data, timestamp, latitude, longitude, altitude, utm_coords, publisher):
    gps_data.header.stamp = timestamp
    gps_data.latitude = latitude
    gps_data.longitude = longitude
    gps_data.altitude = altitude
    gps_data.utm_easting = utm_coords[0]
    gps_data.utm_northing = utm_coords[1]
    gps_data.zone = utm_coords[2]
    gps_data.letter = utm_coords[3]
    rospy.loginfo(utm_coords)
    print(gps_data)
    publisher.publish(gps_data)


if __name__ == '__main__':
   
    gpsData, gpsPublisher, gpsSerialConnection = initialize_gps_node()


    try:
        while not rospy.is_shutdown():
            #print("below while")
            line = gpsSerialConnection.readline().strip()
            line = str(line)
           
            if line == '':
                rospy.logwarn("No GPS data was collected")
                print("No GPS data was collected")
                continue
            elif not line.startswith("b'$GPGGA"):
                continue
        

            else:
                if line.startswith("b'$GPGGA"):
                    line = line.split(",")
                    print(line)

                    timestamp = parse_time(line[1])
                    #######time_str = line[1]  # Extract the time string
                    latitude, longitude, altitude = parse_location_data(line)
                    utm_coords = utm.from_latlon(latitude, longitude)

                    publish_gps_data(gpsData, timestamp, latitude, longitude, altitude, utm_coords, gpsPublisher)
                    #rospy.sleep(0)


                   

    except rospy.ROSInterruptException:
        gpsSerialConnection.close()

    except serial.serialutil.SerialException:
        rospy.loginfo("Shutting down puck")