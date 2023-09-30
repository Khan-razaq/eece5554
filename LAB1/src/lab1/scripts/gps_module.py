#!/usr/bin/env python
import rospy
import sys
import serial
from lab1.msg import gps_msg

def parse_gpgga(data):
    # Split the GPGGA data into fields
    fields = data.split(',')
    
    # Check if the message is formatted correctly
    if len(fields) < 15 or fields[0] != "$GPGGA":
        return None

    # Extract relevant information
    latitude = float(fields[2])
    longitude = float(fields[4])
    altitude = float(fields[9])

    return latitude, longitude, altitude

def main(serial_port):
    rospy.init_node('gps_driver')
    gps_publisher = rospy.Publisher('/gps', gps_message, queue_size=10)

    # Open the serial port for reading GPS data
    try:
        with serial.Serial(serial_port, 9600) as ser:
            rospy.loginfo("GPS driver is reading from serial port: {0}".format(serial_port))

            while not rospy.is_shutdown():
                data = ser.readline().strip()

                # Parse GPGGA data
                parsed_data = parse_gpgga(data)
                if parsed_data:
                    latitude, longitude, altitude = parsed_data

                    # Convert latitude and longitude to UTM
                    utm_coords = utm.from_latlon(latitude, longitude)
                    utm_easting = utm_coords[0]
                    utm_northing = utm_coords[1]
                    utm_zone = utm_coords[2]
                    utm_letter = utm_coords[3]

                    # Create a custom ROS message
                    gps_msg = gps_msg()
                    gps_msg.header = Header()
                    gps_msg.header.stamp = rospy.Time.now()
                    gps_msg.header.frame_id = "GPS1_Frame"
                    gps_msg.latitude = latitude
                    gps_msg.longitude = longitude
                    gps_msg.altitude = altitude
                    gps_msg.utm_easting = utm_easting
                    gps_msg.utm_northing = utm_northing
                    gps_msg.zone = utm_zone
                    gps_msg.letter = utm_letter

                    # Publish the custom ROS message
                    gps_publisher.publish(gps_msg)
    except serial.SerialException as e:
        rospy.logerr("Failed to open serial port {0}: {1}".format(serial_port, e))

if __name__ == '__main__':
    try:
        # Get the serial port from command line arguments
        serial_port = rospy.myargv(argv=sys.argv)[1] if len(sys.argv) > 1 else "/dev/ttyUSB0"
        main(serial_port)
    except rospy.ROSInterruptException:
        pass

