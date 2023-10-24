#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import sys
import serial
import math
from std_msgs.msg import Header
from sensor_msgs.msg import Imu, MagneticField
from imu_driver.msg import imu_msg

def euler_to_quaternion(roll, pitch, yaw):
    """Convert Euler angles to quaternion."""
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    q_w = cr * cp * cy + sr * sp * sy
    q_x = sr * cp * cy - cr * sp * sy
    q_y = cr * sp * cy + sr * cp * sy
    q_z = cr * cp * sy - sr * sp * cy

    return [q_w, q_x, q_y, q_z]

def vnymr_filter(data):
    fields = data.split(',')
    if len(fields) < 13:
        return None
    if "$VNYMR" not in data:
        return None
    
    # Extract data fields
    roll, pitch, yaw = float(fields[3]), float(fields[2]), float(fields[1])
    magnetometer = [float(fields[4]), float(fields[5]), float(fields[6])]
    accel = [float(fields[7]), float(fields[8]), float(fields[9])]
    gyro = [float(fields[10]), float(fields[11]), float(fields[12].split('*')[0])]
    
    return accel, gyro, (roll, pitch, yaw), magnetometer

if __name__ == '__main__':
    rospy.init_node('imu_driver_node')

    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if ":=" in arg:
            key, value = arg.split(":=")
            if key == "port":
                imu_serial_port = value
            else:
                rospy.logerr("Invalid argument provided.")
                sys.exit(1)
        else:
            imu_serial_port = arg
    elif rospy.has_param('~port'):
        imu_serial_port = rospy.get_param('~port')
    else:
        rospy.logerr("No port parameter provided.")
        sys.exit(1)
    rospy.loginfo("IMU Serial Port: %s", imu_serial_port)

    try:
        ser = serial.Serial(imu_serial_port, 115200, timeout=2.)
    except Exception as e:
        rospy.logerr("Error opening serial port: %s", str(e))
        sys.exit(1)

    imu_publisher = rospy.Publisher('/imu', imu_msg, queue_size=10)
    rate = rospy.Rate(40)

    try:
        while not rospy.is_shutdown():
            data = ser.readline().decode('utf-8').strip()
            print(data)
            imu_data = vnymr_filter(data)
            if imu_data:
                print(imu_data)
                accel, gyro, orientation, magnetometer = imu_data
                roll, pitch, yaw = orientation
                quaternion = euler_to_quaternion(roll, pitch, yaw)

                imu_data = imu_msg()
		imu_data.Header.stamp = rospy.Time.now()
		imu_data.Header.frame_id = "IMU1_Frame"

		# IMU data
		imu_data.IMU.angular_velocity.x = gyro[0]                
   	        imu_data.IMU.angular_velocity.y = gyro[1] 
                imu_data.IMU.angular_velocity.z = gyro[2]
		imu_data.IMU.linear_acceleration.x = accel[0]
                imu_data.IMU.linear_acceleration.y = accel[1]
                imu_data.IMU.linear_acceleration.z = accel[2]
		imu_data.IMU.orientation.w = quaternion[0]
                imu_data.IMU.orientation.x = quaternion[1]
                imu_data.IMU.orientation.y = quaternion[2]
                imu_data.IMU.orientation.z = quaternion[3]

		# Magnetic field data
		imu_data.MagField.magnetic_field.x = magnetometer[0]
                imu_data.MagField.magnetic_field.y = magnetometer[1]
                imu_data.MagField.magnetic_field.z = magnetometer[2]

		# Raw data
		imu_data.raw_data = data

                imu_publisher.publish(imu_data)

    except rospy.ROSInterruptException:
        ser.close()
    except serial.serialutil.SerialException:
        rospy.loginfo("Shutting down imu driver")
    except Exception as e:
        rospy.logerr("An error occurred: %s", str(e))

