#!/usr/bin/env python
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

    if not rospy.has_param('~port'):
        rospy.logerr("No port parameter provided.")
        sys.exit(1)

    imu_serial_port = rospy.get_param('~port')

    ser = serial.Serial(imu_serial_port, 115200, timeout=2.)
    
    imu_publisher = rospy.Publisher('/imu_pub', imu_msg, queue_size=10)
    sleep_time = 0.025
    try:
        while not rospy.is_shutdown():
            data = ser.readline().decode('utf-8').strip()
            
            imu_data = vnymr_filter(data)
            if imu_data:
                print(imu_data)
                accel, gyro, orientation, magnetometer = imu_data
                roll, pitch, yaw = orientation
                quaternion = euler_to_quaternion(roll, pitch, yaw)

                header = Header()
                header.stamp = rospy.Time.now()  # Use current system time
                header.frame_id = "IMU1_Frame"
                
                imu_message = Imu()
                imu_message.header = header
                imu_message.angular_velocity.x, imu_message.angular_velocity.y, imu_message.angular_velocity.z = gyro
                imu_message.linear_acceleration.x, imu_message.linear_acceleration.y, imu_message.linear_acceleration.z = accel
                imu_message.orientation.w, imu_message.orientation.x, imu_message.orientation.y, imu_message.orientation.z = quaternion

                mag_field = MagneticField()
                mag_field.header = header
                mag_field.magnetic_field.x, mag_field.magnetic_field.y, mag_field.magnetic_field.z = magnetometer
                
                custom_msg = imu_msg()
                custom_msg.header = header
                custom_msg.IMU = imu_message
                custom_msg.MagField = mag_field
                custom_msg.raw_data = data

                imu_publisher.publish(custom_msg)

    except rospy.ROSInterruptException:
        ser.close()
    except serial.serialutil.SerialException:
        rospy.loginfo("Shutting down imu driver")
    except Exception as e:
        rospy.logerr("An error occurred: %s", str(e))

