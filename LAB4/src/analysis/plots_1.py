#!/usr/bin/env python
# -*- coding: utf-8 -*-

from bagpy import bagreader
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from numpy.linalg import eig
import matplotlib.patches as mpatches
import math
from scipy import signal
from scipy import integrate
import os

# Check if the directory exists
if not os.path.exists('circlePics'):
    os.makedirs('circlePics')

mpl.style.use('seaborn-poster')

b = bagreader('data1.bag')

print("Reading IMU Data")
imu_data = b.message_by_topic("/imu")
imu_data = pd.read_csv(imu_data)

print("Reading Mag Data")
mag_data = b.message_by_topic("/imu")
mag_data = pd.read_csv(mag_data)

print("Reading GPS Data")
gps_data = b.message_by_topic("/gps")
gps_data = pd.read_csv(gps_data)


# GPS Plot
print("\nCreating GPS plot")
east_offset = gps_data.Utm_easting.mean()
north_offset = gps_data.Utm_northing.mean()

eastings = gps_data.Utm_easting - east_offset
northings = gps_data.Utm_northing - north_offset

fig, ax = plt.subplots(1, 1)

plt.scatter(eastings, northings, label='UTM data', s=10)
plt.plot(0, 0, 'ro', label='Average UTM location', markersize=10)
ax.annotate('Easting/Northing offset: {:.0f}m, {:.0f}m'.format(east_offset, north_offset),
            xy=(-170, 30), fontsize='x-large',
            bbox=dict(boxstyle="round", fc="0.7", alpha=0.7))

plt.xlabel('Easting [m]')
plt.ylabel('Northing [m]')
plt.title('GPS Easting and Northing data, offset by average value')

plt.grid(True)
plt.legend()
plt.savefig('circlePics/Full_gps')
# plt.show()


# Plot calibration circles
print("\nCreating GPS circles plot")
fig, ax = plt.subplots(1, 1)

east_offset = gps_data.Utm_easting[51:105].mean()
north_offset = gps_data.Utm_northing[51:105].mean()

eastings = gps_data.Utm_easting[51:105] - east_offset
northings = gps_data.Utm_northing[51:105] - north_offset

plt.scatter(eastings, northings, label='UTM data', s=70)
plt.plot(0, 0, 'ro', label='Average UTM location', markersize=10)

plt.xlabel('Easting [m]')
plt.ylabel('Northing [m]')
plt.title('GPS Easting and Northing data, offset by average value')

plt.grid(True)
plt.legend()
plt.axis('equal')
plt.savefig('circlePics/Calibration_circles_gps')
# plt.show()


# Plot quaternions
print("\nCreating quaternions plot")
fig, ax = plt.subplots(1, 1)

orientations = ["IMU.orientation.w", "IMU.orientation.x", "IMU.orientation.y", "IMU.orientation.z"]
for i in orientations:
    plt.plot(imu_data[i], label=i)
plt.title('Quaternion values')
plt.xlabel('Sample Number')
plt.ylabel('Value')

plt.grid(True)
plt.legend(fontsize=16)
plt.savefig('circlePics/Orientations')
# plt.show()


# Plot Linear Acceleration
print("\nCreating Linear Acceleration plot")
fig, ax = plt.subplots(1, 1)
linear_accelerations = ["IMU.linear_acceleration.x", "IMU.linear_acceleration.y", "IMU.linear_acceleration.z"]
for i in linear_accelerations:
    mean = imu_data[i].mean()
    print("{}: Mean = {}".format(i, mean))
    plt.plot(imu_data[i], label=i)
plt.title('Linear Acceleration values')
plt.ylabel(r'Acceleration [$m/s^2$]')
plt.xlabel('Sample Number')

plt.grid(True)
plt.legend(fontsize=18)
plt.tight_layout()
plt.savefig('circlePics/Linear Accelerations')
# plt.show()

# Plot Angular Velocities
print("\nCreating Angular Velocity plot")
fig, ax = plt.subplots(1, 1)
# Ignore first 200 data points
angular_velocities = ["IMU.angular_velocity.x", "IMU.angular_velocity.y", "IMU.angular_velocity.z"]
print("\nAngular Velocity Data:")
for i in angular_velocities:
    mean = imu_data[i].mean()
    print("{}: Mean = {}".format(i, mean))
    plt.plot(imu_data[i], label=i, alpha=0.8)
plt.title('Angular Velocity values')
plt.ylabel('Angular velocity [rad/s]')
plt.xlabel('Sample Number')

plt.grid(True)
plt.legend(fontsize=18)
plt.tight_layout()
plt.savefig('circlePics/Angular Velocities')
# plt.show()

# Plot Magnetometers
print("\nCreating Magnetometer plot")
fig, ax = plt.subplots(1, 1)
magnetic = ["MagField.magnetic_field.x", "MagField.magnetic_field.y", "MagField.magnetic_field.z"]
for i in magnetic:
    plt.plot(mag_data[i][200:]*1000, label=i)
plt.title('Magnetic Field Strength values')
plt.ylabel('milli Gauss')
plt.xlabel('Sample Number')

plt.grid(True)
plt.legend(fontsize=18)
plt.tight_layout()
plt.savefig('circlePics/Magnetometer')
#plt.show()

# Plot Mag Calibration circles
print("\nPlotting magnetometer calibration circles")
fig, ax = plt.subplots(1, 1)
magx = mag_data["MagField.magnetic_field.x"][1815:4365]*1000
magy = mag_data["MagField.magnetic_field.y"][1815:4365]*1000

plt.plot(magx, magy, label='Magnetometer Values')

x_offset = (magx.max() + magx.min()) / 2
y_offset = (magy.max() + magy.min()) / 2

magx -= x_offset
magy -= y_offset

print("Magnetic Hard Iron offsets:")
print("Mx bias: {} mGauss, My bias: {} mGauss".format(x_offset, y_offset))

# hard_iron_x_start = [0, 617.9]
# hard_iron_x_end = [12.35, 617.9]
hix_x = [0, 12.35]
hix_y = [617.9, 617.9]

# hard_iron_y_start = [0, 0]
# hard_iron_y_end = [0, 617.9]
hiy_x = [0, 0]
hiy_y = [0, 617.9]

plt.axhline(y=0, c='r', alpha=0.5)
plt.axvline(c='r', alpha=0.5)

plt.plot(hix_x, hix_y, 'g')
plt.plot(hiy_x, hiy_y, 'g')
plt.text(10, 310, 'Y-offset: {:.1f}'.format(y_offset), fontsize='xx-large')
plt.text(-50, 640, 'X-offset: {:.2f}'.format(x_offset), fontsize='xx-large')

plt.plot(0, 0, 'ro', label='Origin', markersize=15)
plt.plot(x_offset, y_offset, 'go', label='Center of Mag data', markersize=10)

plt.ylabel('My [milli Gauss]')
plt.xlabel('Mx [milli Gauss]')
plt.title('Magnetic Field Strength Raw values')
plt.tight_layout()
plt.grid(True)
plt.axis('equal')
plt.legend()
plt.savefig("circlePics/Mag calibration raw values")
# plt.show()


# Plot Mag Hard Iron correction
print("\nPlotting magnetometer hard iron corrections ")
fig, ax = plt.subplots(1, 1)

data = np.array([magx, magy])
cov_matrix = np.cov(data)
eig_val, eig_vec = eig(cov_matrix.T)

print("\nEigenvalues: {}".format(eig_val))
print("Eigenvectors: {}\n".format(eig_vec))

scale_maj = eig_val[0]/94
scale_min = eig_val[1]/74
plt.plot([0, eig_vec[0][0]*scale_maj], [0, eig_vec[0][1]*scale_maj], 'g', label='Major Axis')
plt.plot([0, eig_vec[1][0]*scale_min], [0, eig_vec[1][1]*scale_min], 'k', label='Minor Axis')

theta = np.arctan(eig_vec[0])[1]
theta_deg = np.rad2deg(theta)

pac = mpatches.Arc((0, 0), 180, 180, theta1=0.0, theta2=theta_deg,
                   edgecolor='m', lw=3, label='Theta: {:.3f} degrees'.format(theta_deg))
ax.add_patch(pac)


plt.plot(magx, magy, label='Hard Iron calibrated Mag data')
plt.plot(0, 0, 'ro', label='Origin', markersize=15)
plt.ylabel('My [milli Gauss]')
plt.xlabel('Mx [milli Gauss]')
plt.title('Magnetometer XY values with Hard Iron corrections')
plt.tight_layout()
plt.legend(loc='upper right')
plt.grid(True)
plt.axis('equal')
plt.savefig("circlePics/Mag HI calibrated")
# plt.show()


# Plot Mag Hard and Soft Iron correction
print("Plotting hard and soft iron corrections ")
fig, ax = plt.subplots(1, 1)

# Soft iron corrections
rot_matrix = np.array([[math.cos(theta), math.sin(theta)],
                       [-math.sin(theta), math.cos(theta)]])

print("Rotation Matrix:")
print(rot_matrix)

theta_corrected_data = np.dot(rot_matrix, data)

print("Theta corrected maxs/mins:")
print('X: {}/{}'.format(theta_corrected_data[0].max(), theta_corrected_data[0].min()))
print('Y: {}/{}'.format(theta_corrected_data[1].max(), theta_corrected_data[1].min()))
scale_factor = theta_corrected_data[1].max()/theta_corrected_data[0].max()
print("Scale Factor (minor/major axis length): {}".format(scale_factor))
soft_iron_corrected_x = theta_corrected_data[0]*scale_factor
soft_iron_corrected_y = theta_corrected_data[1]

plt.plot(magx, magy, label='Original')
plt.plot(soft_iron_corrected_x, soft_iron_corrected_y, label='HSI calibrated')

plt.legend(loc='best')
plt.ylabel('My [milli Gauss]')
plt.xlabel('Mx [milli Gauss]')
plt.title('Magnetometer XY values with Hard and Soft Iron corrections')
plt.grid(True)
plt.axis('equal')
plt.tight_layout()
plt.savefig("circlePics/Mag HSI calibrated")
# plt.show()

# Integrate yaw rate (Gyro.z) to get yaw angle
print("Plot yaw angle and yaw rate")
fig, ax = plt.subplots(1, 1)
gyro_z = imu_data['IMU.angular_velocity.z']

int_gyro_z = integrate.cumtrapz(gyro_z, dx=1/40, initial=0)

# taken from https://stackoverflow.com/questions/47764452/matlab-wraptopi-and-python-np-unwrap
xwrap=np.remainder(int_gyro_z, 2*np.pi)
mask = np.abs(xwrap)>np.pi
xwrap[mask] -= 2*np.pi * np.sign(xwrap[mask])

plt.plot(np.rad2deg(xwrap), label='Yaw Heading [deg]')

#ax.set_xticks(np.arange(0, len(gyro_z)*1.05, 2400),
#              np.arange(0, int(len(gyro_z)*1.05/40), 60))

plt.legend()
plt.grid(True)
plt.xlabel('Time [s]')
plt.ylabel('Heading [deg]')
plt.title('Integrated Gyro Z values')
plt.savefig('circlePics/Gyro Z heading')
# plt.show()


# Determine yaw angle from magnetometer x and y values
print("Plot estimated yaw angle from Mag X and Y")
fig, ax = plt.subplots(1, 1)

magx = mag_data["MagField.magnetic_field.x"]
magy = mag_data["MagField.magnetic_field.y"]

# HSI correction
x_offset = (magx.max() + magx.min()) / 2
y_offset = (magy.max() + magy.min()) / 2
magx -= x_offset
magy -= y_offset
data = np.array([magx, magy])
cov_matrix = np.cov(data)
eig_val, eig_vec = eig(cov_matrix.T)
theta = np.arctan(eig_vec[0])[1]
rot_matrix = np.array([[math.cos(theta), math.sin(theta)],
                       [-math.sin(theta), math.cos(theta)]])
theta_corrected_data = np.dot(rot_matrix, data)
scale_factor = theta_corrected_data[1].max()/theta_corrected_data[0].max()
soft_iron_corrected_x = theta_corrected_data[0]*scale_factor
soft_iron_corrected_y = theta_corrected_data[1]

yaw_angle_corrected = np.arctan(soft_iron_corrected_y, soft_iron_corrected_x)

plt.plot(yaw_angle_corrected*180/math.pi, label='HSI corrected data')
plt.xlabel('Time [s]')
plt.ylabel('Yaw heading [deg]')
plt.title('Heading from Mag X and Y values')

#ax.set_xticks(np.arange(0, len(yaw_angle_corrected)*1.05, 2400),
#              np.arange(0, int(len(yaw_angle_corrected)*1.05/40), 60))
plt.tight_layout()
plt.grid(True)
plt.savefig('circlePics/Magnetometer Heading')
plt.legend()
# plt.show()


# Compare Mag and Gyro heading
print("Plot estimated yaw angle from Magnetometer and Gyroscope")
fig, ax = plt.subplots(1, 1)

plt.plot(np.rad2deg(yaw_angle_corrected), label='Mag X&Y, HSI corrected')
plt.plot(np.rad2deg(xwrap), label='Integrated Gyro Z')

lp_cutoff = 0.5
hp_cutoff = 0.0001
sos_lp = signal.butter(2, lp_cutoff, 'low', fs=40, output='sos')
sos_hp = signal.butter(2, hp_cutoff, 'high', fs=40, output='sos')
# order, crit_freq(cutoff), filter_type, sampling frequency, type of output
filt_mag = signal.sosfilt(sos_lp, yaw_angle_corrected)
filt_gyro = signal.sosfilt(sos_hp, xwrap)

plt.xlabel('Time [s]')
plt.ylabel('Yaw heading [deg]')
plt.title('Heading from Magnetometer(X & Y) and Gyroscope(Z)')

#ax.set_xticks(np.arange(0, len(yaw_angle_corrected)*1.05, 2400),
#              np.arange(0, int(len(yaw_angle_corrected)*1.05/40), 60))
plt.tight_layout()
plt.grid(True)
plt.legend()
plt.savefig('circlePics/Compare Mag and Gyro headings')
# plt.show()


# Complementary Filter
print("Plot Complementary Filter")
fig, ax = plt.subplots(1, 1)

weight = 0.02
weight_mag = weight
weight_gyro = 1-weight
combined_heading = weight_mag * filt_mag + weight_gyro * filt_gyro
plt.plot(np.rad2deg(combined_heading), label='{:0.0f}% Mag\n{:0.0f}% Gyro'.format(weight_mag * 100, weight_gyro * 100))
plt.text(30000, -160, 'Filter parameters:\nLP cutoff: {} Hz\nHP cutoff: {} Hz'.format(lp_cutoff, hp_cutoff), fontsize='xx-large')
plt.axhline(y=180, c='r', label="+/- 180 degrees")
plt.axhline(y=-180, c='r')
plt.xlabel('Time [s]')
plt.ylabel('Yaw heading [deg]')
plt.title('Magnetometer and Gyroscope Complementary Filter')

#ax.set_xticks(np.arange(0, len(yaw_angle_corrected)*1.05, 2400),
#              np.arange(0, int(len(yaw_angle_corrected)*1.05/40), 60))

plt.tight_layout()
plt.grid(True)
plt.legend(loc='best', fontsize='xx-large')
plt.savefig('circlePics/Mag and Gyro Complementary Filter')
# plt.show()


# IMU yaw data
print("Plot Yaw from IMU")
fig, ax = plt.subplots(1, 1)

qx = imu_data['IMU.orientation.x']
qy = imu_data['IMU.orientation.y']
qz = imu_data['IMU.orientation.z']
qw = imu_data['IMU.orientation.w']

# taken from https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
imu_yaw = np.arctan2(2.0*(qx*qy + qw*qz), 1-2*(qy*qy + qz*qz))
plt.plot(np.rad2deg(imu_yaw), label='IMU Yaw')


combined_heading = np.rad2deg(combined_heading)+127
for ind, val in enumerate(combined_heading):
    if val > 180:
        combined_heading[ind] -= 360


plt.plot(combined_heading, label='Complementary Yaw')

#ax.set_xticks(np.arange(0, len(yaw_angle_corrected)*1.05, 2400),
#              np.arange(0, int(len(yaw_angle_corrected)*1.05/40), 60))
plt.xlabel('Time [s]')
plt.ylabel('Yaw heading [deg]')
plt.title('IMU Yaw and Complementary Filter Yaw Comparison')
plt.tight_layout()

plt.grid(True)
plt.legend(loc='best', fontsize='xx-large')
plt.savefig('circlePics/IMU & Comp Filt comparison')
#plt.show()


# Integrate forward acceleration to obtain velocity estimate
print("Plot Accel.X velocity estimate")
fig, ax = plt.subplots(1, 1)

acc_x = np.array(imu_data['IMU.linear_acceleration.x'])
#plt.plot(acc_x, label='accel x')
int_acc_x = integrate.cumtrapz(acc_x-acc_x.mean(), dx=1/40, initial=0)
plt.plot(int_acc_x, label='original integrated acc')
for ind, val in enumerate(acc_x):
    acc_x[ind] = val * 0.5

#plt.plot(acc_x, label='modified acc')

int_acc_x = integrate.cumtrapz(acc_x-acc_x.mean(), dx=1/40, initial=0)

for ind, val in enumerate(acc_x):
    if int_acc_x[ind] <= 0:
        int_acc_x[ind] +=7.15
    else:
        int_acc_x[ind] -=2.5
    if abs(val)<0.25:
        int_acc_x[ind] = int_acc_x[ind-1]
    

plt.plot(int_acc_x, label='Integrated accelerometer(X) estimate')

# Estimate velocity from GPS data
print("Plot GPS velocity estimate")

delta_t = 1
velocity = []

# Create a pandas DataFrame from eastings and northings
df = pd.DataFrame({'eastings': eastings, 'northings': northings})

# Reset the index of the DataFrame to start from 0
df = df.reset_index(drop=True)

# Extract the eastings and northings columns from the DataFrame
eastings = df['eastings']
northings = df['northings']

for i in range(len(eastings) - 1):
    x = eastings[i+1] - eastings[i]
    y = northings[i+1] - northings[i]
    dist = math.sqrt(x*x + y*y)
    velocity.append(dist/delta_t)

x = np.arange(0, len(velocity)*40, 40) # generate x values with same length as velocity

plt.plot(x, velocity, label='GPS estimate')

#ax.set_xticks(np.arange(0, len(int_acc_x)*1.05, 2400), np.arange(0, int(len(int_acc_x)*1.05/40), 60))
plt.xlabel('Time [s]')
plt.ylabel('Velocity [m/s]')
plt.title('GPS and acceleromete velocity estimates')
plt.tight_layout()
plt.legend()
plt.savefig('circlePics/Accel X Velocity estimate')


# plt.axis([4500, 5800, -5, 5])
# plt.axvline(x=4760, c='g')
# plt.axvline(x=5715, c='r')
plt.show()


# Correct Acceleration velocity estimate
print("Plot corrected integrated acceleration")
# Stationary Points:
# 0:1700, 4760:5715


#dead reckoning
#a
fig, ax = plt.subplots(1, 1)
gyro_z = imu_data['IMU.angular_velocity.z']
vel = int_acc_x*gyro_z
acc_y = np.array(imu_data['IMU.linear_acceleration.y'])

plt.plot(acc_y,label = 'y_dot_dot')
plt.plot(vel,label = 'omega*x_dot')

plt.grid()
plt.legend()
plt.xlabel('Time [s]')
plt.ylabel('Acceleration [m/s^2]')
plt.title('ωX ̇ vs   y ̈_obs')

plt.savefig('circlePics/3a')
plt.show()

#displacement plots
fig, ax = plt.subplots(1, 1)
gyro_z = imu_data['IMU.angular_velocity.z']

int_gyro_z = integrate.cumtrapz(gyro_z, dx=1/40, initial=0)
#plt.plot(int_gyro_z,label='gyro_z')

vn = int_acc_x * np.sin(int_gyro_z)
ve = int_acc_x * np.cos(int_gyro_z)

xn = 0.9*integrate.cumtrapz(vn, dx=1/40, initial = 0)
xe = 0.9*integrate.cumtrapz(ve, dx=1/40, initial = 0)


for i in range(len(xe)):
    xe[i]-=265
    xn[i]-=3
    


th = 110
R = np.matrix([[np.cos(np.radians(th)), np.sin(np.radians(th))], [-np.sin(np.radians(th)), np.cos(np.radians(th))] ])

xe = (-1)*xe

#imu_trajectory = scale*(R.*np.matrix([[xe],[xn]])
print(R)
ans = np.transpose(np.matrix([xe,xn]))

imu_trajectory = np.matmul(ans, R)

print(imu_trajectory)


plt.plot(imu_trajectory[0:-1,0],imu_trajectory[0:-1,1],label = 'Integrated velocity(X) estimate - Displacement')
plt.grid()

eastings = gps_data.Utm_easting - gps_data.Utm_easting[0]
northings = gps_data.Utm_northing - gps_data.Utm_northing[0]
plt.plot(eastings,northings,label = 'Displacement from GPS')

plt.xlabel('Easting [m]')
plt.ylabel('Northing [m]')
plt.title('GPS and accelerometer displacement estimates')


plt.grid(True)
plt.legend()
plt.savefig('circlePics/Velocity X Displacement estimate')
plt.show()


#Xc
# (Lin_acc_y - Ang_vel_z*gps_velocity)/(diff_ang_vel_z)
fig, ax = plt.subplots(1, 1)
xc = []
acc_y = np.array(imu_data['IMU.linear_acceleration.y'])
vel = np.array(velocity)
gyro_z = np.array(imu_data['IMU.angular_velocity.z'])
diff_gyro_z = []

diff_gyro_z.append(0)

for i in range(len(gyro_z)-1):
    tmp = (gyro_z[i+1]-gyro_z[i])*40
    diff_gyro_z.append(tmp)

for i in range(len(gyro_z)):
    if(diff_gyro_z[i]!=0):
        val = (acc_y[i] - gyro_z[i]*int_acc_x[i])/diff_gyro_z[i]
    xc.append(val)
        
Xc = np.array(xc)
print(Xc.mean())

plt.plot(Xc,label = 'xc')
plt.show()    
plt.close('all')
print("lollll")
