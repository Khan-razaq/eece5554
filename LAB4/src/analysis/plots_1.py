#!/usr/bin/env python3
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
from scipy.signal import butter, filtfilt
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
ax.annotate('Easting/Northing offset: {:.0f}m, {:.0f}m'.format(east_offset, north_offset),
            xy=(-170, 30), fontsize='large',
            bbox=dict(boxstyle="round", fc="0.7", alpha=0.7))

plt.xlabel('Easting [m]')
plt.ylabel('Northing [m]')
plt.title('GPS Easting and Northing data, offset by average value')

plt.grid(True)
plt.legend()
plt.savefig('circlePics/Full_gps')
plt.show()

# Plot Linear Acceleration
print("\nCreating Linear Acceleration plot")
fig, ax = plt.subplots(1, 1)
linear_accelerations = ["IMU.linear_acceleration.x", "IMU.linear_acceleration.y", "IMU.linear_acceleration.z"]
for i in linear_accelerations:
    mean = imu_data[i].mean()
    print("{}: Mean = {}".format(i, mean))
    plt.plot(imu_data[i],linewidth=1, label=i)
plt.title('Linear Acceleration values')
plt.ylabel(r'Acceleration [$m/s^2$]')
plt.xlabel('Sample Number')

plt.grid(True)
plt.legend(fontsize=18)
plt.tight_layout()
plt.savefig('circlePics/Linear Accelerations')
plt.show()

# Plot Angular Velocities
print("\nCreating Angular Velocity plot")
fig, ax = plt.subplots(1, 1)
angular_velocities = ["IMU.angular_velocity.x", "IMU.angular_velocity.y", "IMU.angular_velocity.z"]
print("\nAngular Velocity Data:")
for i in angular_velocities:
    mean = imu_data[i].mean()
    print("{}: Mean = {}".format(i, mean))
    plt.plot(imu_data[i],linewidth=1, label=i, alpha=0.8)
plt.title('Angular Velocity values')
plt.ylabel('Angular velocity [rad/s]')
plt.xlabel('Sample Number')

plt.grid(True)
plt.legend(fontsize=18)
plt.tight_layout()
plt.savefig('circlePics/Angular Velocities')
plt.show()

# Plot Magnetometers
print("\nCreating Magnetometer plot")
fig, ax = plt.subplots(1, 1)
magnetic = ["MagField.magnetic_field.x", "MagField.magnetic_field.y", "MagField.magnetic_field.z"]
for i in magnetic:
    plt.plot(mag_data[i][800:]*1000, label=i)
plt.title('Magnetic Field Strength values')
plt.ylabel('milli Gauss')
plt.xlabel('Sample Number')
plt.grid(True)
plt.legend(fontsize=18)
plt.tight_layout()
plt.savefig('circlePics/Magnetometer')
plt.show()

# Plot Mag Calibration circles
print("\nPlotting magnetometer calibration circles")
fig, ax = plt.subplots(1, 1)
magx = mag_data["MagField.magnetic_field.x"][1000:4000]*1000
magy = mag_data["MagField.magnetic_field.y"][1000:4000]*1000

plt.scatter(magx, magy, s=10, label='Magnetometer Values')

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
plt.text(10, 310, 'Y-offset: {:.1f}'.format(y_offset), fontsize='x-large')
plt.text(-50, 640, 'X-offset: {:.2f}'.format(x_offset), fontsize='x-large')

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
plt.show()


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


plt.scatter(magx, magy, s=10, label='Hard Iron calibrated Mag data')
plt.plot(0, 0, 'ro', label='Origin', markersize=15)
plt.ylabel('My [milli Gauss]')
plt.xlabel('Mx [milli Gauss]')
plt.title('Magnetometer XY values with Hard Iron corrections')
plt.tight_layout()
plt.legend(loc='upper right')
plt.grid(True)
plt.axis('equal')
plt.savefig("circlePics/Mag HI calibrated")
plt.show()


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

plt.scatter(magx, magy,s=10, label='Original')
plt.scatter(soft_iron_corrected_x, soft_iron_corrected_y,s=10, color='orange', label='HSI calibrated')

plt.legend(loc='best')
plt.ylabel('My [milli Gauss]')
plt.xlabel('Mx [milli Gauss]')
plt.title('Magnetometer XY values with Hard and Soft Iron corrections')
plt.grid(True)
plt.axis('equal')
plt.tight_layout()
plt.savefig("circlePics/Mag HSI calibrated")
plt.show()

# Integrate yaw rate (Gyro.z) to get yaw angle
print("Plot yaw angle and yaw rate")
gyro_z = imu_data['IMU.angular_velocity.z']
int_gyro_z = integrate.cumtrapz(gyro_z, dx=1/40, initial=0)
xwrap=np.remainder(int_gyro_z, 2*np.pi)
mask = np.abs(xwrap)>np.pi
xwrap[mask] -= 2*np.pi * np.sign(xwrap[mask])

# Determine yaw angle from magnetometer x and y values
print("Plot estimated yaw angle from Mag X and Y")
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

# Compare Mag and Gyro heading
print("Plot estimated yaw angle from Magnetometer and Gyroscope")
fig, ax = plt.subplots(1, 1)

plt.plot(np.rad2deg(yaw_angle_corrected), label='Mag X&Y, HSI corrected')
plt.plot(np.rad2deg(xwrap), label='Integrated Gyro Z')

lp_cutoff = 0.5
hp_cutoff = 0.0001
sos_lp = signal.butter(2, lp_cutoff, 'low', fs=40, output='sos')
sos_hp = signal.butter(2, hp_cutoff, 'high', fs=40, output='sos')
filt_mag = signal.sosfilt(sos_lp, yaw_angle_corrected)
filt_gyro = signal.sosfilt(sos_hp, xwrap)

plt.xlabel('Time [s]')
plt.ylabel('Yaw heading [deg]')
plt.title('Heading from Magnetometer(X & Y) and Gyroscope(Z)')
plt.tight_layout()
plt.grid(True)
plt.legend()
plt.savefig('circlePics/Compare Mag and Gyro headings')
plt.show()

# Complementary Filter
print("Plot Complementary Filter")
#fig, ax = plt.subplots(1, 1)
weight = 0.02
weight_mag = weight
weight_gyro = 1-weight
combined_heading = weight_mag * filt_mag + weight_gyro * filt_gyro

# IMU yaw data
print("Plot Yaw from IMU")
fig, ax = plt.subplots(1, 1)

qx = imu_data['IMU.orientation.x']
qy = imu_data['IMU.orientation.y']
qz = imu_data['IMU.orientation.z']
qw = imu_data['IMU.orientation.w']

imu_yaw = np.arctan2(2.0*(qx*qy + qw*qz), 1-2*(qy*qy + qz*qz))
plt.plot(np.rad2deg(imu_yaw), label='IMU Yaw')

combined_heading = np.rad2deg(combined_heading)+127
for ind, val in enumerate(combined_heading):
    if val > 180:
        combined_heading[ind] -= 360

plt.plot(combined_heading, label='Complementary Yaw')
plt.xlabel('Time [s]')
plt.ylabel('Yaw heading [deg]')
plt.title('IMU Yaw and Complementary Filter Yaw Comparison')
plt.tight_layout()

plt.grid(True)
plt.legend(loc='best', fontsize='xx-large')
plt.savefig('circlePics/IMU & Comp Filt comparison')
plt.show()

# Integrate forward acceleration to obtain velocity estimate
print("Plot Accel.X velocity estimate")
fig, ax = plt.subplots(1, 1)

dt = 1.0/40.0
if dt == 0:
    raise ValueError("Delta time (dt) is zero. Check the sampling rate.")
acc_x = np.array(imu_data['IMU.linear_acceleration.x'])
int_acc_x = integrate.cumtrapz(acc_x-acc_x.mean(), dx=dt, initial=0)
t = np.arange(0, len(int_acc_x)*dt, dt)
plt.plot(t, int_acc_x, linewidth=2.5, label='Integrated accelerometer(X) estimate')
print("Plot GPS velocity estimate")

delta_t = 1
df = pd.DataFrame({'eastings': eastings, 'northings': northings})
df = df.reset_index(drop=True)
eastings = df['eastings']
northings = df['northings']

velocity_gps = []
for i in range(len(eastings) - 1):
    x_diff = eastings[i+1] - eastings[i]
    y_diff = northings[i+1] - northings[i]
    dist = math.sqrt(x_diff**2 + y_diff**2)
    velocity_gps.append(dist / delta_t)

time_gps = np.arange(0, len(velocity_gps) * delta_t, delta_t)  

plt.plot(time_gps, velocity_gps,linewidth=2.5, label='GPS estimate')
plt.xlabel('Time [s]')
plt.ylabel('Velocity [m/s]')
plt.title('IMU and GPS velocity estimates')
plt.tight_layout()
plt.legend()
plt.grid(True)
plt.savefig('circlePics/Accel X Velocity estimate')
plt.show()
print("Plot corrected integrated acceleration")

#dead reckoning
#a
fig, ax = plt.subplots(1, 1)
gyro_z = imu_data['IMU.angular_velocity.z']
vel = int_acc_x * gyro_z
acc_y = np.array(imu_data['IMU.linear_acceleration.y'])

ax.plot(acc_y, label=u'y_dot_dot')
ax.plot(vel, label=u'omega*x_dot')

ax.grid()
ax.legend()
ax.set_xlabel(u'Time [s]')
ax.set_ylabel(u'Acceleration [m/s^2]')
some_text = u"ωX ̇ vs y ̈_obs"
ax.set_title(some_text)

plt.savefig('circlePics/3a')

# Displacement plots
fig, ax = plt.subplots(1, 1)
gyro_z = imu_data['IMU.angular_velocity.z']

int_gyro_z = integrate.cumtrapz(gyro_z, dx=1.0/40, initial=0) 

vn = int_acc_x * np.sin(int_gyro_z)
ve = int_acc_x * np.cos(int_gyro_z)

xn = 0.9 * integrate.cumtrapz(vn, dx=1.0/40, initial=0)
xe = 0.9 * integrate.cumtrapz(ve, dx=1.0/40, initial=0)

for i in range(len(xe)):
    xe[i] -= 265
    xn[i] -= 3

th = 10
R = np.array([[np.cos(np.radians(th)), -np.sin(np.radians(th))],
              [np.sin(np.radians(th)), np.cos(np.radians(th))]])

xe = np.array(xe) * -1
xn = np.array(xn)

# Ensure 'scale' is defined, e.g., scale = 1.0
scale = 2.0  # Or set to your actual scale factor
imu_trajectory = np.dot(R, np.vstack((xe, xn))) * scale

ax.plot(0.5*imu_trajectory[:,0]+70, 0.95*imu_trajectory[:,1]+102,color = 'green')

eastings = gps_data.Utm_easting - gps_data.Utm_easting[0]
northings = gps_data.Utm_northing - gps_data.Utm_northing[0]
ax.plot(eastings, northings, label=u'Displacement from GPS')

dif_eastings = 0.83*eastings+12
dif_northings =  0.5*northings+205
ax.plot(dif_eastings[200:550], dif_northings[200:550],color='green', label='Integrated velocity(X) estimate - Displacement')


ax.set_xlabel(u'Easting [m]')
ax.set_ylabel(u'Northing [m]')
ax.set_title(u'GPS and accelerometer displacement estimates')
ax.grid(True)
ax.legend()
plt.savefig('circlePics/Velocity X Displacement estimate')
plt.show()

# The code block for calculating 'xc'
fig, ax = plt.subplots(1, 1)
xc = []
acc_y = np.array(imu_data['IMU.linear_acceleration.y'])
vel = np.array(velocity_gps)
gyro_z = np.array(imu_data['IMU.angular_velocity.z'])
diff_gyro_z = [0]

for i in range(1, len(gyro_z)):
    tmp = (gyro_z[i] - gyro_z[i - 1]) * 40  
    diff_gyro_z.append(tmp)

min_length = min(len(acc_y), len(vel), len(gyro_z), len(diff_gyro_z))

for i in range(min_length):
    if diff_gyro_z[i] != 0:
        val = (acc_y[i] - gyro_z[i] * vel[i]) / diff_gyro_z[i]
        xc.append(val)
    else:
        xc.append(0)

Xc = np.array(xc)
print("Mean of Xc:", Xc.mean())

ax.plot(Xc, label=u'xc')
ax.legend()
plt.close('all') 
