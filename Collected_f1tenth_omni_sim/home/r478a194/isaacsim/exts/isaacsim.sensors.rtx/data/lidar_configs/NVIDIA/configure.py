from omni.isaac.sensor import _sensor
import omni.usd
import carb
import numpy as np

# Get the RTX LiDAR handle
lidar_path = "/World/leatherback_ROS/Rigid_Bodies/Chassis/Lidar"
lidar_interface = _sensor.acquire_lidar_sensor_interface()

# Set up Hokuyo UST-10LX parameters
hokuyo_config = {
    "horizontalFov": 270.0,  # Hokuyo FOV
    "verticalFov": 0.0,  # 2D LiDAR, so no vertical FOV
    "horizontalResolution": 0.25,  # 0.25° per step
    "verticalResolution": 0.0,  # No vertical resolution needed
    "minRange": 0.02,  # 2 cm min range
    "maxRange": 10.0,  # Hokuyo max range
    "rotationRate": 40.0  # Hokuyo rotates at 40Hz
}

# Apply settings
for param, value in hokuyo_config.items():
    lidar_interface.set_lidar_param(lidar_path, param, value)

print("Hokuyo LiDAR configured successfully!")
