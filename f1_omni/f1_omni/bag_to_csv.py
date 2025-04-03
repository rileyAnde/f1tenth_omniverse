import sqlite3
import pandas as pd
import numpy as np
import struct

# Path to your ROS2 bag file (.db3)
ros2_db_path = "/home/r478a194/f1tenth_omniverse/ros2_ws/new_data/new_data_0.db3"
csv_output_path = "dataset.csv"

# Connect to the SQLite database inside the ROS2 bag
conn = sqlite3.connect(ros2_db_path)
cursor = conn.cursor()

# Query available topics
cursor.execute("SELECT name FROM topics")
topics = [row[0] for row in cursor.fetchall()]
print(f"Available topics: {topics}")

# Extract LiDAR (`/scan`) and drive (`/drive`) data
df_lidar = pd.read_sql_query("SELECT timestamp, data FROM messages WHERE topic_id = (SELECT id FROM topics WHERE name='/scan')", conn)
df_drive = pd.read_sql_query("SELECT timestamp, data FROM messages WHERE topic_id = (SELECT id FROM topics WHERE name='/drive')", conn)

# Convert timestamps to match
df_lidar["timestamp"] = df_lidar["timestamp"].astype(float)
df_drive["timestamp"] = df_drive["timestamp"].astype(float)

def decode_lidar(binary_data):
    """ Extracts 'ranges' from sensor_msgs/LaserScan (float32 array) """
    num_ranges = (len(binary_data) - 28) // 4  # Assuming 28-byte header
    return list(struct.unpack(f"{num_ranges}f", binary_data[28:]))

# Function to decode drive commands (ackermann_msgs/AckermannDrive)
def decode_drive(binary_data):
    """ Extract 'steering_angle' and 'speed' from a 40-byte AckermannDriveStamped message. """
    
    print(f"Raw message size: {len(binary_data)} bytes")  # Debugging line
    
    if len(binary_data) != 40:  # Ensure the message size is what we expect
        print(f"Unexpected message size: {len(binary_data)}")
        return [0.0, 0.0]

    try:
        # Skip the first 20 bytes (header)
        steering_angle, _, speed, _, _ = struct.unpack("fffff", binary_data[20:40])  
        
        return [steering_angle, speed]
    
    except struct.error as e:
        print(f"Failed to decode drive data: {binary_data}, Error: {e}")
        return [0.0, 0.0]

df_lidar["data_x"] = df_lidar["data"].apply(decode_lidar)
df_drive["data_y"] = df_drive["data"].apply(decode_drive)

# Drop old binary 'data' columns
df_lidar = df_lidar.drop(columns=["data"])
df_drive = df_drive.drop(columns=["data"])

print(df_drive.head())  # Check the first few rows
print(type(df_drive.iloc[0, 1]))  # Check the type of the `data` column
print(df_drive.iloc[0, 1])  # Print one raw drive message

min_len = min(len(df_lidar), len(df_drive))
df_lidar = df_lidar.iloc[:min_len].reset_index(drop=True)
df_drive = df_drive.iloc[:min_len].reset_index(drop=True)

#merge data
df = pd.concat([df_lidar["data_x"], df_drive["data_y"]], axis=1)
df.columns = ["data_x", "data_y"]

# Save to CSV
df.to_csv(csv_output_path, index=False)
print(f"dataset saved to {csv_output_path}")
