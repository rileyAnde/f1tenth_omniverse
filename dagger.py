import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import Joy
import numpy as np
import pickle
import os
import math


class HGRecorder(Node):
    def __init__(self):
        super().__init__("hg_dagger_recorder")
        self.speed = 0.0
        self.steering_angle = 0.0
        self.expert_data = []
        self.joy_input_active = False

        self.max_speed = 8.0
        self.max_steering_angle =  math.radians(30)

        self.create_subscription(LaserScan, "/scan", self.lidar_callback, 1)
        self.create_subscription(Joy, "/joy", self.joy_callback, 1)

    def joy_callback(self, msg):

        raw_trigger = msg.axes[5]
        speed = ((-raw_trigger + 1.0) / 2.0) * self.max_speed
        steering_input = msg.axes[0]
        steering_angle = steering_input * self.max_steering_angle

        self.joy_input_active = (abs(speed) > 1e-3) or (abs(steering_angle) > 1e-3)

        if self.joy_input_active:
            self.speed = speed
            self.steering_angle = steering_angle

    def lidar_callback(self, msg):
        if not self.joy_input_active:
            return

        lidar = np.array(msg.ranges, dtype=np.float32)
        action = np.array([self.speed, self.steering_angle], dtype=np.float32)

        self.expert_data.append((lidar.copy(), action.copy()))
        self.get_logger().info(f"Collected sample #{len(self.expert_data)}")

        # cap number of samples
        if len(self.expert_data) >= 2000:
            self.create_timer(1.0, self.save_and_shutdown)

    def save_and_shutdown(self):
        self.get_logger().info("Saving HG-DAgger data...")

        obs_data = np.array([d[0] for d in self.expert_data])
        act_data = np.array([d[1] for d in self.expert_data])

        filename = "dagger_data1.pkl"

        if os.path.exists(filename):
            try:
                with open(filename, "rb") as f:
                    existing_obs, existing_acts = pickle.load(f)
                obs_data = np.concatenate([existing_obs, obs_data])
                act_data = np.concatenate([existing_acts, act_data])
            except Exception as e:
                self.get_logger().warn(f"Could not load existing data: {e}")

        with open(filename, "wb") as f:
            pickle.dump((obs_data, act_data), f)

        self.get_logger().info("Data saved. Shutting down...")
        rclpy.shutdown()


def main():
    rclpy.init()
    node = HGRecorder()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
