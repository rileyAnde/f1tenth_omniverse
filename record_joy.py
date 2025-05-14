import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import Joy
import numpy as np
from collections import deque
import pickle
import os



class TLNRecorder(Node):
    def __init__(self):
        super().__init__("expert_recorder")
        self.speed = 0.0
        self.steering_angle = 0.0
        self.expert_data = []
        self.shutdown_initiated = False
        self.rec = 0

        self.drive_sub = self.create_subscription(AckermannDriveStamped, "/drive", self.drive_callback, 1)
        self.subscription_scan = self.create_subscription(LaserScan, "/scan", self.lidar_callback, 1)
        self.joy_sub = self.create_subscription(Joy, "/joy", self.joy_callback, 1)
        #self.subscription_odom = self.create_subscription(Odometry, "/odom", self.odom_callback, 1)
    
    def joy_callback(self, msg):
        if msg.buttons[0] == 1:
            if self.rec == 0:
                self.rec = 1
            else:
                self.rec = 0

    def drive_callback(self, msg):
        self.steering_angle = msg.drive.steering_angle
        self.speed = msg.drive.speed

    def lidar_callback(self, msg):
        
        if self.rec != 0:
            lidar = np.array(msg.ranges)

            obs = lidar.astype(np.float32)

            # save (obs, action)
            action = np.array([self.speed, self.steering_angle], dtype=np.float32)
            self.expert_data.append((obs.copy(), action.copy()))

            self.get_logger().info(f"Collected sample #{len(self.expert_data)}")

        # Stop after N samples
        if len(self.expert_data) >= 10000:
            # Delay the shutdown to avoid race condition with callbacks
            self.create_timer(1.0, self.save_and_shutdown)

    # def odom_callback(self, msg):
    #     self.speed = msg.twist.twist.linear.x
    #     self.steering_angle = msg.twist.twist.angular.z

    def save_and_shutdown(self):
        self.get_logger().info("Saving expert data...")

        obs_data = np.array([d[0] for d in self.expert_data])
        act_data = np.array([d[1] for d in self.expert_data])

        if os.path.exists("joy_data4.pkl"):
            try:
                with open("joy_data4.pkl", "rb") as f:
                    existing_obs, existing_acts = pickle.load(f)
                obs_data = np.concatenate([existing_obs, obs_data])
                act_data = np.concatenate([existing_acts, act_data])
            except Exception as e:
                self.get_logger().warn(f"Failed to load existing data: {e}")

        with open("joy_data4.pkl", "wb") as f:
            pickle.dump((obs_data, act_data), f)

        self.get_logger().info("Data saved. Shutting down...")
        rclpy.shutdown() 


def main():
    rclpy.init()
    node = TLNRecorder()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
