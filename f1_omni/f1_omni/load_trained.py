import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
import numpy as np
from stable_baselines3 import SAC

class SACInferenceNode(Node):
    def __init__(self):
        super().__init__('sac_lidar_agent')

        # Load the trained SAC model
        model_path = '/home/r478a194/f1tenth_omniverse/ros2_ws/checkpoints/dqn_lidar_100000_steps.zip'
        self.get_logger().info(f'Loading model from {model_path}...')
        self.model = SAC.load(model_path)

        # Subscribe to LiDAR scan data
        self.subscription_scan = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            10
        )

        # Subscribe to Odometry data
        self.subscription_odom = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        # Publisher for driving commands
        self.publisher_cmd = self.create_publisher(AckermannDriveStamped, "/drive", 10)

        # Initialize state variables
        self.lidar_data = np.zeros(1081, dtype=np.float32)
        self.speed = 0.0
        self.steering_angle = 0.0
        self.received_data = False  # Ensure first sensor data is received

    def lidar_callback(self, msg):
        """Receive and store LiDAR scan data."""
        self.lidar_data = np.array(msg.ranges, dtype=np.float32)
        self.lidar_data[np.isinf(self.lidar_data)] = msg.range_max  # Handle infinite values
        self.received_data = True

    def odom_callback(self, msg):
        """Receive and store odometry data."""
        self.speed = msg.twist.twist.linear.x
        self.steering_angle = msg.twist.twist.angular.z

    def run(self):
        """Main loop for inference."""
        while rclpy.ok():
            if not self.received_data:
                self.get_logger().info("Waiting for first LiDAR data...")
                rclpy.spin_once(self)
                continue

            # Prepare observation
            observation = np.concatenate((self.lidar_data, [self.speed, self.steering_angle])).astype(np.float32)

            # Get action from trained model
            action, _ = self.model.predict(observation, deterministic=True)
            steering, throttle = action

            # Create and publish control command
            cmd = AckermannDriveStamped()
            cmd.drive.speed = float(np.clip(throttle, 0.05, 1) * 8.0)
            cmd.drive.steering_angle = float(np.clip(steering, -1, 1) * 0.523599)
            self.publisher_cmd.publish(cmd)

            # Sleep to maintain loop rate
            rclpy.spin_once(self)

def main():
    rclpy.init()
    node = SACInferenceNode()
    node.run()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
