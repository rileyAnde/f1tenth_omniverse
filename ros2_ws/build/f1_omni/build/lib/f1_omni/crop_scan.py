import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

class ScanCrop(Node):
    def __init__(self):
        super().__init__('scan_processor')
        self.scan_publisher = self.create_publisher(LaserScan, '/scan', 10)
        self.scan_subscription = self.create_subscription(LaserScan, 'raw_scan', self.scan_callback, 10)
        self.get_logger().info('scan-croppinator 3000 has been started.')

    def scan_callback(self, msg):
        self.publish_new_scan(msg)

    def publish_new_scan(self, msg):
        newscan = LaserScan()
        newscan.angle_min = -2.3561944902
        newscan.angle_max = 2.3561944902
        newscan.angle_increment = msg.angle_increment
        newscan.time_increment = msg.time_increment
        newscan.scan_time = msg.scan_time
        newscan.ranges = msg.ranges[180:1261]
        newscan.intensities = msg.intensities[180:1261]

        self.scan_publisher.publish(newscan)
        #self.get_logger().info(f'Published AckermannDriveStamped message: speed={speed}, steering_angle={steering_angle}')

def main(args=None):
    rclpy.init(args=args)
    node = ScanCrop()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt (SIGINT)')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()