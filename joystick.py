import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from rosgraph_msgs.msg import Clock
from ackermann_msgs.msg import AckermannDriveStamped
import math

class JoyToAckermann(Node):

    def __init__(self):
        super().__init__('joy_to_ackermann')
        self.publisher_ = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.reset  = self.create_publisher(Clock, '/crash', 10)
        self.subscription = self.create_subscription(Joy, '/joy', self.joy_callback, 10)
        self.max_speed = 8.0
        self.max_steering_angle = math.radians(30)
        self.joy_input_active = False

    def joy_callback(self, msg):
        raw_trigger = msg.axes[5]
        speed = ((-raw_trigger + 1.0) / 2.0) * self.max_speed
        steering_input = msg.axes[0]
        steering_angle = steering_input * self.max_steering_angle

        self.joy_input_active = (abs(speed) > 1e-3) or (abs(steering_angle) > 1e-3)

        if self.joy_input_active:
            drive_msg = AckermannDriveStamped()
            drive_msg.header.stamp = self.get_clock().now().to_msg()
            drive_msg.drive.speed = speed
            drive_msg.drive.steering_angle = steering_angle
            self.publisher_.publish(drive_msg)
            
        if msg.buttons[0] == 1:
            self.reset.publish(Clock())

def main(args=None):
    rclpy.init(args=args)
    node = JoyToAckermann()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
