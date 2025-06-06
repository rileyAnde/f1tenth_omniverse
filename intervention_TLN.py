import rclpy
import numpy as np
import time
import tensorflow as tf
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import Joy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Header
import math

class TLNNode(Node):
    def __init__(self):
        super().__init__('tln_node')
        self.ackermann_publisher = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.scan_subscription = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.joy_sub = self.create_subscription(Joy, '/joy', self.joy_callback, 10)
        self.get_logger().info('TLNNode has been started.')

        self.model_path = "/home/r478a194/f1tenth_omniverse/ros2_ws/models/TLN_dag_noquantized.tflite" #"/home/r478a194/Downloads/f1_tenth_model_small_noquantized.tflite"
        self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        self.input_index = self.interpreter.get_input_details()[0]["index"]
        self.output_index = self.interpreter.get_output_details()[0]["index"]
        self.scan_buffer = np.zeros((2, 20))
        self.joy_input_active = False

        self.min_speed = 1
        self.max_speed = 8
        self.max_steering_angle = math.radians(30)

    def joy_callback(self, msg):

        raw_trigger = msg.axes[5]
        speed = ((-raw_trigger + 1.0) / 2.0) * self.max_speed
        steering_input = msg.axes[0]
        steering_angle = steering_input * self.max_steering_angle

        self.joy_input_active = (abs(speed) > 1e-3) or (abs(steering_angle) > 1e-3)

        if self.joy_input_active:
            self.speed = speed
            self.steering_angle = steering_angle

    def linear_map(self, x, x_min, x_max, y_min, y_max):
        return (x - x_min) / (x_max - x_min) * (y_max - y_min) + y_min    

    def scan_callback(self, msg):
        if self.joy_input_active:
            return
        scans = np.array(msg.ranges)
        scans = np.append(scans, [20])
        self.get_logger().info(f'num scans:{len(scans)}')
        noise = np.random.normal(0, 0.5, scans.shape)
        scans = scans + noise
        scans[scans > 10] = 10
        scans = scans[::2]  # Use every other value
        self.get_logger().info(f'after crop:{len(scans)}')
        scans = np.expand_dims(scans, axis=-1).astype(np.float32)
        scans = np.expand_dims(scans, axis=0)

        self.interpreter.set_tensor(self.input_index, scans)
        start_time = time.time()
        self.interpreter.invoke()
        inf_time = (time.time() - start_time) * 1000  # in milliseconds
        self.get_logger().info(f'Inference time: {inf_time:.2f} ms')

        output = self.interpreter.get_tensor(self.output_index)
        steer = output[0, 0]
        speed = output[0, 1]

        speed = self.linear_map(speed, 0, 1, self.min_speed, self.max_speed)

        self.publish_ackermann_drive(speed, steer)

    def publish_ackermann_drive(self, speed, steering_angle):
        ackermann_msg = AckermannDriveStamped()
        ackermann_msg.header = Header()
        ackermann_msg.header.stamp = self.get_clock().now().to_msg()
        ackermann_msg.drive.speed = float(speed)
        ackermann_msg.drive.steering_angle = float(steering_angle)

        self.ackermann_publisher.publish(ackermann_msg)
        self.get_logger().info(f'Published AckermannDriveStamped message: speed={speed}, steering_angle={steering_angle}')

def main(args=None):
    rclpy.init(args=args)
    node = TLNNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt (SIGINT)')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
