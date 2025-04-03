import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from rosgraph_msgs.msg import Clock
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Bool
import numpy as np
import gym
from gym import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
import torch
import sys
sys.path.append("/home/r478a194/f1tenth_omniverse/ros2_ws/src/f1_omni/f1_omni")
from callable_TLN import Callable_TLN


class dq(Node, gym.Env):
    def __init__(self):
        Node.__init__(self, "dq")
        gym.Env.__init__(self)
        self.crash = False
        self.lap_count = -1
        self.use_expert = True
        self.tln = Callable_TLN()

        #pubs and subs
        self.publisher_cmd = self.create_publisher(AckermannDriveStamped, "/drive", 1)
        self.subscription_scan = self.create_subscription(LaserScan, "/scan", self.lidar_callback, 1)
        self.subscription_odom = self.create_subscription(Odometry, "/odom", self.odom_callback, 1)
        self.crash_sub = self.create_subscription(Clock, "/crash", self.crash_callback, 1)
        self.reset_pub = self.create_publisher(Twist, "/reset", 1)
        self.lap_count = self.create_subscription(Clock, "/clock", self.lap, 1)

        # Initialize state variables
        self.lidar_data = np.zeros(541)
        self.speed = 0.0
        self.steering_angle = 0.0
        self.received_data = False  #got first scan

        #rl env
        self.observation_space = spaces.Box(low=-1, high=1, shape=(len(self.lidar_data) + 2,), dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([-0.523599, 0.0]), high=np.array([0.523599, 1.0]), dtype=np.float32)

        self.max_steps = 5000  #max steps / ep
        self.current_step = 0

    def lidar_callback(self, msg):
        #recieve and store lidar scan
        #self.get_logger().info(f"scan len:{len(msg.ranges)}")
        self.lidar_data = np.array(msg.ranges)[::2]
        #self.get_logger().info(f"post-crop scan len:{len(self.lidar_data)}")
        self.received_data = True

    def odom_callback(self, msg):
        #get odom and store
        self.speed = msg.twist.twist.linear.x
        self.steering_angle = msg.twist.twist.angular.z

    def crash_callback(self, msg):
        if msg != None:
            self.get_logger().info("Crash")
            self.crash = True
    
    def lap(self, msg):
        self.get_logger().info("Lap")
        self.lap_count += 1

    def reset(self):
        self.current_step = 0
        self.received_data = False
        self.reset_pub.publish(Twist())
        self.crash = False

        self.lap_count = -1


        while not self.received_data and rclpy.ok():
            rclpy.spin_once(self)

        return np.concatenate((self.lidar_data, [self.speed, self.steering_angle])).astype(np.float32)

    def step(self, action):

        if self.use_expert:
            action = self.tln.tln_expert(self.lidar_data)

        """apply action, update state, and calculate reward."""
        self.current_step += 1
        steering, throttle = action
        #publish control
        cmd = AckermannDriveStamped()
        cmd.drive.speed = float(np.clip(throttle, 0, 1)*8.0)
        cmd.drive.steering_angle = float(np.clip(steering, -1, 1))
        self.publisher_cmd.publish(cmd)

        #wait for sensor update
        rclpy.spin_once(self)

        # reward calculation
        reward = cmd.drive.speed * 2
        reward -= (abs(cmd.drive.steering_angle) ** 2) * 5

        if cmd.drive.speed > 3.0:
            reward += 10

        if self.crash:
            reward -= 500  # large penalty for crashing
        if self.current_step >= self.max_steps:
            reward += 100  #bonus for completing the episode

        reward += self.lap_count * 20

        # check if episode is done
        done = self.crash or self.current_step >= self.max_steps

        return np.concatenate((self.lidar_data, [self.speed, self.steering_angle])).astype(np.float32), reward, done, {}


def main():
    rclpy.init()
    env = dq()

    #save model every 10,000 steps
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./pt_checkpoints/",
        name_prefix="bc_tln"
    )

    # define model

    policy_kwargs = dict(
    net_arch=[256, 256, 128]
    )

    #from scratch
    model = SAC("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1, learning_rate=1e-3, buffer_size=50000, batch_size=64, target_update_interval=1000)

    #continue training from 10,000 step tln train
    #model = SAC.load("/home/r478a194/f1tenth_omniverse/ros2_ws/pt_checkpoints/dqn_lidar_final.zip", env)

    #pretrain from TLN
    model.learn(total_timesteps=20000)
    model.save("./tln_base_20000/")

    #switch to RL
    env.use_expert = False
    model.learn(total_timesteps=5000000, callback=checkpoint_callback)


    # Save final model
    model.save("./dqn_lidar_final")

    # Cleanup
    env.destroy_node()
    rclpy.shutdown()


# if __name__ == "__main__":
#     main()
