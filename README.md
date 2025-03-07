# f1tenth_omniverse
A robust, physics-accurate omniverse simulation for f1tenth racing.

## Setup:
**The instructions below assume that you are working on a Linux machine with ROS2 Humble and a RTX graphics card installed**

1. Install NVIDIA Omniverse IsaacSim by downloading the most current [version, here] (https://docs.isaacsim.omniverse.nvidia.com/latest/installation/download.html#isaac-sim-latest-release) and installing as below:

```
mkdir ~/isaacsim
cd ~/Downloads
unzip "**file name**" -d ~/isaacsim
cd ~/isaacsim
./post_install.sh
```

2. Start IsaacSim
Source your ROS2 installation. When working from a sourced terminal, ROS2 will automatically be detected by omniverse.

```
source /opt/ros/humble/setup.bash
```

You may then start omniverse simulator, and select **Isaac Sim Full**

```
./isaac-sim.selector.sh
```

**Note that the first start up will take a while. Subsequent starts will be much quicker.**

3. Open USD file from this repository.
Select File>open and open the USD file from this repo in omniverse. No need to save the current blank stage when it asks. The entire sim, along with ROS2 bridge, will load and be ready for use. Press play on left side to begin simulation, and open ROS2 topics.

Topics:
/drive (ackermannStamped message to control car)
/scan (laserscan message from lidar sensor)
/odom (odometry message produced by sim)
/rgb (rgb camera data)
/depth (depth camera data)
