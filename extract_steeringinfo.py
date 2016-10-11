#!/usr/bin/python

# Start up ROS pieces.
PKG = 'my_package'
import roslib; #roslib.load_manifest(PKG)
import rosbag
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from tqdm import tqdm

# Reading bag filename from command line or roslaunch parameter.
import os
import sys

class ImageCreator():
    # Must have __init__(self) function for a class, similar to a C++ class constructor.
    def __init__(self):
        # Get parameters when starting node from a launch file.
        if len(sys.argv) < 1:
            save_dir = rospy.get_param('save_dir')
            filename = rospy.get_param('filename')
            rospy.loginfo("Bag filename = %s", filename)
        # Get parameters as arguments to 'rosrun my_package bag_to_images.py <save_dir> <filename>', where save_dir and filename exist relative to this executable file.
        else:
            save_dir = os.path.join(sys.path[0], sys.argv[1])
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)
            filename = os.path.join(sys.path[0], sys.argv[2])
            rospy.loginfo("Bag filename = %s", filename)

        # Use a CvBridge to convert ROS images to OpenCV images so they can be saved.
        self.bridge = CvBridge()

        steeringCSV = open(os.path.join(save_dir, 'steeringinfo.csv'), 'w+')
        # Open bag file.
        print('Loading the bag file')
        with rosbag.Bag(filename, 'r') as bag:
            progressBar = tqdm(total=bag.get_message_count('/vehicle/steering_report'))
            print('Loaded the bag file')
            for topic, msg, t in bag.read_messages():
                if topic == "/vehicle/steering_report":
                    try:
                        timestr = "%.6f" % msg.header.stamp.to_sec()
                        steering_data = str(msg.steering_wheel_angle)
                        steeringCSV.write(str(timestr) + ',')
                        steeringCSV.write(steering_data + '\n')
                        progressBar.update(1)
                    except CvBridgeError, e:
                        print e
            progressBar.close()
        steeringCSV.close()

# Main function.    
if __name__ == '__main__':
    # Initialize the node and name it.
    rospy.init_node(PKG)
    # Go to class functions that do all the heavy lifting. Do error checking.
    try:
        image_creator = ImageCreator()
    except rospy.ROSInterruptException: pass
