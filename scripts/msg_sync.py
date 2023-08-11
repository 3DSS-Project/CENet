#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Header

class ImageSynchronizer:
    def __init__(self):
        self.range_image_header = None
        self.signal_image_header = None
        
        # Threshold for considering two images as synchronized (in seconds)
        self.sync_threshold = 0.05  

        rospy.Subscriber("/ouster/range_image", Image, self.range_image_callback)
        rospy.Subscriber("/ouster/signal_image", Image, self.signal_image_callback)
        
    def range_image_callback(self, msg):
        self.range_image_header = msg.header.seq
        print(f"range image seq: {self.range_image_header}")
        self.check_synchronization()

    def signal_image_callback(self, msg):
        self.signal_image_header = msg.header.seq
        print(f"signal image seq: {self.signal_image_header}")
        self.check_synchronization()

    def check_synchronization(self):
        if self.range_image_header is None or self.signal_image_header is None:
            return
        
        seq_diff = self.range_image_header - self.signal_image_header
        #time_diff = abs(self.range_image_header.stamp.to_sec() - self.signal_image_header.stamp.to_sec())
        rospy.loginfo("range_image header: %f, signal_image header: %f", self.range_image_header, self.signal_image_header)
        if seq_diff < self.sync_threshold:
            rospy.loginfo("The two images are synchronized!")
        else:
            rospy.logwarn("The images are not synchronized. Time difference: %f seconds", seq_diff)

if __name__ == '__main__':
    rospy.init_node('image_synchronizer_node')
    synchronizer = ImageSynchronizer()
    rospy.spin()
