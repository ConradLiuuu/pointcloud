#!/usr/bin/env python3
import tttp
import rospy
import matplotlib.pyplot as plt

if __name__ == '__main__':
    rospy.init_node('classifier_test_60hz_9step')
    plt.ion()
    rospy.loginfo("init node classifier_test_60hz_9step.py")
    obj = tttp.Listener()
    rospy.spin()
