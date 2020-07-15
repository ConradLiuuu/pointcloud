#!/usr/bin/env python
import rospy
import numpy as np
import pandas as pd
from std_msgs.msg import Float32MultiArray


if __name__ == '__main__':
    rospy.init_node('pub')

    pub = rospy.Publisher('/visual_coordinate', Float32MultiArray, queue_size=1)
    msg_mx = Float32MultiArray()
    msg_mx.data = np.array([1,2,3,4])
    r = rospy.Rate(10)

    while not rospy.is_shutdown():
        pub.publish(msg_mx)
        r.sleep()

'''
    pub1 = rospy.Publisher('mx', Float32MultiArray, queue_size=1)
    pub2 = rospy.Publisher('my', Float32MultiArray, queue_size=1)
    pub3 = rospy.Publisher('mz', Float32MultiArray, queue_size=1)
    pub4 = rospy.Publisher('vx', Float32MultiArray, queue_size=1)
    pub5 = rospy.Publisher('vy', Float32MultiArray, queue_size=1)
    pub6 = rospy.Publisher('vz', Float32MultiArray, queue_size=1)
    r = rospy.Rate(5)

    msg_mx = Float32MultiArray()
    msg_my = Float32MultiArray()
    msg_mz = Float32MultiArray()
    msg_vx = Float32MultiArray()
    msg_vy = Float32MultiArray()
    msg_vz = Float32MultiArray()

    stereo = np.array(pd.read_csv('/home/lab606a/Documents/st.csv'))
    print (stereo.shape)
    mx = stereo[:, 0]
    my = stereo[:, 1]
    mz = stereo[:, 2]
    vx = stereo[:, 3]
    vy = stereo[:, 4]
    vz = stereo[:, 5]


    print(mx)
    msg_mx.data = mx
    msg_my.data = my
    msg_mz.data = mz
    msg_vx.data = vx
    msg_vy.data = vy
    msg_vz.data = vz

    while not rospy.is_shutdown():
        pub1.publish(msg_mx)
        pub2.publish(msg_my)
        pub3.publish(msg_mz)
        pub4.publish(msg_vx)
        pub5.publish(msg_vy)
        pub6.publish(msg_vz)
        r.sleep()
'''
