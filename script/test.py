#!/usr/bin/env python3
import rospy
from std_msgs.msg import String
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from IPython import display

class Listener:
    def __init__(self):
        #self.sub = rospy.Subscriber("/chatter", String, self.callback)
        #plt.figure(figsize=(8,4))
        plt.figure()
        self.x = np.array([[1],[2],[3]])
        self.y = np.array([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]],[[4,5,6],[1,5,3]]])
        print(self.y.shape)
        print(self.y)
        self.summ = np.zeros((self.y.shape[0], 1))
        for i in range(self.y.shape[0]):
            self.summ[i] = np.sum(self.y[i,:,:])
        print(self.summ)
        plt.pause(3)
        #self.tt = np.array([range(5,10,1)])
        #print(self.tt)
        for i in range(10):
            self.summ = self.summ+1
            self.draw()
            plt.pause(1)
            plt.clf()
        #self.summ = self.summ**(1/2)
        #print(self.summ)
        
    def draw(self):
        #print(self.summ.shape)
        #print(self.x.shape)
        plt.plot(self.x, self.summ)
        plt.scatter(self.x, self.summ)
        plt.grid(True)
        plt.title('error between visual measurement and model prediction')
        plt.xlabel('Number of input balls')
        plt.ylabel('error')

        plt.pause(0.00000000001)

        #display.clear_output(wait=True)
        #display.display(plt.gcf())
        plt.gcf()

        #plt.show()
        #plt.pause(0.00000000001)


    #def callback(self, data):
        #rospy.loginfo("I heard %s", data.data)

if __name__ == '__main__':
    plt.ion()
    rospy.init_node('test')
    print("init node test.py")
    Listener()
    rospy.spin()
