#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float32MultiArray
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.preprocessing import sequence
import tensorflow as tf

graph = tf.get_default_graph()

# set GPU memory
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

#model = load_model('/home/lab606a/ML/trajectories/fixed/classification/8classes/saved model/classifier_8classes_fixed_10balls')

class Listener:
    def __init__(self):
        self.__classifier = load_model('/home/lab606a/ML/trajectories/fixed/classification/8classes/saved model/classifier_8classes_fixed_11balls')
        self.__n_balls = 11
        self.__tmp = np.zeros([1,self.__n_balls*3])
        self.__index = 0
        self.__cnt = 0
        self.__max_index = 0
        self.__sub = rospy.Subscriber("/visual_coordinate", Float32MultiArray, self.callback)
        print("already load model")

    def top5(self):
        print("top spin speed 5")
    def top6(self):
        print("top spin speed 6")
    def left5(self):
        print("left spin speed 5")
    def left6(self):
        print("left spin speed 6")
    def right5(self):
        print("right spin speed 5")
    def right6(self):
        print("right spin speed 6")
    def back5(self):
        print("back spin speed 5")
    def back6(self):
        print("back spin speed 6")

    def show_spin_direction(self, max_index):
        ## make dictionary to replace switch case
        dictionary = {0:self.top5, 1:self.top6, 2:self.left5, 3:self.left6, 4:self.right5, 5:self.right6, 6:self.back5, 7:self.back6}
        funcToCall = dictionary[max_index]
        funcToCall()

    
    def classification(self):
        ## call classifier
        with graph.as_default():
            classes = self.__classifier.predict(self.__tmp)
        ## figure out which direction is
        self.__max_index = np.argmax(classes)
        print("cnt = ", self.__cnt)
        ## show result
        self.show_spin_direction(self.__max_index)
        self.__cnt += 1

    def callback(self, data):
        a = data.data
        
        a2 = np.array([a[1:]])
        if ((a[0] == 1) and ((self.__index+3) < (self.__n_balls*3+1))):
            self.__tmp[:,self.__index:self.__index+3] = a2
            self.__index += 3
        if (a[0] == 0):
            self.__tmp = np.zeros([1,self.__n_balls*3])
            self.__index = 0

        if (self.__index == (self.__n_balls*3)):
            self.__index = self.__index + 1
            self.__tmp = self.__tmp.reshape(1,self.__n_balls, 3)

            self.__tmp = self.__tmp.astype('float32')
            a = np.zeros([1,8])

            self.classification()

            '''
            ## switch case
            with graph.as_default():
                classes = self.__classifier.predict(self.__tmp)
            max_index = np.argmax(classes)
            #print("model predict class:", classes)
            print("cnt = ", self.__cnt)
            if max_index == 0:
                print("top spin speed 5")
            if max_index == 1:
                print("top spin speed 6")
            if max_index == 2:
                print("left spin speed 5")
            if max_index == 3:
                print("left spin speed 6")
            if max_index == 4:
                print("right spin speed 5")
            if max_index == 5:
                print("right spin speed 6")
            if max_index == 6:
                print("back spin speed 5")
            if max_index == 7:
                print("back spin speed 6")
            self.__cnt += 1
            '''



if __name__ == '__main__':
    rospy.init_node('test')
    print("init node test.py")
    Listener()
    rospy.spin()
