#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float32MultiArray
import numpy as np
import pandas as pd
import cupy as cp
from keras.models import load_model
from keras.preprocessing import sequence
import tensorflow as tf
import matplotlib.pyplot as plt

graph = tf.get_default_graph()

# set GPU memory
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

np.set_printoptions(suppress=True)

class Listener:
    def __init__(self):
        self.__sub = rospy.Subscriber("/visual_coordinate", Float32MultiArray, self.callback)
        #self.__pub = rospy.Publisher("/prediction_coordinate", Float32MultiArray, queue_size=1)
        self.__tmp = cp.zeros([1,3])
        self.__arr_classification = cp.zeros([1,90])
        self.__vis_point = cp.zeros((1,3))
        self.__vis_balls = cp.zeros((5,3))
        self.__predction_balls = cp.zeros((1,5,3))
        self.__arr_prediction = cp.zeros([1,15])
        self.__padding_done = False
        self.__cnt = 5
        self.__max_index = 0
        self.__pred = cp.zeros([1,5,3])
        self.__time = 0.016667
        self.__delta_T = 0.016667
        #self.__pred_msg = Float32MultiArray()
        #self.__rate = rospy.Rate(100)
        self.__classifier = load_model('/home/lab606a/catkin_ws/src/pointcloud/models/classification_30ball_20200404_filled_v2')
        self.__pred_top5 = load_model('/home/lab606a/catkin_ws/src/pointcloud/models/prediction_top5')
        self.__pred_top6 = load_model('/home/lab606a/catkin_ws/src/pointcloud/models/prediction_top6')
        self.__pred_left5 = load_model('/home/lab606a/catkin_ws/src/pointcloud/models/prediction_left5')
        self.__pred_left6 = load_model('/home/lab606a/catkin_ws/src/pointcloud/models/prediction_left6')
        self.__pred_right5 = load_model('/home/lab606a/catkin_ws/src/pointcloud/models/prediction_right5')
        self.__pred_right6 = load_model('/home/lab606a/catkin_ws/src/pointcloud/models/prediction_right6')
        self.__pred_back5 = load_model('/home/lab606a/catkin_ws/src/pointcloud/models/prediction_back5')
        self.__pred_back6 = load_model('/home/lab606a/catkin_ws/src/pointcloud/models/prediction_back6')
        print("already load model")

    ## print spin direction and speed
    def top5(self):
        print("top spin speed 5")
        with graph.as_default():
            self.__pred = self.__pred_top5.predict(cp.asnumpy(self.__arr_prediction.reshape(1,5,3)))
        #print(self.__pred)
    def top6(self):
        print("top spin speed 6")
        with graph.as_default():
            self.__pred = self.__pred_top6.predict(cp.asnumpy(self.__arr_prediction.reshape(1,5,3)))
        #print(self.__pred)
    def left5(self):
        print("left spin speed 5")
        with graph.as_default():
            self.__pred = self.__pred_left5.predict(cp.asnumpy(self.__arr_prediction.reshape(1,5,3)))
        #print(self.__pred)
    def left6(self):
        print("left spin speed 6")
        with graph.as_default():
            self.__pred = self.__pred_left6.predict(cp.asnumpy(self.__arr_prediction.reshape(1,5,3)))
        #print(self.__pred)
    def right5(self):
        print("right spin speed 5")
        with graph.as_default():
            self.__pred = self.__pred_right5.predict(cp.asnumpy(self.__arr_prediction.reshape(1,5,3)))
        #print(self.__pred)
    def right6(self):
        print("right spin speed 6")
        with graph.as_default():
            self.__pred = self.__pred_right6.predict(cp.asnumpy(self.__arr_prediction.reshape(1,5,3)))
        #print(self.__pred)
    def back5(self):
        print("back spin speed 5")
        with graph.as_default():
            self.__pred = self.__pred_back5.predict(cp.asnumpy(self.__arr_prediction.reshape(1,5,3)))
        #print(self.__pred)
    def back6(self):
        print("back spin speed 6")
        with graph.as_default():
            self.__pred = self.__pred_back6.predict(cp.asnumpy(self.__arr_prediction.reshape(1,5,3)))
        #print(self.__pred)

    def show_spin_direction(self, max_index):
        ## make dictionary to replace switch case
        dictionary = {0:self.top5, 1:self.top6, 2:self.left5, 3:self.left6, 4:self.right5, 5:self.right6, 6:self.back5, 7:self.back6}
        funcToCall = dictionary[max_index]
        funcToCall()

    
    def classification(self):
        ## call classifier
        with graph.as_default():
            classes = self.__classifier.predict(cp.asnumpy(self.__arr_classification.reshape(1,30,3)))
        ## figure out which direction is
        self.__max_index = np.argmax(classes)
        print("number of input balls = ", self.__cnt)
        ## show result
        self.show_spin_direction(self.__max_index)
        self.__cnt += 1

    def padding(self):
        # if __tmp is empty, init array
        if (self.__padding_done == False):
            self.__tmp = self.__vis_point ## pad first point
            self.__padding_done = True
        else:
        # if __tmp is not empty, then filled array
            self.__tmp = cp.hstack((self.__tmp, self.__vis_point))
            if (self.__tmp.shape[1] == 15): ## when colect 5 balls
                self.__arr_classification[:,:self.__tmp.shape[1]] = self.__tmp ## asigne to classification input array
                self.__vis_balls = self.__tmp.reshape(1,5,3)
                self.__arr_prediction[:,:] = self.__tmp ## for predct next 5 steps
            if (self.__tmp.shape[1] > 15): ## when colect over 5 balls
                self.__vis_balls = cp.vstack((self.__vis_balls, self.__tmp[:,self.__tmp.shape[1]-15:].reshape(1,5,3))) ## visual measurement point for calculate error
                self.__arr_prediction[:,:] = self.__tmp[:,self.__tmp.shape[1]-15:] ## rolling visual measurement for predct next 5 steps
                if (self.__tmp.shape[1] <= 90): ## when colect under 31 balls
                    self.__arr_classification[:,:self.__tmp.shape[1]] = self.__tmp ## still asigne to classification input array
                else:
                    self.__cnt = 5
            #print(self.__arr_prediction)

    def pub_prediction(self):
        self.__pred = self.__pred.astype('float32')
        self.__pred_msg.data = self.__pred.reshape(15,1)
        self.__pub.publish(self.__pred_msg)

    def final_padding(self):
        for i in range(4): ## padding zeros 4 times
            self.__tmp = cp.asnumpy(self.__tmp)
            self.__tmp = sequence.pad_sequences(self.__tmp, maxlen=(self.__tmp.shape[1]+3), padding='post', dtype='float32')
            self.__tmp = cp.array(self.__tmp)
            
            self.__vis_balls = cp.vstack((self.__vis_balls, self.__tmp[:,self.__tmp.shape[1]-15:].reshape(1,5,3)))
            self.__arr_prediction[:,:] = self.__tmp[:,self.__tmp.shape[1]-15:] ## rolling visual measurement for predct next 5 steps
            self.show_spin_direction(self.__max_index) ## predct next 5 steps
            #self.pub_prediction() ## publish result to topic
            self.filled_pred_result()

    def filled_pred_result(self):
        if (self.__tmp.shape[1] == 15):
            self.__predction_balls = cp.array(self.__pred)
        if (self.__tmp.shape[1] > 15):
            self.__predction_balls = cp.vstack((self.__predction_balls, cp.array(self.__pred)))

    def calculate_error(self):
        error = self.__vis_balls[5:,:,:] - self.__predction_balls[:self.__predction_balls.shape[0]-5, :, :]
        res = cp.zeros((error.shape[0], 1))
        axis = cp.linspace(1, error.shape[0], error.shape[0])
        axis = axis.reshape(axis.shape[0], 1)
        error = cp.abs(error)
        res = cp.sum(error, axis=-1)
        res = cp.sum(res, axis=-1)

        ## plot error
        plt.clf()
        plt.plot(cp.asnumpy(axis), cp.asnumpy(res.reshape(res.shape[0],1)))
        plt.scatter(cp.asnumpy(axis), cp.asnumpy(res))
        plt.grid(True)
        plt.title('Error between visual measurement and model prediction')
        plt.xlabel('update times')
        plt.ylabel('Error')
        plt.pause(0.00000000001)
        plt.gcf()

    def callback(self, data):
        a = data.data
        self.__vis_point = cp.array([a[1:]])

        #print("time = ", self.__time)
        if (a[0] == 1):
            self.padding()
            if ((self.__tmp.shape[1] >= 15) and (self.__tmp.shape[1] <= 90)):
                self.classification()
            if (self.__tmp.shape[1] > 90):
                self.show_spin_direction(self.__max_index)
            self.filled_pred_result()
            
        else:
            if (self.__padding_done == True):
                self.final_padding()
                print("vis shape = ", self.__vis_balls.shape)
                #print(self.__vis_balls)
                print("ped shape = ", self.__predction_balls.shape)
                #print(self.__predction_balls)
                self.calculate_error()
            self.__padding_done = False
            self.__arr_classification = cp.zeros([1,90])

if __name__ == '__main__':
    plt.ion()
    rospy.init_node('classifier_test_60hz')
    #plt.ion()
    print("init node classifier_test_60hz.py")
    Listener()
    rospy.spin()
