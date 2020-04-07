#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float32MultiArray
import numpy as np
import pandas as pd
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
        self.__pub = rospy.Publisher("/prediction_coordinate", Float32MultiArray, queue_size=1)
        self.__n_balls = 30
        self.__tmp = np.zeros([1,3])
        self.__tmpp = np.zeros([1,15])
        self.__tmp_classification = np.zeros([1,self.__n_balls*3])
        self.__a2 = np.zeros((1,3))
        self.__input_balls = np.zeros((5,3))
        self.__predction_balls = np.zeros((1,5,3))
        self.__tmp_prediction = np.zeros([1,15])
        self.__padding_done = False
        self.__index = 0
        self.__cnt = 5
        self.__ccc = 3
        self.__max_index = 0
        self.__classification_done = False
        self.__done = False
        self.__pred = np.zeros([1,5,3])
        self.__time = 0.016667
        self.__delta_T = 0.016667
        self.__pred_msg = Float32MultiArray()
        self.__rate = rospy.Rate(100)
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
            self.__pred = self.__pred_top5.predict(self.__tmp_prediction.reshape(1,5,3))
        print(self.__pred)
    def top6(self):
        print("top spin speed 6")
        with graph.as_default():
            self.__pred = self.__pred_top6.predict(self.__tmp_prediction.reshape(1,5,3))
        print(self.__pred)
    def left5(self):
        print("left spin speed 5")
        with graph.as_default():
            self.__pred = self.__pred_left5.predict(self.__tmp_prediction.reshape(1,5,3))
        print(self.__pred)
    def left6(self):
        print("left spin speed 6")
        with graph.as_default():
            self.__pred = self.__pred_left6.predict(self.__tmp_prediction.reshape(1,5,3))
        print(self.__pred)
    def right5(self):
        print("right spin speed 5")
        with graph.as_default():
            self.__pred = self.__pred_right5.predict(self.__tmp_prediction.reshape(1,5,3))
        print(self.__pred)
    def right6(self):
        print("right spin speed 6")
        with graph.as_default():
            self.__pred = self.__pred_right6.predict(self.__tmp_prediction.reshape(1,5,3))
        print(self.__pred)
    def back5(self):
        print("back spin speed 5")
        with graph.as_default():
            self.__pred = self.__pred_back5.predict(self.__tmp_prediction.reshape(1,5,3))
        print(self.__pred)
    def back6(self):
        print("back spin speed 6")
        with graph.as_default():
            self.__pred = self.__pred_back6.predict(self.__tmp_prediction.reshape(1,5,3))
        print(self.__pred)

    def show_spin_direction(self, max_index):
        ## make dictionary to replace switch case
        dictionary = {0:self.top5, 1:self.top6, 2:self.left5, 3:self.left6, 4:self.right5, 5:self.right6, 6:self.back5, 7:self.back6}
        funcToCall = dictionary[max_index]
        funcToCall()

    
    def classification(self):
        ## call classifier
        #self.__tmp_classification = self.__tmp_classification.reshape(1,30,3)
        with graph.as_default():
            classes = self.__classifier.predict(self.__tmp_classification.reshape(1,30,3))
        ## figure out which direction is
        self.__max_index = np.argmax(classes)
        print("number of input balls = ", self.__cnt)
        ## show result
        self.show_spin_direction(self.__max_index)
        self.__cnt += 1

    def padding(self):
        # if __tmp is empty, init array
        if (self.__padding_done == False):
            self.__tmp = self.__a2 ## pad first point
            self.__padding_done = True
            #print(self.__tmp.shape)
        else:
        # if __tmp is not empty, then filled array
            self.__tmp = np.append(self.__tmp, self.__a2) ## colect visual measurement points
            #print(self.__tmp.shape)
            if (self.__tmp.shape[0] == 15): ## when colect 5 balls
                self.__tmpp = self.__tmp
                self.__tmp_classification[:,:self.__tmp.shape[0]] = self.__tmp ## asigne to classification input array
                self.__input_balls = self.__tmpp.reshape(1,5,3) ## visual measurement point for calculate error
                self.__tmp_prediction[:,:] = self.__tmp ## for predct next 5 steps
            if (self.__tmp.shape[0] > 15): ## when colect over 5 balls
                self.__tmpp = np.vstack((self.__tmpp,self.__tmp[self.__tmp.shape[0]-15:])) ## append visual measurement point
                self.__input_balls = self.__tmpp.reshape((self.__tmpp.shape[0], 5, 3)) ## visual measurement point for calculate error
                self.__tmp_prediction[:,:] = self.__tmp[self.__tmp.shape[0]-15:] ## rolling visual measurement for predct next 5 steps
                if (self.__tmp.shape[0] <= 90): ## when colect under 31 balls
                    self.__tmp_classification[:,:self.__tmp.shape[0]] = self.__tmp ## still asigne to classification input array
                else:
                    self.__cnt = 5
            #print(self.__tmp_prediction)

    def pub_prediction(self):
        #self.__pred = self.__pred.reshape(15,1)
        self.__pred = self.__pred.astype('float32')
        self.__pred_msg.data = self.__pred.reshape(15,1)
        self.__pub.publish(self.__pred_msg)

    def final_padding(self):
        self.__tmp = self.__tmp.reshape(1, self.__tmp.shape[0])
        for i in range(4):
            #self.__tmp = self.__tmp.reshape(1, self.__tmp.shape[0])
            self.__tmp = sequence.pad_sequences(self.__tmp, maxlen=(self.__tmp.shape[1]+3), padding='post', dtype='float32')
            self.__tmpp = np.vstack((self.__tmpp,self.__tmp[:,self.__tmp.shape[1]-15:]))
            self.__input_balls = self.__tmpp.reshape((self.__tmpp.shape[0], 5, 3))
            self.__tmp_prediction[:,:] = self.__tmp[:,self.__tmp.shape[1]-15:] ## rolling visual measurement for predct next 5 steps
            #self.show_spin_direction(self.__max_index) ## predct next 5 steps
            self.filled_pred_result()
            #print(self.__tmp_prediction)

    def filled_pred_result(self):
        if (self.__tmp.shape[0] == 15):
            self.__predction_balls = self.__pred
        if (self.__tmp.shape[0] > 15):
            self.__predction_balls = np.vstack((self.__predction_balls, self.__pred))

    def calculate_error(self):
        error = self.__input_balls[5:, :, :] - self.__predction_balls[:-1, :, :]
        res = np.zeros((error.shape[0], 1))
        for i in range(error.shape[0]):
            res[i] = np.power(error[i, :, :], 2)
            res[i] = res[i]**(1/2)

    def callback(self, data):
        a = data.data
        self.__a2 = np.array([a[1:]])

        #print("time = ", self.__time)
        if (a[0] == 1):
            self.padding()
            if ((self.__tmp.shape[0] >= 15) and (self.__tmp.shape[0] <= 90)):
                self.classification()
            if (self.__tmp.shape[0] > 90):
                self.show_spin_direction(self.__max_index)
            self.pub_prediction()
            self.filled_pred_result()
            
        else:
            if (self.__padding_done == True):
                #print("sssss")
                self.final_padding()
                self.pub_prediction()
                self.filled_pred_result()
            print("visual measurement points")
            print(self.__input_balls)
            print("prediction points")
            print(self.__predction_balls)
            print("vis shape = ", self.__input_balls.shape)
            print("ped shape = ", self.__predction_balls.shape)
            self.__padding_done = False
            #self.__cnt = 0
            self.__tmp_classification = np.zeros([1,90])
        '''
        if (a[0] == 1):
            print("time = ", self.__time)

            if (self.__padding_done == False):
                if ((self.__index+3) < (self.__n_balls*3+1)):
                    self.__tmp[:,self.__index:self.__index+3] = a2
                    self.__index += 3
                    self.__padding_done = False
                    #print("aaaaa")
                else:
                    self.__padding_done = True
                    #print("aaaaa123")
                    self.__done = True

            if ((self.__padding_done == True) and (self.__classification_done == False)):
                ## do classification
                self.__index = self.__index + 1
                self.__tmp_classification = self.__tmp.reshape(1,self.__n_balls, 3)
                self.__tmp_classification = self.__tmp_classification.astype('float32')
                self.classification()
                self.__classification_done = True
                #print("bbbbb")
            
            if ((self.__classification_done == True)):
                ## prediction
                #print("call prediction")
                if (self.__done == False):
                    #print("ggggg")
                    ## update 
                    self.__tmp_prediction[0,:4,:] = self.__tmp_prediction[0,1:,:]
                    self.__tmp_prediction[0,4,:] = a2
                    print(self.__tmp_prediction)
                if (self.__done == True):
                    #print("ddddd")
                    ## get last five balls by visual measurement
                    self.__tmpp[0,:] = self.__tmp[0,18:]
                    self.__tmpp = self.__tmpp.reshape(1,5,3)
                    self.__tmp_prediction = self.__tmpp
                    self.__done = False
                    print(self.__tmp_prediction)

                self.show_spin_direction(self.__max_index)
                self.__pred = self.__pred.reshape(15,1)
                self.__pred = self.__pred.astype('float32')
                self.__pred_msg.data = self.__pred
                self.__pub.publish(self.__pred_msg)
                self.__rate.sleep()
                self.__pred = self.__pred.reshape(1,5,3)

                #self.__cnt += 1
            self.__time = self.__time + self.__delta_T


        else:
            self.__tmp = np.zeros([1,self.__n_balls*3])
            self.__tmpp = np.zeros([1,15])
            self.__index = 0
            self.__classification_done = False
            self.__padding_done = False
            self.__done = False
            self.__time = 0.016667
            #print("ccccc")
        '''





if __name__ == '__main__':
    rospy.init_node('classifier_test')
    print("init node classifier_test.py")
    Listener()
    rospy.spin()
