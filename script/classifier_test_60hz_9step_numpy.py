#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float32MultiArray
import numpy as np
import pandas as pd
#import cupy as cp
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
        self.__pub_hitting = rospy.Publisher("/hitting", Float32MultiArray, queue_size=1)
        self.__time_step = 9
        self.__tmp = np.zeros([1,3])
        self.__arr_classification = np.zeros([1,90])
        self.__vis_point = np.zeros((1,3))
        self.__vis_balls = np.zeros((self.__time_step,3))
        self.__predction_balls = np.zeros((1,self.__time_step,3))
        self.__arr_prediction = np.zeros([1,self.__time_step*3])
        self.__predction_balls_hat = np.ones((1,self.__time_step,3))
        self.__predction_balls_hat2 = np.ones((1,self.__time_step,3))
        self.__diff = np.zeros([1,3])
        self.__padding_done = False
        self.__cnt = 5
        self.__max_index = 0
        self.__hitting_point = -45
        self.__hitting_timimg = 0
        self.__possible_point = np.zeros((1,3))
        self.__pred = np.zeros([1,self.__time_step,3])
        self.__time = 0.016667
        self.__delta_T = 0.016667
        #self.__pred_msg = Float32MultiArray()
        self.__hitting_msg = Float32MultiArray()
        #self.__rate = rospy.Rate(100)
        self.__classifier = load_model('/home/lab606a/catkin_ws/src/pointcloud/models/60hz_9step_256/classification_30ball_20200404_filled_v2')
        self.__pred_top5 = load_model('/home/lab606a/catkin_ws/src/pointcloud/models/60hz_9step_256/prediction_top5')
        self.__pred_top6 = load_model('/home/lab606a/catkin_ws/src/pointcloud/models/60hz_9step_256/prediction_top6')
        self.__pred_left5 = load_model('/home/lab606a/catkin_ws/src/pointcloud/models/60hz_9step_256/prediction_left5')
        self.__pred_left6 = load_model('/home/lab606a/catkin_ws/src/pointcloud/models/60hz_9step_256/prediction_left6')
        self.__pred_right5 = load_model('/home/lab606a/catkin_ws/src/pointcloud/models/60hz_9step_256/prediction_right5')
        self.__pred_right6 = load_model('/home/lab606a/catkin_ws/src/pointcloud/models/60hz_9step_256/prediction_right6')
        self.__pred_back5 = load_model('/home/lab606a/catkin_ws/src/pointcloud/models/60hz_9step_256/prediction_back5')
        self.__pred_back6 = load_model('/home/lab606a/catkin_ws/src/pointcloud/models/60hz_9step_256/prediction_back6')
        rospy.loginfo("already load model")

    ## print spin direction and speed
    def top5(self):
        print("top spin speed 5")
        with graph.as_default():
            self.__pred = self.__pred_top5.predict(self.__arr_prediction.reshape(1,self.__time_step,3), verbose=1)
            self.__predction_balls_hat = self.__pred_top5.predict(self.__pred, verbose=1)
            self.__predction_balls_hat2 = self.__pred_top5.predict(self.__predction_balls_hat, verbose=1)
        #print(self.__pred)
    def top6(self):
        print("top spin speed 6")
        with graph.as_default():
            self.__pred = self.__pred_top6.predict(self.__arr_prediction.reshape(1,self.__time_step,3), verbose=1)
            self.__predction_balls_hat = self.__pred_top6.predict(self.__pred, verbose=1)
            self.__predction_balls_hat2 = self.__pred_top6.predict(self.__predction_balls_hat, verbose=1)
        #print(self.__pred)
    def left5(self):
        print("left spin speed 5")
        with graph.as_default():
            self.__pred = self.__pred_left5.predict(self.__arr_prediction.reshape(1,self.__time_step,3), verbose=1)
            self.__predction_balls_hat = self.__pred_left5.predict(self.__pred, verbose=1)
            self.__predction_balls_hat2 = self.__pred_left5.predict(self.__predction_balls_hat, verbose=1)
        #print(self.__pred)
    def left6(self):
        print("left spin speed 6")
        with graph.as_default():
            self.__pred = self.__pred_left6.predict(self.__arr_prediction.reshape(1,self.__time_step,3), verbose=1)
            self.__predction_balls_hat = self.__pred_left6.predict(self.__pred, verbose=1)
            self.__predction_balls_hat2 = self.__pred_left6.predict(self.__predction_balls_hat, verbose=1)
        #print(self.__pred)
    def right5(self):
        print("right spin speed 5")
        with graph.as_default():
            self.__pred = self.__pred_right5.predict(self.__arr_prediction.reshape(1,self.__time_step,3), verbose=1)
            self.__predction_balls_hat = self.__pred_right5.predict(self.__pred, verbose=1)
            self.__predction_balls_hat2 = self.__pred_right5.predict(self.__predction_balls_hat, verbose=1)
        #print(self.__pred)
    def right6(self):
        print("right spin speed 6")
        with graph.as_default():
            self.__pred = self.__pred_right6.predict(self.__arr_prediction.reshape(1,self.__time_step,3), verbose=1)
            self.__predction_balls_hat = self.__pred_right6.predict(self.__pred, verbose=1)
            self.__predction_balls_hat2 = self.__pred_right6.predict(self.__predction_balls_hat, verbose=1)
        #print(self.__pred)
    def back5(self):
        print("back spin speed 5")
        with graph.as_default():
            self.__pred = self.__pred_back5.predict(self.__arr_prediction.reshape(1,self.__time_step,3), verbose=1)
            self.__predction_balls_hat = self.__pred_back5.predict(self.__pred, verbose=1)
            self.__predction_balls_hat2 = self.__pred_back5.predict(self.__predction_balls_hat, verbose=1)
        #print(self.__pred)
    def back6(self):
        print("back spin speed 6")
        with graph.as_default():
            self.__pred = self.__pred_back6.predict(self.__arr_prediction.reshape(1,self.__time_step,3), verbose=1)
            self.__predction_balls_hat = self.__pred_back6.predict(self.__pred, verbose=1)
            self.__predction_balls_hat2 = self.__pred_back6.predict(self.__predction_balls_hat, verbose=1)
        #print(self.__pred)

    def show_spin_direction(self, max_index):
        ## make dictionary to replace switch case
        dictionary = {0:self.top5, 1:self.top6, 2:self.left5, 3:self.left6, 4:self.right5, 5:self.right6, 6:self.back5, 7:self.back6}
        funcToCall = dictionary[max_index]
        funcToCall()
        #print(self.__pred)
        #print(self.__predction_balls_hat)
        #print(self.__predction_balls_hat2)
        self.traj_C_possible_point()
        self.traj_B_possible_point()
        self.find_hitting_point_v2()

    def classification(self):
        ## call classifier
        with graph.as_default():
            classes = self.__classifier.predict(self.__arr_classification.reshape(1,30,3), verbose=1)
        ## figure out which direction is
        self.__max_index = np.argmax(classes)
        print("number of input balls = ", self.__cnt)
        ## show result
        if (self.__cnt >= 9):
            self.show_spin_direction(self.__max_index)
        self.__cnt += 1

    def padding(self):
        # if __tmp is empty, init array
        if (self.__padding_done == False):
            self.__tmp = self.__vis_point ## pad first point
            self.__padding_done = True
        else:
        # if __tmp is not empty, then filled array
            self.__tmp = np.hstack((self.__tmp, self.__vis_point))
            if (self.__tmp.shape[1] == 15): ## when colect 5 balls
                self.__arr_classification[:,:self.__tmp.shape[1]] = self.__tmp ## asigne to classification input array
            if (self.__tmp.shape[1] == 27):
                self.__vis_balls = self.__tmp.reshape(1,self.__time_step,3)
                self.__arr_prediction[:,:] = self.__tmp ## for predct next 5 steps
            if (self.__tmp.shape[1] > 27): ## when colect over 5 balls
                self.__vis_balls = np.vstack((self.__vis_balls, self.__tmp[:,self.__tmp.shape[1]-(self.__time_step*3):].reshape(1,self.__time_step,3))) ## visual measurement point for calculate error
                self.__arr_prediction[:,:] = self.__tmp[:,self.__tmp.shape[1]-(self.__time_step*3):] ## rolling visual measurement for predct next 5 steps
                if (self.__tmp.shape[1] <= 90): ## when colect under 31 balls
                    self.__arr_classification[:,:self.__tmp.shape[1]] = self.__tmp ## still asigne to classification input array
                else:
                    self.__cnt = 5

    def pub_prediction(self):
        self.__pred = self.__pred.astype('float32')
        self.__pred_msg.data = self.__pred.reshape((self.__time_step*3),1)
        self.__pub.publish(self.__pred_msg)

    def pub_hit(self):
        self.__possible_point = self.__possible_point.astype('float32')
        self.__hitting_msg.data = self.__possible_point.reshape(4,1)
        self.__pub_hitting.publish(self.__hitting_msg)

    def final_padding(self):
        for i in range((self.__time_step-1)): ## padding zeros 4 times
            print("Time = ", self.__time)
            print("visual measurement = ", self.__vis_point)
            #self.__tmp = cp.asnumpy(self.__tmp)
            self.__tmp = sequence.pad_sequences(self.__tmp, maxlen=(self.__tmp.shape[1]+3), padding='post', dtype='float32')
            #self.__tmp = cp.array(self.__tmp)

            self.__vis_balls = np.vstack((self.__vis_balls, self.__tmp[:,self.__tmp.shape[1]-(self.__time_step*3):].reshape(1,self.__time_step,3)))
            self.__arr_prediction[:,:] = self.__tmp[:,self.__tmp.shape[1]-(self.__time_step*3):] ## rolling visual measurement for predct next 5 steps
            self.show_spin_direction(self.__max_index) ## predct next 5 steps
            #self.pub_prediction() ## publish result to topic
            self.filled_pred_result()
            self.__time += self.__delta_T

    def filled_pred_result(self):
        if (self.__tmp.shape[1] == (self.__time_step*3)):
            self.__predction_balls = self.__pred
        elif (self.__tmp.shape[1] > (self.__time_step*3)):
            self.__predction_balls = np.vstack((self.__predction_balls, self.__pred))

    def find_hitting_point(self):
        #self.__pred = cp.array(self.__pred)
        for i in range((self.__time_step-1)):
            if (self.__pred[:,i+1,1] < self.__hitting_point):
                print("count down ", i+1)
                w1 = (self.__pred[:,i+1,1]-self.__hitting_point)/(self.__pred[:,i+1,1]-self.__pred[:,i,1])
                w2 = 1 - w1
                self.__possible_point = w1*self.__pred[:,i,:] + w2*self.__pred[:,i+1,:]
                #print("hitting point = ", self.__possible_point)
                self.__hitting_timimg = self.__time + (i+1+w1)*self.__delta_T
                #print("hitting timimg = ", self.__hitting_timimg)
                self.__possible_point = np.hstack((self.__hitting_timimg, self.__possible_point.reshape(3,)))
                print("hitting timing and position = ", self.__possible_point)
                #print("\n")
            if ((i == 0) and (self.__pred[:,0,1] < self.__hitting_point)):
                self.__vis_point = self.__vis_point
                print("count down ", i)
                w1 = (self.__hitting_point-self.__vis_point[:,1])/(self.__pred[:,0,1]-self.__vis_point[:,1])
                #print("pred = ", self.__pred[:,0,:])
                self.__possible_point = (1-w1)*self.__vis_point.reshape(1,1,3) + w1*self.__pred[:,0,:]
                #print("hitting point = ", self.__possible_point)
                self.__hitting_timimg = self.__time + w1*self.__delta_T
                #print("hitting timimg = ", self.__hitting_timimg)
                self.__possible_point = np.hstack((self.__hitting_timimg, self.__possible_point.reshape(3,)))
                print("hitting timing and position = ", self.__possible_point)
                #print("\n")

    def find_hitting_point_v2(self):
        self.__pred = self.__pred
        #print("pred = ", self.__pred)
        if (float(np.min(self.__pred)) < -45):
            index = np.argmin(self.__pred)
            row = int((int(index)-1) / 3)
            print("count down ", int(row))
            #print("pred = ", self.__pred[:, row, :])
            if (row == 0):
                self.__vis_point = self.__vis_point
                w1 = (self.__hitting_point-self.__vis_point[:,1])/(self.__pred[:,0,1]-self.__vis_point[:,1])
                self.__possible_point = (1-w1)*self.__vis_point.reshape(1,1,3) + w1*self.__pred[:,0,:]
                self.__hitting_timimg = self.__time + w1*self.__delta_T
                self.__possible_point = np.hstack((self.__hitting_timimg, self.__possible_point.reshape(3,)))
                print("hitting timing and position = ", self.__possible_point)
                #print("\n")
                self.pub_hit()
            else:
                w1 = (self.__pred[:,row,1]-self.__hitting_point)/(self.__pred[:,row,1]-self.__pred[:,row-1,1])
                self.__possible_point = w1*self.__pred[:,row-1,:] + (1-w1)*self.__pred[:,row,:]
                self.__hitting_timimg = self.__time + (row+w1)*self.__delta_T
                self.__possible_point = np.hstack((self.__hitting_timimg, self.__possible_point.reshape(3,)))
                print("hitting timing and position = ", self.__possible_point)
                #print("\n")
                self.pub_hit()
                
        elif (-45 < float(np.min(self.__pred)) < -40):
            index = np.argmin(self.__pred)
            row = int((int(index)-1) / 3)
            print("count down ", int(row))
            #print("pred = ", self.__pred[:, row, :])
            if (row == 0):
                w1 = (self.__hitting_point-self.__pred[:,row,1]) / (self.__diff[:,1])
                self.__possible_point = w1*(self.__pred[:,row,:] + self.__diff) + (1-w1)*self.__pred[:,row,:]
                self.__hitting_timimg = self.__time + (row+1+w1)*self.__delta_T
                self.__possible_point = np.hstack((self.__hitting_timimg, self.__possible_point.reshape(3,)))
                print("hitting timing and position = ", self.__possible_point)
                #print("\n")
                self.pub_hit()
            else:
                self.__diff = self.__pred[:,row,:] - self.__pred[:,row-1,:]
                w1 = (self.__hitting_point-self.__pred[:,row,1]) / (self.__diff[:,1])
                self.__possible_point = w1*(self.__pred[:,row,:] + self.__diff) + (1-w1)*self.__pred[:,row,:]
                self.__hitting_timimg = self.__time + (row+1+w1)*self.__delta_T
                self.__possible_point = np.hstack((self.__hitting_timimg, self.__possible_point.reshape(3,)))
                print("hitting timing and position = ", self.__possible_point)
                #print("\n")
                self.pub_hit()

    def traj_C_possible_point(self):
        self.__predction_balls_hat2 = self.__predction_balls_hat2
        if (float(np.min(self.__predction_balls_hat2)) < -45):
            index = np.argmin(self.__predction_balls_hat2)
            row = int((int(index)-1) / 3)
            print("count down from traj C ", int(row)+18)
            if (row == 0):
                self.__predction_balls_hat = self.__predction_balls_hat
                w1 = (self.__hitting_point-self.__predction_balls_hat[:,self.__time_step-1,1])/(self.__predction_balls_hat2[:,0,1]-self.__predction_balls_hat[:,self.__time_step-1,1])
                self.__possible_point = (1-w1)*self.__predction_balls_hat[:,self.__time_step-1,:].reshape(1,1,3) + w1*self.__predction_balls_hat2[:,0,:]
                self.__hitting_timimg = self.__time + (w1+row+18)*self.__delta_T
                self.__possible_point = np.hstack((self.__hitting_timimg, self.__possible_point.reshape(3,)))
                print("hitting timing and position from traj C = ", self.__possible_point)
                self.pub_hit()
            else:
                w1 = (self.__predction_balls_hat2[:,row,1]-self.__hitting_point)/(self.__predction_balls_hat2[:,row,1]-self.__predction_balls_hat2[:,row-1,1])
                self.__possible_point = w1*self.__predction_balls_hat2[:,row-1,:] + (1-w1)*self.__predction_balls_hat2[:,row,:]
                self.__hitting_timimg = self.__time + (row+w1+18)*self.__delta_T
                self.__possible_point = np.hstack((self.__hitting_timimg, self.__possible_point.reshape(3,)))
                print("hitting timing and position from traj C  = ", self.__possible_point)
                self.pub_hit()

        elif (-45 < float(np.min(self.__predction_balls_hat2)) < -40):
            index = np.argmin(self.__predction_balls_hat2)
            row = int((int(index)-1) / 3)
            print("count down from traj C ", int(row)+18)
            if (row == 0):
                w1 = (self.__hitting_point-self.__predction_balls_hat2[:,row,1]) / (self.__diff[:,1])
                self.__possible_point = w1*(self.__predction_balls_hat2[:,row,:] + self.__diff) + (1-w1)*self.__predction_balls_hat2[:,row,:]
                self.__hitting_timimg = self.__time + (row+1+w1+18)*self.__delta_T
                self.__possible_point = np.hstack((self.__hitting_timimg, self.__possible_point.reshape(3,)))
                print("hitting timing and position from traj C = ", self.__possible_point)
                self.pub_hit()
            else:
                self.__diff = self.__predction_balls_hat2[:,row,:] - self.__predction_balls_hat2[:,row-1,:]
                w1 = (self.__hitting_point-self.__predction_balls_hat2[:,row,1]) / (self.__diff[:,1])
                self.__possible_point = w1*(self.__predction_balls_hat2[:,row,:] + self.__diff) + (1-w1)*self.__predction_balls_hat2[:,row,:]
                self.__hitting_timimg = self.__time + (row+1+w1+18)*self.__delta_T
                self.__possible_point = np.hstack((self.__hitting_timimg, self.__possible_point.reshape(3,)))
                print("hitting timing and position from traj C = ", self.__possible_point)
                self.pub_hit()
    
    def traj_B_possible_point(self):
        self.__predction_balls_hat = self.__predction_balls_hat
        if (float(np.min(self.__predction_balls_hat)) < -45):
            index = np.argmin(self.__predction_balls_hat)
            row = int((int(index)-1) / 3)
            print("count down from traj B ", int(row)+9)
            if (row == 0):
                self.__pred = self.__pred
                w1 = (self.__hitting_point-self.__pred[:,self.__time_step-1,1])/(self.__predction_balls_hat[:,0,1]-self.__pred[:,self.__time_step-1,1])
                self.__possible_point = (1-w1)*self.__pred[:,self.__time_step-1,:].reshape(1,1,3) + w1*self.__predction_balls_hat[:,0,:]
                self.__hitting_timimg = self.__time + (w1+row+9)*self.__delta_T
                self.__possible_point = np.hstack((self.__hitting_timimg, self.__possible_point.reshape(3,)))
                print("hitting timing and position from traj B = ", self.__possible_point)
                self.pub_hit()
            else:
                w1 = (self.__predction_balls_hat[:,row,1]-self.__hitting_point)/(self.__predction_balls_hat[:,row,1]-self.__predction_balls_hat[:,row-1,1])
                self.__possible_point = w1*self.__predction_balls_hat[:,row-1,:] + (1-w1)*self.__predction_balls_hat[:,row,:]
                self.__hitting_timimg = self.__time + (row+w1+9)*self.__delta_T
                self.__possible_point = np.hstack((self.__hitting_timimg, self.__possible_point.reshape(3,)))
                print("hitting timing and position from traj B  = ", self.__possible_point)
                self.pub_hit()

        elif (-45 < float(np.min(self.__predction_balls_hat)) < -40):
            index = np.argmin(self.__predction_balls_hat)
            row = int((int(index)-1) / 3)
            print("count down from traj B ", int(row)+9)
            if (row == 0):
                w1 = (self.__hitting_point-self.__predction_balls_hat[:,row,1]) / (self.__diff[:,1])
                self.__possible_point = w1*(self.__predction_balls_hat[:,row,:] + self.__diff) + (1-w1)*self.__predction_balls_hat[:,row,:]
                self.__hitting_timimg = self.__time + (row+1+w1+9)*self.__delta_T
                self.__possible_point = np.hstack((self.__hitting_timimg, self.__possible_point.reshape(3,)))
                print("hitting timing and position from traj B = ", self.__possible_point)
                self.pub_hit()
            else:
                self.__diff = self.__predction_balls_hat[:,row,:] - self.__predction_balls_hat[:,row-1,:]
                w1 = (self.__hitting_point-self.__predction_balls_hat[:,row,1]) / (self.__diff[:,1])
                self.__possible_point = w1*(self.__predction_balls_hat[:,row,:] + self.__diff) + (1-w1)*self.__predction_balls_hat[:,row,:]
                self.__hitting_timimg = self.__time + (row+1+w1+9)*self.__delta_T
                self.__possible_point = np.hstack((self.__hitting_timimg, self.__possible_point.reshape(3,)))
                print("hitting timing and position from traj B = ", self.__possible_point)
                self.pub_hit()
        

    def calculate_error(self):
        error = self.__vis_balls[self.__time_step:,:,:] - self.__predction_balls[:self.__predction_balls.shape[0]-self.__time_step, :, :]
        error = error[:error.shape[0]-self.__time_step,:,:]
        res = np.zeros((error.shape[0], 1))
        axis = np.linspace(1, error.shape[0], error.shape[0])
        axis = axis.reshape(axis.shape[0], 1)
        #error = cp.abs(error)
        ## .XXXX
        error = np.round_(error, decimals=4)
        ## calculate error by MES
        error = np.power(error, 2)
        #for i in range(error.shape[0]):
            #res[i] = cp.sum(error[i,:,:])
            #print("i = ", i)
            #print("vis = ", self.__vis_balls[i+5,:,:])
            #print("pred = ", self.__predction_balls[i,:,:])
            #print("err = ", error[i,:,:])
        res = np.sum(error, axis=-1)
        res = np.sum(res, axis=-1)
        res = res/(self.__time_step*3)

        ## plot error
        plt.clf()
        plt.plot(axis, res.reshape(res.shape[0],1))
        plt.scatter(axis, res)
        plt.grid(True)
        plt.title('Error between visual measurement and model prediction')
        plt.xlabel('update times')
        plt.ylabel('Error')
        plt.pause(0.00000000001)
        plt.gcf()

    def callback(self, data):
        a = data.data
        self.__vis_point = np.array([a[1:]])

        if (a[0] == 1):
            print("\nTime = ", self.__time) ## show current time
            #rospy.loginfo("\nTime = %f"% self.__time)
            print("visual measurement = ", self.__vis_point) ## show visual measurement coordinate
            self.padding() ## colect 5 balls
            if ((self.__tmp.shape[1] >= 15) and (self.__tmp.shape[1] <= 90)): ## while colect balls over 5 balls and under 30 balls
                self.classification() ## doing classification and prediction
            if (self.__tmp.shape[1] > 90): ## while colect balls over 30 balls, stop doing classification
                self.show_spin_direction(self.__max_index) ## skip classification doing prediction
            self.filled_pred_result() ## colect prediction result for calculate error
            self.__time += self.__delta_T
            
        else:
            if (self.__padding_done == True):
                self.final_padding() ## padding 8 times for predict final predicion coordinate
                #print("vis shape = ", self.__vis_balls.shape)
                #print(self.__vis_balls)
                #print("ped shape = ", self.__predction_balls.shape)
                #print(self.__predction_balls)
                self.calculate_error()
                self.__time = 0.016667 ## reset time
            self.__padding_done = False
            self.__arr_classification = np.zeros([1,90])
            self.__vis_balls = np.zeros((self.__time_step,3))

if __name__ == '__main__':
    plt.ion()
    rospy.init_node('classifier_test_60hz_9step')
    rospy.loginfo("init node classifier_test_60hz_9step.py")
    Listener()
    rospy.spin()
