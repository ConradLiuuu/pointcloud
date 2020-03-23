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
        self.__pred_top5 = load_model('/home/lab606a/ML/trajectories/fixed/prediction/saved model/prediction_top_speed5')
        self.__pred_top6 = load_model('/home/lab606a/ML/trajectories/fixed/prediction/saved model/prediction_top_speed6')
        self.__pred_left5 = load_model('/home/lab606a/ML/trajectories/fixed/prediction/saved model/prediction_left_speed5')
        self.__pred_left6 = load_model('/home/lab606a/ML/trajectories/fixed/prediction/saved model/prediction_left_speed6')
        self.__pred_right5 = load_model('/home/lab606a/ML/trajectories/fixed/prediction/saved model/prediction_right_speed5')
        self.__pred_right6 = load_model('/home/lab606a/ML/trajectories/fixed/prediction/saved model/prediction_right_speed6')
        self.__pred_back5 = load_model('/home/lab606a/ML/trajectories/fixed/prediction/saved model/prediction_back_speed5')
        self.__pred_back6 = load_model('/home/lab606a/ML/trajectories/fixed/prediction/saved model/prediction_back_speed6')
        self.__n_balls = 11
        self.__tmp = np.zeros([1,self.__n_balls*3])
        self.__tmpp = np.zeros([1,15])
        self.__tmp_classification = np.zeros([1,self.__n_balls*3])
        self.__tmp_prediction = np.zeros([1,5,3])
        self.__padding_done = False
        self.__index = 0
        self.__cnt = 0
        self.__max_index = 0
        self.__classification_done = False
        self.__done = False
        self.__pred = np.zeros([1,5,3])
        self.__sub = rospy.Subscriber("/visual_coordinate", Float32MultiArray, self.callback)
        print("already load model")

    ## print spin direction and speed
    def top5(self):
        print("top spin speed 5")
        with graph.as_default():
            self.__pred = self.__pred_top5.predict(self.__tmp_prediction)
        print(self.__pred)
    def top6(self):
        print("top spin speed 6")
        with graph.as_default():
            self.__pred = self.__pred_top6.predict(self.__tmp_prediction)
        print(self.__pred)
    def left5(self):
        print("left spin speed 5")
        with graph.as_default():
            self.__pred = self.__pred_left5.predict(self.__tmp_prediction)
        print(self.__pred)
    def left6(self):
        print("left spin speed 6")
        with graph.as_default():
            self.__pred = self.__pred_left6.predict(self.__tmp_prediction)
        print(self.__pred)
    def right5(self):
        print("right spin speed 5")
        with graph.as_default():
            self.__pred = self.__pred_right5.predict(self.__tmp_prediction)
        print(self.__pred)
    def right6(self):
        print("right spin speed 6")
        with graph.as_default():
            self.__pred = self.__pred_right6.predict(self.__tmp_prediction)
        print(self.__pred)
    def back5(self):
        print("back spin speed 5")
        with graph.as_default():
            self.__pred = self.__pred_back5.predict(self.__tmp_prediction)
        print(self.__pred)
    def back6(self):
        print("back spin speed 6")
        with graph.as_default():
            self.__pred = self.__pred_back6.predict(self.__tmp_prediction)
        print(self.__pred)

    def show_spin_direction(self, max_index):
        ## make dictionary to replace switch case
        dictionary = {0:self.top5, 1:self.top6, 2:self.left5, 3:self.left6, 4:self.right5, 5:self.right6, 6:self.back5, 7:self.back6}
        funcToCall = dictionary[max_index]
        funcToCall()

    
    def classification(self):
        ## call classifier
        with graph.as_default():
            classes = self.__classifier.predict(self.__tmp_classification)
        ## figure out which direction is
        self.__max_index = np.argmax(classes)
        print("cnt = ", self.__cnt)
        ## show result
        #self.show_spin_direction(self.__max_index)
        #self.__cnt += 1

    def callback(self, data):
        a = data.data
        
        a2 = np.array([a[1:]])
        #print("cnt = ", self.__cnt)
        if (a[0] == 1):
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
            '''
            if ((self.__index+3) < (self.__n_balls*3+1)):
                ## filled array
                self.__tmp[:,self.__index:self.__index+3] = a2
                self.__index += 3
                self.__padding_done = False
                print("aaaaa")
            else:
                self.__padding_done = True
                print("aaaaa123")
            '''
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
                    print("ggggg")
                    ## update 
                    self.__tmp_prediction[0,:4,:] = self.__tmp_prediction[0,1:,:]
                    self.__tmp_prediction[0,4,:] = a2
                    print(self.__tmp_prediction)
                if (self.__done == True):
                    print("ddddd")
                    ## get last five balls by visual measurement
                    self.__tmpp[0,:] = self.__tmp[0,18:]
                    self.__tmpp = self.__tmpp.reshape(1,5,3)
                    self.__tmp_prediction = self.__tmpp
                    self.__done = False
                    print(self.__tmp_prediction)

                self.show_spin_direction(self.__max_index)
                self.__cnt += 1


        else:
            self.__tmp = np.zeros([1,self.__n_balls*3])
            self.__tmpp = np.zeros([1,15])
            self.__index = 0
            self.__classification_done = False
            self.__padding_done = False
            self.__done = False
            #print("ccccc")


        '''
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
