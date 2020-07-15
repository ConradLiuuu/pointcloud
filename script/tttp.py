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
import sys

graph = tf.get_default_graph()

# set GPU memory
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

np.set_printoptions(suppress=True)

class Listener:
    def __init__(self):
        self.__sub = rospy.Subscriber("/visual_coordinate", Float32MultiArray, self.callback)
        self.__time_step = 9
        self.__tmp = cp.zeros([1,3])
        self.__arr_classification = cp.zeros([1,90])
        self.__vis_point = cp.zeros((1,3))
        self.__vis_balls = cp.zeros((self.__time_step,3))
        self.__vis_balls2 = cp.zeros((1,4))
        self.__predction_balls = cp.zeros((1,self.__time_step,3))
        self.__arr_prediction = cp.zeros([1,self.__time_step*3])
        self.__arr_pred_possible = cp.zeros((1,4))
        self.__diff = cp.zeros([1,3])
        self.__padding_done = False
        self.__cnt = 1
        self.__num = 1
        self.__max_index = 0
        self.__coor = -10*cp.ones(6)
        self.__pred_msg = Float32MultiArray()
        self.__ttbot_msg = Float32MultiArray()
        self.__pub = rospy.Publisher("/prediction_coordinate", Float32MultiArray, queue_size=1)
        self.__pub_ttbot = rospy.Publisher("/TTbot/hitting_point", Float32MultiArray, queue_size=1)
        self.__find_possible_point = False
        self.__cal_possible_point = False
        self.__vis_possible_point = False
        self.__vis_hitting_point = cp.zeros((1,4))
        self.__hitting_point = -45
        self.__hitting_timimg = 0
        self.__possible_point = cp.zeros((1,4))
        self.__pred = cp.zeros([1,self.__time_step,3])
        self.__pred_for_offline = cp.zeros([1,self.__time_step,3])
        self.__time = 0.016667
        self.__delta_T = 0.016667
        self.__anchor = 0
        self.__rowww = 0
        self.__direction = 'top5'
        self.fig, self.ax = plt.subplots(2,2, figsize=(10.24,7.2))
        self.__csv_path = '/home/lab606a/catkin_ws/src/pointcloud/offline/'
        if sys.argv[1] == 'fixed':
            self.__classifier = load_model('/home/lab606a/catkin_ws/src/pointcloud/models/fixed/classification_fixed')
            rospy.loginfo("loaded classification model")
            self.__pred_top5 = load_model('/home/lab606a/catkin_ws/src/pointcloud/models/fixed/prediction_top5')
            self.__pred_top6 = load_model('/home/lab606a/catkin_ws/src/pointcloud/models/fixed/prediction_top6')
            rospy.loginfo("loaded top prediction model")
            self.__pred_left5 = load_model('/home/lab606a/catkin_ws/src/pointcloud/models/fixed/prediction_left5')
            self.__pred_left6 = load_model('/home/lab606a/catkin_ws/src/pointcloud/models/fixed/prediction_left6')
            rospy.loginfo("loaded left prediction model")
            self.__pred_right5 = load_model('/home/lab606a/catkin_ws/src/pointcloud/models/fixed/prediction_right5')
            self.__pred_right6 = load_model('/home/lab606a/catkin_ws/src/pointcloud/models/fixed/prediction_right6')
            rospy.loginfo("loaded right prediction model")
            self.__pred_back5 = load_model('/home/lab606a/catkin_ws/src/pointcloud/models/fixed/prediction_back5')
            self.__pred_back6 = load_model('/home/lab606a/catkin_ws/src/pointcloud/models/fixed/prediction_back6')
            rospy.loginfo("loaded back prediction model")
            rospy.loginfo("Load fixed model")
        else:
            self.__classifier = load_model('/home/lab606a/catkin_ws/src/pointcloud/models/not_fixed/classification_notfixed_20200505_256to8')
            self.__pred_top5 = load_model('/home/lab606a/catkin_ws/src/pointcloud/models/not_fixed/prediction_not_fixed_top5')
            self.__pred_top6 = load_model('/home/lab606a/catkin_ws/src/pointcloud/models/not_fixed/prediction_not_fixed_top6')
            self.__pred_left5 = load_model('/home/lab606a/catkin_ws/src/pointcloud/models/not_fixed/prediction_not_fixed_left5')
            self.__pred_left6 = load_model('/home/lab606a/catkin_ws/src/pointcloud/models/not_fixed/prediction_not_fixed_left6')
            self.__pred_right5 = load_model('/home/lab606a/catkin_ws/src/pointcloud/models/not_fixed/prediction_not_fixed_right5')
            self.__pred_right6 = load_model('/home/lab606a/catkin_ws/src/pointcloud/models/not_fixed/prediction_not_fixed_right6')
            self.__pred_back5 = load_model('/home/lab606a/catkin_ws/src/pointcloud/models/not_fixed/prediction_not_fixed_back5')
            self.__pred_back6 = load_model('/home/lab606a/catkin_ws/src/pointcloud/models/not_fixed/prediction_not_fixed_back6')
            rospy.loginfo("Load not fixed model")
        #rospy.loginfo("already load model")

    ## print spin direction and speed
    def top5(self):
        print("top spin speed 5")
        self.__model = self.__pred_top5
        self.__direction = '_top5'

    def top6(self):
        print("top spin speed 6")
        self.__model = self.__pred_top6
        self.__direction = '_top6'

    def left5(self):
        print("left spin speed 5")
        self.__model = self.__pred_left5
        self.__direction = '_left5'

    def left6(self):
        print("left spin speed 6")
        self.__model = self.__pred_left6
        self.__direction = '_left6'

    def right5(self):
        print("right spin speed 5")
        self.__model = self.__pred_right5
        self.__direction = '_right5'

    def right6(self):
        print("right spin speed 6")
        self.__model = self.__pred_right6
        self.__direction = '_right6'

    def back5(self):
        print("back spin speed 5")
        self.__model = self.__pred_back5
        self.__direction = '_back5'

    def back6(self):
        print("back spin speed 6")
        self.__model = self.__pred_back6
        self.__direction = '_back6'

    def append_pred(self): ## Exhaustive
        with graph.as_default():
            self.__pred = cp.array(self.__model.predict(cp.asnumpy(self.__arr_prediction.reshape(1,self.__time_step,3)), verbose=1))
        while not self.__find_possible_point == True:
            self.__pred = cp.array(self.__pred)
            if ( -50 < float(cp.min(self.__pred[self.__pred.shape[0]-1,:,:])) < -40): ## find possible hitting point
                index = cp.argmin(self.__pred[self.__pred.shape[0]-1,:,:])
                row = int((index-1)/3)
                count_down = row + (self.__pred.shape[0]-1)*9
                self.__find_possible_point = True
                self.__anchor = self.__pred.shape[0]-1
                print("count down = ", int(count_down))
                #index = cp.argmin(self.__anchor[self.__anchor,:,:])
                #row = int((index-1)/3)
                self.__pred[self.__anchor, row+1:, :] = 0
                self.__pred_for_offline = self.__pred

            else: ## predict next time step
                self.__pred = cp.vstack((self.__pred, cp.zeros([1,self.__time_step,3])))
                with graph.as_default():
                    self.__pred[self.__pred.shape[0]-1,:,:] = cp.array(self.__model.predict(cp.asnumpy(self.__pred[self.__pred.shape[0]-2,:,:].reshape(1,self.__time_step,3)), verbose=1)).reshape(self.__time_step,3)
                if (self.__pred.shape[0] >= 8):
                    self.__pred = cp.zeros([1,self.__time_step,3])
                    self.__find_possible_point = True
                
    def update_prediction(self):
        with graph.as_default():
            self.__pred[0,:,:] = cp.array(self.__model.predict(cp.asnumpy(self.__arr_prediction.reshape(1,self.__time_step,3)), verbose=1)).reshape(self.__time_step,3)
            for i in range(1, self.__pred.shape[0]):
                self.__pred[i,:,:] = cp.array(self.__model.predict(cp.asnumpy(self.__pred[i-1,:,:].reshape(1,self.__time_step,3)), verbose=1)).reshape(self.__time_step,3)
            '''
            index = cp.argmin(self.__pred[self.__anchor,:,:])
            row = int((index-1)/3)
            if (row < 8):
                self.__pred[self.__anchor, row+1:, :] = 0
            if (self.__anchor != self.__pred.shape[0]-1):
                self.__pred[self.__anchor+1:, :, :] = 0
            if (self.__anchor == 0) and ( -10 < float(cp.min(self.__pred[self.__anchor,0,:])) < 10):
                self.__pred[self.__anchor,0,:] = 0
            '''
            #self.__pred_for_offline = cp.vstack((self.__pred_for_offline, self.__pred))
            

    def modify(self):
        #index = cp.argmin(self.__pred[self.__anchor,:,:])
        #row = int((index-1)/3)
        if (self.__rowww < 8):
            self.__pred[self.__anchor, self.__rowww+1:, :] = 0
        if (self.__anchor != self.__pred.shape[0]-1):
            self.__pred[self.__anchor+1:, :, :] = 0
        if (self.__anchor == 0) and (self.__rowww == 0) and ( -10 < float(cp.min(self.__pred[self.__anchor,0,:])) < 10):
            self.__pred[self.__anchor,0,:] = 0
        self.__pred_for_offline = cp.vstack((self.__pred_for_offline, self.__pred))
        self.pub_prediction()

    def for_ttbot(self):
        self.__coor = self.__coor.astype('float32')
        self.__coor[0:3] = self.__vis_point[0,:]
        if self.__tmp.shape[1] > 27:
            self.__coor[3:6] = self.__possible_point[1:4]
        self.__ttbot_msg.data = self.__coor.reshape(6,1)
        self.__pub_ttbot.publish(self.__ttbot_msg)

    def calculate_hitting_point(self, arr):
        #print("cnt = ", self.__cnt)
        arr = cp.array(arr)
        self.__cal_possible_point = False
        while not self.__cal_possible_point == True:
            print(arr[self.__anchor,:,:])
            if (-55 <= float(cp.min(arr[self.__anchor,:,:])) <= -35) and (self.__anchor >= 0):
                #print("anchor = ", self.__anchor)
                index = cp.argmin(arr[self.__anchor,:,:])
                row = int((index-1)/3)
                self.__rowww = row
                if (-55 <= float(cp.min(arr[self.__anchor,:,:])) <= -45): ## interpolation
                    index = cp.argmin(arr[self.__anchor,:,:])
                    row = int((index-1)/3)
                    self.__rowww = row
                    count_down = row + self.__anchor*9
                    print("count down = ", count_down)
                    if (row == 0):
                        print("case a1")
                        self.__vis_point = cp.array(self.__vis_point)
                        w1 = (self.__hitting_point-self.__vis_point[:,1]) / (arr[self.__anchor,row,1]-self.__vis_point[:,1])
                        self.__possible_point = w1*arr[self.__anchor,row,:] + (1-w1)*self.__vis_point.reshape(1,1,3)
                        self.__hitting_timimg = self.__time + (w1+count_down)*self.__delta_T
                        self.__possible_point = cp.hstack((self.__hitting_timimg, self.__possible_point.reshape(3,)))

                        #w1 = (arr[self.__anchor,row,1]-self.__hitting_point) / (arr[self.__anchor,row,1]-arr[self.__anchor-1,self.__time_step-1,1])
                        #self.__possible_point = w1*arr[self.__anchor-1,self.__time_step-1,:] + (1-w1)*arr[self.__anchor,row,:]
                        #self.__hitting_timimg = self.__time + (w1+count_down)*self.__delta_T
                        #self.__possible_point = cp.hstack((self.__hitting_timimg, self.__possible_point.reshape(3,)))
                        print("hitting timing and position = ", self.__possible_point)
                    else:
                        print("case a2")
                        w1 = (arr[self.__anchor,row,1]-self.__hitting_point) / (arr[self.__anchor,row,1]-arr[self.__anchor,row-1,1])
                        self.__possible_point = w1*arr[self.__anchor,row-1,:] + (1-w1)*arr[self.__anchor,row,:]
                        self.__hitting_timimg = self.__time + (w1+count_down)*self.__delta_T
                        self.__possible_point = cp.hstack((self.__hitting_timimg, self.__possible_point.reshape(3,)))
                        print("hitting timing and position = ", self.__possible_point)
                elif (-45 < float(cp.min(arr[self.__anchor,:,:])) <= -40):
                    index = cp.argmin(arr[self.__anchor,:,:])
                    row = int((index-1)/3)
                    self.__rowww = row
                    count_down = row + self.__anchor*9
                    print("count down = ", count_down)
                    if (row == 0):
                        print("case b1")
                        if (self.__diff[:,1] > 0):
                            w1 = (arr[self.__anchor,row,1]-self.__hitting_point) / self.__diff[:,1]
                            if w1 < 1:
                                self.__possible_point = arr[self.__anchor,row,:] - w1*self.__diff
                                self.__hitting_timimg = self.__time + (count_down+1+w1)*self.__delta_T
                                self.__possible_point = cp.hstack((self.__hitting_timimg, self.__possible_point.reshape(3,)))
                                print("hitting timing and position = ", self.__possible_point)
                    else:
                        print("case b2")
                        self.__diff = arr[self.__anchor,row-1,:]-arr[self.__anchor,row,:]
                        self.__diff = self.__diff.reshape(1,3)
                        w1 = (arr[self.__anchor,row,1]-self.__hitting_point) / self.__diff[:,1]
                        if w1 < 1:
                            self.__possible_point = arr[self.__anchor,row,:] - w1*self.__diff
                            self.__hitting_timimg = self.__time + (count_down+1+w1)*self.__delta_T
                            self.__possible_point = cp.hstack((self.__hitting_timimg, self.__possible_point.reshape(3,)))
                            print("hitting timing and position = ", self.__possible_point)
                else:
                    self.__possible_point = self.__possible_point

                self.__cal_possible_point = True
                #self.pub_prediction()
                '''
                if (row < 8):
                    arr[self.__anchor, row+1:, :] = 0
                if (self.__anchor != self.__pred.shape[0]-1):
                    arr[self.__anchor+1:, :, :] = 0
                if (self.__anchor == 0) and ( -10 < float(cp.min(arr[self.__anchor,0,:])) < 10):
                    arr[self.__anchor,0,:] = 0
                '''
                #self.__pred_for_offline = cp.vstack((self.__pred_for_offline, arr))
                #self.__cnt += 1
                if (self.__arr_pred_possible.shape[0] == 1) and (int(self.__arr_pred_possible[0,2]) == 0):
                    self.__arr_pred_possible = self.__possible_point.reshape(1,4)
                else:
                    self.__arr_pred_possible = cp.vstack((self.__arr_pred_possible, self.__possible_point.reshape(1,4)))
            else:
                if ((self.__anchor-1) >= 0) and (abs(float(cp.min(arr[self.__anchor,:,:]))-self.__hitting_point) >= 22.5):
                    self.__anchor = self.__anchor -1
                    #print("case c1")
                else:
                    self.__cal_possible_point = True
    
    def pub_prediction(self):
        '''
        msg = self.__possible_point.astype('float32')
        self.__pred_msg.data = msg.reshape(4,1)
        self.__pub.publish(self.__pred_msg)
        '''
        msg = self.__pred.astype('float32')
        self.__pred_msg.data = msg.reshape(self.__pred.shape[0]*self.__pred.shape[1]*self.__pred.shape[2],1)
        self.__pub.publish(self.__pred_msg)

    def calculate_vis_hitting_point(self):
        if (-50 <= float(cp.min(self.__vis_balls2)) <= -45):
            index = cp.argmin(self.__vis_balls2)
            row = int((index-1)/4)
            w1 = (self.__vis_balls2[row,2]-self.__hitting_point) / (self.__vis_balls2[row,2]-self.__vis_balls2[row-1,2])
            self.__vis_hitting_point = (1-w1)*self.__vis_balls2[row,:] + w1*self.__vis_balls2[row-1,:]
        elif (-45 < float(cp.min(self.__vis_balls2)) < -40):
            index = cp.argmin(self.__vis_balls2)
            row = int((index-1)/4)
            w1 = (self.__vis_balls2[row,2]-self.__hitting_point) / (self.__vis_balls2[row-1,2]-self.__vis_balls2[row,2])
            self.__vis_hitting_point = self.__vis_balls2[row,:] - w1*(self.__vis_balls2[row-1,:]-self.__vis_balls2[row,:])
        print("vis hitting point = ", self.__vis_hitting_point)


    def show_spin_direction(self, max_index):
        ## make dictionary to replace switch case
        dictionary = {0:self.top5, 1:self.top6, 2:self.left5, 3:self.left6, 4:self.right5, 5:self.right6, 6:self.back5, 7:self.back6}
        funcToCall = dictionary[max_index]
        funcToCall()

        if (self.__pred.shape[0] == 1):
            self.append_pred() ## Exhaustive
        elif (self.__pred.shape[0] > 1):
            self.update_prediction() ## update prediction result
            self.calculate_hitting_point(self.__pred) ## calculate hitting timimg and hitting point
            self.modify()

    def classification(self):
        ## call classifier
        with graph.as_default():
            classes = self.__classifier.predict(cp.asnumpy(self.__arr_classification.reshape(1,30,3)), verbose=1)
        ## figure out which direction is
        self.__max_index = np.argmax(classes)
        #print("number of input balls = ", self.__cnt)
        ## show result
        if (self.__cnt >= 9):
            self.show_spin_direction(self.__max_index)
        #self.__cnt += 1

    def padding(self):
        # if __tmp is empty, init array
        if (self.__padding_done == False):
            self.__tmp = self.__vis_point ## pad first point
            self.__padding_done = True
            self.__vis_balls = self.__vis_point.reshape(1,3)
        else:
        # if __tmp is not empty, then filled array
            self.__tmp = cp.hstack((self.__tmp, self.__vis_point))
            self.__vis_balls = cp.vstack((self.__vis_balls, self.__vis_point))
            if (self.__tmp.shape[1] == 15): ## when colect 5 balls
                self.__arr_classification[:,:self.__tmp.shape[1]] = self.__tmp ## asigne to classification input array
            if (self.__tmp.shape[1] == 27):
                #self.__vis_balls = self.__tmp.reshape(1,self.__time_step,3)
                self.__arr_prediction[:,:] = self.__tmp ## for predct next 5 steps
            if (self.__tmp.shape[1] > 27): ## when colect over 5 balls
                #self.__vis_balls = cp.vstack((self.__vis_balls, self.__tmp[:,self.__tmp.shape[1]-(self.__time_step*3):].reshape(1,self.__time_step,3))) ## visual measurement point for calculate error
                self.__arr_prediction[:,:] = self.__tmp[:,self.__tmp.shape[1]-(self.__time_step*3):] ## rolling visual measurement for predct next 5 steps
                if (self.__tmp.shape[1] <= 90): ## when colect under 31 balls
                    self.__arr_classification[:,:self.__tmp.shape[1]] = self.__tmp ## still asigne to classification input array

    def save_data(self):
        visurement_balls = cp.asnumpy(self.__vis_balls)
        pred_trajs = cp.asnumpy(self.__pred_for_offline.reshape(self.__pred_for_offline.shape[0]*self.__pred_for_offline.shape[1], self.__pred_for_offline.shape[2]))

        df_vis = pd.DataFrame(data=visurement_balls)
        df_pred = pd.DataFrame(data=pred_trajs)

        vis_name = self.__csv_path + 'visurement' + str(self.__num) + '.csv'
        pred_name = self.__csv_path + 'prediction' + str(self.__num) + '.csv'

        df_vis.to_csv(vis_name, header=0, index=0)
        df_pred.to_csv(pred_name, header=0, index=0)

    def final_padding(self):
        for i in range((self.__time_step-1)): ## padding zeros 4 times
            print("Time = ", self.__time)
            print("visual measurement = ", self.__vis_point)
            self.__tmp = cp.asnumpy(self.__tmp)
            self.__tmp = sequence.pad_sequences(self.__tmp, maxlen=(self.__tmp.shape[1]+3), padding='post', dtype='float32')
            self.__tmp = cp.array(self.__tmp)

            self.__vis_balls = cp.vstack((self.__vis_balls, self.__tmp[:,self.__tmp.shape[1]-(self.__time_step*3):].reshape(1,self.__time_step,3)))
            self.__arr_prediction[:,:] = self.__tmp[:,self.__tmp.shape[1]-(self.__time_step*3):] ## rolling visual measurement for predct next 5 steps
            self.show_spin_direction(self.__max_index) ## predct next 5 steps
            self.__time += self.__delta_T

    def plot_hp_plane(self):
        pred_hp = self.__arr_pred_possible[-1:,1:]
        vis_hp = self.__vis_hitting_point[1:]
        pred_hp = pred_hp.reshape(1,3)
        vis_hp = vis_hp.reshape(1,3)

        error = vis_hp - pred_hp
        #error = error.reshape(1,3)

        plt.figure(figsize=(8,8))
        circle = plt.Circle((0, 0), 7.5, color='r', fill=False)
        ax = plt.gca()
        ax.add_artist(circle)
        ax.set_xlim((-15, 15))
        ax.set_ylim((-15, 15))

        plt.scatter(0,0,marker='o', color='r')
        plt.scatter(cp.asnumpy(error[0,0]), cp.asnumpy(error[0,2]), marker='x', color='b')

        plt.xlabel('X coordinate error (cm)', fontsize=20, fontname='FreeSerif')
        plt.ylabel('Z coordinate error (cm)', fontsize=20, fontname='FreeSerif')
        plt.xticks(fontsize=14, fontname='FreeSerif')
        plt.yticks(fontsize=14, fontname='FreeSerif')

        plt.title('X Z hitting plane', fontsize=20, fontname='FreeSerif')
        plt.tight_layout(pad=1.5)
        name = '/home/lab606a/catkin_ws/src/pointcloud/fig/' + str(self.__num) + '_XZ_plane' + '.png'
        plt.savefig(name)

    def plot_res(self):
        self.__arr_pred_possible = cp.round_(self.__arr_pred_possible, 4)
        self.__vis_hitting_point = cp.round_(self.__vis_hitting_point, 4)

        update_times = cp.linspace(1, self.__arr_pred_possible.shape[0], self.__arr_pred_possible.shape[0])
        t = self.__arr_pred_possible[:,0]
        x = self.__arr_pred_possible[:,1]
        z = self.__arr_pred_possible[:,3]
        Euclidean_vis = cp.sqrt(cp.sum(cp.power(self.__vis_hitting_point[1:],2)))
        Euclidean_pred = cp.sqrt(cp.sum(cp.power(self.__arr_pred_possible[:,1:],2), axis=1))

        plt.figure(figsize=(8,7))

        ## plot hitting timing
        plt.clf()
        plt.plot(cp.asnumpy(update_times), cp.asnumpy(cp.ones((self.__arr_pred_possible.shape[0],))*self.__vis_hitting_point[0]), color='green') #vis
        plt.plot(cp.asnumpy(update_times), cp.asnumpy(t), color='blue') #pred
        plt.scatter(cp.asnumpy(update_times), cp.asnumpy(t), color='blue')
        plt.grid(True)
        plt.xticks(fontsize=14, fontname='FreeSerif')
        plt.yticks(fontsize=14, fontname='FreeSerif')
        plt.xlabel('Update times', fontsize=24, fontname='FreeSerif')
        plt.ylabel('Hitting timimg (sec)', fontsize=24, fontname='FreeSerif')
        plt.title('Relationship between \n update times and hitting timing', fontsize=24, fontweight='bold', fontname='FreeSerif')
        name = '/home/lab606a/catkin_ws/src/pointcloud/fig/' + str(self.__num) + '_timing' + '.png'
        plt.savefig(name)

        plt.clf()
        err = cp.abs((t-self.__vis_hitting_point[0])*1000)
        plt.bar(cp.asnumpy(update_times), cp.asnumpy(err), color='blue', edgecolor='black', width=0.5)
        plt.xticks(fontsize=18, fontname='FreeSerif')
        plt.yticks(fontsize=18, fontname='FreeSerif')
        plt.xlabel('Update times', fontsize=24, fontname='FreeSerif')
        plt.ylabel('Hitting timimg error (ms)', fontsize=24, fontname='FreeSerif')
        plt.title('Relationship between \n update times and hitting timing error', fontsize=24, fontweight='bold', fontname='FreeSerif')
        name = '/home/lab606a/catkin_ws/src/pointcloud/fig/' + str(self.__num) + '_timing_error' + '.png'
        plt.savefig(name)

        ## plot Euclidean distance
        plt.clf()
        plt.plot(cp.asnumpy(update_times), cp.asnumpy(cp.ones((self.__arr_pred_possible.shape[0],))*Euclidean_vis), color='green') #vis
        plt.plot(cp.asnumpy(update_times), cp.asnumpy(Euclidean_pred), color='blue') #pred
        plt.scatter(cp.asnumpy(update_times), cp.asnumpy(Euclidean_pred), color='blue')
        plt.grid(True)
        plt.xticks(fontsize=14, fontname='FreeSerif')
        plt.yticks(fontsize=14, fontname='FreeSerif')
        plt.xlabel('Update times', fontsize=24, fontname='FreeSerif')
        plt.ylabel('Euclidean distance (cm)', fontsize=24, fontname='FreeSerif')
        plt.title('Relationship between \n update times and Euclidean distance', fontsize=24, fontweight='bold', fontname='FreeSerif')
        name = '/home/lab606a/catkin_ws/src/pointcloud/fig/' + str(self.__num) + '_distance' + '.png'
        plt.savefig(name)

        plt.clf()
        err = cp.abs((Euclidean_pred-Euclidean_vis))
        plt.bar(cp.asnumpy(update_times), cp.asnumpy(err), color='blue', edgecolor='black', width=0.5)
        plt.xticks(fontsize=18, fontname='FreeSerif')
        plt.yticks(fontsize=18, fontname='FreeSerif')
        plt.xlabel('Update times', fontsize=24, fontname='FreeSerif')
        plt.ylabel('Euclidean distance error (cm)', fontsize=24, fontname='FreeSerif')
        plt.title('Relationship between \n update times and Euclidean distance error', fontsize=24, fontweight='bold', fontname='FreeSerif')
        name = '/home/lab606a/catkin_ws/src/pointcloud/fig/' + str(self.__num) + '_distance_error' + '.png'
        plt.savefig(name)

        ## plot x coordinate
        plt.clf()
        plt.plot(cp.asnumpy(update_times), cp.asnumpy(cp.ones((self.__arr_pred_possible.shape[0],))*self.__vis_hitting_point[1]), color='green') #vis
        plt.plot(cp.asnumpy(update_times), cp.asnumpy(x), color='blue') #pred
        plt.scatter(cp.asnumpy(update_times), cp.asnumpy(x), color='blue')
        plt.grid(True)
        plt.xticks(fontsize=14, fontname='FreeSerif')
        plt.yticks(fontsize=14, fontname='FreeSerif')
        plt.xlabel('Update times', fontsize=24, fontname='FreeSerif')
        plt.ylabel('X-coordinate (cm)', fontsize=24, fontname='FreeSerif')
        plt.title('Relationship between \n update times and X-coordinate', fontsize=24, fontweight='bold', fontname='FreeSerif')
        name = '/home/lab606a/catkin_ws/src/pointcloud/fig/' + str(self.__num) + '_X-coordinate' + '.png'
        plt.savefig(name)

        plt.clf()
        err = cp.abs((x-self.__vis_hitting_point[1]))
        plt.bar(cp.asnumpy(update_times), cp.asnumpy(err), color='blue', edgecolor='black', width=0.5)
        plt.xticks(fontsize=18, fontname='FreeSerif')
        plt.yticks(fontsize=18, fontname='FreeSerif')
        plt.xlabel('Update times', fontsize=24, fontname='FreeSerif')
        plt.ylabel('X-coordinate error (cm)', fontsize=24, fontname='FreeSerif')
        plt.title('Relationship between \n update times and X-coordinate error', fontsize=24, fontweight='bold', fontname='FreeSerif')
        name = '/home/lab606a/catkin_ws/src/pointcloud/fig/' + str(self.__num) + '_X-coordinate_error' + '.png'
        plt.savefig(name)

        ## plot z coordinate
        plt.clf()
        plt.plot(cp.asnumpy(update_times), cp.asnumpy(cp.ones((self.__arr_pred_possible.shape[0],))*self.__vis_hitting_point[3]), color='green') #vis
        plt.plot(cp.asnumpy(update_times), cp.asnumpy(z), color='blue') #pred
        plt.scatter(cp.asnumpy(update_times), cp.asnumpy(z), color='blue')
        plt.grid(True)
        plt.xticks(fontsize=14, fontname='FreeSerif')
        plt.yticks(fontsize=14, fontname='FreeSerif')
        plt.xlabel('Update times', fontsize=24, fontname='FreeSerif')
        plt.ylabel('Z-coordinate (cm)', fontsize=24, fontname='FreeSerif')
        plt.title('Relationship between \n update times and Z-coordinate', fontsize=24, fontweight='bold', fontname='FreeSerif')
        name = '/home/lab606a/catkin_ws/src/pointcloud/fig/' + str(self.__num) + '_Z-coordinate' + '.png'
        plt.savefig(name)

        plt.clf()
        err = cp.abs((z-self.__vis_hitting_point[3]))
        plt.bar(cp.asnumpy(update_times), cp.asnumpy(err), color='blue', edgecolor='black', width=0.5)
        plt.xticks(fontsize=18, fontname='FreeSerif')
        plt.yticks(fontsize=18, fontname='FreeSerif')
        plt.xlabel('Update times', fontsize=24, fontname='FreeSerif')
        plt.ylabel('Z-coordinate error (cm)', fontsize=24, fontname='FreeSerif')
        plt.title('Relationship between \n update times and Z-coordinate error', fontsize=24, fontweight='bold', fontname='FreeSerif')
        name = '/home/lab606a/catkin_ws/src/pointcloud/fig/' + str(self.__num) + '_Z-coordinate_error' + '.png'
        plt.savefig(name)
    
    def plot_error(self):
        self.ax[0,0].remove()
        self.ax[0,1].remove()
        self.ax[1,0].remove()
        self.ax[1,1].remove()
        #self.fig.clf()
        self.__arr_pred_possible = cp.round_(self.__arr_pred_possible, 4)
        self.__vis_hitting_point = cp.round_(self.__vis_hitting_point, 4)
        #self.__vis_hitting_point = self.__vis_hitting_point.reshape(1,self.__vis_hitting_point.shape[0])

        update_times = cp.linspace(1, self.__arr_pred_possible.shape[0], self.__arr_pred_possible.shape[0])
        t = self.__arr_pred_possible[:,0]
        x = self.__arr_pred_possible[:,1]
        z = self.__arr_pred_possible[:,3]
        Euclidean_vis = cp.sqrt(cp.sum(cp.power(self.__vis_hitting_point[1:],2)))
        Euclidean_pred = cp.sqrt(cp.sum(cp.power(self.__arr_pred_possible[:,1:],2), axis=1))
        
        #fig, ax = plt.subplots(2,2, figsize=(10.24,7.2))
        
        self.fig, self.ax = plt.subplots(2,2, figsize=(12.0,10.0)) ##(10.24,7.2)

        self.ax[0,0].plot(cp.asnumpy(update_times), cp.asnumpy(cp.ones((self.__arr_pred_possible.shape[0],))*self.__vis_hitting_point[0]), color='green')
        self.ax[0,0].plot(cp.asnumpy(update_times), cp.asnumpy(t), color='blue')
        self.ax[0,0].scatter(cp.asnumpy(update_times), cp.asnumpy(t), color='blue')
        self.ax[0,0].grid(True)
        self.ax[0,0].set_xlabel('update times')
        self.ax[0,0].set_ylabel('hitting timimg (sec)')
        self.ax[0,0].set_title('Hitting timimg')

        self.ax[0,1].plot(cp.asnumpy(update_times), cp.asnumpy(cp.ones((self.__arr_pred_possible.shape[0],))*Euclidean_vis), color='green')
        self.ax[0,1].plot(cp.asnumpy(update_times), cp.asnumpy(Euclidean_pred), color='blue')
        self.ax[0,1].scatter(cp.asnumpy(update_times), cp.asnumpy(Euclidean_pred), color='blue')
        self.ax[0,1].grid(True)
        self.ax[0,1].set_xlabel('update times')
        self.ax[0,1].set_ylabel('Euclidean distance (cm)')
        self.ax[0,1].set_title('Euclidean distance')

        self.ax[1,0].plot(cp.asnumpy(update_times), cp.asnumpy(cp.ones((self.__arr_pred_possible.shape[0],))*self.__vis_hitting_point[1]), color='green')
        self.ax[1,0].plot(cp.asnumpy(update_times), cp.asnumpy(x), color='blue')
        self.ax[1,0].scatter(cp.asnumpy(update_times), cp.asnumpy(x), color='blue')
        self.ax[1,0].grid(True)
        self.ax[1,0].set_xlabel('update times')
        self.ax[1,0].set_ylabel('X-coordinate (cm)')
        self.ax[1,0].set_title('X-coordinate')

        self.ax[1,1].plot(cp.asnumpy(update_times), cp.asnumpy(cp.ones((self.__arr_pred_possible.shape[0],))*self.__vis_hitting_point[3]), color='green')
        self.ax[1,1].plot(cp.asnumpy(update_times), cp.asnumpy(z), color='blue')
        self.ax[1,1].scatter(cp.asnumpy(update_times), cp.asnumpy(z), color='blue')
        self.ax[1,1].grid(True)
        self.ax[1,1].set_xlabel('update times')
        self.ax[1,1].set_ylabel('Z-Coordinate (cm)')
        self.ax[1,1].set_title('Z-Coordinate')

        '''
        #plt.clf()
        plt.figure(figsize=(10.24,7.2))
        plt.subplot(2,2,1)
        plt.plot(cp.asnumpy(update_times), cp.asnumpy(cp.ones((self.__arr_pred_possible.shape[0],))*self.__vis_hitting_point[0]), color='green')
        plt.plot(cp.asnumpy(update_times), cp.asnumpy(t), color='blue')
        plt.scatter(cp.asnumpy(update_times), cp.asnumpy(t), color='blue')
        plt.grid(True)
        plt.xlabel('update times')
        plt.ylabel('hitting timimg')
        
        plt.subplot(2,2,2)
        plt.plot(cp.asnumpy(update_times), cp.asnumpy(cp.ones((self.__arr_pred_possible.shape[0],))*Euclidean_vis), color='green')
        plt.plot(cp.asnumpy(update_times), cp.asnumpy(Euclidean_pred), color='blue')
        plt.scatter(cp.asnumpy(update_times), cp.asnumpy(Euclidean_pred), color='blue')
        plt.grid(True)
        plt.xlabel('update times')
        plt.ylabel('Euclidean distance')
        
        plt.subplot(2,2,3)
        plt.plot(cp.asnumpy(update_times), cp.asnumpy(cp.ones((self.__arr_pred_possible.shape[0],))*self.__vis_hitting_point[1]), color='green')
        plt.plot(cp.asnumpy(update_times), cp.asnumpy(x), color='blue')
        plt.scatter(cp.asnumpy(update_times), cp.asnumpy(x), color='blue')
        plt.grid(True)
        plt.xlabel('update times')
        plt.ylabel('Coordinate X')

        plt.subplot(2,2,4)
        plt.plot(cp.asnumpy(update_times), cp.asnumpy(cp.ones((self.__arr_pred_possible.shape[0],))*self.__vis_hitting_point[3]), color='green')
        plt.plot(cp.asnumpy(update_times), cp.asnumpy(z), color='blue')
        plt.scatter(cp.asnumpy(update_times), cp.asnumpy(z), color='blue')
        plt.grid(True)
        plt.xlabel('update times')
        plt.ylabel('Coordinate Z')

        #plt.pause(0.00000000001)
        plt.show()
        '''
        #name = '/home/lab606a/catkin_ws/src/pointcloud/fig/' + self.__direction + str(rospy.get_time()) + '.png'
        name = '/home/lab606a/catkin_ws/src/pointcloud/fig/' + str(self.__num) + self.__direction + '.png'
        
        self.fig.savefig(name)

        #self.fig.pause(0.001)
        #plt.savefig(name)
        #plt.clf()
        
        #self.fig.clf()
        #self.ax[0,0].remove()
        #self.ax[0,1].remove()
        #self.ax[1,0].remove()
        #self.ax[1,1].remove()


        #self.fig.clf()
        #self.fig.show()

    def callback(self, data):
        a = data.data
        self.__vis_point = cp.array([a[1:]])

        if (-50 < self.__vis_point[:,1] < -30):
            tmp = cp.hstack((self.__time, self.__vis_point.reshape(3,)))
            if self.__vis_possible_point == False:
                self.__vis_balls2 = tmp.reshape(1,4)
                self.__vis_possible_point = True
            else:
                self.__vis_balls2 = cp.vstack((self.__vis_balls2, tmp.reshape(1,4)))


        if (a[0] == 1):
            print("\nTime = ", self.__time) ## show current time
            print("visual measurement = ", self.__vis_point) ## show visual measurement coordinate
            self.padding() ## colect 5 balls
            if ((self.__tmp.shape[1] >= 15) and (self.__tmp.shape[1] <= 90)): ## while colect balls over 5 balls and under 30 balls
                self.classification() ## doing classification and prediction
            if (self.__tmp.shape[1] > 90): ## while colect balls over 30 balls, stop doing classification
                self.show_spin_direction(self.__max_index) ## skip classification doing prediction
            self.for_ttbot()
            self.__time += self.__delta_T
            self.__cnt += 1
            
        else:
            if (self.__padding_done == True):
                self.calculate_vis_hitting_point()
                print("pred hitting point = ", self.__arr_pred_possible)
                if (self.__pred_for_offline.shape[0] != 1):
                    #self.plot_error()
                    self.plot_res()
                    self.plot_hp_plane()
                    self.save_data()
                self.__num += 1

            self.__padding_done = False
            self.__arr_classification = cp.zeros([1,90])
            self.__coor = -10*cp.ones(6)
            self.__vis_balls = cp.zeros((self.__time_step,3))
            self.__cnt = 1
            self.__pred = cp.zeros([1,self.__time_step,3])
            self.__vis_balls2 = cp.zeros((1,4))
            self.__arr_pred_possible = cp.zeros((1,4))
            self.__find_possible_point = False
            self.__vis_possible_point = False
            self.__pred_for_offline = cp.zeros([1,self.__time_step,3])
            self.__time = 0.016667 ## reset time