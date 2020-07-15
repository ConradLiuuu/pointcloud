#include <ros/ros.h>
#include <std_msgs/Float32MultiArray.h>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>
#include <iostream>
using namespace std;
using namespace ros;

float mx[66] = {0};
float my[66] = {0};
float mz[66] = {0};
float vx[66] = {0};
float vy[66] = {0};
float vz[66] = {0};

void callback1(const std_msgs::Float32MultiArray::ConstPtr& arr_msg){
  for (int i = 0; i < 66; i++){
    mx[i] = arr_msg->data[i];
    cout << mx << endl;
  }
}

int main(int argc, char** argv)
{
  init(argc, argv, "sub");
  NodeHandle nh;
  Subscriber sub1 = nh.subscribe("mx",10,callback1);
  /*Subscriber sub2 = nh.subscribe("my",10,callback2);
  Subscriber sub3 = nh.subscribe("mz",10,callback3);
  Subscriber sub4 = nh.subscribe("vx",10,callback4);
  Subscriber sub5 = nh.subscribe("vy",10,callback5);
  Subscriber sub6 = nh.subscribe("vz",10,callback6);*/

  spin();
  return 0;
}
