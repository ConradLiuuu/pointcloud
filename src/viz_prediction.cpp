#include <ros/ros.h>
#include <std_msgs/Float32MultiArray.h>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>
#include <iostream>

using namespace ros;
using namespace std;

class Show_prediction
{
private:
  NodeHandle nh;
  Subscriber sub;
  Publisher pcl_pub;
  pcl::PointCloud<pcl::PointXYZ> cloud;
  sensor_msgs::PointCloud2 output;
  int n = 0;
  int cnt = 0;

public:
  Show_prediction(){
    cloud.points.resize(1000);
    pcl_pub = nh.advertise<sensor_msgs::PointCloud2> ("/prediction", 1);
    sub = nh.subscribe("/prediction_coordinate", 1, &Show_prediction::callback, this);
  }

  void callback(const std_msgs::Float32MultiArray::ConstPtr& msg){
    //cout << "aaa" << endl;
    cout << "length = " << msg->data.size() << endl;
    while(cnt < msg->data.size()){
      cout << msg->data[cnt] << ", " << msg->data[cnt+1] << ", " << msg->data[cnt+2] << endl;
      cloud.points[n].x = msg->data[cnt]/100;
      cloud.points[n].y = msg->data[cnt+1]/100;
      cloud.points[n].z = msg->data[cnt+2]/100;
      n += 1;
      cnt += 3;
    }
    n = 0;
    cnt = 0;
    /*
    cloud.points[0].x = msg->data[0];
    cloud.points[0].y = msg->data[1];
    cloud.points[0].z = msg->data[2];
    cloud.points[1].x = msg->data[3];
    cloud.points[1].y = msg->data[4];
    cloud.points[1].z = msg->data[5];
    cloud.points[2].x = msg->data[6];
    cloud.points[2].y = msg->data[7];
    cloud.points[2].z = msg->data[8];
    cloud.points[3].x = msg->data[9];
    cloud.points[3].y = msg->data[10];
    cloud.points[3].z = msg->data[11];
    cloud.points[4].x = msg->data[12];
    cloud.points[4].y = msg->data[13];
    cloud.points[4].z = msg->data[14];
    */
    pcl::toROSMsg(cloud, output);
    output.header.frame_id = "world";
    pcl_pub.publish(output);

    cloud.points.clear();
    cloud.points.resize(1000);
  }

};

int main(int argc, char** argv)
{
  init(argc, argv, "viz_prediction");
  Show_prediction pred;
  spin();
  return 0;
}