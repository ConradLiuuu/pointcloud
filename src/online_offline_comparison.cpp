#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <ros/ros.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>
using namespace std;

int main(int argc, char** argv)
{
  vector<double> mea, vis;

  ros::init(argc, argv, "show_offline");
  ros::NodeHandle n;

  ros::Publisher measurement_pub = n.advertise<sensor_msgs::PointCloud2> ("online_output", 1);
  ros::Publisher visual_pub = n.advertise<sensor_msgs::PointCloud2> ("offline_output", 1);

  pcl::PointCloud<pcl::PointXYZ> measurement;
  sensor_msgs::PointCloud2 measurement_output;
  pcl::PointCloud<pcl::PointXYZ> visual;
  sensor_msgs::PointCloud2 visual_output;

  measurement.width  = 10000;
  measurement.height = 1;
  measurement.points.resize(measurement.width * measurement.height);
  visual.width  = 10000;
  visual.height = 1;
  visual.points.resize(visual.width * visual.height);

  fstream myfile;
  myfile.open ("/home/lab606a/Documents/offline/top2_re.csv");

  string line;

  while (getline(myfile, line, '\n')){
    istringstream templine(line);
    string data;
    while (getline(templine, data, ',')){
      mea.push_back(atof(data.c_str()));
    }
  }

  myfile.close();

  cout << mea.size() << endl;


  //int j = 5;
  //int k = 31 * (j+1);
  //int i = 9*3*31*j;
  int i = 0;

  while(i < mea.size()){

    measurement.points[i].x = mea[i]/100;
    measurement.points[i].y = mea[i+1]/100;
    measurement.points[i].z = mea[i+2]/100;

    //cout << mea[i] << endl;

    visual.points[i].x = mea[i+3]/100;
    visual.points[i].y = mea[i+4]/100;
    visual.points[i].z = mea[i+5]/100;

    i = i + 6;
  }
  cout << "aa" << endl;
/*
  i = 0;
  while(i < vis.size()){
    visual.points[i].x = vis[i];
    visual.points[i].y = vis[i+1];
    visual.points[i].z = vis[i+2];

    i = i + 3;
  }
*/
  
  while(ros::ok()){
    pcl::toROSMsg(measurement, measurement_output);
    measurement_output.header.frame_id = "world";
    measurement_pub.publish(measurement_output);

    pcl::toROSMsg(visual, visual_output);
    visual_output.header.frame_id = "world";
    visual_pub.publish(visual_output);
  }
  
  return 0;
}
