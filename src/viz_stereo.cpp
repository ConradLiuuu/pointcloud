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

  ros::init(argc, argv, "viz_stereo");
  ros::NodeHandle n;

  ros::Publisher measurement_pub = n.advertise<sensor_msgs::PointCloud2> ("measurement_output", 1);
  ros::Publisher visual_pub = n.advertise<sensor_msgs::PointCloud2> ("visual_output", 1);

  pcl::PointCloud<pcl::PointXYZ> measurement;
  sensor_msgs::PointCloud2 measurement_output;
  pcl::PointCloud<pcl::PointXYZ> visual;
  sensor_msgs::PointCloud2 visual_output;

  measurement.width  = 198;
  measurement.height = 1;
  measurement.points.resize(measurement.width * measurement.height);
  visual.width  = 198;
  visual.height = 1;
  visual.points.resize(visual.width * visual.height);

  fstream myfile;
  myfile.open ("/home/lab606a/Documents/stereo_m.csv");

  string line;

  while (getline(myfile, line, '\n')){
    istringstream templine(line);
    string data;
    while (getline(templine, data, ',')){
      mea.push_back(atof(data.c_str()));
    }
  }

  myfile.close();

  myfile.open ("/home/lab606a/Documents/stereo_v.csv");
  while (getline(myfile, line, '\n')){
    istringstream templine(line);
    string data;
    while (getline(templine, data, ',')){
      vis.push_back(atof(data.c_str()));
    }
  }

  myfile.close();



  //cout << matrix.size() << endl;
/*
  for (int i = 0; i < matrix.size(); i++){
    cout << matrix[i] << endl;
  }
*/
/*
  for (int i = 0; i < matrix.size(); i++){
    if (((i+1)%3) == 0){
      cout << matrix[i] << ",";
      cout << endl;
    }
    else{
      cout << matrix[i] << ",";
    }
  }
*/
/*
  for (int i = 0; i < matrix.size(); i++){
    for (int j = 0; j < 66; j++){
      if ((i%3) == 0){
        measurement.points[j].x = matrix[i];
      }
      else if ((i%3) == 1){
        measurement.points[j].y = matrix[i];
      }
      else{
        measurement.points[j].z = matrix[i];
      }
    }
  }
*/
  //cout << mea.size() << endl;

  int i = 0;
  while(i < mea.size()){
    measurement.points[i].x = mea[i]/100;
    measurement.points[i].y = mea[i+1]/100;
    measurement.points[i].z = mea[i+2]/100;

    visual.points[i].x = vis[i]/100;
    visual.points[i].y = vis[i+1]/100;
    visual.points[i].z = vis[i+2]/100;

    i = i + 3;
  }

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
