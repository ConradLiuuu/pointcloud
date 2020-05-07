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
  string num_ = argv[1];
  //cout << num_ << endl;

  vector<double> pred, vis;

  ros::init(argc, argv, "predction_9step_offline");
  ros::NodeHandle n;

  ros::Publisher measurement_pub = n.advertise<sensor_msgs::PointCloud2> ("measurement_output", 1);
  ros::Publisher visual_pub = n.advertise<sensor_msgs::PointCloud2> ("visual_output", 1);

  ros::Rate rate(4);

  pcl::PointCloud<pcl::PointXYZ> measurement;
  sensor_msgs::PointCloud2 measurement_output;
  pcl::PointCloud<pcl::PointXYZ> visual;
  sensor_msgs::PointCloud2 visual_output;

  measurement.width  = 1000;
  measurement.height = 1;
  measurement.points.resize(measurement.width * measurement.height);
  visual.width  = 2000;
  visual.height = 1;
  visual.points.resize(visual.width * visual.height);

  string path = "/home/lab606a/catkin_ws/src/pointcloud/offline/";
  string pred_path = path + "prediction" + num_ + ".csv";
  string vis_path = path + "visurement" + num_ + ".csv";

  fstream myfile;
  myfile.open (pred_path);

  string line;

  while (getline(myfile, line, '\n')){
    istringstream templine(line);
    string data;
    while (getline(templine, data, ',')){
      pred.push_back(atof(data.c_str()));
    }
  }

  myfile.close();

  cout << "pred shape = " << pred.size() << endl;
  int pred_row = pred.size()/3;

  myfile.open (vis_path);
  while (getline(myfile, line, '\n')){
    istringstream templine(line);
    string data;
    while (getline(templine, data, ',')){
      vis.push_back(atof(data.c_str()));
    }
  }

  myfile.close();

  cout << "vis shape = " << vis.size() << endl;
  int vis_row = vis.size()/3;
  int move_times = vis_row-8;
  cout << "move times = " << move_times << endl;
  int pred_append = pred_row/move_times/9;
  cout << "pred append = " << pred_append << endl;

  int anchor = 0;
  int anchor2 = 0;
  int i = 0;
  int j = 0;
  int cnt = 1;
  while (ros::ok())
  {
    while (cnt <= move_times){
      cout << cnt << endl;
      while (anchor < 9){
        //cout << i << endl;
        //cout << vis[i] << endl;
        /*
        measurement.points[anchor].x = pred[i];
        measurement.points[anchor].y = pred[i+1];
        measurement.points[anchor].z = pred[i+2];
        */
        visual.points[anchor].x = vis[i]/100;
        visual.points[anchor].y = vis[i+1]/100;
        visual.points[anchor].z = vis[i+2]/100;

        i = i+3;
        anchor += 1;
        //cout << i << ", " << anchor << endl;
      }
      while (anchor2 < (9*pred_append)){
        //cout << anchor2 << endl;
        //cout << pred[j] << endl;
        //cout << "anchor2 = " << anchor2 << endl;
        
        measurement.points[anchor2].x = pred[j]/100;
        measurement.points[anchor2].y = pred[j+1]/100;
        measurement.points[anchor2].z = pred[j+2]/100;
        /*
        if ((anchor2 == (9*pred_append-1)) && (cnt == move_times)){
          cout << "----------" << endl;
          cout << j/3 << endl;
          cout << (j-3*(pred_append-2)*(9*pred_append)+2*3)/3 << endl;
          cout << pred[(j-3*(pred_append-2)*(9*pred_append)+3)] << endl;
          cout << pred[(j-3*(pred_append-2)*(9*pred_append)+3)+1] << endl;
          cout << pred[(j-3*(pred_append-2)*(9*pred_append)+3)+2] << endl;
          measurement.points[anchor2+1].x = pred[(j-3*(pred_append-2)*(9*pred_append)+3)];
          measurement.points[anchor2+1].y = pred[(j-3*(pred_append-2)*(9*pred_append)+3)+1];
          measurement.points[anchor2+1].z = pred[(j-3*(pred_append-2)*(9*pred_append)+3)+2];
          
        }
        */
        j = j+3;
        anchor2 += 1;
        //cout << i << ", " << anchor << endl;
      }

      anchor = 0;
      anchor2 = 0;
      cnt += 1;
      i = i - 24;
      //j = j - 27;

      pcl::toROSMsg(measurement, measurement_output);
      measurement_output.header.frame_id = "world";
      measurement_pub.publish(measurement_output);

      pcl::toROSMsg(visual, visual_output);
      visual_output.header.frame_id = "world";
      visual_pub.publish(visual_output);

      rate.sleep();
      
      measurement.points.clear();
      measurement.points.resize(measurement.width * measurement.height);

      visual.points.clear();
      visual.points.resize(visual.width * visual.height);
      
    }
    cnt = 1;
    i = 0;
    j = 0;
    //measurement.points.clear();
    //measurement.points.resize(measurement.width * measurement.height);
      
    //ros::spin();
  }
  ros::spin();

  /*
  int i = 0;
  while(i < mea.size()){
    measurement.points[i].x = mea[i];
    measurement.points[i].y = mea[i+1];
    measurement.points[i].z = mea[i+2];

    visual.points[i].x = vis[i];
    visual.points[i].y = vis[i+1];
    visual.points[i].z = vis[i+2];

    i = i + 3;
  }

  while(ros::ok()){
    pcl::toROSMsg(measurement, measurement_output);
    measurement_output.header.frame_id = "map";
    measurement_pub.publish(measurement_output);

    pcl::toROSMsg(visual, visual_output);
    visual_output.header.frame_id = "map";
    visual_pub.publish(visual_output);
  }
  */

  return 0;
}
