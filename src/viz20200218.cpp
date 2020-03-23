#include <ros/ros.h>
#include <std_msgs/Int64MultiArray.h>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>
#include <iostream>
#include <math.h>
#include <vector>
#include "Eigen/Eigen"
#include <thread>
#include <fstream>
using namespace std;
using namespace ros;

string camera_L = "left", camera_R = "right";
// left camera intrinsic parameter
double fu_L = 1766.56232, fv_L = 1766.80239;  // focal length
double u0_L = 1038.99491, v0_L = 783.18243;  // principal point
double kc_L[8] = {-0.00825394043969168, 0.0179401516569629, 4.99404147750032e-05, 0.00191928903162996, 0.00543957313843740, -0.0137634713444645, -3.37506702874865e-05, -0.00143562480046507};
// right camera intrinsic parameter
double fu_R = 1764.19837, fv_R = 1765.54600; // focal length
double u0_R = 1056.77639, v0_R = 773.98008; // principal point
double kc_R[8] = {-0.00771656075847792, 0.0111416719316138, 0.000739495171748185, 0.000840103654848698, 0.00584317345777879, -0.00874157051790497, -0.000384902445181700, -0.000593299662265151};

//double a = 0,b = 0, c = 0, dd = 0;
int dis_Ix_L = -1, dis_Iy_L = -1, dis_Ix_R = -1, dis_Iy_R = -1;
double Ix_L, Iy_L, Ix_R, Iy_R;
//int ID_L, ID_R;
int ID_L = 0, ID_R = 0;
int id_L, id_R;
int contour_L, contour_R;


class Sub_ball_center
{
private:
  NodeHandle nh;
  Subscriber sub_left, sub_right;

public:
  Sub_ball_center(){
    sub_left = nh.subscribe("ball_center_left", 1, &Sub_ball_center::callback_left, this);
    sub_right = nh.subscribe("ball_center_right", 1, &Sub_ball_center::callback_right, this);
  }

  void operator()(){
    sub_left = nh.subscribe("ball_center_left", 1, &Sub_ball_center::callback_left, this);
    sub_right = nh.subscribe("ball_center_right", 1, &Sub_ball_center::callback_right, this);
    ros::spin();
  }

  void callback_left(const std_msgs::Int64MultiArray::ConstPtr& msg_left)
  {
    ID_L = msg_left->data[0];
    dis_Ix_L = msg_left->data[1];
    dis_Iy_L = msg_left->data[2];
    contour_L = msg_left->data[3];
  }

  void callback_right(const std_msgs::Int64MultiArray::ConstPtr& msg_right)
  {
    ID_R = msg_right->data[0];
    dis_Ix_R = msg_right->data[1];
    dis_Iy_R = msg_right->data[2];
    contour_R = msg_right->data[3];
  }
};




void correction_img(string camera, int dis_Ix, int dis_Iy, double fu, double fv, double u0, double v0, double kc[8])
{
  double dis_hxz, dis_hyz, rd ,G, hxz, hyz;

  if (camera == "left"){
    // calculate distortion ray vector
    dis_hxz = (dis_Ix - u0) / fu;
    dis_hyz = (dis_Iy - v0) / fv;

    // calcuate correction parameter
    rd = sqrt(pow(dis_hxz,2)+pow(dis_hyz,2));
    G = 4*kc[4]*pow(rd,2) + 6*kc[5]*pow(rd,4) + 8*kc[6]*dis_hyz + 8*kc[7]*dis_hxz + 1;

    // calculate correction sight vector
    hxz = dis_hxz + (1/G)*(kc[0]*pow(rd,2)+kc[1]*pow(rd,4)*dis_hxz+2*kc[2]*dis_hxz*dis_hyz+kc[3]*(pow(rd,2)+2*pow(dis_hxz,2)));
    hyz = dis_hyz + (1/G)*(kc[0]*pow(rd,2)+kc[1]*pow(rd,4)*dis_hyz+kc[2]*(pow(rd,2)+2*pow(dis_hyz,2))+2*kc[3]*dis_hxz*dis_hyz);

    // calculate correction position
    Ix_L = u0 + fu*hxz;
    Iy_L = v0 + fv*hyz;

  }

  if (camera == "right"){
    // calculate distortion sight vector
    dis_hxz = (dis_Ix - u0) / fu;
    dis_hyz = (dis_Iy - v0) / fv;

    // calcuate correction parameter
    rd = sqrt(pow(dis_hxz,2)+pow(dis_hyz,2));
    G = 4*kc[4]*pow(rd,2) + 6*kc[5]*pow(rd,4) + 8*kc[6]*dis_hyz + 8*kc[7]*dis_hxz + 1;

    // calculate correction ray vector
    hxz = dis_hxz + (1/G)*(kc[0]*pow(rd,2)+kc[1]*pow(rd,4)*dis_hxz+2*kc[2]*dis_hxz*dis_hyz+kc[3]*(pow(rd,2)+2*pow(dis_hxz,2)));
    hyz = dis_hyz + (1/G)*(kc[0]*pow(rd,2)+kc[1]*pow(rd,4)*dis_hyz+kc[2]*(pow(rd,2)+2*pow(dis_hyz,2))+2*kc[3]*dis_hxz*dis_hyz);

    // calculate correction position
    Ix_R = u0 + fu*hxz;
    Iy_R = v0 + fv*hyz;

  }

}

void trajectory(){
  NodeHandle n;
  ros::Publisher pcl_pub = n.advertise<sensor_msgs::PointCloud2> ("pcl_output", 1);
  ros::Publisher KF_pub = n.advertise<sensor_msgs::PointCloud2> ("KF_trajectory", 1);
  ros::Publisher landing_pub = n.advertise<sensor_msgs::PointCloud2> ("landing_point", 1);

  pcl::PointCloud<pcl::PointXYZ> cloud;
  sensor_msgs::PointCloud2 output;

  pcl::PointCloud<pcl::PointXYZ> cloud_KF;
  sensor_msgs::PointCloud2 output_KF;

  pcl::PointCloud<pcl::PointXYZ> cloud_landing;
  sensor_msgs::PointCloud2 output_landing;

  double R_R2L[3][3] = {{0.9633, -0.1044, 0.2474},{0.1166, 0.9925, -0.0353},{-0.2419, 0.0628, 0.9683}}; // matlab given
  double R_L2R[3][3] = {{R_R2L[0][0], R_R2L[1][0], R_R2L[2][0]},{R_R2L[0][1], R_R2L[1][1], R_R2L[2][1]},{R_R2L[0][2], R_R2L[1][2], R_R2L[2][2]}};

  double b_R2L[3] = {-840.38437, -115.52910, 232.14452}; // matlab given
  double b_L2R[3] = {879.1440, 12.3325, -20.9244}; // -R_L2R * b_R2L

  double d[3];
  d[0] = (R_R2L[0][0]*b_L2R[0]) + (R_R2L[0][1]*b_L2R[1]) + (R_R2L[0][2]*b_L2R[2]);
  d[1] = (R_R2L[1][0]*b_L2R[0]) + (R_R2L[1][1]*b_L2R[1]) + (R_R2L[1][2]*b_L2R[2]);
  d[2] = (R_R2L[2][0]*b_L2R[0]) + (R_R2L[2][1]*b_L2R[1]) + (R_R2L[2][2]*b_L2R[2]);

  double b_L2W[3] = {-699.620721, 450.703227, 2042.738938}; // matlab given
  double R_W2L[3][3] = {{0.999942, 0.001900, -0.010572},{-0.009106, -0.372230, -0.928096},{-0.005698, 0.928139, -0.372191}}; // matlab given
  double R_L2W[3][3] = {{R_W2L[0][0], R_W2L[1][0], R_W2L[2][0]},{R_W2L[0][1], R_W2L[1][1], R_W2L[2][1]},{R_W2L[0][2], R_W2L[1][2], R_W2L[2][2]}};

  double hx_L, hy_L, hz_L;

  double hx_W, hy_W, hz_W;
  double hx, hy, hz;
  double dif_L2W[3] = {0};

  int y1;
  int y2 = 0;
  int i = 0;
  double k = 0;

  cloud.width  = 1000;
  cloud.height = 1;
  cloud.points.resize(cloud.width * cloud.height);

  std::ofstream myfile;
  myfile.open ("/home/lab606a/Documents/tmp.csv");

  while (ros::ok()) {
    id_L = ID_L;
    id_R = ID_R;

    if (id_L == id_R){

      if ((dis_Ix_L >= 0) && (dis_Iy_L >= 0) && (dis_Ix_R >= 0) && (dis_Iy_R >= 0) && (ID_L == ID_R)){
        correction_img(camera_L, dis_Ix_L, dis_Iy_L, fu_L, fv_L, u0_L, v0_L, kc_L);
        correction_img(camera_R, dis_Ix_R, dis_Iy_R, fu_R, fv_R, u0_R, v0_R, kc_R);

        // calcuate k
        k = ((R_R2L[0][0]*(Ix_L-u0_L)/fu_L) + (R_R2L[0][1]*(Iy_L-v0_L)/fv_L) + R_R2L[0][2]) - ((Ix_R-u0_R)/fu_R)*((R_R2L[2][0]*(Ix_L-u0_L)/fu_L) + (R_R2L[2][1]*(Iy_L-v0_L)/fv_L) + R_R2L[2][2]);

        // calculate left ray vector
        hz_L = (d[0] - (d[2]*(Ix_R-u0_R)/fu_R)) / k;
        hx_L = hz_L*(Ix_L-u0_L)/fu_L;
        hy_L = hz_L*(Iy_L-v0_L)/fv_L;

        dif_L2W[0] = hx_L-b_L2W[0];
        dif_L2W[1] = hy_L-b_L2W[1];
        dif_L2W[2] = hz_L-b_L2W[2];

        hx_W = R_L2W[0][0] * dif_L2W[0] + R_L2W[0][1] * dif_L2W[1] + R_L2W[0][2] * dif_L2W[2] - (-5.717688);
        hy_W = R_L2W[1][0] * dif_L2W[0] + R_L2W[1][1] * dif_L2W[1] + R_L2W[1][2] * dif_L2W[2] - (-0.331069);
        hz_W = R_L2W[2][0] * dif_L2W[0] + R_L2W[2][1] * dif_L2W[1] + R_L2W[2][2] * dif_L2W[2] - (5.301289) + 16;

        hx = hx_W / 10;
        hy = hy_W / 10;
        hz = hz_W / 10;

        y1 = int(hy_W);

        //cout << hx << ", " << hy << ", " << hz << endl;

        if ((y1 != y2) && (ID_L == ID_R )){
          //cout << hx << ", " << hy << ", " << hz << endl;
          cout << hx << "," << hy << "," << hz << ",";
          //myfile << hx << "," << hy << "," << hz << ",";

          if (hy <= (-50)){ // far away ping pong table
          //if ((contour_L == 0) && (contour_R == 0)){

            cloud.points.clear();
            cloud.points.resize(cloud.width * cloud.height);
            i = 0;
            cout << endl;
            myfile << "\n";

          }
          else{ // nearby ping pong table
            //cout << hx << ", " << hy << ", " << hz << endl;
            // Measurement trajectory
            myfile << hx << "," << hy << "," << hz << ",";
            cloud.points[i].x = hx;
            cloud.points[i].y = hy;
            cloud.points[i].z = hz;
            pcl::toROSMsg(cloud, output);
            output.header.frame_id = "map";
            pcl_pub.publish(output);
          }

          y2 = y1;
          i = i+1;
        }

      }
    }
  }
  myfile.close();
}

int main(int argc, char** argv)
{
  init(argc, argv, "viz20200218");
  //Stereo sterro;
  Sub_ball_center sub;

  thread t1(ref(sub));
  thread t2(trajectory);

  t2.join();
  t1.join();


  return 0;
}
