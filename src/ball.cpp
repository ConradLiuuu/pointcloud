#include <ros/ros.h>
#include <std_msgs/Int64MultiArray.h>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>
#include <iostream>
#include <math.h>
#include <thread>
using namespace std;
using namespace ros;

string camera_L = "left", camera_R = "right";
// left camera intrinsic parameter
double fu_L = 1767.89133, fv_L = 1768.29877;  // focal length
double u0_L = 1032.41658, v0_L = 779.77186;  // principal point
double kc_L[8] = {0.00305139641514509, -0.00765590095180579, 0.00129026164796253, 0.00304992343089967, -0.00232649720836786, 0.00625583703031773, -0.00076244567038416, -0.00229054173473991};
// right camera intrinsic parameter
double fu_R = 1767.11269, fv_R = 1768.73380; // focal length
double u0_R = 1055.03490, v0_R = 775.48461; // principal point
double kc_R[8] = {-0.00240554144590306, 0.00415861668132687, 0.00124996506239818, 0.000930166885290533, 0.00202439715420488, -0.00351884072634534, -0.000806762468456028, -0.000668423449138214};

//double a = 0,b = 0, c = 0, dd = 0;
int dis_Ix_L = -1, dis_Iy_L = -1, dis_Ix_R = -1, dis_Iy_R = -1;
double Ix_L, Iy_L, Ix_R, Iy_R;
//int ID_L, ID_R;
int ID_L = 0, ID_R = 0;
int id_L, id_R;
bool isDone = true;
int bound = -26;

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
    if ((msg_left->data[1] >= 0) && (msg_left->data[2] >= 0)){
      ID_L = msg_left->data[0];
      dis_Ix_L = msg_left->data[1];
      dis_Iy_L = msg_left->data[2];
      //cout << "ID_L = " << ID_L << endl;
      //isDone = false;
    }
    else {
      ID_L = 0;
      dis_Ix_L = -1;
      dis_Iy_L = -1;
    }

    //cout << "dis_Ix_L = " << dis_Ix_L << endl;
    //cout << "dis_Iy_L = " << dis_Iy_L << endl;
  }

  void callback_right(const std_msgs::Int64MultiArray::ConstPtr& msg_right)
  {
    if ((msg_right->data[1] >= 0) && (msg_right->data[2] >= 0)){
      ID_R = msg_right->data[0];
      dis_Ix_R = msg_right->data[1];
      dis_Iy_R = msg_right->data[2];
      //cout << "ID_R = " << ID_R << endl;
      //isDone = false;
    }
    else{
      ID_R = -1;
      dis_Ix_R = -1;
      dis_Iy_R = -1;
    }

    //cout << "dis_Ix_R = " << dis_Ix_R << endl;
    //cout << "dis_Iy_R = " << dis_Iy_R << endl;
  }
};

void correction_img(string camera, int dis_Ix, int dis_Iy, double fu, double fv, double u0, double v0, double kc[8])
{
  double dis_hxz, dis_hyz, rd ,G, hxz, hyz;

  if (camera == "left"){
    // calculate distortion ray vector
    dis_hxz = (dis_Ix - u0) / fu;
    dis_hyz = (dis_Iy - v0) / fv;
    //cout << "distortion hxz_L = " << dis_hxz << endl;
    //cout << "distortion hyz_L = " << dis_hyz << endl;


    // calcuate correction parameter
    rd = sqrt(pow(dis_hxz,2)+pow(dis_hyz,2));
    G = 4*kc[4]*pow(rd,2) + 6*kc[5]*pow(rd,4) + 8*kc[6]*dis_hyz + 8*kc[7]*dis_hxz + 1;

    // calculate correction sight vector
    hxz = dis_hxz + (1/G)*(kc[0]*pow(rd,2)+kc[1]*pow(rd,4)*dis_hxz+2*kc[2]*dis_hxz*dis_hyz+kc[3]*(pow(rd,2)+2*pow(dis_hxz,2)));
    hyz = dis_hyz + (1/G)*(kc[0]*pow(rd,2)+kc[1]*pow(rd,4)*dis_hyz+kc[2]*(pow(rd,2)+2*pow(dis_hyz,2))+2*kc[3]*dis_hxz*dis_hyz);

    // calculate correction position
    Ix_L = u0 + fu*hxz;
    Iy_L = v0 + fv*hyz;
    //cout << "correct Ix_L Iy_L = [" << Ix_L << "," << Iy_L << "]" << endl;
    //cout << "Ix_L by func = " << Ix_L << endl;
    //cout << "Iy_L by func= " << Iy_L << endl;

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
    //cout << "correct Ix_R Iy_R = [" << Ix_R << "," << Iy_R << "]" << endl;
    //cout << "Ix_R by func = " << Ix_R << endl;
    //cout << "Iy_R by func= " << Iy_R << endl;

  }

}

void tt(){
  NodeHandle n;
  ros::Publisher pcl_pub = n.advertise<sensor_msgs::PointCloud2> ("pcl_output", 1);
  pcl::PointCloud<pcl::PointXYZ> cloud;
  sensor_msgs::PointCloud2 output;

  double R_R2L[3][3] = {{0.973111335157242, 0.00164499724031902, 0.230329380176672},{-0.00161841567681531, 0.999998644055752, -0.000304330996052487},{-0.230329568486522, -0.0000766207378107115, 0.973112677961846}}; // matlab given
  double R_L2R[3][3] = {{R_R2L[0][0], R_R2L[1][0], R_R2L[2][0]},{R_R2L[0][1], R_R2L[1][1], R_R2L[2][1]},{R_R2L[0][2], R_R2L[1][2], R_R2L[2][2]}};

  double b_R2L[3] = {-320.44171, -0.50178, 79.70245}; // matlab given
  double b_L2R[3] = {330.18247908537, 1.03501190877172, -3.75247681977456}; // -R_L2R * b_R2L

  double d[3];
  d[0] = (R_R2L[0][0]*b_L2R[0]) + (R_R2L[0][1]*b_L2R[1]) + (R_R2L[0][2]*b_L2R[2]);
  d[1] = (R_R2L[1][0]*b_L2R[0]) + (R_R2L[1][1]*b_L2R[1]) + (R_R2L[1][2]*b_L2R[2]);
  d[2] = (R_R2L[2][0]*b_L2R[0]) + (R_R2L[2][1]*b_L2R[1]) + (R_R2L[2][2]*b_L2R[2]);

  double b_L2W[3] = {-819.727918, 437.972549, 1550.559426};
  double R_L2W[3][3] = {{0.999860, -0.010543, 0.012973}, {0.006686, -0.459035, -0.888393}, {0.015321, 0.888356, -0.458900}};
  double R_W2L[3][3] = {{R_L2W[0][0], R_L2W[1][0], R_L2W[2][0]},{R_L2W[0][1], R_L2W[1][1], R_L2W[2][1]},{R_L2W[0][2], R_L2W[1][2], R_L2W[2][2]}};

  double hx_L, hy_L, hz_L;

  double hx_W, hy_W, hz_W;
  double hx, hy, hz;
  double dif_L2W[3] = {0};

  double y1;
  double y2 = 0;
  int diff = 0;
  int i = 0;

  cloud.width  = 300;
  cloud.height = 1;
  cloud.points.resize(cloud.width * cloud.height);


  while (ros::ok()) {
    //ROS_INFO("dis_Ix_L = %f, dis_Iy_L = %f \n", a,b);
    //ROS_INFO("dis_Ix_R = %f, dis_Iy_R = %f \n", c,dd);
    //cout << dis_Ix_L << "," << dis_Iy_L << "|" << dis_Ix_R << "," << dis_Iy_R << endl;
    //cout << "ID_L = " << ID_L  << ", ID_R = " << ID_R << endl;

    //diff = ID_L - ID_R;
    id_L = ID_L;
    id_R = ID_R;

    if (id_L == id_R){

      if ((dis_Ix_L >= 0) && (dis_Iy_L >= 0) && (dis_Ix_R >= 0) && (dis_Iy_R >= 0) && (ID_L == ID_R)){
        correction_img(camera_L, dis_Ix_L, dis_Iy_L, fu_L, fv_L, u0_L, v0_L, kc_L);
        correction_img(camera_R, dis_Ix_R, dis_Iy_R, fu_R, fv_R, u0_R, v0_R, kc_R);

        // calcuate k
        double k = ((R_R2L[0][0]*(Ix_L-u0_L)/fu_L) + (R_R2L[0][1]*(Iy_L-v0_L)/fv_L) + R_R2L[0][2]) - ((Ix_R-u0_R)/fu_R)*((R_R2L[2][0]*(Ix_L-u0_L)/fu_L) + (R_R2L[2][1]*(Iy_L-v0_L)/fv_L) + R_R2L[2][2]);

        // calculate left ray vector
        hz_L = (d[0] - (d[2]*(Ix_R-u0_R)/fu_R)) / k;
        hx_L = hz_L*(Ix_L-u0_L)/fu_L;
        hy_L = hz_L*(Iy_L-v0_L)/fv_L;

        dif_L2W[0] = hx_L-b_L2W[0];
        dif_L2W[1] = hy_L-b_L2W[1];
        dif_L2W[2] = hz_L-b_L2W[2];
/*
        hx_W = R_W2L[0][0]*dif_L2W[0] + R_W2L[0][1]*dif_L2W[1] + R_W2L[0][2]*dif_L2W[2];
        hy_W = R_W2L[1][0]*dif_L2W[0] + R_W2L[1][1]*dif_L2W[1] + R_W2L[1][2]*dif_L2W[2];
        hz_W = R_W2L[2][0]*dif_L2W[0] + R_W2L[2][1]*dif_L2W[1] + R_W2L[2][2]*dif_L2W[2];
        //cout << "object position = [" << hx_W / 10 << "," << hy_W / 10 << "," << hz_W / 10 << "]" << endl;
*/
        hx_W = R_W2L[0][0] * dif_L2W[0] + R_W2L[0][1] * dif_L2W[1] + R_W2L[0][2] * dif_L2W[2] - 17.6031;
        hy_W = R_W2L[1][0] * dif_L2W[0] + R_W2L[1][1] * dif_L2W[1] + R_W2L[1][2] * dif_L2W[2] - (-23.1113);
        hz_W = R_W2L[2][0] * dif_L2W[0] + R_W2L[2][1] * dif_L2W[1] + R_W2L[2][2] * dif_L2W[2] - 18.5448 + 16;

        hx = hx_W / 10;
        hy = hy_W / 10;
        hz = hz_W / 10;

        y1 = hy_W;
        //diff = y1-y2;
        if ((y1 != y2) && (ID_L == ID_R )){
          //ROS_INFO("ID_L = %d, ID_R = %d", ID_L, ID_R);
          /*
          if (hy >= (-20)){
            //cout << "ID_L = " << ID_L  << ", ID_R = " << ID_R << endl;;
            //ROS_INFO("correction position = [%f, %f, %f]", hx, hy,hz);
            cout << hx << ", " << hy << ", " << hz << endl;
          }
          */

          if (hy <= (-20)){
            cloud.points.clear();
            cloud.width  = 300;
            cloud.height = 1;
            cloud.points.resize(cloud.width * cloud.height);
            i = 0;
          }
          else{
            cout << hx << ", " << hy << ", " << hz << endl;
            cloud.points[i].x = hx ;
            cloud.points[i].y = hy;
            cloud.points[i].z = hz;
            pcl::toROSMsg(cloud, output);
            output.header.frame_id = "map";
            pcl_pub.publish(output);
          }

          //cout << "correction position = [" << hx_W / 10 << "," << hy_W / 10 << "," << hz_W / 10 << "]" << endl;
          y2 = y1;
          i = i+1;
          //ros::spinOnce();
        }
        //isDone = true;
      }

    }
    //cloud.points.clear();
    //cloud.width  = 30;
    //cloud.height = 1;

/*
    if ((dis_Ix_L >= 0) && (dis_Iy_L >= 0) && (dis_Ix_R >= 0) && (dis_Iy_R >= 0)){
      correction_img(camera_L, dis_Ix_L, dis_Iy_L, fu_L, fv_L, u0_L, v0_L, kc_L);
      correction_img(camera_R, dis_Ix_R, dis_Iy_R, fu_R, fv_R, u0_R, v0_R, kc_R);

      // calcuate k
      double k = ((R_R2L[0][0]*(Ix_L-u0_L)/fu_L) + (R_R2L[0][1]*(Iy_L-v0_L)/fv_L) + R_R2L[0][2]) - ((Ix_R-u0_R)/fu_R)*((R_R2L[2][0]*(Ix_L-u0_L)/fu_L) + (R_R2L[2][1]*(Iy_L-v0_L)/fv_L) + R_R2L[2][2]);

      // calculate left ray vector
      hz_L = (d[0] - (d[2]*(Ix_R-u0_R)/fu_R)) / k;
      hx_L = hz_L*(Ix_L-u0_L)/fu_L;
      hy_L = hz_L*(Iy_L-v0_L)/fv_L;

      dif_L2W[0] = hx_L-b_L2W[0];
      dif_L2W[1] = hy_L-b_L2W[1];
      dif_L2W[2] = hz_L-b_L2W[2];

      hx_W = R_W2L[0][0]*dif_L2W[0] + R_W2L[0][1]*dif_L2W[1] + R_W2L[0][2]*dif_L2W[2];
      hy_W = R_W2L[1][0]*dif_L2W[0] + R_W2L[1][1]*dif_L2W[1] + R_W2L[1][2]*dif_L2W[2];
      hz_W = R_W2L[2][0]*dif_L2W[0] + R_W2L[2][1]*dif_L2W[1] + R_W2L[2][2]*dif_L2W[2];
      //cout << "object position = [" << hx_W / 10 << "," << hy_W / 10 << "," << hz_W / 10 << "]" << endl;

      hx_W = R_W2L[0][0] * dif_L2W[0] + R_W2L[0][1] * dif_L2W[1] + R_W2L[0][2] * dif_L2W[2] - 17.6031;
      hy_W = R_W2L[1][0] * dif_L2W[0] + R_W2L[1][1] * dif_L2W[1] + R_W2L[1][2] * dif_L2W[2] - (-23.1113);
      hz_W = R_W2L[2][0] * dif_L2W[0] + R_W2L[2][1] * dif_L2W[1] + R_W2L[2][2] * dif_L2W[2] - 18.5448 + 16;

      y1 = hy_W;
      //diff = y1-y2;
      if (y1 != y2){
        cout << "ID_L = " << ID_L  << ", ID_R = " << ID_R << endl;;
        cout << "correction position = [" << hx_W / 10 << "," << hy_W / 10 << "," << hz_W / 10 << "]" << endl;
        y2 = y1;
      }

    }
    */
    /*else{
      cout << "not found ball" << endl;
    }*/

  }

}

int main(int argc, char** argv)
{
  init(argc, argv, "ball");
  //Stereo sterro;
  Sub_ball_center sub;

  thread t1(ref(sub));
  thread t2(tt);

  t2.join();
  t1.join();


  return 0;
}
