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

struct point
{
  double p = 0;
  double t = 0;
};
struct param
{
  double a;
  double b;
  double c;
};
struct param_rebound
{
  double x;
  double y;
  double z;
};
struct position
{
  double x;
  double y;
  double z;
};
struct velocity
{
  double vx;
  double vy;
  double vz;
};

class KalmanFilterMagnus
{
private:
  double Fmagnus_z = 550;
  double Fmagnus_y = 550;

public:
  Eigen::Matrix <double, 6, 1> X;
  Eigen::Matrix <double, 6, 6> A;
  Eigen::Matrix <double, 6, 1> B;
  Eigen::Matrix <double, 6, 1> Y;

  Eigen::Matrix <double, 6, 6> P;
  Eigen::Matrix <double, 6, 6> Q; // weight for prediction

  Eigen::Matrix <double, 3, 6> H;
  Eigen::Matrix <double, 3, 1> Z;

  Eigen::Matrix <double, 6, 3> K;
  Eigen::Matrix <double, 3, 3> R; // weight for measurement
  Eigen::Matrix <double, 6, 6> I;

  double Km = -0.12e-2;
  double Fgravity = 9.81 * 100; // 9.81 (cm/s^2)
  double dt;
  double V, vx, vy, vz;
  int cnt = 1;
  int q = 4;
  int r = 5;

  KalmanFilterMagnus() {
    // initial
    I = Eigen::Matrix<double, 6, 6>::Identity();

    A << 1, 0, 0, dt, 0, 0,
      0, 1, 0, 0, dt, 0,
      0, 0, 1, 0, 0, dt,
      0, 0, 0, (1 + Km * V)*dt, 0, 0,
      0, 0, 0, 0, (1 + Km * V)*dt, 0,
      0, 0, 0, 0, 0, (1 + Km * V)*dt;

    X << 0, 0, 0, 0, 0, 0;
    B << 0, 0, 0, 0, 0, -dt;
    Y << 0, 0, 0, 0, -dt, 0;


    P << 10, 0, 0, 0, 0, 0,
      0, 10, 0, 0, 0, 0,
      0, 0, 10, 0, 0, 0,
      0, 0, 0, 10, 0, 0,
      0, 0, 0, 0, 10, 0,
      0, 0, 0, 0, 0, 10;


    Q << q, 0, 0, 0, 0, 0,
      0, q, 0, 0, 0, 0,
      0, 0, q, 0, 0, 0,
      0, 0, 0, q, 0, 0,
      0, 0, 0, 0, q, 0,
      0, 0, 0, 0, 0, q;

    H << 1, 0, 0, 0, 0, 0,
      0, 1, 0, 0, 0, 0,
      0, 0, 1, 0, 0, 0;

    R << r, 0, 0,
      0, r, 0,
      0, 0, r;

    Z << 0, 0, 0;

  }

  void UpdateWeight(int q, int r) {
    Q << q, 0, 0, 0, 0, 0,
      0, q, 0, 0, 0, 0,
      0, 0, q, 0, 0, 0,
      0, 0, 0, q, 0, 0,
      0, 0, 0, 0, q, 0,
      0, 0, 0, 0, 0, q;

    R << r, 0, 0,
      0, r, 0,
      0, 0, r;
  }

  void UpdateVariables() {
    A(0, 3) = dt;
    A(1, 4) = dt;
    A(2, 5) = dt;

    //V = sqrt(pow(Vx, 2) + pow(Vy, 2) + pow(Vz, 2));
    // x = [x, y, z, vx, vy, vz]
    V = sqrt(pow(X[3], 2) + pow(X[4], 2) + pow(X[5], 2));
    A(3, 3) = 1 + Km * V*dt;
    A(4, 4) = 1 + Km * V*dt;
    A(5, 5) = 1 + Km * V*dt;

    B(5) = -dt;

  }

  void kalmanfilter() {
    // Update variables
    UpdateVariables();

    // prediction
    X = A * X + B * (Fgravity + Fmagnus_z);
    P = A * P*A.transpose() + Q;

    // update
    K = P * H.transpose()*(H*P*H.transpose() + R).inverse();
    X = X + K * (Z - H * X);
    P = (I - K * H)*P;
  }

  void reset(){
    X << 0, 0, 0, 0, 0, 0;
    Z << 0, 0, 0;
  }
};

class Reg_2nd //for trajectory 2nd regression by Least Square Method (LSM)
{
public:
  int nn = 8; // number of points to do regression
  int cnt; // push back point until cnt equal nn
  double root, root1, root2; // root1 and root2 are solution of z direction equation. root will equal root1 or root2 which at right hand side.
  vector <point> pointArray; // store input points
  double m[3][3] = { 0 }; // for Cramer's rule delta
  double m0[3][3], m1[3][3], m2[3][3]; // for Cramer's rule delta0, deta1 and delta2
  double v[3] = { 0 }; // store output points
  double det_m, det_m0, det_m1, det_m2;

  struct param paramer;
  double vel;

  void init(){
    pointArray.clear();
    //m[3][3] = {0};
    //m0[3][3] = {0};
    //m1[3][3] = {0};
    //m2[3][3] = {0};
    v[0] = 0;
    v[1] = 0;
    v[2] = 0;
    for (int i = 0; i < 3; i++){
      for (int j = 0; j < 3; j++){
        m[i][j] = 0;
        m0[i][j] = 0;
        m1[i][j] = 0;
        m2[i][j] = 0;
      }
    }
    //paramer.a = 0;
    //paramer.b = 0;
    //paramer.c = 0;
    //det_m = 0;
    //det_m0 = 0;
    //det_m1= 0;
    //det_m2 = 0;
  }

  void addPoint(double t_in, double p_in) {
    //cnt += 1;
    struct point tmp;
    tmp.t = t_in;
    tmp.p = p_in;

    pointArray.push_back(tmp);
  }

  void sum_y() {
    for (int i = 0; i < nn; i++) {
      v[0] += pointArray[i].p;
      v[1] += pointArray[i].t*pointArray[i].p;
      v[2] += pow(pointArray[i].t, 2)*pointArray[i].p;
    }
  }

  void filled_m() {
    m[0][0] = nn;
    for (int i = 0; i < nn; i++) {
      m[0][1] += pointArray[i].t;
      m[0][2] += pow(pointArray[i].t, 2);
      m[1][2] += pow(pointArray[i].t, 3);
      m[2][2] += pow(pointArray[i].t, 4);
    }
    m[1][0] = m[0][1];
    m[1][1] = m[0][2];
    m[2][0] = m[1][1];
    m[2][1] = m[1][2];
  }

  double calDetermin(double a[3][3]) {
    double det;
    det = (a[0][0] * a[1][1] * a[2][2] + a[0][1] * a[1][2] * a[2][0] + a[0][2] * a[1][0] * a[2][1]) - (a[0][2] * a[1][1] * a[2][0] + a[0][0] * a[2][1] * a[1][2] + a[0][1] * a[1][0] * a[2][2]);
    return det;
  }

  void filed_mx() {
    for (int i = 0; i < 3; i++) {
      m0[i][0] = v[i];
      m0[i][1] = m[i][1];
      m0[i][2] = m[i][2];

      m1[i][0] = m[i][0];
      m1[i][1] = v[i];
      m1[i][2] = m[i][2];

      m2[i][0] = m[i][0];
      m2[i][1] = m[i][1];
      m2[i][2] = v[i];
    }
  }

  void calParam() {
    paramer.c = det_m0 / det_m;
    paramer.b = det_m1 / det_m;
    paramer.a = det_m2 / det_m;
  }

  double getRegRes(double t) {
    double res = paramer.a*pow(t, 2) + paramer.b*t + paramer.c;
    return res;
  }

  void calVel(int n1, int n2) {
    //cout << pointArray[n2].p << "-" << pointArray[n1].p << endl;
    vel = (pointArray[n2].p - pointArray[n1].p) / (pointArray[n2].t - pointArray[n1].t) / 100;
  }

  bool calRoots() {
    double discriminant = pow(paramer.b, 2) - 4 * paramer.a*paramer.c; // b^2-4ac>0 has two roots, when b^2-4ac=0 , root is peak
    if ((discriminant > 0) && (paramer.a < 0)) { //parabola opens downwards
      root1 = (-paramer.b + sqrt(discriminant)) / (2 * paramer.a);
      root2 = (-paramer.b - sqrt(discriminant)) / (2 * paramer.a);
      // chose root which at right hand side
      if (root1 < root2) {
        root = root2;
      }
      else {
        root = root1;
      }
      return true; //has two solution
    }
    else {
      return false;
    }
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

void tt(){
  NodeHandle n;
  ros::Publisher pcl_pub = n.advertise<sensor_msgs::PointCloud2> ("pcl_output", 1);
  ros::Publisher KF_pub = n.advertise<sensor_msgs::PointCloud2> ("KF_trajectory", 1);
  ros::Publisher landing_pub = n.advertise<sensor_msgs::PointCloud2> ("landing_point", 1);

  pcl::PointCloud<pcl::PointXYZ> cloud;
  sensor_msgs::PointCloud2 output;

  pcl::PointCloud<pcl::PointXYZ> cloud_KF;
  sensor_msgs::PointCloud2 output_KF;

  //pcl::PointCloud<pcl::PointXYZ> cloud_landing;
  //sensor_msgs::PointCloud2 output_landing;

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

  cloud.width  = 1000;
  cloud.height = 1;
  cloud.points.resize(cloud.width * cloud.height);

  cloud_KF.width  = 1000;
  cloud_KF.height = 1;
  cloud_KF.points.resize(cloud_KF.width * cloud_KF.height);

  //cloud_landing.points.resize(30);

  // trajectory setting
  double T = 0.02;
  double delta_T = 0.02;
  const int reg_num = 8;
  int cnt_reg = 0;
  bool reg_done = false;

  Reg_2nd reg_x; // for X direction
  Reg_2nd reg_y; // for Y direction
  Reg_2nd reg_z; // for Z direction

  struct position ti;
  struct velocity vi;

  KalmanFilterMagnus KF;

  // 1st fly model & KF
  double time = delta_T * 8;
  //double distance_y = 50;
  double land_position_x, land_position_y, land_position_z;
  double landing_time = 0;
  bool islanding = false;

  /* Rebound model parameters */
  struct param_rebound Kr;
  Kr.x = 0.57923;
  Kr.y = 3.512691; //-0.112691(origin) //3.512691(better)
  Kr.z = -0.598816; // 0.598816(origin) //-0.598816(better)
  struct param_rebound b;
  b.x = 5.46866; //5.46866
  b.y = 460.66244; //460.66244
  b.z = 85.20345; //85.20345
  struct param_rebound landing_pose;
  struct param_rebound vin, vout;
  bool hit = false;
  bool isRebound = false;

  double strike_point = -5;
  struct param_rebound pos_now;
  double vy_now = 0;

  while (ros::ok()) {
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

        hx_W = R_W2L[0][0] * dif_L2W[0] + R_W2L[0][1] * dif_L2W[1] + R_W2L[0][2] * dif_L2W[2] - 17.6031;
        hy_W = R_W2L[1][0] * dif_L2W[0] + R_W2L[1][1] * dif_L2W[1] + R_W2L[1][2] * dif_L2W[2] - (-23.1113);
        hz_W = R_W2L[2][0] * dif_L2W[0] + R_W2L[2][1] * dif_L2W[1] + R_W2L[2][2] * dif_L2W[2] - 18.5448 + 16;

        hx = hx_W / 10;
        hy = hy_W / 10;
        hz = hz_W / 10;

        y1 = hy_W;
        //diff = y1-y2;
        if ((y1 != y2) && (ID_L == ID_R )){
          //cout << "T = " << T << endl;
          // curve fitting using 1st~10th
          cout << "time = " << T << endl;
          cout << hx << ", " << hy << ", " << hz << endl;
          if (cnt_reg != reg_num && hy > 50) {
            reg_x.addPoint(T, hx);
            reg_y.addPoint(T, hy);
            reg_z.addPoint(T, hz);
            //cout << "Y dir = " << T << ", " << hy << endl;
            //cout << "add point:" << reg_x.pointArray[cnt_reg].t << ", " << reg_x.pointArray[cnt_reg].p << endl;
            //T += delta_T;
            cnt_reg += 1;
          }
          if (cnt_reg == reg_num && reg_done == false){
            reg_x.sum_y();
            reg_y.sum_y();
            reg_z.sum_y();

            reg_x.filled_m();
            reg_y.filled_m();
            reg_z.filled_m();

            reg_x.det_m = reg_x.calDetermin(reg_x.m);
            reg_y.det_m = reg_x.calDetermin(reg_y.m);
            reg_z.det_m = reg_x.calDetermin(reg_z.m);
            //cout << reg_x.det_m << endl;

            reg_x.filed_mx();
            reg_y.filed_mx();
            reg_z.filed_mx();

            reg_x.det_m0 = reg_x.calDetermin(reg_x.m0);
            reg_x.det_m1 = reg_x.calDetermin(reg_x.m1);
            reg_x.det_m2 = reg_x.calDetermin(reg_x.m2);
            reg_y.det_m0 = reg_x.calDetermin(reg_y.m0);
            reg_y.det_m1 = reg_x.calDetermin(reg_y.m1);
            reg_y.det_m2 = reg_x.calDetermin(reg_y.m2);
            reg_z.det_m0 = reg_x.calDetermin(reg_z.m0);
            reg_z.det_m1 = reg_x.calDetermin(reg_z.m1);
            reg_z.det_m2 = reg_x.calDetermin(reg_z.m2);

            reg_x.calParam();
            reg_y.calParam();
            reg_z.calParam();

            /*
            // show X Y Z equarion parameters
            cout << "x parameters = ";
            cout << reg_x.paramer.a << "," << reg_x.paramer.b << "," << reg_x.paramer.c << endl;
            cout << "y parameters = ";
            cout << reg_y.paramer.a << "," << reg_y.paramer.b << "," << reg_y.paramer.c << endl;
            cout << "z parameters = ";
            cout << reg_z.paramer.a << "," << reg_z.paramer.b << "," << reg_z.paramer.c << endl;
            */
            reg_done = true;


            bool sol = reg_z.calRoots();
            landing_time = reg_z.root;
            cout << "landing time = " << landing_time << endl;
            islanding = true;
            //cout << "numer of time = " << landing_time / delta_T << endl;

            land_position_x = reg_x.getRegRes(landing_time);
            land_position_y = reg_y.getRegRes(landing_time);
            land_position_z = reg_z.getRegRes(landing_time);
            cout << "Landing position by regression = " << land_position_x << ", " << land_position_y << ", " << land_position_z << endl;
            /*
            cloud_landing.points[0].x = land_position_x;
            cloud_landing.points[0].y = land_position_y;
            cloud_landing.points[0].z = land_position_z;
            pcl::toROSMsg(cloud_landing, output_landing);
            output_landing.header.frame_id = "map";
            KF_pub.publish(output_landing);
*/
            // Input position and velocity
            ti.x = reg_x.pointArray[7].p;
            ti.y = reg_y.pointArray[7].p;
            ti.z = reg_z.pointArray[7].p;

            // calculate veocity by (x10-x9)/t , unit(cm/s)
            vi.vx = (reg_x.pointArray[7].p - reg_x.pointArray[6].p) / delta_T;
            vi.vy = (reg_y.pointArray[7].p - reg_y.pointArray[6].p) / delta_T;
            vi.vz = (reg_z.pointArray[7].p - reg_z.pointArray[6].p) / delta_T;

            KF.X << ti.x, ti.y, ti.z, vi.vx, vi.vy, vi.vz;
            KF.dt = delta_T;

          }

          T += delta_T;

          // show point in rviz
          if (hy <= (-20)){
            cloud.points.clear();
            //cloud.width  = 300;
            //cloud.height = 1;
            cloud.points.resize(cloud.width * cloud.height);
            cloud_KF.points.clear();
            cloud_KF.points.resize(1000);
            //cloud_landing.clear();
            //cloud_landing.points.resize(30);
            i = 0;
            //reg_x.pointArray.clear();
            //reg_y.pointArray.clear();
            //reg_z.pointArray.clear();
            reg_x.init();
            reg_y.init();
            reg_z.init();
            cnt_reg = 0;
            reg_done = false;
            T = 0.02;
            time = delta_T * 8;
            KF.reset();
            islanding = false;
            isRebound = false;
            hit = false;
          }
          else{
            //cout << hx << ", " << hy << ", " << hz << endl;
            //if (time < (landing_time - delta_T) && islanding == true){
            if (0 < (KF.X[2] + KF.X[5]*delta_T) && islanding == true){
              // 1st fly model
              time += delta_T;
              // measurement position
              KF.Z << hx, hy, hz;
              // prediction
              KF.kalmanfilter();
              //cout << "1st fly model by KF = " << KF.X[0] << ", " << KF.X[1] << ", " << KF.X[2] << endl;
              cout << KF.X[0] << ", " << KF.X[1] << ", " << KF.X[2]<< " <-- 1st fly model by KF"  << endl;
              //distance_y = KF.X[1] - land_position_y;
              //cout << "con Z = " << KF.X[2] + KF.X[5]*delta_T << endl;


              cloud_KF.points[i].x = KF.X[0];
              cloud_KF.points[i].y = KF.X[1];
              cloud_KF.points[i].z = KF.X[2];
              pcl::toROSMsg(cloud_KF, output_KF);
              output_KF.header.frame_id = "map";
              KF_pub.publish(output_KF);

              if (0 > (KF.X[2] + KF.X[5]*delta_T)){
                cout << "Rebound !!!" << endl;
                isRebound = true;
                KF.UpdateWeight(40, 1);
                // rebound model
                landing_pose.x = KF.X[0];
                landing_pose.y = KF.X[1];
                landing_pose.z = KF.X[2];
                vin.x = KF.X[3];
                vin.y = KF.X[4];
                vin.z = KF.X[5];
                vout.x = Kr.x*vin.x + b.x;
                vout.y = Kr.y*vin.y + b.y;
                vout.z = Kr.z*vin.z + b.z;
                islanding = false;
                KF.X << land_position_x, land_position_y, land_position_z, vout.x, vout.y, vout.z;

              }

            }
            else if (isRebound == true){
              //islanding = false;
              /*
              // rebound model
              landing_pose.x = KF.X[0];
              landing_pose.y = KF.X[1];
              landing_pose.z = KF.X[2];
              vin.x = KF.X[3];
              vin.y = KF.X[4];
              vin.z = KF.X[5];
              vout.x = Kr.x*vin.x + b.x;
              vout.y = Kr.y*vin.y + b.y;
              vout.z = Kr.z*vin.z + b.z;
*/

              // 2nd fly model
              //pos_now.x = land_position_x;
              //pos_now.y = land_position_y;
              //pos_now.z = land_position_z;
              //KF.X << land_position_x, land_position_y, land_position_z, vout.x, vout.y, vout.z;

              //KF.UpdateWeight(40, 1);
              //if (KF.X[1] > -20){
                // measurement position
                KF.Z << hx, hy, hz;
                KF.kalmanfilter();
                cout << KF.X[0] << ", " << KF.X[1] << ", " << KF.X[2]<< " <-- 2nd fly model by KF"  << endl;

                //cout << "pos now y = " << pos_now.y << endl;
                //cout << "test yyy = " << KF.X[1] + KF.X[4]*delta_T/2 << ", pos y = " << KF.X[1] << ",  vy =  " << KF.X[4] << endl;

                //if (((pos_now.y + KF.X[1]) <= strike_point) && (hit == false)){
                if (((KF.X[1] + KF.X[4]*delta_T/2) <= strike_point) && (hit == false)){
                  //double tt = delta_T * (pos_now.y - strike_point) / (pos_now.y - KF.X[1]); //v1
                  //double tt = delta_T * (KF.X[1] - strike_point) / (KF.X[4]*delta_T); //v2
                  //double tt = delta_T * (strike_point-KF.X[1]) / (KF.X[4]*delta_T);
                  double tt = delta_T * (strike_point-KF.X[1]) / (KF.X[4]*delta_T);
                  //cout << "found strike point: " << pos_now.x + KF.X[3] * tt << ", " << pos_now.y + KF.X[4] * tt << ", " << pos_now.z + KF.X[5] * tt << endl;
                  cout << "found strike point: " << KF.X[0] + KF.X[3] * tt << ", " << KF.X[1] + KF.X[4] * tt << ", " << KF.X[2] + KF.X[5] * tt << endl;
                  hit = true;
                  //cout << "strike timing = " << time + delta_T / KF.X[1] << endl; //time + delta_T/KF.X[1]
                  cout << "strike timing = " << time + tt << endl;
                }

                //pos_now.x = KF.X[0];
                //pos_now.y = KF.X[1];
                //pos_now.z = KF.X[2];
                //pos_now.x = hx;
                //pos_now.y = KF.X[1];
                //pos_now.z = hz;
                //vy_now = KF.X[4];
                //cout << "2nd fly model by KF = " << KF.X[0] << ", " << KF.X[1] << ", " << KF.X[2] << endl;
                //cout << KF.X[0] << ", " << KF.X[1] << ", " << KF.X[2]<< " <-- 2nd fly model by KF"  << endl;
                time += delta_T;

/*
                cloud_KF.points[i].x = KF.X[0];
                cloud_KF.points[i].y = KF.X[1];
                cloud_KF.points[i].z = KF.X[2];
                pcl::toROSMsg(cloud_KF, output_KF);
                output_KF.header.frame_id = "map";
                KF_pub.publish(output_KF);
*/
              //}
            }

            //cout << hx << ", " << hy << ", " << hz << endl;
            cloud.points[i].x = hx;
            cloud.points[i].y = hy;
            cloud.points[i].z = hz;
            pcl::toROSMsg(cloud, output);
            output.header.frame_id = "map";
            pcl_pub.publish(output);
          }

          //cout << "correction position = [" << hx_W / 10 << "," << hy_W / 10 << "," << hz_W / 10 << "]" << endl;
          y2 = y1;
          i = i+1;
        }

      }

    }

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
