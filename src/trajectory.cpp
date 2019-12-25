#include <math.h>
#include <iostream>
#include <vector>
#include "Eigen/Eigen"
//#include <Eigen/eigen>
using namespace std;

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
};

class Reg_2nd //for trajectory 2nd regression by Least Square Method (LSM)
{
public:
	int nn = 10; // number of points to do regression
	int cnt; // push back point until cnt equal nn
	double root, root1, root2; // root1 and root2 are solution of z direction equation. root will equal root1 or root2 which at right hand side.
	vector <point> pointArray; // store input points
	double m[3][3] = { 0 }; // for Cramer's rule delta
	double m0[3][3], m1[3][3], m2[3][3]; // for Cramer's rule delta0, deta1 and delta2
	double v[3] = { 0 }; // store output points
	double det_m, det_m0, det_m1, det_m2;

	struct param paramer;
	double vel;

	void addPoint(double t_in, double p_in) {
		cnt += 1;
		struct point tmp;
		tmp.p = p_in;
		tmp.t = t_in;

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

int main()
{
	//double T[37] = { 0.02,0.04,0.06,0.08,0.1,0.12,0.14,0.16,0.18,0.2,0.22,0.24,0.26,0.28,0.3,0.32,0.34,0.36,0.38,0.4,0.42,0.44,0.46,0.48,0.5,0.52,0.54,0.56,0.58,0.6,0.62,0.64,0.66,0.68,0.7,0.72,0.74 };
	/* 50fps trajectory 1 */
	//double x[37] = { 76.012,76.4635,77.3149,77.9225,78.5032,79.2523,79.9433,80.6208,80.9351,81.3967,82.1633,82.5829,83.4009,83.6243,84.3198,84.8259,85.1707,85.7791,86.1378,86.7173,87.1616,87.8222,88.4844,88.6905,89.4689,89.8115,90.5321,90.9586,91.291,92.0108,92.5398,92.9491,93.5573,93.8638,94.2613,94.5931,95.1705 };
	//double y[37] = { 281.716,274,261.823,254.661,245.53,236.231,223.11,214.8140,208.3890,200.267,190.111,182.402,169.52,161.961,153.285,146.195,138.898,129.108,123.049,113.299,106.074,96.3075,87.2749,82.1069,73.3407,66.3015,56.1067,50.477,43.1776,34.2219,24.647,18.6609,9.48888,3.19528,3.0456,-9.27062,-17.7579 };
	//double z[37] = { 51.2488,52.2095,53.8564,54.3703,55.0211,54.4546,53.7271,53.2172,51.9337,50.5427,48.0232,46.1822,42.9557,40.0389,34.4766,31.693,28.0718,21.8586,17.5168,10.6226,5.91822,4.59611,11.2713,14.3604,19.9589,23.8486,28.721,31.0643,34.1226,36.8667,39.294,40.4087,41.633,42.1667,42.4861,42.4898 };
	/* 50fps trajectory 2 */
	//double x[40] = { 76.236,76.6858,77.3164,77.9222,78.496,78.873,79.5775,79.758,80.2574,81.06,81.531,82.2902,82.5669,82.8644,83.4636,84.1152,84.2712,84.6689,85.2614,85.5547,85.996,86.447,86.875,87.5222,87.8067,88.2561,88.6317,88.868,89.3948,89.7087,90.1695,90.4056,90.8964,91.2951,91.4889,91.932,92.2405,92.6056,92.7421,93.2956 };
	//double y[40] = { 279.053,268.99,261.566,252.09,240.635,235.685,222.504,219.516,211.183,198.686,192.504,182.036,175.606,170.647,163.125,151.657,147.578,141.232,131.002,123.429,114.847,108.411,102.828,93.0844,87.9754,79.0017,72.0433,65.4706,58.1241,51.9142,43.2666,37.076,29.8481,23.9237,18.1484,10.4595,4.57208,-3.21712,-7.34797,-15.9573 };
	//double z[40] = { 51.4468,52.7669,53.3101,54.0003,54.2962,53.2524,52.3992,50.4317,49.6723,47.2742,45.6716,41.9653,39.0328,35.519,30.0353,25.3761,20.232,16.3093,12.6462,10.8047,6.51029,3.56368,1.94435,6.65069,8.76613,12.0146,14.2016,16.1472,17.2047,18.0782,18.6925,19.1275,18.0029,17.5173,16.7414,14.5966,13.3023,10.3699,8.96656,4.08201 };
	/* 50fps trajectory 3 */
	//double x[36] = { 76.0218,76.2515,76.6954,77.5207,78.0988,78.4642,78.8217,79.1542,79.6538,79.9624,80.4048,80.8503,81.0965,81.5153,81.6159,82.0065,82.2264,82.6,82.6667,82.9173,83.2772,83.5981,83.9638,84.0849,84.4032,84.5891,84.5626,85.0246,85.281,85.4164,85.6321,85.8122,85.9347,86.1783,86.3849,86.3266 };
	//double y[36] = { 279.149,274.184,264.368,250.31,236.99,228.349,221.799,211.255,203.166,195.259,183.724,176.292,164.986,157.61,150.299,140.756,132.557,122.976,114.921,108.897,99.2879,92.0725,84.0223,76.9304,66.3345,59.5971,50.9656,42.388,32.9981,26.2003,19.6665,10.0738,3.47035,-5.78944,-12.1497,-19.4116 };
	//double z[36] = { 51.6333,52.5813,53.8566,55.2172,55.9688,55.8858,55.1728,53.8184,52.6965,51.3128,48.8231,47.0032,43.0593,40.1858,37.0041,31.7721,28.586,22.2446,18.4169,13.5685,6.09937,2.82507,9.01045,13.582,19.7006,23.3176,27.501,30.9996,34.3529,36.617,38.159,40.1393,41.2394,42.0466,42.2175,42.9739 };
	/* 50fps trajectory 4 */
	double x[34] = { 76.252,76.2682,76.9169,77.3253,77.5245,78.0851,78.4395,78.7641,78.9252,79.0678,79.5152,79.4997,79.7889,79.9199,80.3205,80.5739,80.6918,81.0759,81.2098,81.4462,81.5766,81.8463,82.1527,82.2663,82.6092,82.7677,83.2147,83.384,83.6503,83.8294,83.8134,84.153,84.2502,84.4525 };
	double y[34] = { 274.097,269.177,255.032,245.885,234.971,224.252,216.087,202.37,196.136,186.766,174.386,168.492,158.865,151.513,139.762,131.64,124.71,114.345,105.854,97.9086,91.9258,82.5249,74.68,65.4035,57.5051,48.3767,39.9324,33.5742,23.1098,16.4103,8.17426,0.611449,-5.83463,-15.4137 };
	double z[34] = { 52.3932,53.1304,55.0666,55.6995,56.275,56.0029,55.7617,54.8573,53.1367,51.6233,49.6352,47.1191,42.8138,39.592,35.2574,32.0324,28.2082,22.053,14.5727,10.0261,4.82118,4.60794,9.6381,15.2178,19.7585,24.3124,27.9679,30.5849,34.1575,36.0806,38.7691,39.3157,40.0654,40.8284 };

	double T = 0.02;
	double delta_T = 0.02;

	const int n = 10;

	Reg_2nd reg_x; // for X direction
	Reg_2nd reg_y; // for Y direction
	Reg_2nd reg_z; // for Z direction

	// curve fitting using 1st~10th
	for (int i = 0; i < n; i++) {
		reg_x.addPoint(T, x[i]);
		reg_y.addPoint(T, y[i]);
		reg_z.addPoint(T, z[i]);
		T += delta_T;
	}

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

	// show X Y Z equarion parameters
	cout << "x parameters = ";
	cout << reg_x.paramer.a << "," << reg_x.paramer.b << "," << reg_x.paramer.c << endl;
	cout << "y parameters = ";
	cout << reg_y.paramer.a << "," << reg_y.paramer.b << "," << reg_y.paramer.c << endl;
	cout << "z parameters = ";
	cout << reg_z.paramer.a << "," << reg_z.paramer.b << "," << reg_z.paramer.c << endl;

	/* Obtain landing time by curve fitting. Let eqation of z = 0, will obtain two solution, choose root which at right hand side */
// calculate prediction landing point and time
	bool sol = reg_z.calRoots();
	/*
	if (sol == true) {
		cout << "root1 = " << reg_z.root1 << endl;
		cout << "root2 = " << reg_z.root2 << endl;
	}
	else{
		cout << "not found solutions" << endl;
	}*/

	double landing_time = reg_z.root;
	cout << "landing time = " << landing_time << endl;
	cout << "numer of time = " << landing_time / delta_T << endl;

	double land_position_x, land_position_y, land_position_z;

	land_position_x = reg_x.getRegRes(landing_time);
	land_position_y = reg_y.getRegRes(landing_time);
	land_position_z = reg_z.getRegRes(landing_time);
	cout << "Landing position by regression = " << land_position_x << "," << land_position_y << ", " << land_position_z << endl;

	/*
	struct position ti, ti1;
	struct velocity vi, vi1;
	double Km = -0.12e-2;
	double Fgravity = 9.81 * 100; // 9.81 (cm/s^2)
	double Fmagnus_z = 550; // Fmagnus ~= 0.01N = 0.01 * 2.7e3 * 1e2 (cm/s^2) //472.5
	double Fmagnus_y = 550;
	*/

	struct position ti;
	struct velocity vi;

	// Input position and velocity
	ti.x = reg_x.pointArray[9].p;
	ti.y = reg_y.pointArray[9].p;
	ti.z = reg_z.pointArray[9].p;

	// calculate velocity by differencial
	//v0.vx = 2 * reg_x.paramer.a * T[8] + reg_x.paramer.b;
	//v0.vy = 2 * reg_y.paramer.a * T[8] + reg_y.paramer.b;
	//v0.vz = 2 * reg_z.paramer.a * T[8] + reg_z.paramer.b;

	// calculate veocity by (x10-x9)/t , unit(cm/s)
	vi.vx = (reg_x.pointArray[9].p - reg_x.pointArray[8].p) / delta_T;
	vi.vy = (reg_y.pointArray[9].p - reg_y.pointArray[8].p) / delta_T;
	vi.vz = (reg_z.pointArray[9].p - reg_z.pointArray[8].p) / delta_T;
	//cout << vi.vx << ", " << vi.vy << ", " << vi.vz << endl;

	/* fly model */
	/*
	int cnt = 13;
	while (ti.z >= 0) {
		ti1.x = ti.x + vi.vx * delta_T;
		ti1.y = ti.y + vi.vy * delta_T;
		ti1.z = ti.z + vi.vz * delta_T;
		//cout << cnt << "th. point position = " << endl;
		//cout << ti1.x << ", " << ti1.y << ", " << ti1.z << endl;
		cout << ti1.z << " ";

		vi1.vx = vi.vx + (Km * sqrt(pow(vi.vx, 2) + pow(vi.vy, 2) + pow(vi.vz, 2)) * vi.vx) * delta_T;
		vi1.vy = vi.vy + (Km * sqrt(pow(vi.vx, 2) + pow(vi.vy, 2) + pow(vi.vz, 2)) * vi.vy - Fmagnus_y) * delta_T;
		vi1.vz = vi.vz + (Km * sqrt(pow(vi.vx, 2) + pow(vi.vy, 2) + pow(vi.vz, 2)) * vi.vz - Fgravity - Fmagnus_z) * delta_T;

		//ti.x = ti1.x;
		cnt += 1;
		ti = ti1;
		vi = vi1;
	}
	*/
	/* Kalman Filter and fly model */

	KalmanFilterMagnus KF;
	// 1st fly model & KF
	int j = 10;
	double time = delta_T * n;
	double distance_y = 50;
	KF.X << ti.x, ti.y, ti.z, vi.vx, vi.vy, vi.vz;
	KF.dt = delta_T;
	//while (KF.X[1] > land_position_y && distance_y > 0) { // when prediction y of kalman not close enough prediction y of curve fitting
	while (time < (landing_time - delta_T)) { // when time now not close enough to landing time
		//cout << "time = " << time << endl;
		time += delta_T;

		// measurement position
		KF.Z << x[j], y[j], z[j];

		// prediction
		KF.kalmanfilter();
		//cout << KF.K << endl;

		cout << "1st fly model by KF = " << KF.X[0] << ", " << KF.X[1] << ", " << KF.X[2] << endl;
		//cout << KF.X[2] << " ";
		distance_y = KF.X[1] - land_position_y;
		//cout << j << endl;
		j += 1;
	}

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
	landing_pose.x = KF.X[0];
	landing_pose.y = KF.X[1];
	landing_pose.z = KF.X[2];
	struct param_rebound vin, vout;
	vin.x = KF.X[3];
	vin.y = KF.X[4];
	vin.z = KF.X[5];
	vout.x = Kr.x*vin.x + b.x;
	vout.y = Kr.y*vin.y + b.y;
	vout.z = Kr.z*vin.z + b.z;

	bool hit = false;

	cout << endl;
	// 2nd fly model & KF
	//KalmanFilter KF2nd;
	double strike_point = 0;
	struct param_rebound pos_now;
	pos_now.x = land_position_x;
	pos_now.y = land_position_y;
	pos_now.z = land_position_z;
	//KF.X << landing_pose.x, landing_pose.y, landing_pose.z, vout.x, vout.y, vout.z;
	KF.X << land_position_x, land_position_y, land_position_z, vout.x, vout.y, vout.z;

	KF.UpdateWeight(40, 1); //update(prediction, measurement) //(8,4)
	while (KF.X[1] > -8) {
		
		// measurement position
		KF.Z << x[j], y[j], z[j];
		KF.kalmanfilter();

		if (((pos_now.y + KF.X[1]) <= strike_point) && (hit == false)) {
			double tt = delta_T * (pos_now.y - strike_point) / (pos_now.y - KF.X[1]);
			cout << "found strike point: " << pos_now.x + KF.X[3] * tt << ", " << pos_now.y + KF.X[4] * tt << ", " << pos_now.z + KF.X[5] * tt << endl;
			hit = true;
			//cout << "position = " << KF.X[0] << ", " << KF.X[1] << ", " << KF.X[2] << endl;
			cout << "strike timing = " << time + delta_T / KF.X[1] << endl; //time + delta_T/KF.X[1]
			//cout << "position = " << KF.X[0] + KF.X[3] * delta_T / KF.X[1] << ", " << KF.X[1] + KF.X[4] * delta_T / KF.X[1] << ", " << KF.X[2] + KF.X[5] * delta_T / KF.X[1] << endl;
			//cout << "strike timing = " << time + tt << endl;
			//cout << "position = " << pos_now.x + KF.X[3] * tt << ", " << pos_now.y + KF.X[4] * tt << ", " << pos_now.z + KF.X[5] * tt << endl;
		}
		pos_now.x = KF.X[0];
		pos_now.y = KF.X[1];
		pos_now.z = KF.X[2];
		cout << "2nd fly model by KF = " << KF.X[0] << ", " << KF.X[1] << ", " << KF.X[2] << endl;
		//cout << "d = " << KF.X[4] * delta_T << endl;
		//cout << KF.X[2] << " ";
		//cout << j << endl;
		time += delta_T;
		j += 1;
	}

	return 0;
}


