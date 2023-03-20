#ifndef FEATURE_MANAGER_H
#define FEATURE_MANAGER_H

#include <list>
#include <algorithm>
#include <vector>
#include <numeric>
using namespace std;

#include <eigen3/Eigen/Dense>
using namespace Eigen;

#include <ros/console.h>
#include <ros/assert.h>

#include "parameters.h"

class FeaturePerFrame
{
  public:
    FeaturePerFrame(const Eigen::Matrix<double, 8, 1> &_point, double td)
    {
        point.x() = _point(0);
        point.y() = _point(1);
        point.z() = _point(2);
        uv.x() = _point(3);
        uv.y() = _point(4);
        velocity.x() = _point(5); 
        velocity.y() = _point(6); 
        depth = _point(7);
        cur_td = td;
    }
    double cur_td; // 时间偏移
    Vector3d point; // 归一化坐标
    Vector2d uv; // 像素坐标
    Vector2d velocity; // 像素速度
    double z;
    bool is_used;
    double parallax; // 视差
    MatrixXd A;
    VectorXd b;
    double dep_gradient;
    double depth; // lidar depth, initialized with -1 from feature points in feature tracker node  雷达深度
};

class FeaturePerId
{
  public:
    const int feature_id; // 特征点ID
    int start_frame; // 第一次观察到本特征点的ID
    vector<FeaturePerFrame> feature_per_frame; // 本特征点的所有图片观测帧

    int used_num; // 本特征点总计观测次数
    bool is_outlier;
    bool is_margin;
    double estimated_depth; // 特征点逆深度值，用于优化
    bool lidar_depth_flag; // 是否有雷达辅助深度
    int solve_flag; // 0 haven't solve yet; 1 solve succ; 2 solve fail;

    Vector3d gt_p;

    FeaturePerId(int _feature_id, int _start_frame, double _measured_depth)
        : feature_id(_feature_id), start_frame(_start_frame),
          used_num(0), estimated_depth(-1.0), lidar_depth_flag(false), solve_flag(0) 
    {
        if (_measured_depth > 0)
        {
            estimated_depth = _measured_depth;
            lidar_depth_flag = true;
        }
        else
        {
            estimated_depth = -1;
            lidar_depth_flag = false;
        }
    }

    int endFrame();
};

class FeatureManager
{
  public:
    FeatureManager(Matrix3d _Rs[]);

    void setRic(Matrix3d _ric[]);

    void clearState();

    int getFeatureCount();

    bool addFeatureCheckParallax(int frame_count, const map<int, vector<pair<int, Eigen::Matrix<double, 8, 1>>>> &image, double td);
    void debugShow();
    vector<pair<Vector3d, Vector3d>> getCorresponding(int frame_count_l, int frame_count_r);

    //void updateDepth(const VectorXd &x);
    void setDepth(const VectorXd &x);
    void removeFailures();
    void clearDepth(const VectorXd &x);
    VectorXd getDepthVector();
    void triangulate(Vector3d Ps[], Vector3d tic[], Matrix3d ric[]);
    void removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, Eigen::Matrix3d new_R, Eigen::Vector3d new_P);
    void removeBack();
    void removeFront(int frame_count);
    void removeOutlier();
    list<FeaturePerId> feature; // 储存所有特征点的链表
    int last_track_num; // 跟踪成功的特征点数量

  private:
    double compensatedParallax2(const FeaturePerId &it_per_id, int frame_count);
    const Matrix3d *Rs; // 旋转矩阵数组
    Matrix3d ric[NUM_OF_CAM]; //相机IMU外参数组
};

#endif