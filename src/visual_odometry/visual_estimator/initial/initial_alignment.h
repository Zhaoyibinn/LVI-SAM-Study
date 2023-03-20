#pragma once
#include <eigen3/Eigen/Dense>
#include <iostream>
#include "../factor/imu_factor.h"
#include "../utility/utility.h"
#include <ros/ros.h>
#include <map>
#include "../feature_manager.h"

using namespace Eigen;
using namespace std;

class ImageFrame
{
    public:
        ImageFrame(){};
        ImageFrame(const map<int, vector<pair<int, Eigen::Matrix<double, 8, 1>>>>& _points, 
                   const vector<float> &_lidar_initialization_info,
                   double _t):
        t{_t}, is_key_frame{false}, reset_id{-1}, gravity{9.805}
        {
            points = _points;
            
            // reset id in case lidar odometry relocate
            reset_id = (int)round(_lidar_initialization_info[0]);
            // Pose
            T.x() = _lidar_initialization_info[1];
            T.y() = _lidar_initialization_info[2];
            T.z() = _lidar_initialization_info[3];
            // Rotation
            Eigen::Quaterniond Q = Eigen::Quaterniond(_lidar_initialization_info[7],
                                                      _lidar_initialization_info[4],
                                                      _lidar_initialization_info[5],
                                                      _lidar_initialization_info[6]);
            R = Q.normalized().toRotationMatrix();
            // Velocity
            V.x() = _lidar_initialization_info[8];
            V.y() = _lidar_initialization_info[9];
            V.z() = _lidar_initialization_info[10];
            // Acceleration bias
            Ba.x() = _lidar_initialization_info[11];
            Ba.y() = _lidar_initialization_info[12];
            Ba.z() = _lidar_initialization_info[13];
            // Gyroscope bias
            Bg.x() = _lidar_initialization_info[14];
            Bg.y() = _lidar_initialization_info[15];
            Bg.z() = _lidar_initialization_info[16];
            // Gravity
            gravity = _lidar_initialization_info[17];
        };

        map<int, vector<pair<int, Eigen::Matrix<double, 8, 1>> > > points;
        double t;
        
        IntegrationBase *pre_integration;
        bool is_key_frame;

        // Lidar odometry info
        int reset_id; // 雷达信息的reset_id
        Vector3d T; // 雷达信息的平移向量
        Matrix3d R; // 雷达信息的旋转矩阵
        Vector3d V; // 雷达信息的速度
        Vector3d Ba; // 雷达信息的加速度计bias
        Vector3d Bg; // 雷达信息的陀螺仪bias
        double gravity; // 雷达信息的重力加速度
};


bool VisualIMUAlignment(map<double, ImageFrame> &all_image_frame, Vector3d* Bgs, Vector3d &g, VectorXd &x);


class odometryRegister
{
public:

    ros::NodeHandle n;
    tf::Quaternion q_lidar_to_cam;
    Eigen::Quaterniond q_lidar_to_cam_eigen;

    ros::Publisher pub_latest_odometry; 

    odometryRegister(ros::NodeHandle n_in):
    // n(n_in)
    // {
    //     q_lidar_to_cam = tf::Quaternion(0, 1, 0, 0); // rotate orientation // mark: camera - lidar
    //     q_lidar_to_cam_eigen = Eigen::Quaterniond(0, 0, 0, 1); // rotate position by pi, (w, x, y, z) // mark: camera - lidar
    //     // pub_latest_odometry = n.advertise<nav_msgs::Odometry>("odometry/test", 1000);
    // }
    n(n_in)
    {
        q_lidar_to_cam = tf::Quaternion(0, 0, 0, 1);
        q_lidar_to_cam_eigen = Eigen::Quaterniond(1,0,0,0); 
    }

    // convert odometry from ROS Lidar frame to VINS camera frame
    //获取与图片特征时间戳最近的LiDAR位姿并转换到相机坐标系下（odom-->LiDAR-->camera），将LiDAR的信息（除位姿以外还包含id(1), P(3), Q(4), V(3), Ba(3), Bg(3), gravity(1)信息）转换成数组格式返回。
    //值得注意的是，此处的LiDAR位姿实际上是由LIS子系统的imu预积分得来的，因此频率很高，为500Hz，因此正常运行的情况下能够获取十分相近的位姿信息。
    vector<float> getOdometry(deque<nav_msgs::Odometry>& odomQueue, double img_time)
    {
        vector<float> odometry_channel;
        odometry_channel.resize(18, -1); // reset id(1), P(3), Q(4), V(3), Ba(3), Bg(3), gravity(1)

        nav_msgs::Odometry odomCur;
        
        // pop old odometry msg
        while (!odomQueue.empty()) 
        {
            if (odomQueue.front().header.stamp.toSec() < img_time - 0.05)
                odomQueue.pop_front();
            else
                break;
        }

        if (odomQueue.empty())
        {
            return odometry_channel;
        }

        // find the odometry time that is the closest to image time
        for (int i = 0; i < (int)odomQueue.size(); ++i)
        {
            odomCur = odomQueue[i];

            if (odomCur.header.stamp.toSec() < img_time - 0.002) // 500Hz imu
                continue;
            else
                break;
        }

        // time stamp difference still too large
        if (abs(odomCur.header.stamp.toSec() - img_time) > 0.05)
        {
            return odometry_channel;
        }

        // convert odometry rotation from lidar ROS frame to VINS camera frame (only rotation, assume lidar, camera, and IMU are close enough)  将里程计旋转信息从雷达系转到vins相机系
        tf::Quaternion q_odom_lidar;
        tf::quaternionMsgToTF(odomCur.pose.pose.orientation, q_odom_lidar);

        // tf::Quaternion q_odom_cam = tf::createQuaternionFromRPY(0, 0, M_PI) * (q_odom_lidar * q_lidar_to_cam); // global rotate by pi // mark: camera - lidar
        tf::Quaternion q_odom_cam = tf::createQuaternionFromRPY(0, 0, M_PI / 2.0) * (q_odom_lidar * q_lidar_to_cam);
        tf::quaternionTFToMsg(q_odom_cam, odomCur.pose.pose.orientation);

        // convert odometry position from lidar ROS frame to VINS camera frame  将里程计位置和速度信息 从雷达系转到vins相机系
        Eigen::Vector3d p_eigen(odomCur.pose.pose.position.x, odomCur.pose.pose.position.y, odomCur.pose.pose.position.z);
        Eigen::Vector3d v_eigen(odomCur.twist.twist.linear.x, odomCur.twist.twist.linear.y, odomCur.twist.twist.linear.z);
        Eigen::Vector3d p_eigen_new = q_lidar_to_cam_eigen * p_eigen;
        Eigen::Vector3d v_eigen_new = q_lidar_to_cam_eigen * v_eigen;

        odomCur.pose.pose.position.x = p_eigen_new.x();
        odomCur.pose.pose.position.y = p_eigen_new.y();
        odomCur.pose.pose.position.z = p_eigen_new.z();

        odomCur.twist.twist.linear.x = v_eigen_new.x();
        odomCur.twist.twist.linear.y = v_eigen_new.y();
        odomCur.twist.twist.linear.z = v_eigen_new.z();

        // odomCur.header.stamp = ros::Time().fromSec(img_time);
        // odomCur.header.frame_id = "vins_world";
        // odomCur.child_frame_id = "vins_body";
        // pub_latest_odometry.publish(odomCur);

        //将转换后的信息存储到odometry_channel
        odometry_channel[0] = odomCur.pose.covariance[0];
        odometry_channel[1] = odomCur.pose.pose.position.x;
        odometry_channel[2] = odomCur.pose.pose.position.y;
        odometry_channel[3] = odomCur.pose.pose.position.z;
        odometry_channel[4] = odomCur.pose.pose.orientation.x;
        odometry_channel[5] = odomCur.pose.pose.orientation.y;
        odometry_channel[6] = odomCur.pose.pose.orientation.z;
        odometry_channel[7] = odomCur.pose.pose.orientation.w;
        odometry_channel[8]  = odomCur.twist.twist.linear.x;
        odometry_channel[9]  = odomCur.twist.twist.linear.y;
        odometry_channel[10] = odomCur.twist.twist.linear.z;
        odometry_channel[11] = odomCur.pose.covariance[1];
        odometry_channel[12] = odomCur.pose.covariance[2];
        odometry_channel[13] = odomCur.pose.covariance[3];
        odometry_channel[14] = odomCur.pose.covariance[4];
        odometry_channel[15] = odomCur.pose.covariance[5];
        odometry_channel[16] = odomCur.pose.covariance[6];
        odometry_channel[17] = odomCur.pose.covariance[7];

        return odometry_channel;
    }
};