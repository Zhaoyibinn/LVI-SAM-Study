#include <stdio.h>
#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include "estimator.h"
#include "parameters.h"
#include "utility/visualization.h"


Estimator estimator;

std::condition_variable con;
double current_time = -1;
queue<sensor_msgs::ImuConstPtr> imu_buf;
queue<sensor_msgs::PointCloudConstPtr> feature_buf;

// global variable saving the lidar odometry
deque<nav_msgs::Odometry> odomQueue;
odometryRegister *odomRegister;

std::mutex m_buf;
std::mutex m_state;
std::mutex m_estimator;
std::mutex m_odom;

double latest_time;
Eigen::Vector3d tmp_P;
Eigen::Quaterniond tmp_Q;
Eigen::Vector3d tmp_V;
Eigen::Vector3d tmp_Ba;
Eigen::Vector3d tmp_Bg;
Eigen::Vector3d acc_0;
Eigen::Vector3d gyr_0;
bool init_feature = 0;
bool init_imu = 1;
double last_imu_t = 0;

// 从当前的imu测量值和上一时刻的PQV进行递推得到当前时刻的PQV
void predict(const sensor_msgs::ImuConstPtr &imu_msg)
{
    double t = imu_msg->header.stamp.toSec();
    if (init_imu)
    {
        latest_time = t;
        init_imu = 0;
        return;
    }
    double dt = t - latest_time;// 当前imu_msg 距离上一个时刻之间的时间间隔
    latest_time = t;

    double dx = imu_msg->linear_acceleration.x;
    double dy = imu_msg->linear_acceleration.y;
    double dz = imu_msg->linear_acceleration.z;
    Eigen::Vector3d linear_acceleration{dx, dy, dz};

    double rx = imu_msg->angular_velocity.x;
    double ry = imu_msg->angular_velocity.y;
    double rz = imu_msg->angular_velocity.z;
    Eigen::Vector3d angular_velocity{rx, ry, rz};

    Eigen::Vector3d un_acc_0 = tmp_Q * (acc_0 - tmp_Ba) - estimator.g;// 采用最新bias重新计算上一帧加速度  acc_0为机体坐标系，un_acc_0为世界坐标系，所以要乘tmp_Q转换坐标系

    Eigen::Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - tmp_Bg;// 计算角速度中值
    tmp_Q = tmp_Q * Utility::deltaQ(un_gyr * dt);// 更新imu旋转

    Eigen::Vector3d un_acc_1 = tmp_Q * (linear_acceleration - tmp_Ba) - estimator.g;// 计算当前帧加速度

    Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);// 计算加速度中值

    tmp_P = tmp_P + dt * tmp_V + 0.5 * dt * dt * un_acc;
    tmp_V = tmp_V + dt * un_acc;

    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
}

void update()
{
    TicToc t_predict;
    latest_time = current_time;
    tmp_P = estimator.Ps[WINDOW_SIZE];
    tmp_Q = estimator.Rs[WINDOW_SIZE];
    tmp_V = estimator.Vs[WINDOW_SIZE];
    tmp_Ba = estimator.Bas[WINDOW_SIZE];
    tmp_Bg = estimator.Bgs[WINDOW_SIZE];
    acc_0 = estimator.acc_0;
    gyr_0 = estimator.gyr_0;

    queue<sensor_msgs::ImuConstPtr> tmp_imu_buf = imu_buf;
    for (sensor_msgs::ImuConstPtr tmp_imu_msg; !tmp_imu_buf.empty(); tmp_imu_buf.pop())
        predict(tmp_imu_buf.front());
}

// 对imu数据和图像数据进行时间对齐
std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>>
getMeasurements()
{
    std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> measurements;

    while (ros::ok())
    {
        if (imu_buf.empty() || feature_buf.empty())
            return measurements;

        // imu 太慢，等imu数据
        // 如果最晚的一帧imu测量仍然早于最早的图片特征，说明imu测量队列中的所有帧均早于图片特征队列中的所有帧
        if (!(imu_buf.back()->header.stamp.toSec() > feature_buf.front()->header.stamp.toSec() + estimator.td))
        {
            return measurements;
        }

        // 图像太老，扔掉图像
        // 如果最早一帧的imu测量仍然晚于最早的图片特征，说明imu测量队里中的所有帧均晚于最早的图片特征
        if (!(imu_buf.front()->header.stamp.toSec() < feature_buf.front()->header.stamp.toSec() + estimator.td))
        {
            ROS_WARN("throw img, only should happen at the beginning");
            feature_buf.pop();
            continue;
        }
        // 排除上述情况后，现在的情况是：最早一帧的imu测量早于最早的图片特征，最晚一帧的imu测量晚于最早的图片特征
        // 提取早于最早图片特征的imu测量和第一个晚于图片特征的imu测量，组成数组
        sensor_msgs::PointCloudConstPtr img_msg = feature_buf.front();
        feature_buf.pop();

        std::vector<sensor_msgs::ImuConstPtr> IMUs;
        while (imu_buf.front()->header.stamp.toSec() < img_msg->header.stamp.toSec() + estimator.td)
        {
            IMUs.emplace_back(imu_buf.front());
            imu_buf.pop();
        }
        IMUs.emplace_back(imu_buf.front());
        if (IMUs.empty())
            ROS_WARN("no imu between two image");
        measurements.emplace_back(IMUs, img_msg);
    }
    return measurements;
}

//imu回调函数，将imu_msg 保存到 imu_buf，并且将IMU进行状态递推并发布。获取IMU原始信息，中值积分算位姿，用于计算下一帧图像未到来之前的高频位姿，包含predict()和pubLatestOdometry()两个关键函数。
void imu_callback(const sensor_msgs::ImuConstPtr &imu_msg)
{
    // 对imu数据进行检查（imu数据应该按时间序列的顺序被订阅）
    if (imu_msg->header.stamp.toSec() <= last_imu_t)
    {
        ROS_WARN("imu message in disorder!");
        return;
    }
    last_imu_t = imu_msg->header.stamp.toSec();

    m_buf.lock();
    imu_buf.push(imu_msg);
    m_buf.unlock();
    con.notify_one();// 唤醒条件变量con阻塞线程，继续执行process

    last_imu_t = imu_msg->header.stamp.toSec();

    {
        std::lock_guard<std::mutex> lg(m_state);
        predict(imu_msg);// 递推得到IMU的当前时刻的PQV（初值）
        std_msgs::Header header = imu_msg->header;
        // 如果进入了非线性优化阶段，即稳定的VIO过程，会发布最新的里程计(和IMU同频率)  发布最新的由IMU直接递推得到的PQV （用于RVIZ可视化）
        if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
            pubLatestOdometry(tmp_P, tmp_Q, tmp_V, header, estimator.failureCount);
    }
}

void odom_callback(const nav_msgs::Odometry::ConstPtr& odom_msg)
{
    m_odom.lock();
    odomQueue.push_back(*odom_msg);
    m_odom.unlock();
}

void feature_callback(const sensor_msgs::PointCloudConstPtr &feature_msg)
{
    if (!init_feature)
    {
        //skip the first detected feature, which doesn't contain optical flow speed
        init_feature = 1;
        return;
    }
    m_buf.lock();
    feature_buf.push(feature_msg);
    m_buf.unlock();
    con.notify_one();
}

void restart_callback(const std_msgs::BoolConstPtr &restart_msg)
{
    if (restart_msg->data == true)
    {
        ROS_WARN("restart the estimator!");
        m_buf.lock();
        while(!feature_buf.empty())
            feature_buf.pop();
        while(!imu_buf.empty())
            imu_buf.pop();
        m_buf.unlock();
        m_estimator.lock();
        estimator.clearState();
        estimator.setParameter();
        m_estimator.unlock();
        current_time = -1;
        last_imu_t = 0;
    }
    return;
}

// thread: visual-inertial odometry  处理IMU，odom，feature信息，完成VIO融合估计位姿。
void process()
{
    while (ros::ok())
    {
        // vector<pair<imu测量的vector，图片特征>>
        std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> measurements;
        std::unique_lock<std::mutex> lk(m_buf);
        con.wait(lk, [&]
                 {
            return (measurements = getMeasurements()).size() != 0;
                 });
        lk.unlock();

        m_estimator.lock();
        for (auto &measurement : measurements)
        {
            auto img_msg = measurement.second;

            // 1. IMU pre-integration  进行 IMU 预积分
            double dx = 0, dy = 0, dz = 0, rx = 0, ry = 0, rz = 0;
            for (auto &imu_msg : measurement.first)
            {
                double t = imu_msg->header.stamp.toSec();
                double img_t = img_msg->header.stamp.toSec() + estimator.td;
                // 对于早于图片特征的imu测量
                if (t <= img_t)
                { 
                    if (current_time < 0)
                        current_time = t;
                    double dt = t - current_time;
                    ROS_ASSERT(dt >= 0);
                    current_time = t;
                    dx = imu_msg->linear_acceleration.x;
                    dy = imu_msg->linear_acceleration.y;
                    dz = imu_msg->linear_acceleration.z;
                    rx = imu_msg->angular_velocity.x;
                    ry = imu_msg->angular_velocity.y;
                    rz = imu_msg->angular_velocity.z;
                    estimator.processIMU(dt, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));// 对imu数据进行预积分 并利用imu对系统最新状态进行传播，为视觉三角化及重投影提供位姿初值；
                    //printf("imu: dt:%f a: %f %f %f w: %f %f %f\n",dt, dx, dy, dz, rx, ry, rz);
                }
                // 对于晚于图片特征的imu测量（只有一个）
                else
                {
                    double dt_1 = img_t - current_time;
                    double dt_2 = t - img_t;
                    current_time = img_t;
                    ROS_ASSERT(dt_1 >= 0);
                    ROS_ASSERT(dt_2 >= 0);
                    ROS_ASSERT(dt_1 + dt_2 > 0);
                    // 计算权重值
                    double w1 = dt_2 / (dt_1 + dt_2);
                    double w2 = dt_1 / (dt_1 + dt_2);
                    // 按照权重计算加速度和角速度，插值计算图片特征时间戳对应的imu测量
                    dx = w1 * dx + w2 * imu_msg->linear_acceleration.x;
                    dy = w1 * dy + w2 * imu_msg->linear_acceleration.y;
                    dz = w1 * dz + w2 * imu_msg->linear_acceleration.z;
                    rx = w1 * rx + w2 * imu_msg->angular_velocity.x;
                    ry = w1 * ry + w2 * imu_msg->angular_velocity.y;
                    rz = w1 * rz + w2 * imu_msg->angular_velocity.z;
                    estimator.processIMU(dt_1, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz)); // 预积分计算状态[P,V,Q]
                    //printf("dimu: dt:%f a: %f %f %f w: %f %f %f\n",dt_1, dx, dy, dz, rx, ry, rz);
                }
            }

            // 2. VINS Optimization
            // TicToc t_s;
            // 图像特征数据不是opencv，而是包含许多通道的msg
            // 每帧图像特征的特征信息，由一个map的数据结构组成；
            // 键为关键点ID，值为该特征每个相机中的x,y,z,u,v,velocity_x,velocity_y,depth 8个变量;
            // map< 关键点ID, vector< pair<相机ID, 八维向量(x,y,z,u,v,vel_x,vel_y,depth) >>>
            map<int, vector<pair<int, Eigen::Matrix<double, 8, 1>>>> image;
            for (unsigned int i = 0; i < img_msg->points.size(); i++)
            {
                //以下数值来源于feature_tracker_node.cpp的图像回调函数的feature_points，最后都存到了下面的image里
                int v = img_msg->channels[0].values[i] + 0.5; //关键点id
                int feature_id = v / NUM_OF_CAM;
                int camera_id = v % NUM_OF_CAM;
                double x = img_msg->points[i].x; // 归一化平面坐标
                double y = img_msg->points[i].y; // 归一化平面坐标
                double z = img_msg->points[i].z; // 归一化平面坐标
                double p_u = img_msg->channels[1].values[i]; // 像素坐标
                double p_v = img_msg->channels[2].values[i]; // 像素坐标
                double velocity_x = img_msg->channels[3].values[i]; // 速度
                double velocity_y = img_msg->channels[4].values[i]; // 速度
                double depth = img_msg->channels[5].values[i]; // 关键点深度值

                ROS_ASSERT(z == 1);
                Eigen::Matrix<double, 8, 1> xyz_uv_velocity_depth;
                xyz_uv_velocity_depth << x, y, z, p_u, p_v, velocity_x, velocity_y, depth;
                image[feature_id].emplace_back(camera_id,  xyz_uv_velocity_depth);
            }

            // Get initialization info from lidar odometry  初始化信息的获得 通过LIS的预积分模块得来
            vector<float> initialization_info; // 雷达里程计位姿，来源于LIS子系统的imu预积分:包括 id(1), P(3), Q(4), V(3), Ba(3), Bg(3), gravity(1)
            m_odom.lock();
            initialization_info = odomRegister->getOdometry(odomQueue, img_msg->header.stamp.toSec() + estimator.td); // odomQueue来源于imuPreintegration.cpp的pubImuOdometry，对里面的协方差赋了自己的数值
            m_odom.unlock();


            estimator.processImage(image, initialization_info, img_msg->header);
            // double whole_t = t_s.toc();
            // printStatistics(estimator, whole_t);

            // 3. Visualization
            std_msgs::Header header = img_msg->header;
            pubOdometry(estimator, header);
            pubKeyPoses(estimator, header);
            pubCameraPose(estimator, header);
            pubPointCloud(estimator, header);
            pubTF(estimator, header);
            pubKeyframe(estimator);
        }
        m_estimator.unlock();

        m_buf.lock();
        m_state.lock();
        if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
            update();
        m_state.unlock();
        m_buf.unlock();
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "vins");
    ros::NodeHandle n;
    ROS_INFO("\033[1;32m----> Visual Odometry Estimator Started.\033[0m");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Warn);

    readParameters(n);
    estimator.setParameter();

    registerPub(n);// 注册各种话题发布器

    odomRegister = new odometryRegister(n);

    ros::Subscriber sub_imu     = n.subscribe(IMU_TOPIC,      5000, imu_callback,  ros::TransportHints().tcpNoDelay());
    ros::Subscriber sub_odom    = n.subscribe("odometry/imu", 5000, odom_callback);
    ros::Subscriber sub_image   = n.subscribe(PROJECT_NAME + "/vins/feature/feature", 1, feature_callback);
    ros::Subscriber sub_restart = n.subscribe(PROJECT_NAME + "/vins/feature/restart", 1, restart_callback);
    if (!USE_LIDAR)
        sub_odom.shutdown();

    std::thread measurement_process{process};

    ros::MultiThreadedSpinner spinner(4);
    spinner.spin();

    return 0;
}