#include "utility.h"
#include "lvi_sam/cloud_info.h"

// Velodyne
struct PointXYZIRT
{
    PCL_ADD_POINT4D // x、y、z和一个对齐变量
    PCL_ADD_INTENSITY;
    uint16_t ring; // 点云所在的圈数（因为雷达是一圈一圈扫描的）
    float time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW // 保证在内存中是对齐的
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT (PointXYZIRT,  
    (float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity)
    (uint16_t, ring, ring) (float, time, time)
)

// Ouster
// struct PointXYZIRT {
//     PCL_ADD_POINT4D;
//     float intensity;
//     uint32_t t;
//     uint16_t reflectivity;
//     uint8_t ring;
//     uint16_t noise;
//     uint32_t range;
//     EIGEN_MAKE_ALIGNED_OPERATOR_NEW
// }EIGEN_ALIGN16;

// POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZIRT,
//     (float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity)
//     (uint32_t, t, t) (uint16_t, reflectivity, reflectivity)
//     (uint8_t, ring, ring) (uint16_t, noise, noise) (uint32_t, range, range)
// )

const int queueLength = 500;

class ImageProjection : public ParamServer
{
private:

    std::mutex imuLock;
    std::mutex odoLock;

    ros::Subscriber subLaserCloud;
    ros::Publisher  pubLaserCloud;
    
    ros::Publisher pubExtractedCloud;
    ros::Publisher pubLaserCloudInfo;

    ros::Subscriber subImu;
    std::deque<sensor_msgs::Imu> imuQueue;

    ros::Subscriber subOdom;
    std::deque<nav_msgs::Odometry> odomQueue;

    std::deque<sensor_msgs::PointCloud2> cloudQueue;
    sensor_msgs::PointCloud2 currentCloudMsg;
    
    double *imuTime = new double[queueLength]; // imu时间戳数组
    double *imuRotX = new double[queueLength]; // imu欧拉角数组
    double *imuRotY = new double[queueLength]; // imu欧拉角数组
    double *imuRotZ = new double[queueLength]; // imu欧拉角数组

    int imuPointerCur; // imu当前帧的数组下标
    bool firstPointFlag; // 第一帧点云标志位
    Eigen::Affine3f transStartInverse; // 上一帧时的位姿的逆

    pcl::PointCloud<PointXYZIRT>::Ptr laserCloudIn; // 转换为pcl格式的原始点云
    pcl::PointCloud<PointType>::Ptr   fullCloud; // 去畸变后的有序的全部点云
    pcl::PointCloud<PointType>::Ptr   extractedCloud; // 去畸变的、有序的、可计算曲率的点云

    int deskewFlag; // 去畸变标志位
    cv::Mat rangeMat; // Mat格式存储点云投影  保存的是点的距离，行代表第几个扫描线，列代表的是这个扫描线的第几个扫描点

    bool odomDeskewFlag; // 获取帧间位姿变换标志位
    float odomIncreX; // 位姿增量
    float odomIncreY; // 位姿增量
    float odomIncreZ; // 位姿增量

    lvi_sam::cloud_info cloudInfo; // 自定义点云格式
    double timeScanCur; // 当前帧点云时间戳
    double timeScanNext; // 下一帧点云时间戳
    std_msgs::Header cloudHeader; // 点云标头


public:
    ImageProjection():
    deskewFlag(0)
    {
        // 订阅imu原始数据，回调函数负责将测量信息坐标变换到激光雷达并存储到队列
        subImu        = nh.subscribe<sensor_msgs::Imu>        (imuTopic, 2000, &ImageProjection::imuHandler, this, ros::TransportHints().tcpNoDelay());
        // 订阅Odom原始数据，此Odom来自于VIS，可能早于点云采集时间，也可能稍晚于点云采集时间，回调函数负责将位姿信息存放到队列
        subOdom       = nh.subscribe<nav_msgs::Odometry>      (PROJECT_NAME + "/vins/odometry/imu_propagate_ros", 2000, &ImageProjection::odometryHandler, this, ros::TransportHints().tcpNoDelay());
        // 订阅原始点云，回调函数负责检查、获取对齐的imu帧旋转矩阵、获取相邻帧的位姿变换、点云投影成图片从而使点云有序化，最后把去畸变有序化点云和自定义格式点云发布出去
        subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>(pointCloudTopic, 5, &ImageProjection::cloudHandler, this, ros::TransportHints().tcpNoDelay());

        pubExtractedCloud = nh.advertise<sensor_msgs::PointCloud2> (PROJECT_NAME + "/lidar/deskew/cloud_deskewed", 5);
        pubLaserCloudInfo = nh.advertise<lvi_sam::cloud_info>      (PROJECT_NAME + "/lidar/deskew/cloud_info", 5);

        // 为下一次处理分配内存空间并重置所有参数
        allocateMemory();
        resetParameters();

        pcl::console::setVerbosityLevel(pcl::console::L_ERROR); // 设置控制台输出信息（等级）
    }

    void allocateMemory()
    {
        laserCloudIn.reset(new pcl::PointCloud<PointXYZIRT>());
        fullCloud.reset(new pcl::PointCloud<PointType>());
        extractedCloud.reset(new pcl::PointCloud<PointType>());

        fullCloud->points.resize(N_SCAN*Horizon_SCAN);

        cloudInfo.startRingIndex.assign(N_SCAN, 0);
        cloudInfo.endRingIndex.assign(N_SCAN, 0);

        cloudInfo.pointColInd.assign(N_SCAN*Horizon_SCAN, 0);
        cloudInfo.pointRange.assign(N_SCAN*Horizon_SCAN, 0);

        resetParameters();
    }

    void resetParameters()
    {
        laserCloudIn->clear();
        extractedCloud->clear();
        // reset range matrix for range image projection
        rangeMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32F, cv::Scalar::all(FLT_MAX));

        imuPointerCur = 0;
        firstPointFlag = true;
        odomDeskewFlag = false;

        for (int i = 0; i < queueLength; ++i)
        {
            imuTime[i] = 0;
            imuRotX[i] = 0;
            imuRotY[i] = 0;
            imuRotZ[i] = 0;
        }
    }

    ~ImageProjection(){}

    void imuHandler(const sensor_msgs::Imu::ConstPtr& imuMsg)
    {
        sensor_msgs::Imu thisImu = imuConverter(*imuMsg); // 作用是把imu数据转换到lidar坐标系
        std::lock_guard<std::mutex> lock1(imuLock);
        imuQueue.push_back(thisImu);
    }

    void odometryHandler(const nav_msgs::Odometry::ConstPtr& odometryMsg)
    {
        std::lock_guard<std::mutex> lock2(odoLock);
        odomQueue.push_back(*odometryMsg);
    }

    void cloudHandler(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg)
    {
        // 添加一帧激光点云到队列，取最早一帧作为当前帧，计算起止时间戳，检查数据有效性
        if (!cachePointCloud(laserCloudMsg))
            return;

        // 当前帧起止时刻对应的imu数据，imu里程计数据处理
        if (!deskewInfo())
            return;

        // 当前帧激光点云运动畸变矫正
        projectPointCloud();

        // 提取有效激光点，存extractedCloud
        cloudExtraction();

        // 发布当前帧矫正后的点云，有效点
        publishClouds();

        // 重置参数，接收每帧lidar数据都要重置这些参数
        resetParameters();
    }

    // 添加一帧激光点云到队列，取最早一帧作为当前帧，计算起止时间戳，检查数据有效性
    bool cachePointCloud(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg)
    {
        // cache point cloud
        cloudQueue.push_back(*laserCloudMsg);

        if (cloudQueue.size() <= 2)
            return false;
        else
        {
            currentCloudMsg = cloudQueue.front();
            cloudQueue.pop_front();

            cloudHeader = currentCloudMsg.header;
            timeScanCur = cloudHeader.stamp.toSec();
            timeScanNext = cloudQueue.front().header.stamp.toSec();
        }

        // convert cloud类型
        pcl::fromROSMsg(currentCloudMsg, *laserCloudIn);

        // check dense flag  存在无效点，Nan或Inf
        if (laserCloudIn->is_dense == false)
        {
            ROS_ERROR("Point cloud is not in dense format, please remove NaN points first!");
            ros::shutdown();
        }

        // check ring channel 检查是否存在ring通道，注意static只检查一次
        static int ringFlag = 0;
        if (ringFlag == 0)
        {
            ringFlag = -1;
            for (int i = 0; i < (int)currentCloudMsg.fields.size(); ++i)
            {
                if (currentCloudMsg.fields[i].name == "ring") // ring代表点是第几束光发出的
                {
                    ringFlag = 1;
                    break;
                }
            }
            if (ringFlag == -1)
            {
                ROS_ERROR("Point cloud ring channel not available, please configure your point cloud data!");
                ros::shutdown();
            }
        }     

        // check point time 检查是否存在time通道 因为去畸变需要知道时间
        if (deskewFlag == 0)
        {
            deskewFlag = -1;
            for (int i = 0; i < (int)currentCloudMsg.fields.size(); ++i)
            {
                if (currentCloudMsg.fields[i].name == timeField)
                {
                    deskewFlag = 1;
                    break;
                }
            }
            if (deskewFlag == -1)
                ROS_WARN("Point cloud timestamp not available, deskew function disabled, system will drift significantly!");
        }

        return true;
    }

    // 处理当前激光帧起止时刻对应的IMU数据、IMU里程计数据。根据IMU或vins的odom信息计算相对扫描开始时刻的位姿变换
    bool deskewInfo()
    {
        // 这两个锁使得imuHandler和odometryHandler函数的传感器信息不push到队列内
        std::lock_guard<std::mutex> lock1(imuLock);
        std::lock_guard<std::mutex> lock2(odoLock);

        // make sure IMU data available for the scan 要求imu数据包含激光数据，否则不往下处理
        if (imuQueue.empty() || imuQueue.front().header.stamp.toSec() > timeScanCur || imuQueue.back().header.stamp.toSec() < timeScanNext)
        {
            ROS_DEBUG("Waiting for IMU data ...");
            return false;
        }

        imuDeskewInfo();

        odomDeskewInfo();

        return true;
    }

    //当前帧对应imu数据处理
    //1.遍历当前激光帧起止时刻之间的imu数据，初始时刻对应的imu姿态角RPY设为当前帧的初始姿态角
    //2.用角速度、时间积分，计算每一时刻相对于前一时刻旋转量，初始时刻旋转设为0
    //注:imu数据都已经转换到lidar坐标系下了
    void imuDeskewInfo()
    {



//从imu队列中删除当前激光帧0.01s前面时刻的imu数据，避免不必要的计算
// 如果imu队列为空，说明没有可用的IMU数据，直接返回
// 初始化一个指针imuPointerCur，用于遍历imu队列
// 对于每一个IMU数据，获取其时间戳和姿态角（roll, pitch, yaw）
// 如果当前IMU时间小于等于当前激光帧时间，就将其姿态角作为当前激光帧的初始姿态角
// 如果当前IMU时间大于下一帧激光帧时间加上0.01s，就跳出循环
// 如果是第一帧IMU数据，就将其旋转角初始化为0，并记录其时间戳
// 如果不是第一帧IMU数据，就根据前一帧IMU数据和当前IMU数据之间的角速度和时间差来积分得到当前旋转角，并记录其时间戳
// 最后将指针减一，并判断是否有有效的IMU数据
        cloudInfo.imuAvailable = false;

        while (!imuQueue.empty())
        {
            // 从imu队列中删除当前激光帧0.01s前面时刻的imu数据
            if (imuQueue.front().header.stamp.toSec() < timeScanCur - 0.01)
                imuQueue.pop_front();
            else
                break;
        }

        if (imuQueue.empty())
            return;

        imuPointerCur = 0;

        for (int i = 0; i < (int)imuQueue.size(); ++i)
        {
            sensor_msgs::Imu thisImuMsg = imuQueue[i];
            double currentImuTime = thisImuMsg.header.stamp.toSec();

            // get roll, pitch, and yaw estimation for this scan  提取imu姿态角RPY，作为当前lidar初始姿态角
            if (currentImuTime <= timeScanCur)
                imuRPY2rosRPY(&thisImuMsg, &cloudInfo.imuRollInit, &cloudInfo.imuPitchInit, &cloudInfo.imuYawInit);

            if (currentImuTime > timeScanNext + 0.01)
                break;

            // 第一帧imu旋转角初始化
            if (imuPointerCur == 0){
                imuRotX[0] = 0;
                imuRotY[0] = 0;
                imuRotZ[0] = 0;
                imuTime[0] = currentImuTime;
                ++imuPointerCur;
                continue;
            }

            // get angular velocity  提取imu角速度
            double angular_x, angular_y, angular_z;
            imuAngular2rosAngular(&thisImuMsg, &angular_x, &angular_y, &angular_z);

            // integrate rotation  当前时刻旋转角 = 前一时刻旋转角 + 角速度*时差
            double timeDiff = currentImuTime - imuTime[imuPointerCur-1];
            imuRotX[imuPointerCur] = imuRotX[imuPointerCur-1] + angular_x * timeDiff;
            imuRotY[imuPointerCur] = imuRotY[imuPointerCur-1] + angular_y * timeDiff;
            imuRotZ[imuPointerCur] = imuRotZ[imuPointerCur-1] + angular_z * timeDiff;
            imuTime[imuPointerCur] = currentImuTime;
            ++imuPointerCur;
        }

        --imuPointerCur;

        if (imuPointerCur <= 0)
            return;

        cloudInfo.imuAvailable = true;
    }

    //vins的odom_ros计算位姿变换
    //1.截取当前帧前0.01s后的vins的odom_ros数据
    //2.第一个odom位姿startOdomMsg赋值给cloudInfo.odomX/Y/Z/R/P/Y
    //3.最后一个odom位姿endOdomMsg
    //4.第一个odom位姿startOdomMsg和最后一个odom位姿endOdomMsg，计算transBt
    //  转换为odomIncreX/Y/Z/R/P/Y
    void odomDeskewInfo()
    {
        cloudInfo.odomAvailable = false;

        // 从vins里程计队列中删除当前激光帧0.01s前面时刻的imu数据
        while (!odomQueue.empty())
        {
            if (odomQueue.front().header.stamp.toSec() < timeScanCur - 0.01)
                odomQueue.pop_front();
            else
                break;
        }

        if (odomQueue.empty())
            return;

        if (odomQueue.front().header.stamp.toSec() > timeScanCur)
            return;

        // get start odometry at the beinning of the scan  当前激光帧起始时刻的vins里程计
        nav_msgs::Odometry startOdomMsg;

        for (int i = 0; i < (int)odomQueue.size(); ++i)
        {
            startOdomMsg = odomQueue[i];

            if (ROS_TIME(&startOdomMsg) < timeScanCur)
                continue;
            else
                break;
        }

        // 提取vins里程计姿态角
        tf::Quaternion orientation;//四元数
        tf::quaternionMsgToTF(startOdomMsg.pose.pose.orientation, orientation);

        double roll, pitch, yaw; // vins里程计的RPY
        tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);

        // Initial guess used in mapOptimization  用当前激光帧起始时刻的vins里程计，初始化lidar位姿，后面用于mapOptmization
        cloudInfo.odomX = startOdomMsg.pose.pose.position.x;
        cloudInfo.odomY = startOdomMsg.pose.pose.position.y;
        cloudInfo.odomZ = startOdomMsg.pose.pose.position.z;
        cloudInfo.odomRoll  = roll;
        cloudInfo.odomPitch = pitch;
        cloudInfo.odomYaw   = yaw;
        cloudInfo.odomResetId = (int)round(startOdomMsg.pose.covariance[0]); // 存的vins里程计里的失败id

        cloudInfo.odomAvailable = true;

        // get end odometry at the end of the scan
        odomDeskewFlag = false;

        // 如果当前激光帧结束时刻之后没有vins里程计数据，返回
        if (odomQueue.back().header.stamp.toSec() < timeScanNext)
            return;

        // 当前激光帧结束时刻的imu里程计
        nav_msgs::Odometry endOdomMsg;

        for (int i = 0; i < (int)odomQueue.size(); ++i)
        {
            endOdomMsg = odomQueue[i];

            if (ROS_TIME(&endOdomMsg) < timeScanNext)
                continue;
            else
                break;
        }

        // failureCount不一致说明雷达在这一帧内视觉里程计至少重启了一次，数值不准确，所以不进行去畸变了
        if (int(round(startOdomMsg.pose.covariance[0])) != int(round(endOdomMsg.pose.covariance[0])))
            return;

        Eigen::Affine3f transBegin = pcl::getTransformation(startOdomMsg.pose.pose.position.x, startOdomMsg.pose.pose.position.y, startOdomMsg.pose.pose.position.z, roll, pitch, yaw);//4X4矩阵

        tf::quaternionMsgToTF(endOdomMsg.pose.pose.orientation, orientation);
        tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
        Eigen::Affine3f transEnd = pcl::getTransformation(endOdomMsg.pose.pose.position.x, endOdomMsg.pose.pose.position.y, endOdomMsg.pose.pose.position.z, roll, pitch, yaw);

        Eigen::Affine3f transBt = transBegin.inverse() * transEnd;//变换矩阵

        // 相对变换，提取两个激光帧之间的vins里程计增量平移，旋转（欧拉角）
        float rollIncre, pitchIncre, yawIncre;
        pcl::getTranslationAndEulerAngles(transBt, odomIncreX, odomIncreY, odomIncreZ, rollIncre, pitchIncre, yawIncre);

        odomDeskewFlag = true;
    }

    void findRotation(double pointTime, float *rotXCur, float *rotYCur, float *rotZCur)
    {
        *rotXCur = 0; *rotYCur = 0; *rotZCur = 0;

        int imuPointerFront = 0;
        while (imuPointerFront < imuPointerCur)//找到激光点采集的时间的IMU
        {
            if (pointTime < imuTime[imuPointerFront])
                break;
            ++imuPointerFront;
        }

        if (pointTime > imuTime[imuPointerFront] || imuPointerFront == 0)//超过就最后一个
        {
            *rotXCur = imuRotX[imuPointerFront];
            *rotYCur = imuRotY[imuPointerFront];
            *rotZCur = imuRotZ[imuPointerFront];
        } else {
            int imuPointerBack = imuPointerFront - 1;
            double ratioFront = (pointTime - imuTime[imuPointerBack]) / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
            double ratioBack = (imuTime[imuPointerFront] - pointTime) / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
            *rotXCur = imuRotX[imuPointerFront] * ratioFront + imuRotX[imuPointerBack] * ratioBack;
            *rotYCur = imuRotY[imuPointerFront] * ratioFront + imuRotY[imuPointerBack] * ratioBack;
            *rotZCur = imuRotZ[imuPointerFront] * ratioFront + imuRotZ[imuPointerBack] * ratioBack;
        }
    }

    void findPosition(double relTime, float *posXCur, float *posYCur, float *posZCur)
    {
        *posXCur = 0; *posYCur = 0; *posZCur = 0;

        // if (cloudInfo.odomAvailable == false || odomDeskewFlag == false)
        //     return;

        // float ratio = relTime / (timeScanNext - timeScanCur);

        // *posXCur = ratio * odomIncreX;
        // *posYCur = ratio * odomIncreY;
        // *posZCur = ratio * odomIncreZ;
    }

    PointType deskewPoint(PointType *point, double relTime)//畸变矫正
    {
        if (deskewFlag == -1 || cloudInfo.imuAvailable == false)//是否去畸变
            return *point;

        // relTime 是当前激光点相对于激光帧起始时刻的时间，pointTime则是当前激光点的时间戳，
        // 激光点是在激光点云开始采集之后一个一个采集的
        double pointTime = timeScanCur + relTime;

        // 在当前激光帧起止时间范围内，计算某一时刻的旋转（相对于起始时刻的旋转增量)
        float rotXCur, rotYCur, rotZCur;
        findRotation(pointTime, &rotXCur, &rotYCur, &rotZCur);//因为采集激光点的时间不完全精准，插值计算IMU

        // 在当前激光帧起止时间范围内，计算某一时刻的平移（相对于起始时刻的平移增量）
        float posXCur, posYCur, posZCur;
        findPosition(relTime, &posXCur, &posYCur, &posZCur);//此处全取0

        // 第一个点的位姿增量（0），求逆
        if (firstPointFlag == true)
        {
            transStartInverse = (pcl::getTransformation(posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur)).inverse();
            firstPointFlag = false;
        }

        // transform points to start 当前时刻激光点与第一个激光点的位姿变换
        Eigen::Affine3f transFinal = pcl::getTransformation(posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur);
        Eigen::Affine3f transBt = transStartInverse * transFinal;

        // 当前激光点在第一个激光点坐标系下的坐标
        PointType newPoint;
        newPoint.x = transBt(0,0) * point->x + transBt(0,1) * point->y + transBt(0,2) * point->z + transBt(0,3);
        newPoint.y = transBt(1,0) * point->x + transBt(1,1) * point->y + transBt(1,2) * point->z + transBt(1,3);
        newPoint.z = transBt(2,0) * point->x + transBt(2,1) * point->y + transBt(2,2) * point->z + transBt(2,3);
        newPoint.intensity = point->intensity;

        return newPoint;
    }

    // 当前帧激光点云运动畸变矫正
    // 1.检查激光点距离，扫描线是否合规
    // 2.激光运动畸变矫正，保存激光点
    void projectPointCloud()
    {
        int cloudSize = (int)laserCloudIn->points.size();
        // range image projection
        for (int i = 0; i < cloudSize; ++i)
        {
            //一个个点读取
            PointType thisPoint;
            thisPoint.x = laserCloudIn->points[i].x;
            thisPoint.y = laserCloudIn->points[i].y;
            thisPoint.z = laserCloudIn->points[i].z;
            thisPoint.intensity = laserCloudIn->points[i].intensity;//强度

            // 扫描线检查
            int rowIdn = laserCloudIn->points[i].ring; // 第几根扫描线
            if (rowIdn < 0 || rowIdn >= N_SCAN) // N_SCAN = 16
                continue;

            // 扫描线如果有降采样，跳过采样的扫描线这里要跳过
            if (rowIdn % downsampleRate != 0)
                continue;

            float horizonAngle = atan2(thisPoint.x, thisPoint.y) * 180 / M_PI;

            // 水平扫描角度步长，例如一周扫描1800次，则两次扫描间隔角度0.2°
            static float ang_res_x = 360.0/float(Horizon_SCAN); // Horizon_SCAN = 1800
            int columnIdn = -round((horizonAngle-90.0)/ang_res_x) + Horizon_SCAN/2;//奇奇怪怪的计算，坐标系标准不一样，反正是列索引
            if (columnIdn >= Horizon_SCAN)
                columnIdn -= Horizon_SCAN;

            if (columnIdn < 0 || columnIdn >= Horizon_SCAN)
                continue;

            float range = pointDistance(thisPoint);//重载过，这里计算到原点距离
            
            if (range < 1.0)
                continue;

            // 已经存过该点，不再处理
            if (rangeMat.at<float>(rowIdn, columnIdn) != FLT_MAX)
                continue;

            // for the amsterdam dataset
            // if (range < 6.0 && rowIdn <= 7 && (columnIdn >= 1600 || columnIdn <= 200))
            //     continue;
            // if (thisPoint.z < -2.0)
            //     continue;

            // 矩阵存激光点的距离
            rangeMat.at<float>(rowIdn, columnIdn) = range;

            // 激光运动畸变矫正
            // 利用当前帧起止时刻之间的imu数据计算旋转增量，vins里程计数据计算平移增量，进而将每一时刻激光点位置变换到第一个激光点坐标系下，进行运动补偿
            thisPoint = deskewPoint(&thisPoint, laserCloudIn->points[i].time); // Velodyne
            // thisPoint = deskewPoint(&thisPoint, (float)laserCloudIn->points[i].t / 1000000000.0); // Ouster

            // 转换成一维索引，存矫正之后的激光点
            int index = columnIdn  + rowIdn * Horizon_SCAN; // 第几列 + 第几行 * 每行有几列
            fullCloud->points[index] = thisPoint;
        }
    }

    // 提取有效激光点（去掉很远很远探测不到的），存extractedCloud
    void cloudExtraction()
    {
        int count = 0; // 有效激光点的数量
        // extract segmented cloud for lidar odometry  遍历所有激光点
        for (int i = 0; i < N_SCAN; ++i)
        {
            // 记录每根扫描线起始第五个激光点在一维数组中的索引
            cloudInfo.startRingIndex[i] = count - 1 + 5;

            for (int j = 0; j < Horizon_SCAN; ++j)
            {
                // 有效激光点
                if (rangeMat.at<float>(i,j) != FLT_MAX)
                {
                    // mark the points' column index for marking occlusion later  记录激光点对应的Horizon_SCAN方向上的索引
                    cloudInfo.pointColInd[count] = j;
                    // save range info  激光点距离
                    cloudInfo.pointRange[count] = rangeMat.at<float>(i,j);
                    // save extracted cloud  加入有效激光点
                    extractedCloud->push_back(fullCloud->points[j + i*Horizon_SCAN]);
                    // size of extracted cloud
                    ++count;
                }
            }
            // 记录每根扫描线倒数第五个激光点在一维数组中的索引
            cloudInfo.endRingIndex[i] = count -1 - 5;
        }
    }
    
    void publishClouds()
    {
        cloudInfo.header = cloudHeader;
        cloudInfo.cloud_deskewed  = publishCloud(&pubExtractedCloud, extractedCloud, cloudHeader.stamp, "base_link");//转一下数据类型
        pubLaserCloudInfo.publish(cloudInfo);
    }
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "lidar");

    ImageProjection IP;
    
    ROS_INFO("\033[1;32m----> Lidar Cloud Deskew Started.\033[0m");

    ros::MultiThreadedSpinner spinner(3); // 创建三个相同的线程，在A线程忙碌的时候启用B线程，A,B线程都忙碌的时候启用C线程
    spinner.spin();
    
    return 0;
}