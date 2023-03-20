#include "utility.h"
#include "lvi_sam/cloud_info.h"

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>

#include <gtsam/nonlinear/ISAM2.h>

using namespace gtsam;

using symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)
using symbol_shorthand::V; // Vel   (xdot,ydot,zdot)
using symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)
using symbol_shorthand::G; // GPS pose

/*
    * A point cloud type that has 6D pose info ([x,y,z,roll,pitch,yaw] intensity is time stamp)
    */
struct PointXYZIRPYT
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;
    float roll;
    float pitch;
    float yaw;
    double time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT (PointXYZIRPYT,
                                   (float, x, x) (float, y, y)
                                   (float, z, z) (float, intensity, intensity)
                                   (float, roll, roll) (float, pitch, pitch) (float, yaw, yaw)
                                   (double, time, time))

typedef PointXYZIRPYT  PointTypePose;


class mapOptimization : public ParamServer
{

public:

    // gtsam
    NonlinearFactorGraph gtSAMgraph; // 本轮图优化因子图
    Values initialEstimate; // 初始估计
    Values optimizedEstimate;
    ISAM2 *isam; // 图优化器
    Values isamCurrentEstimate; // 当前图优化结果
    Eigen::MatrixXd poseCovariance; // 因子图最新关键帧的协方差

    ros::Publisher pubLaserCloudSurround; // 地图发布器
    ros::Publisher pubOdomAftMappedROS; // LIS里程计位姿发布器
    ros::Publisher pubKeyPoses; // 关键帧位姿发布器
    ros::Publisher pubPath; // 关键帧位移发布器

    ros::Publisher pubHistoryKeyFrames; // 回环帧的过去帧的局部地图发布器
    ros::Publisher pubIcpKeyFrames; // 回环帧当前帧组成的局部地图发布器
    ros::Publisher pubRecentKeyFrames; // 当前帧附近的局部地图发布器
    ros::Publisher pubRecentKeyFrame; // 当前帧特征点云发布器
    ros::Publisher pubCloudRegisteredRaw; // 当前帧整体点云关键帧发布器
    ros::Publisher pubLoopConstraintEdge; // 可视化的回环约束发布器

    ros::Subscriber subLaserCloudInfo;
    ros::Subscriber subGPS;
    ros::Subscriber subLoopInfo; // 视觉回环订阅器

    std::deque<nav_msgs::Odometry> gpsQueue;
    lvi_sam::cloud_info cloudInfo;

    vector<pcl::PointCloud<PointType>::Ptr> cornerCloudKeyFrames; // 角点点云容器(雷达坐标系)
    vector<pcl::PointCloud<PointType>::Ptr> surfCloudKeyFrames; // 面点点云容器(雷达坐标系)
    
    pcl::PointCloud<PointType>::Ptr cloudKeyPoses3D; // 用点云格式储存关键帧的位置(xyz) 注意，不是存储的关键帧对应的点云！而存储的是产生关键帧时机器人的位置xyz
    pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D; // 用点云格式储存关键帧位姿

    pcl::PointCloud<PointType>::Ptr laserCloudCornerLast; // corner feature set from odoOptimization 当前帧角点点云
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLast; // surf feature set from odoOptimization 当前帧面点点云
    pcl::PointCloud<PointType>::Ptr laserCloudCornerLastDS; // downsampled corner featuer set from odoOptimization 当前帧角点点云降采样
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLastDS; // downsampled surf featuer set from odoOptimization 当前帧面点点云降采样

    pcl::PointCloud<PointType>::Ptr laserCloudOri; // 将优化的点云特征点
    pcl::PointCloud<PointType>::Ptr coeffSel; // 将优化的特征点对应的残差

    std::vector<PointType> laserCloudOriCornerVec; // corner point holder for parallel computation  当前帧的一个角点容器(雷达坐标系)
    std::vector<PointType> coeffSelCornerVec; // 与角点容器对应的残差
    std::vector<bool> laserCloudOriCornerFlag; // 当前角点是否将被优化的标记数组
    std::vector<PointType> laserCloudOriSurfVec; // surf point holder for parallel computation 当前帧的一个面点容器(雷达坐标系)
    std::vector<PointType> coeffSelSurfVec; // 与面点容器对应的残差
    std::vector<bool> laserCloudOriSurfFlag; // 面点容器是否将被优化的标记数组

    pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMap; // 当前帧附近的局部角点地图
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMap; // 当前帧附近的局部面点地图
    pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMapDS; // 当前帧附近的局部角点地图（降采样）
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMapDS; // 当前帧附近的局部面点地图（降采样）

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerFromMap; // 当前帧附近的局部角点地图（kdtree）
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromMap; // 当前帧附近的局部面点地图（kdtree）

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurroundingKeyPoses; // 关键帧位置坐标数组（kdtree）（scan2map）
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeHistoryKeyPoses; // 关键帧位置坐标数组（kdtree）(LIS回环检测)

    pcl::PointCloud<PointType>::Ptr latestKeyFrameCloud;
    pcl::PointCloud<PointType>::Ptr nearHistoryKeyFrameCloud;

    pcl::VoxelGrid<PointType> downSizeFilterCorner;
    pcl::VoxelGrid<PointType> downSizeFilterSurf;
    pcl::VoxelGrid<PointType> downSizeFilterICP;
    pcl::VoxelGrid<PointType> downSizeFilterSurroundingKeyPoses; // for surrounding key poses of scan-to-map optimization
    
    ros::Time timeLaserInfoStamp; // 点云时间戳
    double timeLaserInfoCur; // 点云时间戳

    float transformTobeMapped[6]; // 当前帧位姿

    std::mutex mtx;

    bool isDegenerate = false; // 非线性优化过程中点云是否退化
    cv::Mat matP; // 用来让退化部分的自由度不再迭代添加增量

    int laserCloudCornerLastDSNum = 0; // 当前帧的角点点云数量
    int laserCloudSurfLastDSNum = 0; // 当前帧的面点点云数量

    bool aLoopIsClosed = false; // 是否存在没有纠正过位姿的回环
    int imuPreintegrationResetId = 0; // IMU预积分的索引

    nav_msgs::Path globalPath; // 全局位移

    Eigen::Affine3f transPointAssociateToMap; // 位姿变换矩阵

    map<int, int> loopIndexContainer; // from new to old 存储每对回环帧
    vector<pair<int, int>> loopIndexQueue; // 存每对回环帧
    vector<gtsam::Pose3> loopPoseQueue; // 回环帧之间的位姿变换
    vector<gtsam::noiseModel::Diagonal::shared_ptr> loopNoiseQueue; // 回环帧之间的噪声（相似度）

    mapOptimization()
    {
        ISAM2Params parameters; // isam2参数类
        parameters.relinearizeThreshold = 0.1; // 重线性化阈值
        parameters.relinearizeSkip = 1; // 重线性化的频率
        isam = new ISAM2(parameters);

        pubKeyPoses           = nh.advertise<sensor_msgs::PointCloud2>(PROJECT_NAME + "/lidar/mapping/trajectory", 1);
        pubLaserCloudSurround = nh.advertise<sensor_msgs::PointCloud2>(PROJECT_NAME + "/lidar/mapping/map_global", 1);
        pubOdomAftMappedROS   = nh.advertise<nav_msgs::Odometry>      (PROJECT_NAME + "/lidar/mapping/odometry", 1);
        pubPath               = nh.advertise<nav_msgs::Path>          (PROJECT_NAME + "/lidar/mapping/path", 1);

        subLaserCloudInfo     = nh.subscribe<lvi_sam::cloud_info>     (PROJECT_NAME + "/lidar/feature/cloud_info", 5, &mapOptimization::laserCloudInfoHandler, this, ros::TransportHints().tcpNoDelay());
        subGPS                = nh.subscribe<nav_msgs::Odometry>      (gpsTopic,                                   50, &mapOptimization::gpsHandler, this, ros::TransportHints().tcpNoDelay());
        subLoopInfo           = nh.subscribe<std_msgs::Float64MultiArray>(PROJECT_NAME + "/vins/loop/match_frame", 5, &mapOptimization::loopHandler, this, ros::TransportHints().tcpNoDelay());

        pubHistoryKeyFrames   = nh.advertise<sensor_msgs::PointCloud2>(PROJECT_NAME + "/lidar/mapping/loop_closure_history_cloud", 1);
        pubIcpKeyFrames       = nh.advertise<sensor_msgs::PointCloud2>(PROJECT_NAME + "/lidar/mapping/loop_closure_corrected_cloud", 1);
        pubLoopConstraintEdge = nh.advertise<visualization_msgs::MarkerArray>(PROJECT_NAME + "/lidar/mapping/loop_closure_constraints", 1);

        pubRecentKeyFrames    = nh.advertise<sensor_msgs::PointCloud2>(PROJECT_NAME + "/lidar/mapping/map_local", 1);
        pubRecentKeyFrame     = nh.advertise<sensor_msgs::PointCloud2>(PROJECT_NAME + "/lidar/mapping/cloud_registered", 1);
        pubCloudRegisteredRaw = nh.advertise<sensor_msgs::PointCloud2>(PROJECT_NAME + "/lidar/mapping/cloud_registered_raw", 1);

        downSizeFilterCorner.setLeafSize(mappingCornerLeafSize, mappingCornerLeafSize, mappingCornerLeafSize);
        downSizeFilterSurf.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
        downSizeFilterICP.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
        downSizeFilterSurroundingKeyPoses.setLeafSize(surroundingKeyframeDensity, surroundingKeyframeDensity, surroundingKeyframeDensity); // for surrounding key poses of scan-to-map optimization

        allocateMemory();
    }

    void allocateMemory()
    {
        cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
        cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());

        kdtreeSurroundingKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());
        kdtreeHistoryKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());

        laserCloudCornerLast.reset(new pcl::PointCloud<PointType>()); // corner feature set from odoOptimization
        laserCloudSurfLast.reset(new pcl::PointCloud<PointType>()); // surf feature set from odoOptimization
        laserCloudCornerLastDS.reset(new pcl::PointCloud<PointType>()); // downsampled corner featuer set from odoOptimization
        laserCloudSurfLastDS.reset(new pcl::PointCloud<PointType>()); // downsampled surf featuer set from odoOptimization

        laserCloudOri.reset(new pcl::PointCloud<PointType>());
        coeffSel.reset(new pcl::PointCloud<PointType>());

        laserCloudOriCornerVec.resize(N_SCAN * Horizon_SCAN);
        coeffSelCornerVec.resize(N_SCAN * Horizon_SCAN);
        laserCloudOriCornerFlag.resize(N_SCAN * Horizon_SCAN);
        laserCloudOriSurfVec.resize(N_SCAN * Horizon_SCAN);
        coeffSelSurfVec.resize(N_SCAN * Horizon_SCAN);
        laserCloudOriSurfFlag.resize(N_SCAN * Horizon_SCAN);

        std::fill(laserCloudOriCornerFlag.begin(), laserCloudOriCornerFlag.end(), false);
        std::fill(laserCloudOriSurfFlag.begin(), laserCloudOriSurfFlag.end(), false);

        laserCloudCornerFromMap.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfFromMap.reset(new pcl::PointCloud<PointType>());
        laserCloudCornerFromMapDS.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfFromMapDS.reset(new pcl::PointCloud<PointType>());

        kdtreeCornerFromMap.reset(new pcl::KdTreeFLANN<PointType>());
        kdtreeSurfFromMap.reset(new pcl::KdTreeFLANN<PointType>());

        latestKeyFrameCloud.reset(new pcl::PointCloud<PointType>());
        nearHistoryKeyFrameCloud.reset(new pcl::PointCloud<PointType>());

        for (int i = 0; i < 6; ++i){
            transformTobeMapped[i] = 0;
        }

        matP = cv::Mat(6, 6, CV_32F, cv::Scalar::all(0));
    }

    void laserCloudInfoHandler(const lvi_sam::cloud_infoConstPtr& msgIn)
    {
        // extract time stamp
        timeLaserInfoStamp = msgIn->header.stamp;
        timeLaserInfoCur = msgIn->header.stamp.toSec();

        // extract info ana feature cloud
        cloudInfo = *msgIn;
        pcl::fromROSMsg(msgIn->cloud_corner,  *laserCloudCornerLast);
        pcl::fromROSMsg(msgIn->cloud_surface, *laserCloudSurfLast);

        std::lock_guard<std::mutex> lock(mtx);

        // mapping执行频率控制
        static double timeLastProcessing = -1;
        if (timeLaserInfoCur - timeLastProcessing >= mappingProcessInterval) {

            timeLastProcessing = timeLaserInfoCur;

            updateInitialGuess();

            extractSurroundingKeyFrames();

            downsampleCurrentScan();

            scan2MapOptimization();

            saveKeyFramesAndFactor();

            correctPoses();

            publishOdometry();

            publishFrames();
        }
    }

    void gpsHandler(const nav_msgs::Odometry::ConstPtr& gpsMsg)
    {
        std::lock_guard<std::mutex> lock(mtx);
        gpsQueue.push_back(*gpsMsg);
    }

    // 利用transformTobeMapped的数据(位姿)，把雷达坐标系转换到世界坐标系上
    void pointAssociateToMap(PointType const * const pi, PointType * const po)
    {
        po->x = transPointAssociateToMap(0,0) * pi->x + transPointAssociateToMap(0,1) * pi->y + transPointAssociateToMap(0,2) * pi->z + transPointAssociateToMap(0,3);
        po->y = transPointAssociateToMap(1,0) * pi->x + transPointAssociateToMap(1,1) * pi->y + transPointAssociateToMap(1,2) * pi->z + transPointAssociateToMap(1,3);
        po->z = transPointAssociateToMap(2,0) * pi->x + transPointAssociateToMap(2,1) * pi->y + transPointAssociateToMap(2,2) * pi->z + transPointAssociateToMap(2,3);
        po->intensity = pi->intensity;
    }

    pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn, PointTypePose* transformIn)
    {
        pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

        PointType *pointFrom;

        int cloudSize = cloudIn->size();
        cloudOut->resize(cloudSize);

        Eigen::Affine3f transCur = pcl::getTransformation(transformIn->x, transformIn->y, transformIn->z, transformIn->roll, transformIn->pitch, transformIn->yaw);
        
        for (int i = 0; i < cloudSize; ++i)
        {
            pointFrom = &cloudIn->points[i];
            cloudOut->points[i].x = transCur(0,0) * pointFrom->x + transCur(0,1) * pointFrom->y + transCur(0,2) * pointFrom->z + transCur(0,3);
            cloudOut->points[i].y = transCur(1,0) * pointFrom->x + transCur(1,1) * pointFrom->y + transCur(1,2) * pointFrom->z + transCur(1,3);
            cloudOut->points[i].z = transCur(2,0) * pointFrom->x + transCur(2,1) * pointFrom->y + transCur(2,2) * pointFrom->z + transCur(2,3);
            cloudOut->points[i].intensity = pointFrom->intensity;
        }
        return cloudOut;
    }

    gtsam::Pose3 affine3fTogtsamPose3(const Eigen::Affine3f& thisPose)
    {
        float x, y, z, roll, pitch, yaw;
        pcl::getTranslationAndEulerAngles(thisPose, x, y, z, roll, pitch, yaw);
        return gtsam::Pose3(gtsam::Rot3::RzRyRx(double(roll), double(pitch), double(yaw)),
                                  gtsam::Point3(double(x),    double(y),     double(z)));
    }

    gtsam::Pose3 pclPointTogtsamPose3(PointTypePose thisPoint)
    {
        return gtsam::Pose3(gtsam::Rot3::RzRyRx(double(thisPoint.roll), double(thisPoint.pitch), double(thisPoint.yaw)),
                                  gtsam::Point3(double(thisPoint.x),    double(thisPoint.y),     double(thisPoint.z)));
    }

    gtsam::Pose3 trans2gtsamPose(float transformIn[])
    {
        return gtsam::Pose3(gtsam::Rot3::RzRyRx(transformIn[0], transformIn[1], transformIn[2]), 
                                  gtsam::Point3(transformIn[3], transformIn[4], transformIn[5]));
    }

    Eigen::Affine3f pclPointToAffine3f(PointTypePose thisPoint)
    { 
        return pcl::getTransformation(thisPoint.x, thisPoint.y, thisPoint.z, thisPoint.roll, thisPoint.pitch, thisPoint.yaw);
    }

    Eigen::Affine3f trans2Affine3f(float transformIn[])
    {
        return pcl::getTransformation(transformIn[3], transformIn[4], transformIn[5], transformIn[0], transformIn[1], transformIn[2]);
    }

    PointTypePose trans2PointTypePose(float transformIn[])
    {
        PointTypePose thisPose6D;
        thisPose6D.x = transformIn[3];
        thisPose6D.y = transformIn[4];
        thisPose6D.z = transformIn[5];
        thisPose6D.roll  = transformIn[0];
        thisPose6D.pitch = transformIn[1];
        thisPose6D.yaw   = transformIn[2];
        return thisPose6D;
    }

    














    void visualizeGlobalMapThread()
    {
        ros::Rate rate(0.2);
        while (ros::ok()){
            rate.sleep();
            publishGlobalMap();
        }

        if (savePCD == false)
            return;

        cout << "****************************************************" << endl;
        cout << "Saving map to pcd files ..." << endl;
        // create directory and remove old files;
        savePCDDirectory = std::getenv("HOME") + savePCDDirectory;
        int unused = system((std::string("exec rm -r ") + savePCDDirectory).c_str());
        unused = system((std::string("mkdir ") + savePCDDirectory).c_str()); ++unused;
        // save key frame transformations
        pcl::io::savePCDFileASCII(savePCDDirectory + "trajectory.pcd", *cloudKeyPoses3D);
        pcl::io::savePCDFileBinary(savePCDDirectory + "transformations.pcd", *cloudKeyPoses6D);
        // extract global point cloud map        
        pcl::PointCloud<PointType>::Ptr globalCornerCloud(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalSurfCloud(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapCloud(new pcl::PointCloud<PointType>());
        for (int i = 0; i < (int)cloudKeyPoses3D->size(); i++) 
        {
            // clip cloud
            // pcl::PointCloud<PointType>::Ptr cornerTemp(new pcl::PointCloud<PointType>());
            // pcl::PointCloud<PointType>::Ptr cornerTemp2(new pcl::PointCloud<PointType>());
            // *cornerTemp = *transformPointCloud(cornerCloudKeyFrames[i],  &cloudKeyPoses6D->points[i]);
            // for (int j = 0; j < (int)cornerTemp->size(); ++j)
            // {
            //     if (cornerTemp->points[j].z > cloudKeyPoses6D->points[i].z && cornerTemp->points[j].z < cloudKeyPoses6D->points[i].z + 5)
            //         cornerTemp2->push_back(cornerTemp->points[j]);
            // }
            // pcl::PointCloud<PointType>::Ptr surfTemp(new pcl::PointCloud<PointType>());
            // pcl::PointCloud<PointType>::Ptr surfTemp2(new pcl::PointCloud<PointType>());
            // *surfTemp = *transformPointCloud(surfCloudKeyFrames[i],  &cloudKeyPoses6D->points[i]);
            // for (int j = 0; j < (int)surfTemp->size(); ++j)
            // {
            //     if (surfTemp->points[j].z > cloudKeyPoses6D->points[i].z && surfTemp->points[j].z < cloudKeyPoses6D->points[i].z + 5)
            //         surfTemp2->push_back(surfTemp->points[j]);
            // }
            // *globalCornerCloud += *cornerTemp2;
            // *globalSurfCloud   += *surfTemp2;

            // origin cloud
            *globalCornerCloud += *transformPointCloud(cornerCloudKeyFrames[i],  &cloudKeyPoses6D->points[i]);
            *globalSurfCloud   += *transformPointCloud(surfCloudKeyFrames[i],    &cloudKeyPoses6D->points[i]);
            cout << "\r" << std::flush << "Processing feature cloud " << i << " of " << cloudKeyPoses6D->size() << " ...";
        }
        // down-sample and save corner cloud
        downSizeFilterCorner.setInputCloud(globalCornerCloud);
        pcl::io::savePCDFileASCII(savePCDDirectory + "cloudCorner.pcd", *globalCornerCloud);
        // down-sample and save surf cloud
        downSizeFilterSurf.setInputCloud(globalSurfCloud);
        pcl::io::savePCDFileASCII(savePCDDirectory + "cloudSurf.pcd", *globalSurfCloud);
        // down-sample and save global point cloud map
        *globalMapCloud += *globalCornerCloud;
        *globalMapCloud += *globalSurfCloud;
        downSizeFilterSurf.setInputCloud(globalMapCloud);
        pcl::io::savePCDFileASCII(savePCDDirectory + "cloudGlobal.pcd", *globalMapCloud);
        cout << "****************************************************" << endl;
        cout << "Saving map to pcd files completed" << endl;
    }

    // 只发布一公里以内的
    void publishGlobalMap()
    {
        if (pubLaserCloudSurround.getNumSubscribers() == 0)
            return;

        if (cloudKeyPoses3D->points.empty() == true)
            return;

        pcl::KdTreeFLANN<PointType>::Ptr kdtreeGlobalMap(new pcl::KdTreeFLANN<PointType>());;
        pcl::PointCloud<PointType>::Ptr globalMapKeyPoses(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyPosesDS(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyFrames(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyFramesDS(new pcl::PointCloud<PointType>());

        // kd-tree to find near key frames to visualize
        std::vector<int> pointSearchIndGlobalMap;
        std::vector<float> pointSearchSqDisGlobalMap;
        // search near key frames to visualize
        mtx.lock();
        kdtreeGlobalMap->setInputCloud(cloudKeyPoses3D);
        // 这里只找一公里以内的，发布出去globalMapVisualizationSearchRadius = 1000
        kdtreeGlobalMap->radiusSearch(cloudKeyPoses3D->back(), globalMapVisualizationSearchRadius, pointSearchIndGlobalMap, pointSearchSqDisGlobalMap, 0);
        mtx.unlock();

        for (int i = 0; i < (int)pointSearchIndGlobalMap.size(); ++i)
            globalMapKeyPoses->push_back(cloudKeyPoses3D->points[pointSearchIndGlobalMap[i]]);
        // downsample near selected key frames
        pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyPoses; // for global map visualization
        downSizeFilterGlobalMapKeyPoses.setLeafSize(globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity); // for global map visualization
        downSizeFilterGlobalMapKeyPoses.setInputCloud(globalMapKeyPoses);
        downSizeFilterGlobalMapKeyPoses.filter(*globalMapKeyPosesDS);

        // extract visualized and downsampled key frames
        for (int i = 0; i < (int)globalMapKeyPosesDS->size(); ++i){
            if (pointDistance(globalMapKeyPosesDS->points[i], cloudKeyPoses3D->back()) > globalMapVisualizationSearchRadius)
                continue;
            int thisKeyInd = (int)globalMapKeyPosesDS->points[i].intensity;
            *globalMapKeyFrames += *transformPointCloud(cornerCloudKeyFrames[thisKeyInd],  &cloudKeyPoses6D->points[thisKeyInd]);
            *globalMapKeyFrames += *transformPointCloud(surfCloudKeyFrames[thisKeyInd],    &cloudKeyPoses6D->points[thisKeyInd]);
        }
        // downsample visualized points
        pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyFrames; // for global map visualization
        downSizeFilterGlobalMapKeyFrames.setLeafSize(globalMapVisualizationLeafSize, globalMapVisualizationLeafSize, globalMapVisualizationLeafSize); // for global map visualization
        downSizeFilterGlobalMapKeyFrames.setInputCloud(globalMapKeyFrames);
        downSizeFilterGlobalMapKeyFrames.filter(*globalMapKeyFramesDS);
        publishCloud(&pubLaserCloudSurround, globalMapKeyFramesDS, timeLaserInfoStamp, "odom");    
    }


















    // vins传来的回环信息的处理
    void loopHandler(const std_msgs::Float64MultiArray::ConstPtr& loopMsg)
    {
        // control loop closure frequency
        static double last_loop_closure_time = -1;
        {
            // std::lock_guard<std::mutex> lock(mtx);
            if (timeLaserInfoCur - last_loop_closure_time < 5.0)
                return;
            else
                last_loop_closure_time = timeLaserInfoCur;
        }

        performLoopClosure(*loopMsg);
    }

    // 闭环检测 (通过 距离内搜索 或者 vins 得到的闭环候选帧), loopMsg保存的是回环帧的时间戳(当前帧, 闭环帧);
    // 在空间和相似度上验证回环，如果正确则添加闭环约束到容器，发布可视化闭环约束
    void performLoopClosure(const std_msgs::Float64MultiArray& loopMsg)
    {
        // 获取所有关键帧的位姿
        pcl::PointCloud<PointTypePose>::Ptr copy_cloudKeyPoses6D(new pcl::PointCloud<PointTypePose>());
        {
            std::lock_guard<std::mutex> lock(mtx);
            *copy_cloudKeyPoses6D = *cloudKeyPoses6D;
        }

        // get lidar keyframe id 通过loopMsg的时间戳来寻找 闭环候选帧对应的关键帧点云
        int key_cur = -1; // latest lidar keyframe id 上一个回环雷达关键帧索引
        int key_pre = -1; // previous lidar keyframe id 当前回环雷达关键帧索引
        {
            loopFindKey(loopMsg, copy_cloudKeyPoses6D, key_cur, key_pre);
            if (key_cur == -1 || key_pre == -1 || key_cur == key_pre)// || abs(key_cur - key_pre) < 25)
                return;
        }

        // check if loop added before 检查是否已经添加过这个回环
        {
            // if image loop closure comes at high frequency, many image loop may point to the same key_cur
            auto it = loopIndexContainer.find(key_cur);
            if (it != loopIndexContainer.end())
                return;
        }
        
        // get lidar keyframe cloud 分别为当前帧和闭环帧构造局部地图, 进行map to map的闭环匹配，用来验证闭环
        pcl::PointCloud<PointType>::Ptr cureKeyframeCloud(new pcl::PointCloud<PointType>()); // 当前关键帧的局部地图
        pcl::PointCloud<PointType>::Ptr prevKeyframeCloud(new pcl::PointCloud<PointType>()); // 候选关键帧的局部地图
        {
            loopFindNearKeyframes(copy_cloudKeyPoses6D, cureKeyframeCloud, key_cur, 0); // 这里参数为0，所以只保留当前帧，也可以设置其他值，取当前帧附近帧组成局部地图
            loopFindNearKeyframes(copy_cloudKeyPoses6D, prevKeyframeCloud, key_pre, historyKeyframeSearchNum); // historyKeyframeSearchNum为25，用闭环帧附近的前后25个帧组成局部地图
            if (cureKeyframeCloud->size() < 300 || prevKeyframeCloud->size() < 1000)
                return;
            if (pubHistoryKeyFrames.getNumSubscribers() != 0)
                publishCloud(&pubHistoryKeyFrames, prevKeyframeCloud, timeLaserInfoStamp, "odom");
        }

        // get keyframe pose
        Eigen::Affine3f pose_cur; // 当前帧pose
        Eigen::Affine3f pose_pre; // 闭环帧pose
        Eigen::Affine3f pose_diff_t; // serves as initial guess  将两者的相对位姿作为初始位姿
        {
            pose_cur = pclPointToAffine3f(copy_cloudKeyPoses6D->points[key_cur]);
            pose_pre = pclPointToAffine3f(copy_cloudKeyPoses6D->points[key_pre]);

            Eigen::Vector3f t_diff;
            t_diff.x() = - (pose_cur.translation().x() - pose_pre.translation().x());
            t_diff.y() = - (pose_cur.translation().y() - pose_pre.translation().y());
            t_diff.z() = - (pose_cur.translation().z() - pose_pre.translation().z());
            // 如果它们之间的距离相差很远，说明漂移很大，不适合用它作为初值
            if (t_diff.norm() < historyKeyframeSearchRadius)
                t_diff.setZero();
            pose_diff_t = pcl::getTransformation(t_diff.x(), t_diff.y(), t_diff.z(), 0, 0, 0);
        }

        // transform and rotate cloud for matching  使用icp进行闭环匹配(map to map)
        pcl::IterativeClosestPoint<PointType, PointType> icp;
        // pcl::GeneralizedIterativeClosestPoint<PointType, PointType> icp;
        // 设置对应点最大距离
        icp.setMaxCorrespondenceDistance(historyKeyframeSearchRadius * 2);
        // 设置最大迭代次数
        icp.setMaximumIterations(100);
        icp.setRANSACIterations(0);
        // 两次迭代变化值小于Epsilon则停止迭代
        icp.setTransformationEpsilon(1e-3);
        icp.setEuclideanFitnessEpsilon(1e-3);

        // initial guess cloud  根据初始相对位姿, 对当前帧点云进行坐标变换
        pcl::PointCloud<PointType>::Ptr cureKeyframeCloud_new(new pcl::PointCloud<PointType>()); // 当前关键帧的局部地图（坐标变换后）
        pcl::transformPointCloud(*cureKeyframeCloud, *cureKeyframeCloud_new, pose_diff_t); // 这里好像是把最新的回环帧继续平移，让其拉近到回环帧。这是因为最新帧的局部地图只由一帧组成，而且其位姿未经过回环优化，不能精准的转换到世界坐标系上，所以通过粗暴的平移将其拉近到回环帧附近

        // match using icp
        icp.setInputSource(cureKeyframeCloud_new);
        icp.setInputTarget(prevKeyframeCloud);
        pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
        icp.align(*unused_result);

        if (pubIcpKeyFrames.getNumSubscribers() != 0)
        {
            pcl::PointCloud<PointType>::Ptr closed_cloud(new pcl::PointCloud<PointType>()); // 真正的转换到世界坐标系的回环雷达帧
            pcl::transformPointCloud(*cureKeyframeCloud_new, *closed_cloud, icp.getFinalTransformation());
            publishCloud(&pubIcpKeyFrames, closed_cloud, timeLaserInfoStamp, "odom");
        }

        // add graph factor// 将闭环保存至loopIndexQueue loopPoseQueue loopNoiseQueue中供addLoopFactor()使用
        // 关键帧和候选帧距离小并且收敛
        if (icp.getFitnessScore() < historyKeyframeFitnessScore && icp.hasConverged() == true)
        {
            // get gtsam pose
            gtsam::Pose3 poseFrom = affine3fTogtsamPose3(Eigen::Affine3f(icp.getFinalTransformation()) * pose_diff_t * pose_cur); // 这里连续乘了多个变换，与其icp匹配前的多个变换相对应
            gtsam::Pose3 poseTo   = pclPointTogtsamPose3(copy_cloudKeyPoses6D->points[key_pre]);
            // get noise 把相似性分数作为方差噪声
            float noise = icp.getFitnessScore();
            gtsam::Vector Vector6(6);
            Vector6 << noise, noise, noise, noise, noise, noise;
            noiseModel::Diagonal::shared_ptr constraintNoise = noiseModel::Diagonal::Variances(Vector6);
            // save pose constraint
            mtx.lock();
            // 添加闭环约束
            loopIndexQueue.push_back(make_pair(key_cur, key_pre));
            loopPoseQueue.push_back(poseFrom.between(poseTo));
            loopNoiseQueue.push_back(constraintNoise);
            mtx.unlock();
            // add loop pair to container
            loopIndexContainer[key_cur] = key_pre;
        }

        // visualize loop constraints 发布 所有闭环约束
        if (!loopIndexContainer.empty())
        {
            visualization_msgs::MarkerArray markerArray;
            // loop nodes
            visualization_msgs::Marker markerNode;
            markerNode.header.frame_id = "odom";
            markerNode.header.stamp = timeLaserInfoStamp;
            markerNode.action = visualization_msgs::Marker::ADD;
            markerNode.type = visualization_msgs::Marker::SPHERE_LIST;
            markerNode.ns = "loop_nodes";
            markerNode.id = 0;
            markerNode.pose.orientation.w = 1;
            markerNode.scale.x = 0.3; markerNode.scale.y = 0.3; markerNode.scale.z = 0.3; 
            markerNode.color.r = 0; markerNode.color.g = 0.8; markerNode.color.b = 1;
            markerNode.color.a = 1;
            // loop edges
            visualization_msgs::Marker markerEdge;
            markerEdge.header.frame_id = "odom";
            markerEdge.header.stamp = timeLaserInfoStamp;
            markerEdge.action = visualization_msgs::Marker::ADD;
            markerEdge.type = visualization_msgs::Marker::LINE_LIST;
            markerEdge.ns = "loop_edges";
            markerEdge.id = 1;
            markerEdge.pose.orientation.w = 1;
            markerEdge.scale.x = 0.1;
            markerEdge.color.r = 0.9; markerEdge.color.g = 0.9; markerEdge.color.b = 0;
            markerEdge.color.a = 1;

            for (auto it = loopIndexContainer.begin(); it != loopIndexContainer.end(); ++it)
            {
                int key_cur = it->first;
                int key_pre = it->second;
                geometry_msgs::Point p;
                p.x = copy_cloudKeyPoses6D->points[key_cur].x;
                p.y = copy_cloudKeyPoses6D->points[key_cur].y;
                p.z = copy_cloudKeyPoses6D->points[key_cur].z;
                markerNode.points.push_back(p);
                markerEdge.points.push_back(p);
                p.x = copy_cloudKeyPoses6D->points[key_pre].x;
                p.y = copy_cloudKeyPoses6D->points[key_pre].y;
                p.z = copy_cloudKeyPoses6D->points[key_pre].z;
                markerNode.points.push_back(p);
                markerEdge.points.push_back(p);
            }

            markerArray.markers.push_back(markerNode);
            markerArray.markers.push_back(markerEdge);
            pubLoopConstraintEdge.publish(markerArray);
        }
    }

    // 根据索引构造局部地图
    void loopFindNearKeyframes(const pcl::PointCloud<PointTypePose>::Ptr& copy_cloudKeyPoses6D,
                               pcl::PointCloud<PointType>::Ptr& nearKeyframes, 
                               const int& key, const int& searchNum)
    {
        // extract near keyframes
        nearKeyframes->clear();
        int cloudSize = copy_cloudKeyPoses6D->size();
        for (int i = -searchNum; i <= searchNum; ++i)
        {
            int key_near = key + i;
            if (key_near < 0 || key_near >= cloudSize )
                continue;
            *nearKeyframes += *transformPointCloud(cornerCloudKeyFrames[key_near], &copy_cloudKeyPoses6D->points[key_near]);
            *nearKeyframes += *transformPointCloud(surfCloudKeyFrames[key_near],   &copy_cloudKeyPoses6D->points[key_near]);
        }

        if (nearKeyframes->empty())
            return;

        // downsample near keyframes
        pcl::PointCloud<PointType>::Ptr cloud_temp(new pcl::PointCloud<PointType>());
        downSizeFilterICP.setInputCloud(nearKeyframes);
        downSizeFilterICP.filter(*cloud_temp);
        *nearKeyframes = *cloud_temp;
    }

    // 根据loopMsg时间戳和copy_cloudKeyPoses6D的时间戳，寻找时间相近的关键帧索引key_cur、key_pre
    void loopFindKey(const std_msgs::Float64MultiArray& loopMsg, 
                     const pcl::PointCloud<PointTypePose>::Ptr& copy_cloudKeyPoses6D,
                     int& key_cur, int& key_pre)
    {
        if (loopMsg.data.size() != 2)
            return;

        double loop_time_cur = loopMsg.data[0];
        double loop_time_pre = loopMsg.data[1];

        if (abs(loop_time_cur - loop_time_pre) < historyKeyframeSearchTimeDiff)
            return;

        int cloudSize = copy_cloudKeyPoses6D->size();
        if (cloudSize < 2)
            return;

        // latest key
        key_cur = cloudSize - 1;
        // 从后往前遍历，找的快
        for (int i = cloudSize - 1; i >= 0; --i)
        {
            if (copy_cloudKeyPoses6D->points[i].time > loop_time_cur)
                key_cur = round(copy_cloudKeyPoses6D->points[i].intensity);
            else
                break;
        }

        // previous key
        key_pre = 0;
        for (int i = 0; i < cloudSize; ++i)
        {
            if (copy_cloudKeyPoses6D->points[i].time < loop_time_pre)
                key_pre = round(copy_cloudKeyPoses6D->points[i].intensity);
            else
                break;
        }
    }

    // 线程: 通过距离进行闭环检测,LIO自己的回环检测
    void loopClosureThread()
    {
        if (loopClosureEnableFlag == false)
            return;

        ros::Rate rate(0.5); // 每2s进行一次回环检测
        while (ros::ok())
        {
            rate.sleep();
            performLoopClosureDetection();
        }
    }

    // 通过距离进行闭环检测
    void performLoopClosureDetection()
    {
        std::vector<int> pointSearchIndLoop;
        std::vector<float> pointSearchSqDisLoop;

        // 通过距离找到的闭环候选帧
        int key_cur = -1;
        int key_pre = -1;

        double loop_time_cur = -1;
        double loop_time_pre = -1;

        // find latest key and time  1.使用kdtree寻找最近的keyframes, 作为闭环检测的候选关键帧 (半径20m以内)
        {
            std::lock_guard<std::mutex> lock(mtx);

            if (cloudKeyPoses3D->empty())
                return;
            // 当前的关键帧位移数组，构建kdtree
            kdtreeHistoryKeyPoses->setInputCloud(cloudKeyPoses3D);
            // 按照半径20m搜索
            kdtreeHistoryKeyPoses->radiusSearch(cloudKeyPoses3D->back(), historyKeyframeSearchRadius, pointSearchIndLoop, pointSearchSqDisLoop, 0);

            key_cur = cloudKeyPoses3D->size() - 1;
            loop_time_cur = cloudKeyPoses6D->points[key_cur].time;
        }

        // find previous key and time // 2.在候选关键帧集合中，找到与当前帧时间相隔较远的最近帧，设为候选匹配帧 (30s之前)
        {
            for (int i = 0; i < (int)pointSearchIndLoop.size(); ++i)
            {
                int id = pointSearchIndLoop[i];
                if (abs(cloudKeyPoses6D->points[id].time - loop_time_cur) > historyKeyframeSearchTimeDiff)
                {
                    key_pre = id;
                    loop_time_pre = cloudKeyPoses6D->points[key_pre].time;
                    break;
                }
            }
        }

        if (key_cur == -1 || key_pre == -1 || key_pre == key_cur ||
            loop_time_cur < 0 || loop_time_pre < 0)
            return;

        std_msgs::Float64MultiArray match_msg;
        match_msg.data.push_back(loop_time_cur);
        match_msg.data.push_back(loop_time_pre);
        performLoopClosure(match_msg);
    }






















    



    // 当前位姿初始化
    // 从VINS部分的IMU里程计或IMU提供的姿态变换，求解当前帧位姿的初值估计，存到transformTobeMapped
    void updateInitialGuess()
    {        
        static Eigen::Affine3f lastImuTransformation; // 前一帧的初始化姿态角（来自原始imu数据），用于估计第一帧的位姿（旋转部分）
        // system initialization  如果关键帧集合为空，即为第一帧点云，直接用IMU初始化
        if (cloudKeyPoses3D->points.empty())
        {
            // 当前帧位姿的旋转部分，用激光帧信息中的RPY（来自IMU原始数据）初始化
            transformTobeMapped[0] = cloudInfo.imuRollInit;
            transformTobeMapped[1] = cloudInfo.imuPitchInit;
            transformTobeMapped[2] = cloudInfo.imuYawInit;

            // 可以选择不使用9轴imu提供的yaw角初始化姿态
            if (!useImuHeadingInitialization)
                transformTobeMapped[2] = 0;

            //保存上一帧姿态给下一帧用
            lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit); // save imu before return;
            return;
        }

        // use VINS odometry estimation for pose guess  非第一帧，优先采用VIS的里程计预测
        // 用当前帧和前一帧对应的里程计计算相对位姿变换，再用前一帧的位姿与相对变换，计算当前帧的位姿，存transformTobeMapped
        static int odomResetId = 0; // 用来检测VIS是否被启动/重启过
        static bool lastVinsTransAvailable = false;
        static Eigen::Affine3f lastVinsTransformation;
        // VIS预测可用，并且VIS已经被启动/重启过
        if (cloudInfo.odomAvailable == true && cloudInfo.odomResetId == odomResetId)
        {
            // ROS_INFO("Using VINS initial guess");
            // 上次预测时VIS重新启动了，保存本次的预测值，尝试采用IMU的预测。lastVinsTransAvailable == false说明第一次收到VINS的里程计消息
            if (lastVinsTransAvailable == false)
            {
                // ROS_INFO("Initializing VINS initial guess");
                lastVinsTransformation = pcl::getTransformation(cloudInfo.odomX,    cloudInfo.odomY,     cloudInfo.odomZ, 
                                                                cloudInfo.odomRoll, cloudInfo.odomPitch, cloudInfo.odomYaw);
                lastVinsTransAvailable = true;
            } 
            // 上次预测成功使用了VIS的预测
            // 用上次预测的位姿与本次预测的位姿之间的位姿变换作为点云配准初值
            else {
                // ROS_INFO("Obtaining VINS incremental guess");
                Eigen::Affine3f transBack = pcl::getTransformation(cloudInfo.odomX,    cloudInfo.odomY,     cloudInfo.odomZ, 
                                                                   cloudInfo.odomRoll, cloudInfo.odomPitch, cloudInfo.odomYaw);
                Eigen::Affine3f transIncre = lastVinsTransformation.inverse() * transBack; // 当前帧相对于前一帧的位姿变换---VINS里程计计算得到

                Eigen::Affine3f transTobe = trans2Affine3f(transformTobeMapped); // 前一帧的位姿（经过优化的前一个关键帧位姿估计结果）
                Eigen::Affine3f transFinal = transTobe * transIncre; // 当前帧的位姿
                pcl::getTranslationAndEulerAngles(transFinal, transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5], 
                                                              transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);

                lastVinsTransformation = pcl::getTransformation(cloudInfo.odomX,    cloudInfo.odomY,     cloudInfo.odomZ, 
                                                                cloudInfo.odomRoll, cloudInfo.odomPitch, cloudInfo.odomYaw);

                lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit); // save imu before return; 存一下imu数据以免下一次VINS失效了使用IMU预测没数据了
                return;
            }
        } 
        // vins跟丢了, 准备重启
        else {
            // ROS_WARN("VINS failure detected.");
            lastVinsTransAvailable = false;
            odomResetId = cloudInfo.odomResetId;
        }

        // use imu incremental estimation for pose guess (only rotation)  imu预测可用，则利用imu预测的旋转矩阵与上次预测时的旋转矩阵之间的位姿变换，作为点云配准的初值
        if (cloudInfo.imuAvailable == true)
        {
            // ROS_INFO("Using IMU initial guess");
            Eigen::Affine3f transBack = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit);
            Eigen::Affine3f transIncre = lastImuTransformation.inverse() * transBack;

            Eigen::Affine3f transTobe = trans2Affine3f(transformTobeMapped); // 前一帧的位姿
            Eigen::Affine3f transFinal = transTobe * transIncre; // 当前帧的位姿
            pcl::getTranslationAndEulerAngles(transFinal, transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5], 
                                                          transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);

            lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit); // save imu before return;
            return;
        }
    }

    // 局部关键帧提取角点、平面点集合
    // 将最新关键帧周围50m内的关键帧提取出来，并将这些关键帧对应的点云组合成局部地图
    void extractNearby()
    {
        pcl::PointCloud<PointType>::Ptr surroundingKeyPoses(new pcl::PointCloud<PointType>()); // 定义储存当前帧附近的 关键帧位置信息 的点云指针集合
        pcl::PointCloud<PointType>::Ptr surroundingKeyPosesDS(new pcl::PointCloud<PointType>());
        std::vector<int> pointSearchInd; // 定义储存邻近关键帧在cloudKeyPoses3D的索引的容器
        std::vector<float> pointSearchSqDis; // 距离关键帧的距离

        // extract all the nearby key poses and downsample them  搜索最后一个关键帧附近的50m以内的关键帧
        kdtreeSurroundingKeyPoses->setInputCloud(cloudKeyPoses3D); // create kd-tree  kdtree的输入，全局关键帧位姿集合（历史所有关键帧集合）
        kdtreeSurroundingKeyPoses->radiusSearch(cloudKeyPoses3D->back(), (double)surroundingKeyframeSearchRadius, pointSearchInd, pointSearchSqDis); // 对最新的关键帧，在半径区域内(50m)搜索空间区域上相邻的关键帧集合
        // 遍历搜索结果，pointSearchInd存的是结果在cloudKeyPoses3D下面的索引
        for (int i = 0; i < (int)pointSearchInd.size(); ++i)
        {
            int id = pointSearchInd[i];
            surroundingKeyPoses->push_back(cloudKeyPoses3D->points[id]);
        }

        // 降采样主要是为了避免50m内的关键帧过于密集，每2m*2m*2m选出一个关键帧代表
        downSizeFilterSurroundingKeyPoses.setInputCloud(surroundingKeyPoses);
        downSizeFilterSurroundingKeyPoses.filter(*surroundingKeyPosesDS); // 降采样后不是真正的关键帧了，而是类似多个关键帧“取平均”后的结果，因此降采样后的位置未必是之前关键帧的位置了。lio-sam内后面还有一些处理，以保证降采样后的点仍然是之前关键帧的位置

        // also extract some latest key frames in case the robot rotates in one position  同时也将时间较近的关键帧位置加入进去，防止传感器仅仅在一个位置做纯旋转。因为纯旋转的话，50m内只能搜索出一个关键帧的位置
        int numPoses = cloudKeyPoses3D->size();
        for (int i = numPoses-1; i >= 0; --i)
        {
            if (timeLaserInfoCur - cloudKeyPoses6D->points[i].time < 10.0)
                surroundingKeyPosesDS->push_back(cloudKeyPoses3D->points[i]);
            else
                break;
        }

        extractCloud(surroundingKeyPosesDS);
    }

    // 将相邻关键帧集合对应的角点、平面点，拼接为局部map中，作为scan-to-map匹配的局部点云地图
    void extractCloud(pcl::PointCloud<PointType>::Ptr cloudToExtract)
    {
        std::vector<pcl::PointCloud<PointType>> laserCloudCornerSurroundingVec;
        std::vector<pcl::PointCloud<PointType>> laserCloudSurfSurroundingVec;

        laserCloudCornerSurroundingVec.resize(cloudToExtract->size());
        laserCloudSurfSurroundingVec.resize(cloudToExtract->size());

        // extract surrounding map 利用处理器并行计算，将所有关键帧特征点云组合起来
        #pragma omp parallel for num_threads(numberOfCores)
        for (int i = 0; i < (int)cloudToExtract->size(); ++i)
        {
            int thisKeyInd = (int)cloudToExtract->points[i].intensity; // 关键帧的索引
            // 剔除距离过远的关键帧
            if (pointDistance(cloudKeyPoses3D->points[thisKeyInd], cloudKeyPoses3D->back()) > surroundingKeyframeSearchRadius)
                continue;
            // 全都变换到map坐标系下。从这里开始处理关键帧对应的点云
            laserCloudCornerSurroundingVec[i]  = *transformPointCloud(cornerCloudKeyFrames[thisKeyInd],  &cloudKeyPoses6D->points[thisKeyInd]);
            laserCloudSurfSurroundingVec[i]    = *transformPointCloud(surfCloudKeyFrames[thisKeyInd],    &cloudKeyPoses6D->points[thisKeyInd]);
        }

        // fuse the map 将上面的点云全都保存到一个点云中，也就是局部地图中
        laserCloudCornerFromMap->clear();
        laserCloudSurfFromMap->clear(); 
        // 组成局部地图
        for (int i = 0; i < (int)cloudToExtract->size(); ++i)
        {
            *laserCloudCornerFromMap += laserCloudCornerSurroundingVec[i];
            *laserCloudSurfFromMap   += laserCloudSurfSurroundingVec[i];
        }

        // Downsample the surrounding corner key frames (or map)
        downSizeFilterCorner.setInputCloud(laserCloudCornerFromMap);
        downSizeFilterCorner.filter(*laserCloudCornerFromMapDS);
        // Downsample the surrounding surf key frames (or map)
        downSizeFilterSurf.setInputCloud(laserCloudSurfFromMap);
        downSizeFilterSurf.filter(*laserCloudSurfFromMapDS);
    }

    // 提取局部角点、平面点云集合，加入局部地图
    // 提取与当前帧相邻的关键帧surroundingKeyPosesDS，组成局部地图laserCloudCornerFromMapDS、laserCloudSurfFromMapDS。以用于之后的scan to map
    void extractSurroundingKeyFrames()
    {
        if (cloudKeyPoses3D->points.empty() == true)
            return; 
        
        extractNearby();
    }

    void downsampleCurrentScan()
    {
        // Downsample cloud from current scan
        laserCloudCornerLastDS->clear();
        downSizeFilterCorner.setInputCloud(laserCloudCornerLast);
        downSizeFilterCorner.filter(*laserCloudCornerLastDS);
        laserCloudCornerLastDSNum = laserCloudCornerLastDS->size();

        laserCloudSurfLastDS->clear();
        downSizeFilterSurf.setInputCloud(laserCloudSurfLast);
        downSizeFilterSurf.filter(*laserCloudSurfLastDS);
        laserCloudSurfLastDSNum = laserCloudSurfLastDS->size();
    }

    // 更新点云点投影到地图坐标系的变换矩阵，实际上也就是当前帧位姿
    // 把transformTobeMapped转变成Eigen::Affine3f的变量类型
    void updatePointAssociateToMap()
    {
        transPointAssociateToMap = trans2Affine3f(transformTobeMapped);
    }

    // 构造点到直线的残差约束
    void cornerOptimization()
    {
        // 将前面updateInitialGuess()函数预测的初值，转换为transPointAssociateToMap变量
        updatePointAssociateToMap();

        // 并行计算每个角点的残差
        #pragma omp parallel for num_threads(numberOfCores)
        for (int i = 0; i < laserCloudCornerLastDSNum; i++)
        {
            PointType pointOri, pointSel, coeff; // pointOri当前帧的一个角点(雷达坐标系)、pointSel当前帧的一个角点(世界坐标系)、coeff当前点到直线的垂线段单位向量*权重后的点，可以理解为残差，构建雅可比矩阵时会用。
            std::vector<int> pointSearchInd;
            std::vector<float> pointSearchSqDis;

            pointOri = laserCloudCornerLastDS->points[i]; // 角点（坐标还是lidar系）
            // 当前角点变换到地图坐标系下，利用transPointAssociateToMap进行坐标变换，从雷达系转换到世界系
            pointAssociateToMap(&pointOri, &pointSel);
            // 在局部角点map中查找当前角点相邻的5个角点
            kdtreeCornerFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

            cv::Mat matA1(3, 3, CV_32F, cv::Scalar::all(0)); // 五个点的协方差矩阵(五个点到其中心点的距离)
            cv::Mat matD1(1, 3, CV_32F, cv::Scalar::all(0)); // PCA分析后的特征值
            cv::Mat matV1(3, 3, CV_32F, cv::Scalar::all(0)); // PCA分析后的特征向量
                    
            // 如果前五个点与当前点距离都小于1m，开始PCA分析
            if (pointSearchSqDis[4] < 1.0) {
                // 求五个点的中心点
                float cx = 0, cy = 0, cz = 0;
                for (int j = 0; j < 5; j++) {
                    cx += laserCloudCornerFromMapDS->points[pointSearchInd[j]].x;
                    cy += laserCloudCornerFromMapDS->points[pointSearchInd[j]].y;
                    cz += laserCloudCornerFromMapDS->points[pointSearchInd[j]].z;
                }
                cx /= 5; cy /= 5;  cz /= 5;

                // 求五个点的协方差矩阵matA1(这五个点到中心点的距离)
                float a11 = 0, a12 = 0, a13 = 0, a22 = 0, a23 = 0, a33 = 0;
                for (int j = 0; j < 5; j++) {
                    // 计算点与中心点之间的距离
                    float ax = laserCloudCornerFromMapDS->points[pointSearchInd[j]].x - cx;
                    float ay = laserCloudCornerFromMapDS->points[pointSearchInd[j]].y - cy;
                    float az = laserCloudCornerFromMapDS->points[pointSearchInd[j]].z - cz;

                    a11 += ax * ax; a12 += ax * ay; a13 += ax * az;
                    a22 += ay * ay; a23 += ay * az;
                    a33 += az * az;
                }
                a11 /= 5; a12 /= 5; a13 /= 5; a22 /= 5; a23 /= 5; a33 /= 5;

                // 构建协方差矩阵
                matA1.at<float>(0, 0) = a11; matA1.at<float>(0, 1) = a12; matA1.at<float>(0, 2) = a13;
                matA1.at<float>(1, 0) = a12; matA1.at<float>(1, 1) = a22; matA1.at<float>(1, 2) = a23;
                matA1.at<float>(2, 0) = a13; matA1.at<float>(2, 1) = a23; matA1.at<float>(2, 2) = a33;

                // 协方差矩阵与点云中角点面点之间的关系:
                // 1.假设点云序列为S，计算 S 的协方差矩阵，记为 cov_mat ，cov_mat 的特征值记为 V ，特征向量记为 E 。
                // 2.如果 S 分布在一条线段上，那么 V 中一个特征值就会明显比其他两个大，E 中与较大特征值相对应的特征向量代表边缘线的方向。(一大两小，大的代表直线方向)
                // 3.如果 S 分布在一块平面上，那么 V 中一个特征值就会明显比其他两个小，E 中与较小特征值相对应的特征向量代表平面片的方向。(一小两大，小方向)边缘线或平面块的位置通过穿过 S 的几何中心来确定。
                
                // 调用cv库计算协方差矩阵的特征值和特征向量，用于判断这5个点是不是呈线状分布，此为PCA的原理；
                cv::eigen(matA1, matD1, matV1);

                // 如果5个点呈线状分布，最大的特征值对应的特征向量就是该线的方向向量；
                // 如果五个点的最大特征值 远大于 第二大的特征值
                if (matD1.at<float>(0, 0) > 3 * matD1.at<float>(0, 1)) {

                    float x0 = pointSel.x;
                    float y0 = pointSel.y;
                    float z0 = pointSel.z;
                    // 从中心点沿着方向向量向两端移动0.1m，构造线上的两个点A(x1 y1 z1)、B(x2 y2 z2)；
                    float x1 = cx + 0.1 * matV1.at<float>(0, 0);
                    float y1 = cy + 0.1 * matV1.at<float>(0, 1);
                    float z1 = cz + 0.1 * matV1.at<float>(0, 2);
                    float x2 = cx - 0.1 * matV1.at<float>(0, 0);
                    float y2 = cy - 0.1 * matV1.at<float>(0, 1);
                    float z2 = cz - 0.1 * matV1.at<float>(0, 2);

                    // 向量OA = (x0 - x1, y0 - y1, z0 - z1), 向量OB = (x0 - x2, y0 - y2, z0 - z2)，向量AB = （x1 - x2, y1 - y2, z1 - z2）;
                    // 点到线的距离，d = |向量OA 叉乘 向量OB|/|AB|;
                    // 向量OA 叉乘 向量OB 得到的向量模长 ： 是垂直a、b所在平面，且以|b|·sinθ为高、|a|为底的平行四边形的面积(即三角形OAB面积的二倍)，
                    // 因此|向量OA 叉乘 向量OB|再除以|AB|的模长，则得到高度，即点到线的距离；

                    // OA×OB叉乘向量的模长，也就是平行四边形面积
                    float a012 = sqrt(((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) * ((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                                    + ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) * ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) 
                                    + ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1)) * ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1)));

                    // AB的模长，也就是对角线长度
                    float l12 = sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));

                    // 残差对当前点x0，y0，z0的偏导。计算O到AB垂线的反方向的单位向量，坐标分别为la、lb、lc
                    float la = ((y1 - y2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                              + (z1 - z2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))) / a012 / l12;

                    float lb = -((x1 - x2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                               - (z1 - z2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;

                    float lc = -((x1 - x2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) 
                               + (y1 - y2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;

                    // 三角形的高，即为点线之间的距离
                    float ld2 = a012 / l12;

                    // 点线距离权重，距离越近，权重越大。有点类似鲁棒核函数的感觉或信息矩阵的感觉  fabs()表示浮点数的绝对值
                    float s = 1 - 0.9 * fabs(ld2);

                    // 点到直线的垂线段单位向量，可理解为残差的方向
                    coeff.x = s * la;
                    coeff.y = s * lb;
                    coeff.z = s * lc;
                    // 点到直线距离，可理解为残差的大小
                    coeff.intensity = s * ld2;

                    // 如果点线距离小于1m
                    if (s > 0.1) {
                        // 保存优化点，残差（带权重），标记这个被优化点索引
                        laserCloudOriCornerVec[i] = pointOri;
                        coeffSelCornerVec[i] = coeff;
                        laserCloudOriCornerFlag[i] = true;
                    }
                }
            }
        }
    }

    // 构造面点残差
    void surfOptimization()
    {
        updatePointAssociateToMap();

        #pragma omp parallel for num_threads(numberOfCores)
        for (int i = 0; i < laserCloudSurfLastDSNum; i++)
        {
            PointType pointOri, pointSel, coeff; // pointOri当前帧的一个面点(雷达坐标系)、pointSel当前帧的一个面点(世界坐标系)、coeff带有权重的平面法向量，可以理解为残差的方向和大小
            std::vector<int> pointSearchInd;
            std::vector<float> pointSearchSqDis;

            pointOri = laserCloudSurfLastDS->points[i];
            pointAssociateToMap(&pointOri, &pointSel); 
            kdtreeSurfFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

            Eigen::Matrix<float, 5, 3> matA0;
            Eigen::Matrix<float, 5, 1> matB0;
            Eigen::Vector3f matX0;

            matA0.setZero();
            matB0.fill(-1); // 设为-1的原因是Ax + By + Cz = -1
            matX0.setZero();

            // 距离小于1的话
            if (pointSearchSqDis[4] < 1.0) {
                // 求面的法向量不是用的PCA，使用的是最小二乘拟合；
                // 假设平面不通过原点，则平面的一般方程为Ax + By + Cz + 1 = 0，用这个假设可以少算一个参数；
                for (int j = 0; j < 5; j++) {
                    /*  matA0 5*3
                        x1 y1 z1
                        x2 y2 z2
                        x3 y3 z3
                        x4 y4 z4
                        x5 y5 z5
                    */
                    matA0(j, 0) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].x;
                    matA0(j, 1) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].y;
                    matA0(j, 2) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].z;
                }

                // 构建超定方程组： matA0 * norm（A, B, C） = matB0；(五个方程，三个未知数ABC)
                // 求解这个最小二乘问题，可得平面的法向量norm（A, B, C）
                matX0 = matA0.colPivHouseholderQr().solve(matB0);

                // 平面的一般方程Ax + By + Cz + 1 = 0的A、B、C分别为pa、pb、pc。
                float pa = matX0(0, 0);
                float pb = matX0(1, 0);
                float pc = matX0(2, 0);
                float pd = 1;

                // Ax + By + Cz + 1 = 0，全部除以法向量的模长，方程依旧成立，而且使得法向量归一化了；
                float ps = sqrt(pa * pa + pb * pb + pc * pc);
                pa /= ps; pb /= ps; pc /= ps; pd /= ps;

                // 点(x0, y0, z0)到平面Ax + By + Cz + D = 0 的距离 = fabs(A*x0 + B*y0 + C*z0 + D) / sqrt(A^2 + B^2 + C^2)；
                // 因为法向量（A, B, C）已经归一化了，所以距离公式可以简写为：距离 = fabs(A*x0 + B*y0 + C*z0 + D) ；
                // 检查平面是否合格，如果5个点中有点到平面的距离超过0.2m，那么认为这些点太分散了，不构成平面
                bool planeValid = true;
                for (int j = 0; j < 5; j++) {
                    if (fabs(pa * laserCloudSurfFromMapDS->points[pointSearchInd[j]].x +
                             pb * laserCloudSurfFromMapDS->points[pointSearchInd[j]].y +
                             pc * laserCloudSurfFromMapDS->points[pointSearchInd[j]].z + pd) > 0.2) {
                        planeValid = false;
                        break;
                    }
                }

                // 平面合格
                if (planeValid) {
                    // 点到平面的距离，带代入平面方程即可。注意这里没取绝对值
                    float pd2 = pa * pointSel.x + pb * pointSel.y + pc * pointSel.z + pd;

                    // 权重，距离雷达越远，权重约大 有点类似鲁棒核函数的感觉或信息矩阵的感觉
                    float s = 1 - 0.9 * fabs(pd2) / sqrt(sqrt(pointSel.x * pointSel.x
                            + pointSel.y * pointSel.y + pointSel.z * pointSel.z));

                    // 残差的方向(平面法向量)
                    coeff.x = s * pa;
                    coeff.y = s * pb;
                    coeff.z = s * pc;
                    // 残差的大小(权重 * 距离)
                    coeff.intensity = s * pd2;

                    if (s > 0.1) {
                        laserCloudOriSurfVec[i] = pointOri;
                        coeffSelSurfVec[i] = coeff;
                        laserCloudOriSurfFlag[i] = true;
                    }
                }
            }
        }
    }

    // 联合两类残差
    void combineOptimizationCoeffs()
    {
        // combine corner coeffs 把所有将要被优化的角点及其残差放入容器
        for (int i = 0; i < laserCloudCornerLastDSNum; ++i){
            if (laserCloudOriCornerFlag[i] == true){
                laserCloudOri->push_back(laserCloudOriCornerVec[i]);
                coeffSel->push_back(coeffSelCornerVec[i]);
            }
        }
        // combine surf coeffs 把所有将要被优化的面点及其残差放入容器
        for (int i = 0; i < laserCloudSurfLastDSNum; ++i){
            if (laserCloudOriSurfFlag[i] == true){
                laserCloudOri->push_back(laserCloudOriSurfVec[i]);
                coeffSel->push_back(coeffSelSurfVec[i]);
            }
        }
        // reset flag for next iteration  重置标记数组，保证下一轮优化
        std::fill(laserCloudOriCornerFlag.begin(), laserCloudOriCornerFlag.end(), false);
        std::fill(laserCloudOriSurfFlag.begin(), laserCloudOriSurfFlag.end(), false);
    }

    // 非线性优化
    bool LMOptimization(int iterCount)
    {
        // This optimization is from the original loam_velodyne, need to cope with coordinate transformation
        // lidar <- camera      ---     camera <- lidar
        // x = z                ---     x = y
        // y = x                ---     y = z
        // z = y                ---     z = x
        // roll = yaw           ---     roll = pitch
        // pitch = roll         ---     pitch = yaw
        // yaw = pitch          ---     yaw = roll

        // lidar -> camera  转换到相机坐标系下，与论文公式对应，在建立雅克比矩阵时转换回来
        // 用于表示雅克比矩阵
        float srx = sin(transformTobeMapped[1]);
        float crx = cos(transformTobeMapped[1]);
        float sry = sin(transformTobeMapped[2]);
        float cry = cos(transformTobeMapped[2]);
        float srz = sin(transformTobeMapped[0]);
        float crz = cos(transformTobeMapped[0]);

        // 如果当前帧匹配特征数量太少
        int laserCloudSelNum = laserCloudOri->size();
        if (laserCloudSelNum < 50) {
            return false;
        }

        // matA雅克比矩阵 matB代价函数误差矩阵  matAtA为Jt*信息矩阵*J
        // 残差(这里是点到线、面的距离)对优化变量(x、y、z、rx、ry、rz)求导，根据链式法则，转换成残差对方向求导*方向对点求导*点对优化变量求导。
        // 其中残差对方向求导*方向对点求导即为前面的coeff.x y z，也就是残差的方向，也就是点到线、面的垂线向量
        // 点对优化变量求导：点Pw = R*P+t，如果对旋转求导，则比较复杂，就是下面的arx、ary、arz内的部分内容，对位置求导，直接为1
        cv::Mat matA(laserCloudSelNum, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat matAt(6, laserCloudSelNum, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtA(6, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat matB(laserCloudSelNum, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtB(6, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matX(6, 1, CV_32F, cv::Scalar::all(0));

        PointType pointOri, coeff;

        // 遍历匹配特征点，构建雅可比矩阵
        for (int i = 0; i < laserCloudSelNum; i++) {
            // lidar -> camera
            pointOri.x = laserCloudOri->points[i].y;
            pointOri.y = laserCloudOri->points[i].z;
            pointOri.z = laserCloudOri->points[i].x;
            // lidar -> camera
            coeff.x = coeffSel->points[i].y;
            coeff.y = coeffSel->points[i].z;
            coeff.z = coeffSel->points[i].x;
            coeff.intensity = coeffSel->points[i].intensity;
            // in camera 旋转矩阵对x方向旋转的偏导 即雅可比
            float arx = (crx*sry*srz*pointOri.x + crx*crz*sry*pointOri.y - srx*sry*pointOri.z) * coeff.x
                      + (-srx*srz*pointOri.x - crz*srx*pointOri.y - crx*pointOri.z) * coeff.y
                      + (crx*cry*srz*pointOri.x + crx*cry*crz*pointOri.y - cry*srx*pointOri.z) * coeff.z;
            // 旋转矩阵对y方向旋转的偏导 即雅可比
            float ary = ((cry*srx*srz - crz*sry)*pointOri.x 
                      + (sry*srz + cry*crz*srx)*pointOri.y + crx*cry*pointOri.z) * coeff.x
                      + ((-cry*crz - srx*sry*srz)*pointOri.x 
                      + (cry*srz - crz*srx*sry)*pointOri.y - crx*sry*pointOri.z) * coeff.z;
            // 旋转矩阵对z方向旋转的偏导 即雅可比
            float arz = ((crz*srx*sry - cry*srz)*pointOri.x + (-cry*crz-srx*sry*srz)*pointOri.y)*coeff.x
                      + (crx*crz*pointOri.x - crx*srz*pointOri.y) * coeff.y
                      + ((sry*srz + cry*crz*srx)*pointOri.x + (crz*sry-cry*srx*srz)*pointOri.y)*coeff.z;
            // lidar -> camera 从相机坐标系转换回LiDAR坐标系，z，x，y对应x，y，z
            matA.at<float>(i, 0) = arz;
            matA.at<float>(i, 1) = arx;
            matA.at<float>(i, 2) = ary;
            matA.at<float>(i, 3) = coeff.z;
            matA.at<float>(i, 4) = coeff.x;
            matA.at<float>(i, 5) = coeff.y;
            matB.at<float>(i, 0) = -coeff.intensity; // 点到直线距离、平面距离，作为观测值
        }

        cv::transpose(matA, matAt);
        matAtA = matAt * matA;
        matAtB = matAt * matB;
        // J^T·J·delta_x = -J^T·f 高斯牛顿
        cv::solve(matAtA, matAtB, matX, cv::DECOMP_QR);

        // 首次迭代，检查近似Hessian矩阵（J^T·J）是否退化，或者称为奇异，行列式值=0
        if (iterCount == 0) 
        {
            cv::Mat matE(1, 6, CV_32F, cv::Scalar::all(0));
            cv::Mat matV(6, 6, CV_32F, cv::Scalar::all(0));
            cv::Mat matV2(6, 6, CV_32F, cv::Scalar::all(0));

            // 为了防止场景几何特征退化，用如下方式进行验证（视觉特征退化只需要统计特征点数量和特征点被跟踪次数）
            // 对AtA进行特征分解

            cv::eigen(matAtA, matE, matV);
            matV.copyTo(matV2);

            isDegenerate = false;
            float eignThre[6] = {100, 100, 100, 100, 100, 100}; // 理想情况是6个自由度的特征值都很大（都大于某个常数），否则就退化了
            for (int i = 5; i >= 0; i--) {
                if (matE.at<float>(0, i) < eignThre[i]) {
                    for (int j = 0; j < 6; j++) {
                        matV2.at<float>(i, j) = 0;
                    }
                    isDegenerate = true;
                } else {
                    break;
                }
            }
            matP = matV.inv() * matV2;
        }

        // 点云退化了
        if (isDegenerate) {
            cv::Mat matX2(6, 1, CV_32F, cv::Scalar::all(0));
            matX.copyTo(matX2);
            matX = matP * matX2; // 更新matX，让退化部分增量为0，从而让退化自由度保持原来的值
        }

        // 更新当前帧位姿 x = x + delta_x
        transformTobeMapped[0] += matX.at<float>(0, 0);
        transformTobeMapped[1] += matX.at<float>(1, 0);
        transformTobeMapped[2] += matX.at<float>(2, 0);
        transformTobeMapped[3] += matX.at<float>(3, 0);
        transformTobeMapped[4] += matX.at<float>(4, 0);
        transformTobeMapped[5] += matX.at<float>(5, 0);

        // 统计增量的值，保证迭代是收敛的
        float deltaR = sqrt(
                            pow(pcl::rad2deg(matX.at<float>(0, 0)), 2) +
                            pow(pcl::rad2deg(matX.at<float>(1, 0)), 2) +
                            pow(pcl::rad2deg(matX.at<float>(2, 0)), 2));
        float deltaT = sqrt(
                            pow(matX.at<float>(3, 0) * 100, 2) +
                            pow(matX.at<float>(4, 0) * 100, 2) +
                            pow(matX.at<float>(5, 0) * 100, 2));

        // delta_x很小，认为收敛
        if (deltaR < 0.05 && deltaT < 0.05) {
            return true; // converged
        }
        return false; // keep optimizing
    }

    // 当前帧位姿估计优化，求解当前帧位姿transformTobeMapped
    void scan2MapOptimization()
    {
        // 验证是否是第一帧，第一帧的话这个是空的
        if (cloudKeyPoses3D->points.empty())
            return;

        // 特征点数量足够
        if (laserCloudCornerLastDSNum > edgeFeatureMinValidNum && laserCloudSurfLastDSNum > surfFeatureMinValidNum)
        {
            // 设置局部地图的kdtree，方便构建残差
            kdtreeCornerFromMap->setInputCloud(laserCloudCornerFromMapDS);
            kdtreeSurfFromMap->setInputCloud(laserCloudSurfFromMapDS);

            for (int iterCount = 0; iterCount < 30; iterCount++)
            {
                laserCloudOri->clear();
                coeffSel->clear();

                cornerOptimization();
                surfOptimization();

                combineOptimizationCoeffs();

                if (LMOptimization(iterCount) == true)
                    break;              
            }

            transformUpdate();
        } else {
            ROS_WARN("Not enough features! Only %d edge and %d planar features available.", laserCloudCornerLastDSNum, laserCloudSurfLastDSNum);
        }
    }

    // 位姿更新
    // 用imu原始RPY数据与scan-to-map优化后的位姿进行加权融合，更新当前帧位姿的roll、pitch，约束z坐标
    void transformUpdate()
    {
        if (cloudInfo.imuAvailable == true)
        {
            // 俯仰角小于1.4
            if (std::abs(cloudInfo.imuPitchInit) < 1.4)
            {
                double imuWeight = 0.01;
                tf::Quaternion imuQuaternion;
                tf::Quaternion transformQuaternion;
                double rollMid, pitchMid, yawMid;

                // slerp roll  roll角求加权均值，用scan-to-map优化得到的位姿与imu原始RPY数据，进行加权平均
                transformQuaternion.setRPY(transformTobeMapped[0], 0, 0);
                imuQuaternion.setRPY(cloudInfo.imuRollInit, 0, 0);
                tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
                transformTobeMapped[0] = rollMid;

                // slerp pitch   pitch角加权均值，用scan-to-map优化得到的位姿与imu原始RPY数据，进行加权平均
                transformQuaternion.setRPY(0, transformTobeMapped[1], 0);
                imuQuaternion.setRPY(0, cloudInfo.imuPitchInit, 0);
                tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
                transformTobeMapped[1] = pitchMid;
            }
        }

        // 更新当前帧位姿的roll，pitch，z坐标；因为是小车，roll、pitch是相对稳定的，不会有很大变动，一定程度上可以信赖imu数据，z是进行高度约束
        transformTobeMapped[0] = constraintTransformation(transformTobeMapped[0], rotation_tollerance);
        transformTobeMapped[1] = constraintTransformation(transformTobeMapped[1], rotation_tollerance);
        transformTobeMapped[5] = constraintTransformation(transformTobeMapped[5], z_tollerance);
    }

    float constraintTransformation(float value, float limit)
    {
        if (value < -limit)
            value = -limit;
        if (value > limit)
            value = limit;

        return value;
    }

    // 根据当前帧与上一个关键帧之间的运动幅度的大小，判断是否将当前帧设为关键帧
    bool saveFrame()
    {
        // 第一帧不必判定，直接认定为关键帧
        if (cloudKeyPoses3D->points.empty())
            return true;

        // 计算上一个关键帧与当前帧之间的变换矩阵
        Eigen::Affine3f transStart = pclPointToAffine3f(cloudKeyPoses6D->back()); // 上一个关键帧的位姿
        Eigen::Affine3f transFinal = pcl::getTransformation(transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5], 
                                                            transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]); // 当前帧经过LMOptimization优化后的位姿
        Eigen::Affine3f transBetween = transStart.inverse() * transFinal;
        float x, y, z, roll, pitch, yaw;
        pcl::getTranslationAndEulerAngles(transBetween, x, y, z, roll, pitch, yaw);

        // 如果旋转角和位移有全都小于阈值，则判定非关键帧，否则判定为关键帧
        if (abs(roll)  < surroundingkeyframeAddingAngleThreshold &&
            abs(pitch) < surroundingkeyframeAddingAngleThreshold && 
            abs(yaw)   < surroundingkeyframeAddingAngleThreshold &&
            sqrt(x*x + y*y + z*z) < surroundingkeyframeAddingDistThreshold)
            return false;

        return true;
    }

    // 加入里程计因子
    void addOdomFactor()
    {
        // 第一帧时的特殊对待
        if (cloudKeyPoses3D->points.empty())
        {
            // 第一帧采用的就是IMU的旋转测量，
            // 重力在yaw角平面的投影为0，因此yaw角是不可观的，误差较大，因此方差设置较大
            // 初始启动时运动速度较慢，因此初始位移设为0，0，0，并且方差设置较大
            noiseModel::Diagonal::shared_ptr priorNoise = noiseModel::Diagonal::Variances((Vector(6) << 1e-2, 1e-2, M_PI*M_PI, 1e8, 1e8, 1e8).finished()); // rad*rad, meter*meter 顺序为rpyxyz
            gtSAMgraph.add(PriorFactor<Pose3>(0, trans2gtsamPose(transformTobeMapped), priorNoise));
            initialEstimate.insert(0, trans2gtsamPose(transformTobeMapped));
        }
        // 普通关键帧就按照一般情况设置方差即可
        else{
            noiseModel::Diagonal::shared_ptr odometryNoise = noiseModel::Diagonal::Variances((Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());
            gtsam::Pose3 poseFrom = pclPointTogtsamPose3(cloudKeyPoses6D->points.back());
            gtsam::Pose3 poseTo   = trans2gtsamPose(transformTobeMapped);
            // 参数：前一帧id，当前帧id，前一帧与当前帧的位姿变换（作为观测值），噪声协方差
            gtSAMgraph.add(BetweenFactor<Pose3>(cloudKeyPoses3D->size()-1, cloudKeyPoses3D->size(), poseFrom.between(poseTo), odometryNoise));
            initialEstimate.insert(cloudKeyPoses3D->size(), poseTo);
            std::cout << "odo的 " << transformTobeMapped[3] << " " << transformTobeMapped[4] << " " << transformTobeMapped[5] << std::endl; // NNS
            // if (isDegenerate)
            // {
                // adding VINS constraints is deleted as benefits are not obvious, disable for now
                // gtSAMgraph.add(BetweenFactor<Pose3>(cloudKeyPoses3D->size()-1, cloudKeyPoses3D->size(), vinsPoseFrom.between(vinsPoseTo), odometryNoise));
            // }
        }
    }

    // 加入GPS因子，GPS的绝对位置测量的误差大约在3~5m
    void addGPSFactor()
    {
        if (gpsQueue.empty())
            return;

        // wait for system initialized and settles down 等待系统运行一段距离，短距离内漂移不会太大
        if (cloudKeyPoses3D->points.empty())
            return;
        else if (pointDistance(cloudKeyPoses3D->front(), cloudKeyPoses3D->back()) < 5.0)
            return;

        // pose covariance small, no need to correct  查看最新关键帧的协方差，如果协方差小于阈值，则不需要GPS因子进行修正
        if (poseCovariance(3,3) < poseCovThreshold && poseCovariance(4,4) < poseCovThreshold)
            return;

        // last gps position
        static PointType lastGPSPoint;

        while (!gpsQueue.empty())
        {
            // GPS时间戳与点云时间戳对齐
            if (gpsQueue.front().header.stamp.toSec() < timeLaserInfoCur - 0.2)
            {
                // message too old
                gpsQueue.pop_front();
            }
            else if (gpsQueue.front().header.stamp.toSec() > timeLaserInfoCur + 0.2)
            {
                // message too new
                break;
            }
            else
            {
                nav_msgs::Odometry thisGPS = gpsQueue.front();
                gpsQueue.pop_front();

                // GPS too noisy, skip  GPS噪声太大时放弃这一次测量
                float noise_x = thisGPS.pose.covariance[0];
                float noise_y = thisGPS.pose.covariance[7];
                float noise_z = thisGPS.pose.covariance[14];
                if (noise_x > gpsCovThreshold || noise_y > gpsCovThreshold)
                    continue;

                float gps_x = thisGPS.pose.pose.position.x;
                float gps_y = thisGPS.pose.pose.position.y;
                float gps_z = thisGPS.pose.pose.position.z;
                // 可以设置不使用GPS的z坐标，因为通常路面起伏不会太大
                if (!useGpsElevation)
                {
                    gps_z = transformTobeMapped[5];
                    noise_z = 0.01;
                }

                // GPS not properly initialized (0,0,0) GPS测量接近于0时，说明没有正确初始化，放弃添加
                if (abs(gps_x) < 1e-6 && abs(gps_y) < 1e-6)
                    continue;

                // Add GPS every a few meters
                PointType curGPSPoint;
                curGPSPoint.x = gps_x;
                curGPSPoint.y = gps_y;
                curGPSPoint.z = gps_z;
                // 本身GPS误差就在3~5m，因此当两次测量距离在5m以内时，放弃本次添加
                if (pointDistance(curGPSPoint, lastGPSPoint) < 5.0)
                    continue;
                else
                    lastGPSPoint = curGPSPoint;

                gtsam::Vector Vector3(3);
                // 设置最低噪声为1.0
                Vector3 << max(noise_x, 1.0f), max(noise_y, 1.0f), max(noise_z, 1.0f);
                noiseModel::Diagonal::shared_ptr gps_noise = noiseModel::Diagonal::Variances(Vector3);
                gtsam::GPSFactor gps_factor(cloudKeyPoses3D->size(), gtsam::Point3(gps_x, gps_y, gps_z), gps_noise);
                gtSAMgraph.add(gps_factor);

                aLoopIsClosed = true;

                break;
            }
        }
    }

    // 加入回环因子
    void addLoopFactor()
    {
        if (loopIndexQueue.empty())
            return;

        for (size_t i = 0; i < loopIndexQueue.size(); ++i)
        {
            int indexFrom = loopIndexQueue[i].first;
            int indexTo = loopIndexQueue[i].second;
            gtsam::Pose3 poseBetween = loopPoseQueue[i];
            gtsam::noiseModel::Diagonal::shared_ptr noiseBetween = loopNoiseQueue[i];
            // 把闭环约束添加到图中
            gtSAMgraph.add(BetweenFactor<Pose3>(indexFrom, indexTo, poseBetween, noiseBetween));
        }

        loopIndexQueue.clear();
        loopPoseQueue.clear();
        loopNoiseQueue.clear();
        // 设置回环标志位，允许根据回环纠正位姿
        aLoopIsClosed = true;
    }

    // 保存关键帧和因子图优化
    // 对于关键帧加入里程计因子、回环因子，进入全局位姿图优化，最新帧位姿保存至transformTobeMapper，更新globalPath中的最新位姿
    void saveKeyFramesAndFactor()
    {
        if (saveFrame() == false)
            return;

        // odom factor
        addOdomFactor();

        // gps factor
        // addGPSFactor();

        // loop factor
        addLoopFactor();

        // update iSAM 保存因子图到isam中，进行因子图优化
        isam->update(gtSAMgraph, initialEstimate);
        isam->update();
        
        // 这一段因子图已经被isam保存了  所以重置gtSAMgraph因子图和initialEstimate初始估计，方便建立下一段因子图
        gtSAMgraph.resize(0);
        initialEstimate.clear();

        //save key poses  保存优化结果
        PointType thisPose3D;
        PointTypePose thisPose6D;
        Pose3 latestEstimate;

        isamCurrentEstimate = isam->calculateEstimate();
        latestEstimate = isamCurrentEstimate.at<Pose3>(isamCurrentEstimate.size()-1);
        // cout << "****************************************************" << endl;
        // isamCurrentEstimate.print("Current estimate: ");

        thisPose3D.x = latestEstimate.translation().x();
        thisPose3D.y = latestEstimate.translation().y();
        thisPose3D.z = latestEstimate.translation().z();
        thisPose3D.intensity = cloudKeyPoses3D->size(); // this can be used as index 借用PCL点云点反射强度的数据结构保存关键帧索引
        cloudKeyPoses3D->push_back(thisPose3D);

        thisPose6D.x = thisPose3D.x;
        thisPose6D.y = thisPose3D.y;
        thisPose6D.z = thisPose3D.z;
        thisPose6D.intensity = thisPose3D.intensity ; // this can be used as index
        thisPose6D.roll  = latestEstimate.rotation().roll();
        thisPose6D.pitch = latestEstimate.rotation().pitch();
        thisPose6D.yaw   = latestEstimate.rotation().yaw();
        thisPose6D.time = timeLaserInfoCur;
        cloudKeyPoses6D->push_back(thisPose6D);

        // cout << "****************************************************" << endl;
        // cout << "Pose covariance:" << endl;
        // cout << isam->marginalCovariance(isamCurrentEstimate.size()-1) << endl << endl;
        // 取出最新关键帧的位姿协方差，与GPS对比的
        poseCovariance = isam->marginalCovariance(isamCurrentEstimate.size()-1);

        // save updated transform
        transformTobeMapped[0] = latestEstimate.rotation().roll();
        transformTobeMapped[1] = latestEstimate.rotation().pitch();
        transformTobeMapped[2] = latestEstimate.rotation().yaw();
        transformTobeMapped[3] = latestEstimate.translation().x();
        transformTobeMapped[4] = latestEstimate.translation().y();
        transformTobeMapped[5] = latestEstimate.translation().z();

        // save all the received edge and surf points  保存关键帧特征点云数组
        pcl::PointCloud<PointType>::Ptr thisCornerKeyFrame(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr thisSurfKeyFrame(new pcl::PointCloud<PointType>());
        pcl::copyPointCloud(*laserCloudCornerLastDS,  *thisCornerKeyFrame);
        pcl::copyPointCloud(*laserCloudSurfLastDS,    *thisSurfKeyFrame);

        // save key frame cloud
        cornerCloudKeyFrames.push_back(thisCornerKeyFrame);
        surfCloudKeyFrames.push_back(thisSurfKeyFrame);

        // save path for visualization 保存位姿到可视化的topic
        updatePath(thisPose6D);
        // ！！！isam优化后，实际上将所有的关键帧位姿都优化过了，但是这里只push_back了最新的位姿，而没将之前的位姿更新到最新优化后的结果。这样是因为处理量太大了，因此只在出现回环时，才更新所有优化的位姿，也就是correctPoses()函数。
    }

    // 遇到回环时纠正位姿
    void correctPoses()
    {
        if (cloudKeyPoses3D->points.empty())
            return;

        // 如果存在回环，则将关键帧位姿数组更新为因子图优化后的结果
        if (aLoopIsClosed == true)
        {
            // clear path
            globalPath.poses.clear();

            // update key poses 更新因子图中所有变量节点的位姿，也就是所有历史关键帧的位姿
            int numPoses = isamCurrentEstimate.size();
            for (int i = 0; i < numPoses; ++i)
            {
                cloudKeyPoses3D->points[i].x = isamCurrentEstimate.at<Pose3>(i).translation().x();
                cloudKeyPoses3D->points[i].y = isamCurrentEstimate.at<Pose3>(i).translation().y();
                cloudKeyPoses3D->points[i].z = isamCurrentEstimate.at<Pose3>(i).translation().z();

                cloudKeyPoses6D->points[i].x = cloudKeyPoses3D->points[i].x;
                cloudKeyPoses6D->points[i].y = cloudKeyPoses3D->points[i].y;
                cloudKeyPoses6D->points[i].z = cloudKeyPoses3D->points[i].z;
                cloudKeyPoses6D->points[i].roll  = isamCurrentEstimate.at<Pose3>(i).rotation().roll();
                cloudKeyPoses6D->points[i].pitch = isamCurrentEstimate.at<Pose3>(i).rotation().pitch();
                cloudKeyPoses6D->points[i].yaw   = isamCurrentEstimate.at<Pose3>(i).rotation().yaw();

                updatePath(cloudKeyPoses6D->points[i]);
            }

            aLoopIsClosed = false; // 回环纠正位姿结束，关闭回环纠正的标志位
            // ID for reseting IMU pre-integration  增加预积分的索引序号
            ++imuPreintegrationResetId;
        }
    }

    void publishOdometry()
    {
        // Publish odometry for ROS
        nav_msgs::Odometry laserOdometryROS;
        laserOdometryROS.header.stamp = timeLaserInfoStamp;
        laserOdometryROS.header.frame_id = "odom";
        laserOdometryROS.child_frame_id = "odom_mapping";
        laserOdometryROS.pose.pose.position.x = transformTobeMapped[3];
        laserOdometryROS.pose.pose.position.y = transformTobeMapped[4];
        laserOdometryROS.pose.pose.position.z = transformTobeMapped[5];
        laserOdometryROS.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
        laserOdometryROS.pose.covariance[0] = double(imuPreintegrationResetId);
        pubOdomAftMappedROS.publish(laserOdometryROS);
        // Publish TF
        static tf::TransformBroadcaster br;
        tf::Transform t_odom_to_lidar = tf::Transform(tf::createQuaternionFromRPY(transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]),
                                                      tf::Vector3(transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5]));
        tf::StampedTransform trans_odom_to_lidar = tf::StampedTransform(t_odom_to_lidar, timeLaserInfoStamp, "odom", "lidar_link");
        br.sendTransform(trans_odom_to_lidar);
    }

    void updatePath(const PointTypePose& pose_in)
    {
        geometry_msgs::PoseStamped pose_stamped;
        pose_stamped.header.stamp = ros::Time().fromSec(pose_in.time);
        pose_stamped.header.frame_id = "odom";
        pose_stamped.pose.position.x = pose_in.x;
        pose_stamped.pose.position.y = pose_in.y;
        pose_stamped.pose.position.z = pose_in.z;
        tf::Quaternion q = tf::createQuaternionFromRPY(pose_in.roll, pose_in.pitch, pose_in.yaw);
        pose_stamped.pose.orientation.x = q.x();
        pose_stamped.pose.orientation.y = q.y();
        pose_stamped.pose.orientation.z = q.z();
        pose_stamped.pose.orientation.w = q.w();

        globalPath.poses.push_back(pose_stamped);
    }

    void publishFrames()
    {
        if (cloudKeyPoses3D->points.empty())
            return;
        // publish key poses 发布关键帧位姿
        publishCloud(&pubKeyPoses, cloudKeyPoses6D, timeLaserInfoStamp, "odom");
        // Publish surrounding key frames  发布当前帧的附近帧聚合的局部面点地图
        publishCloud(&pubRecentKeyFrames, laserCloudSurfFromMapDS, timeLaserInfoStamp, "odom");
        // publish registered key frame  如果有订阅者，发布当前帧特征点云（角点、面点）
        if (pubRecentKeyFrame.getNumSubscribers() != 0)
        {
            pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
            // 将点云转换到世界坐标系发出去
            PointTypePose thisPose6D = trans2PointTypePose(transformTobeMapped);
            *cloudOut += *transformPointCloud(laserCloudCornerLastDS,  &thisPose6D);
            *cloudOut += *transformPointCloud(laserCloudSurfLastDS,    &thisPose6D);
            publishCloud(&pubRecentKeyFrame, cloudOut, timeLaserInfoStamp, "odom");
        }
        // publish registered high-res raw cloud 如果有订阅者，发布当前帧原始点云
        if (pubCloudRegisteredRaw.getNumSubscribers() != 0)
        {
            pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
            pcl::fromROSMsg(cloudInfo.cloud_deskewed, *cloudOut);
            // 将点云转换到世界坐标系发出去
            PointTypePose thisPose6D = trans2PointTypePose(transformTobeMapped);
            *cloudOut = *transformPointCloud(cloudOut,  &thisPose6D);
            publishCloud(&pubCloudRegisteredRaw, cloudOut, timeLaserInfoStamp, "odom");
        }
        // publish path 发布位移
        if (pubPath.getNumSubscribers() != 0)
        {
            globalPath.header.stamp = timeLaserInfoStamp;
            globalPath.header.frame_id = "odom";
            pubPath.publish(globalPath);
        }
    }
};


int main(int argc, char** argv)
{
    ros::init(argc, argv, "lidar");

    mapOptimization MO; // 主线程通过mapOptimization的构造函数完成地图优化

    ROS_INFO("\033[1;32m----> Lidar Map Optimization Started.\033[0m");
    
    std::thread loopDetectionthread(&mapOptimization::loopClosureThread, &MO); // LIO回环检测线程
    std::thread visualizeMapThread(&mapOptimization::visualizeGlobalMapThread, &MO); // 点云和地图保存线程

    ros::spin();

    loopDetectionthread.join();
    visualizeMapThread.join();

    return 0;
}
