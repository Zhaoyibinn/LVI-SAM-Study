#include "utility.h"
#include "lvi_sam/cloud_info.h"

struct smoothness_t{ 
    float value; // 曲率值
    size_t ind; // 激光点一维索引
};

// 曲率比较函数，从小到大排列，左边小返回1
struct by_value{ 
    bool operator()(smoothness_t const &left, smoothness_t const &right) { 
        return left.value < right.value;
    }
};

class FeatureExtraction : public ParamServer
{

public:

    ros::Subscriber subLaserCloudInfo;

    ros::Publisher pubLaserCloudInfo;
    ros::Publisher pubCornerPoints;
    ros::Publisher pubSurfacePoints;

    pcl::PointCloud<PointType>::Ptr extractedCloud; // 存放去畸变点云
    pcl::PointCloud<PointType>::Ptr cornerCloud; // 存放角点点云
    pcl::PointCloud<PointType>::Ptr surfaceCloud; // 存放面点点云

    pcl::VoxelGrid<PointType> downSizeFilter;

    lvi_sam::cloud_info cloudInfo; // 当前激光帧点云信息，包括的历史数据有：运动畸变矫正，点云数据，初始位姿，姿态角，有效点云数据，角点点云，平面点点云等
    std_msgs::Header cloudHeader;

    std::vector<smoothness_t> cloudSmoothness; // 存放所有点云平滑度及其序号
    float *cloudCurvature; // 存放所有点云的平滑度（曲率）
    int *cloudNeighborPicked; // 标记不再被提取的特征点（包含遮挡点和已被提取过的点及其附近点） 1表示遮挡、平行，或者已经进行特征提取的点，0表示还未进行特征提取的点
    int *cloudLabel; // 1表示角点，-1表示平面点

    FeatureExtraction()
    {
        subLaserCloudInfo = nh.subscribe<lvi_sam::cloud_info>(PROJECT_NAME + "/lidar/deskew/cloud_info", 5, &FeatureExtraction::laserCloudInfoHandler, this, ros::TransportHints().tcpNoDelay()); // 订阅当前激光帧运动畸变矫正后的点云信息

        pubLaserCloudInfo = nh.advertise<lvi_sam::cloud_info> (PROJECT_NAME + "/lidar/feature/cloud_info", 5);
        pubCornerPoints = nh.advertise<sensor_msgs::PointCloud2>(PROJECT_NAME + "/lidar/feature/cloud_corner", 5);
        pubSurfacePoints = nh.advertise<sensor_msgs::PointCloud2>(PROJECT_NAME + "/lidar/feature/cloud_surface", 5);
        
        initializationValue();
    }

    void initializationValue()
    {
        cloudSmoothness.resize(N_SCAN*Horizon_SCAN);

        downSizeFilter.setLeafSize(odometrySurfLeafSize, odometrySurfLeafSize, odometrySurfLeafSize);

        extractedCloud.reset(new pcl::PointCloud<PointType>());
        cornerCloud.reset(new pcl::PointCloud<PointType>());
        surfaceCloud.reset(new pcl::PointCloud<PointType>());

        cloudCurvature = new float[N_SCAN*Horizon_SCAN];
        cloudNeighborPicked = new int[N_SCAN*Horizon_SCAN];
        cloudLabel = new int[N_SCAN*Horizon_SCAN];
    }

    void laserCloudInfoHandler(const lvi_sam::cloud_infoConstPtr& msgIn)
    {
        cloudInfo = *msgIn; // new cloud info
        cloudHeader = msgIn->header; // new cloud header
        pcl::fromROSMsg(msgIn->cloud_deskewed, *extractedCloud); // new cloud for extraction

        calculateSmoothness();

        markOccludedPoints();

        extractFeatures();

        publishFeatureCloud();
    }

    // 计算当前激光帧点云中每个点的曲率
    void calculateSmoothness()
    {
        int cloudSize = extractedCloud->points.size();
        for (int i = 5; i < cloudSize - 5; i++)
        {
            float diffRange = cloudInfo.pointRange[i-5] + cloudInfo.pointRange[i-4]
                            + cloudInfo.pointRange[i-3] + cloudInfo.pointRange[i-2]
                            + cloudInfo.pointRange[i-1] - cloudInfo.pointRange[i] * 10
                            + cloudInfo.pointRange[i+1] + cloudInfo.pointRange[i+2]
                            + cloudInfo.pointRange[i+3] + cloudInfo.pointRange[i+4]
                            + cloudInfo.pointRange[i+5];            

            cloudCurvature[i] = diffRange*diffRange;//diffX * diffX + diffY * diffY + diffZ * diffZ;

            cloudNeighborPicked[i] = 0;
            cloudLabel[i] = 0;
            // cloudSmoothness for sorting
            cloudSmoothness[i].value = cloudCurvature[i];
            cloudSmoothness[i].ind = i;
        }
    }

    // 标记属于遮挡、平行两种情况的点，不做特征提取
    void markOccludedPoints()
    {
        int cloudSize = extractedCloud->points.size();
        // mark occluded points and parallel beam points 标记遮挡点和平行光束点
        for (int i = 5; i < cloudSize - 6; ++i)
        {
            // occluded points 当前点和下一个点的range值
            float depth1 = cloudInfo.pointRange[i];
            float depth2 = cloudInfo.pointRange[i+1];
            int columnDiff = std::abs(int(cloudInfo.pointColInd[i+1] - cloudInfo.pointColInd[i]));

            // 两个激光点之间的一维索引值，如果在一条扫描线上，那么值为1，如果两个点之间有一些无效点被剔除，可能会比1大，但不会特别大
            // 如果恰好前一个点在扫描一周的结束时刻，下一个点是另一条扫描线的起始时刻，那么值会更大
            if (columnDiff < 10){//判断是否在一个扫描线上
                // 10 pixel diff in range image
                if (depth1 - depth2 > 0.3){
                    cloudNeighborPicked[i - 5] = 1;
                    cloudNeighborPicked[i - 4] = 1;
                    cloudNeighborPicked[i - 3] = 1;
                    cloudNeighborPicked[i - 2] = 1;
                    cloudNeighborPicked[i - 1] = 1;
                    cloudNeighborPicked[i] = 1;
                }else if (depth2 - depth1 > 0.3){
                    cloudNeighborPicked[i + 1] = 1;
                    cloudNeighborPicked[i + 2] = 1;
                    cloudNeighborPicked[i + 3] = 1;
                    cloudNeighborPicked[i + 4] = 1;
                    cloudNeighborPicked[i + 5] = 1;
                    cloudNeighborPicked[i + 6] = 1;
                }
            }
            // parallel beam  用前后相邻点判断当前点所在平面是否与激光束方向平行 对于平行于激光光束的表面来说，即使距离激光雷达很近，依然会导致两个相邻点很远
            float diff1 = std::abs(float(cloudInfo.pointRange[i-1] - cloudInfo.pointRange[i]));
            float diff2 = std::abs(float(cloudInfo.pointRange[i+1] - cloudInfo.pointRange[i]));

            if (diff1 > 0.02 * cloudInfo.pointRange[i] && diff2 > 0.02 * cloudInfo.pointRange[i])
                cloudNeighborPicked[i] = 1;
        }
    }

    // 点云角点、平面点提取
    void extractFeatures()
    {
        cornerCloud->clear();
        surfaceCloud->clear();

        pcl::PointCloud<PointType>::Ptr surfaceCloudScan(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr surfaceCloudScanDS(new pcl::PointCloud<PointType>());

        for (int i = 0; i < N_SCAN; i++)
        {
            surfaceCloudScan->clear();

            // 将一条扫描线扫描一周的点云数据，划分6段，每段分开提取有限数量的特征，保证特征均匀分布
            for (int j = 0; j < 6; j++)
            {

                // sp、ep是这6段点云中每段点云的起始、结束索引；startRingIndex为扫描线起始第5个激光点在一维数组中的索引
                int sp = (cloudInfo.startRingIndex[i] * (6 - j) + cloudInfo.endRingIndex[i] * j) / 6;
                int ep = (cloudInfo.startRingIndex[i] * (5 - j) + cloudInfo.endRingIndex[i] * (j + 1)) / 6 - 1;

                if (sp >= ep)
                    continue;

                // 按照曲率从小到大排序点云
                std::sort(cloudSmoothness.begin()+sp, cloudSmoothness.begin()+ep, by_value());

                // 按照曲率从大到小遍历每段点云
                int largestPickedNum = 0;
                for (int k = ep; k >= sp; k--)
                {
                    int ind = cloudSmoothness[k].ind; // 激光点一维索引
                    // 当前激光点还未被处理，且曲率大于阈值，则认为是角点
                    if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] > edgeThreshold)
                    {
                        // 每段只提取20个角点，如果单条扫描线扫描一周是1800个点，则划分6段，每段最多300个点，从中提取20个角点
                        largestPickedNum++;
                        if (largestPickedNum <= 20){
                            cloudLabel[ind] = 1;
                            cornerCloud->push_back(extractedCloud->points[ind]);
                        } else {
                            break;
                        }

                        cloudNeighborPicked[ind] = 1; // 标记此点已被处理
                        // 同一条扫描线上后5个点标记
                        for (int l = 1; l <= 5; l++)
                        {
                            int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] - cloudInfo.pointColInd[ind + l - 1]));
                            if (columnDiff > 10)
                                break;
                            cloudNeighborPicked[ind + l] = 1;
                        }
                        // 同一条扫描线上前5个点标记
                        for (int l = -1; l >= -5; l--)
                        {
                            int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] - cloudInfo.pointColInd[ind + l + 1]));
                            if (columnDiff > 10)
                                break;
                            cloudNeighborPicked[ind + l] = 1;
                        }
                    }
                }

                // 按照曲率从小到大遍历
                for (int k = sp; k <= ep; k++)
                {
                    // 激光点的索引
                    int ind = cloudSmoothness[k].ind;
                    // 当前激光点还未被处理，且曲率小于阈值，则认为是平面点
                    if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] < surfThreshold)
                    {

                        cloudLabel[ind] = -1;
                        cloudNeighborPicked[ind] = 1; // 标记此点已被处理

                        // 同一条扫描线上后5个点标记
                        for (int l = 1; l <= 5; l++) {

                            int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] - cloudInfo.pointColInd[ind + l - 1]));
                            if (columnDiff > 10)
                                break;

                            cloudNeighborPicked[ind + l] = 1;
                        }
                        // 同一条扫描线上前5个点标记
                        for (int l = -1; l >= -5; l--) {

                            int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] - cloudInfo.pointColInd[ind + l + 1]));
                            if (columnDiff > 10)
                                break;

                            cloudNeighborPicked[ind + l] = 1;
                        }
                    }
                }
                // 平面点和未被处理的点，都认为是平面点，加入平面点云集合
                for (int k = sp; k <= ep; k++)
                {
                    if (cloudLabel[k] <= 0){
                        surfaceCloudScan->push_back(extractedCloud->points[k]);
                    }
                }extractedCloud
            }

            surfaceCloudScanDS->clear();
            downSizeFilter.setInputCloud(surfaceCloudScan);
            downSizeFilter.filter(*surfaceCloudScanDS);

            *surfaceCloud += *surfaceCloudScanDS;
        }
    }

    void freeCloudInfoMemory()
    {
        cloudInfo.startRingIndex.clear();
        cloudInfo.startRingIndex.shrink_to_fit();
        cloudInfo.endRingIndex.clear();
        cloudInfo.endRingIndex.shrink_to_fit();
        cloudInfo.pointColInd.clear();
        cloudInfo.pointColInd.shrink_to_fit();
        cloudInfo.pointRange.clear();
        cloudInfo.pointRange.shrink_to_fit();
    }

    void publishFeatureCloud()
    {
        // free cloud info memory
        freeCloudInfoMemory();
        // save newly extracted features
        cloudInfo.cloud_corner  = publishCloud(&pubCornerPoints,  cornerCloud,  cloudHeader.stamp, "base_link");
        cloudInfo.cloud_surface = publishCloud(&pubSurfacePoints, surfaceCloud, cloudHeader.stamp, "base_link");
        // publish to mapOptimization
        pubLaserCloudInfo.publish(cloudInfo);
    }
};


int main(int argc, char** argv)
{
    ros::init(argc, argv, "lidar");

    FeatureExtraction FE;

    ROS_INFO("\033[1;32m----> Lidar Feature Extraction Started.\033[0m");
   
    ros::spin();

    return 0;
}