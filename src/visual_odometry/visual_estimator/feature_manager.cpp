#include "feature_manager.h"

int FeaturePerId::endFrame()
{
    return start_frame + feature_per_frame.size() - 1;
}

FeatureManager::FeatureManager(Matrix3d _Rs[])
    : Rs(_Rs)
{
    for (int i = 0; i < NUM_OF_CAM; i++)
        ric[i].setIdentity();
}

void FeatureManager::setRic(Matrix3d _ric[])
{
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ric[i] = _ric[i];
    }
}

void FeatureManager::clearState()
{
    feature.clear();
}

int FeatureManager::getFeatureCount()
{
    int cnt = 0;
    for (auto &it : feature)
    {

        it.used_num = it.feature_per_frame.size();

        if (it.used_num >= 2 && it.start_frame < WINDOW_SIZE - 2)
        {
            cnt++;
        }
    }
    return cnt;
}


/**
 * @brief 将特征点放入list容器，计算每一个点的跟踪次数和它在次新帧和次次新帧间的视差，返回是否是关键帧
 *
 * @param frame_count 窗口内帧的个数
 * @param image 某帧的所有特征点的[camera_id, [x, y, z, u, v, vx, vy]] 构成的map, 索引为feature_id
 * @param td IMU和cam同步的时间差
 * @return true 次新帧是关键帧
 * @return false 次新帧非关键帧
 */
bool FeatureManager::addFeatureCheckParallax(int frame_count, const map<int, vector<pair<int, Eigen::Matrix<double, 8, 1>>>> &image, double td)
{
    ROS_DEBUG("input feature: %d", (int)image.size());
    ROS_DEBUG("num of feature: %d", getFeatureCount());
    double parallax_sum = 0; // 用于记录所有特征点的视差和
    int parallax_num = 0;  // 计算视差的次数
    last_track_num = 0; // 在此帧上被跟踪到的点的个数
    for (auto &id_pts : image)
    {
        FeaturePerFrame f_per_fra(id_pts.second[0].second, td); // id_pts是map的每个键值对，id_pts.second[0]是第一个相机的观测信息：Eigen::Matrix<double, 8, 1>

        // find feature id in the feature bucket  查找特征list中是否包含当前的关键点
        int feature_id = id_pts.first; //应该是map的索引
        // 在feature容器中找到feature_id
        auto it = find_if(feature.begin(), feature.end(), [feature_id](const FeaturePerId &it)
                          {return it.feature_id == feature_id;});

        // 没有找到，是新的特征点，加入到特征list，并加入这次第一个相机观测
        if (it == feature.end())
        {
            // this feature in the image is observed for the first time, create a new feature object
            feature.push_back(FeaturePerId(feature_id, frame_count, f_per_fra.depth));
            feature.back().feature_per_frame.push_back(f_per_fra);
        }
        // 找到了，就是跟踪上的点
        else if (it->feature_id == feature_id)
        {
            // this feature in the image has been observed before
            it->feature_per_frame.push_back(f_per_fra);
            last_track_num++;
            // sometimes the feature is first observed without depth 
            // (initialize initial feature depth with current image depth is not exactly accurate if camera moves very fast, then lines bebow can be commented out)
            if (f_per_fra.depth > 0 && it->lidar_depth_flag == false)
            {
                it->estimated_depth = f_per_fra.depth;
                it->lidar_depth_flag = true;
                it->feature_per_frame[0].depth = f_per_fra.depth;
            }
        }
    }

    // 如果当前帧是第0帧或第1帧 或 跟踪到的特征点数小于20，说明该图片帧为关键帧，应当边缘化旧帧
    if (frame_count < 2 || last_track_num < 20)
        return true;

    for (auto &it_per_id : feature)
    {
        // 如果这个特征的起始观测帧在上上帧以前，最终观测帧是上一帧或当前帧，视差计数+1
        if (it_per_id.start_frame <= frame_count - 2 &&
            it_per_id.start_frame + int(it_per_id.feature_per_frame.size()) - 1 >= frame_count - 1)
        {
            parallax_sum += compensatedParallax2(it_per_id, frame_count);
            parallax_num++;
        }
    }

    // 如果视差计数为0，说明上一帧和当前帧基本都是新特征，当前帧必然是关键帧，应当边缘化旧帧
    if (parallax_num == 0)
    {
        return true;
    }
    else
    {
        // 否则，根据平均视差决定marge最老帧还是次新帧
        ROS_DEBUG("parallax_sum: %lf, parallax_num: %d", parallax_sum, parallax_num);
        ROS_DEBUG("current parallax: %lf", parallax_sum / parallax_num * FOCAL_LENGTH);
        return parallax_sum / parallax_num >= MIN_PARALLAX;
    }
}

void FeatureManager::debugShow()
{
    ROS_DEBUG("debug show");
    for (auto &it : feature)
    {
        ROS_ASSERT(it.feature_per_frame.size() != 0);
        ROS_ASSERT(it.start_frame >= 0);
        ROS_ASSERT(it.used_num >= 0);

        ROS_DEBUG("%d,%d,%d ", it.feature_id, it.used_num, it.start_frame);
        int sum = 0;
        for (auto &j : it.feature_per_frame)
        {
            ROS_DEBUG("%d,", int(j.is_used));
            sum += j.is_used;
            printf("(%lf,%lf) ",j.point(0), j.point(1));
        }
        ROS_ASSERT(it.used_num == sum);
    }
}

vector<pair<Vector3d, Vector3d>> FeatureManager::getCorresponding(int frame_count_l, int frame_count_r)
{
    vector<pair<Vector3d, Vector3d>> corres;
    for (auto &it : feature)
    {
        if (it.start_frame <= frame_count_l && it.endFrame() >= frame_count_r)
        {
            Vector3d a = Vector3d::Zero(), b = Vector3d::Zero();
            int idx_l = frame_count_l - it.start_frame;
            int idx_r = frame_count_r - it.start_frame;

            a = it.feature_per_frame[idx_l].point;

            b = it.feature_per_frame[idx_r].point;
            
            corres.push_back(make_pair(a, b));
        }
    }
    return corres;
}

void FeatureManager::setDepth(const VectorXd &x)
{
    int feature_index = -1;
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;

        it_per_id.estimated_depth = 1.0 / x(++feature_index);

        if (it_per_id.estimated_depth < 0)
        {
            it_per_id.solve_flag = 2;
        }
        else
            it_per_id.solve_flag = 1;
    }
}

void FeatureManager::removeFailures()
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;
        if (it->solve_flag == 2)
            feature.erase(it);
    }
}

void FeatureManager::clearDepth(const VectorXd &x)
{
    int feature_index = -1;
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;

        it_per_id.estimated_depth = 1.0 / x(++feature_index);
        it_per_id.lidar_depth_flag = false;
    }
}

VectorXd FeatureManager::getDepthVector()
{
    VectorXd dep_vec(getFeatureCount());
    int feature_index = -1;
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;

        // optimized depth after ceres maybe negative, initialize them with default value for this optimization
        if (it_per_id.estimated_depth > 0)
            dep_vec(++feature_index) = 1. / it_per_id.estimated_depth;
        else
            dep_vec(++feature_index) = 1. / INIT_DEPTH;
    }
    return dep_vec;
}

void FeatureManager::triangulate(Vector3d Ps[], Vector3d tic[], Matrix3d ric[])
{
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        // 至少两帧观测得到这个特征点  且 不能是滑窗中的最后两帧
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;

        // depth is available, skip triangulation (trust the first estimate)
        if (it_per_id.estimated_depth > 0)
            continue;

        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;

        ROS_ASSERT(NUM_OF_CAM == 1);
        Eigen::MatrixXd svd_A(2 * it_per_id.feature_per_frame.size(), 4);
        int svd_idx = 0;

        // R0 t0为第i帧cam--->world的变换矩阵
        Eigen::Matrix<double, 3, 4> P0;
        Eigen::Vector3d t0 = Ps[imu_i] + Rs[imu_i] * tic[0];
        Eigen::Matrix3d R0 = Rs[imu_i] * ric[0];
        // 投影矩阵
        P0.leftCols<3>() = Eigen::Matrix3d::Identity();
        P0.rightCols<1>() = Eigen::Vector3d::Zero();

        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;

            // R1 t1为第j帧cam--->world的变换矩阵
            Eigen::Vector3d t1 = Ps[imu_j] + Rs[imu_j] * tic[0];
            Eigen::Matrix3d R1 = Rs[imu_j] * ric[0];
            // R t为cam(j)--->cam(i)的变换矩阵
            Eigen::Vector3d t = R0.transpose() * (t1 - t0);
            Eigen::Matrix3d R = R0.transpose() * R1;
            // 若以上坐标系正确的话，就是将相机的世界坐标系的三维点投影到当前相机坐标系下
            Eigen::Matrix<double, 3, 4> P;
            // P为cam(i)--->cam(j)的变换矩阵
            P.leftCols<3>() = R.transpose();
            P.rightCols<1>() = -R.transpose() * t;
            // 获取归一化坐标系下的位置
            // 只保留方向信息 去除尺度信息
            Eigen::Vector3d f = it_per_frame.point.normalized();
            svd_A.row(svd_idx++) = f[0] * P.row(2) - f[2] * P.row(0);
            svd_A.row(svd_idx++) = f[1] * P.row(2) - f[2] * P.row(1);

            if (imu_i == imu_j)
                continue;
        }
        ROS_ASSERT(svd_idx == svd_A.rows());
        // 对A的SVD分解得到其最小奇异值对应的单位奇异向量(x,y,z,w)，深度为z/w
        Eigen::Vector4d svd_V = Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV).matrixV().rightCols<1>();
        double svd_method = svd_V[2] / svd_V[3]; // svd方法计算出的深度

        // update depth from triangulation 得到的深度值实际上就是第一个观察到这个特征点的相机坐标系下的深度值
        it_per_id.estimated_depth = svd_method;
        // check if triangulation failed
        if (it_per_id.estimated_depth < 0)
        {
            it_per_id.estimated_depth = INIT_DEPTH;
        }
    }
}

void FeatureManager::removeOutlier()
{
    ROS_BREAK();
    int i = -1;
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;
        i += it->used_num != 0;
        if (it->used_num != 0 && it->is_outlier == true)
        {
            feature.erase(it);
        }
    }
}
/**
 * 边缘化最老帧时，处理特帧点保存的帧号，将第一次观测到的帧是最老帧的特征点的深度值进行转移
 * @param marg_R 被边缘化的位姿
 * @param marg_P 被边缘化的位置
 * @param new_R 下一帧的位姿
 * @param new_P 下一帧的位置
*/
void FeatureManager::removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, Eigen::Matrix3d new_R, Eigen::Vector3d new_P)
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;

        // 首次观测帧不是第0帧，则直接帧号-1
        if (it->start_frame != 0)
            it->start_frame--;
        else
        {
            // feature point and depth in old local camera frame 取出归一化坐标
            Eigen::Vector3d uv_i = it->feature_per_frame[0].point;
            double depth = -1;
            if (it->feature_per_frame[0].depth > 0) 
                // if lidar depth available at this frame for feature 如果有雷达的测量
                depth = it->feature_per_frame[0].depth;
            else if (it->estimated_depth > 0) 
                // if estimated depth available 如果有三角化的测量
                depth = it->estimated_depth;

            // delete current feature in the old local camera frame 删除特征点观测的最老帧
            it->feature_per_frame.erase(it->feature_per_frame.begin());

            if (it->feature_per_frame.size() < 2) // 当特征点的共视帧小于2
            {
                // delete feature from feature manager 直接删除这个特征
                feature.erase(it);
                continue;
            }
            else
            {
                Eigen::Vector3d pts_i = uv_i * depth; // feature in cartisian space in old local camera frame
                Eigen::Vector3d w_pts_i = marg_R * pts_i + marg_P; // feautre in cartisian space in world frame
                Eigen::Vector3d pts_j = new_R.transpose() * (w_pts_i - new_P); // feature in cartisian space in shifted local camera frame
                double dep_j = pts_j(2);

                // after deletion, the feature has lidar depth in the first of the remaining frame
                if (it->feature_per_frame[0].depth > 0)
                {
                    it->estimated_depth = it->feature_per_frame[0].depth;
                    it->lidar_depth_flag = true;
                } 
                // calculated depth in the current frame
                else if (dep_j > 0)
                {
                    it->estimated_depth = dep_j;
                    it->lidar_depth_flag = false;
                } 
                // non-positive depth, invalid
                else 
                {
                    it->estimated_depth = INIT_DEPTH;
                    it->lidar_depth_flag = false;
                }
            }
        }
    }
}

// 如果初始化没结束，此时进行边缘化时，则直接将帧号前移，不换算深度
void FeatureManager::removeBack()
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;

        if (it->start_frame != 0)
            it->start_frame--;
        else
        {
            it->feature_per_frame.erase(it->feature_per_frame.begin());
            if (it->feature_per_frame.size() == 0)
                feature.erase(it);
        }
    }
}

// 边缘化次新帧时，对特征点在次新帧的信息进行移除
void FeatureManager::removeFront(int frame_count)
{
    for (auto it = feature.begin(), it_next = feature.begin(); it != feature.end(); it = it_next)
    {
        it_next++;

        // 如果观测帧是最新帧，直接改为次新帧
        if (it->start_frame == frame_count)
        {
            it->start_frame--;
        }
        else
        {
            int j = WINDOW_SIZE - 1 - it->start_frame;
            if (it->endFrame() < frame_count - 1)
                continue;
            it->feature_per_frame.erase(it->feature_per_frame.begin() + j);
            if (it->feature_per_frame.size() == 0)
                feature.erase(it);
        }
    }
}

double FeatureManager::compensatedParallax2(const FeaturePerId &it_per_id, int frame_count)
{
    //check the second last frame is keyframe or not
    //parallax betwwen seconde last frame and third last frame
    const FeaturePerFrame &frame_i = it_per_id.feature_per_frame[frame_count - 2 - it_per_id.start_frame];
    const FeaturePerFrame &frame_j = it_per_id.feature_per_frame[frame_count - 1 - it_per_id.start_frame];

    double ans = 0;
    Vector3d p_j = frame_j.point; //倒数第二帧j的3D路标点

    double u_j = p_j(0);
    double v_j = p_j(1);

    Vector3d p_i = frame_i.point;
    Vector3d p_i_comp;

    //int r_i = frame_count - 2;
    //int r_j = frame_count - 1;
    //p_i_comp = ric[camera_id_j].transpose() * Rs[r_j].transpose() * Rs[r_i] * ric[camera_id_i] * p_i;
    p_i_comp = p_i;
    double dep_i = p_i(2);
    // 归一化平面坐标
    double u_i = p_i(0) / dep_i;
    double v_i = p_i(1) / dep_i;
    double du = u_i - u_j, dv = v_i - v_j;

    double dep_i_comp = p_i_comp(2);
    double u_i_comp = p_i_comp(0) / dep_i_comp;
    double v_i_comp = p_i_comp(1) / dep_i_comp;
    double du_comp = u_i_comp - u_j, dv_comp = v_i_comp - v_j;

    // 求得两个点的相对位移，即视差
    ans = max(ans, sqrt(min(du * du + dv * dv, du_comp * du_comp + dv_comp * dv_comp)));

    return ans;
}