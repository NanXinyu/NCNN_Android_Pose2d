#ifndef POSEDET_H
#define POSEDET_H

#include <opencv2/core/core.hpp>

#include <net.h>
struct KeyPoint
{
    float x;
    float y;
    float maxval;
};

class PoseDet
{
public:
    PoseDet();

    int load(AAssetManager* mgr, const char* modeltype, int target_width, int target_height, const float* mean_vals, const float* norm_vals, bool use_gpu = false);
    int detect(const cv::Mat& rgb);
    //int draw(cv::Mat& rgb);

private:

    int detect_pose(cv::Mat& roi, ncnn::Net& poseNet, std::vector<KeyPoint>& keypoints);
//    ncnn::Net detectNet;
    ncnn::Net poseNet;
    int target_width;
    int target_height;

    float mean_vals[3];
    float norm_vals[3];

    ncnn::UnlockedPoolAllocator blob_pool_allocator;
    ncnn::PoolAllocator workspace_pool_allocator;
};

#endif // POSEDET_H
