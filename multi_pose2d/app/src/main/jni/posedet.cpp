#include "posedet.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "cpu.h"

const int num_joints = 17;

template<class ForwardIterator>
inline static size_t argmax(ForwardIterator first, ForwardIterator last) {
    return std::distance(first, std::max_element(first, last));
}


PoseDet::PoseDet()
{
    blob_pool_allocator.set_size_compare_ratio(0.f);
    workspace_pool_allocator.set_size_compare_ratio(0.f);

}


int PoseDet::load(AAssetManager* mgr, const char* modeltype, int _target_width, int _target_height, const float* _mean_vals, const float* _norm_vals, bool use_gpu)
{
    detectNet.clear();
    poseNet.clear();
    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    detectNet.opt = ncnn::Option();
    poseNet.opt = ncnn::Option();

#if NCNN_VULKAN
    detectNet.opt.use_vulkan_compute = use_gpu;
    poseNet.opt.use_vulkan_compute = use_gpu;
#endif

    detectNet.opt.num_threads = ncnn::get_big_cpu_count();
    poseNet.opt.num_threads = ncnn::get_big_cpu_count();

    detectNet.opt.blob_allocator = &blob_pool_allocator;
    poseNet.opt.blob_allocator = &blob_pool_allocator;

    detectNet.opt.workspace_allocator = &workspace_pool_allocator;
    poseNet.opt.workspace_allocator = &workspace_pool_allocator;

    char parampath[256];// = "D://ncnn_pose2d//app//src//main//assets//";
    char modelpath[256];// = "D://ncnn_pose2d//app//src//main//assets//";
    sprintf(parampath, "%s.param", modeltype);
    sprintf(modelpath, "%s.bin", modeltype);

    detectNet.load_param(mgr,"person_detector.param");
    detectNet.load_model(mgr,"person_detector.bin"),
    poseNet.load_param(mgr,parampath);
    poseNet.load_model(mgr,modelpath);
    
    target_width = _target_width;
    target_height = _target_height;
    mean_vals[0] = _mean_vals[0];
    mean_vals[1] = _mean_vals[1];
    mean_vals[2] = _mean_vals[2];
    norm_vals[0] = _norm_vals[0];
    norm_vals[1] = _norm_vals[1];
    norm_vals[2] = _norm_vals[2];

    return 0;
}

static void draw_pose(const cv::Mat& img, const std::vector<KeyPoint>& keypoints)
{
    // draw bone
    static const int kps_lines[][2] = {
            {15, 13}, {13, 11}, {16, 14}, {14, 12}, {11, 12}, {5, 11}, {6, 12}, {5, 6},
            {5, 7}, {6, 8}, {7, 9}, {8, 10}, {1, 2}, {0, 1},{0, 2},{1, 3},{2, 4},{3, 5},{4, 6}
    };
    for (int l = 0; l < 19; l++)
    {
        const KeyPoint& p1 = keypoints[kps_lines[l][0]];
        const KeyPoint& p2 = keypoints[kps_lines[l][1]];

        if ( p1.maxval > 4.f && p2.maxval > 4.f )
            cv::line(img, cv::Point(p1.x, p1.y), cv::Point(p2.x, p2.y), cv::Scalar(0, 255, 0), 2);

        if ( p1.maxval > 4.f )
            cv::circle(img, cv::Point(p1.x, p1.y), 3, cv::Scalar(255, 0, 0), -1);

        if ( p2.maxval > 4.f )
            cv::circle(img, cv::Point(p2.x, p2.y), 3, cv::Scalar(255, 0, 0), -1);
    }
}

int PoseDet::detect_pose(cv::Mat& roi, ncnn::Net& poseNet, std::vector<KeyPoint>& keypoints, float xc, float yc)
{
    int w = roi.cols;
    int h = roi.rows;
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(roi.data, ncnn::Mat::PIXEL_BGR2RGB, \
        roi.cols, roi.rows, target_width, target_height);
    //数据预处理
    const float mean_vals[3] = { 0.485f * 255.f, 0.456f * 255.f, 0.406f * 255.f };
    const float norm_vals[3] = { 1 / 0.229f / 255.f, 1 / 0.224f / 255.f, 1 / 0.225f / 255.f };
    in.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = poseNet.create_extractor();
    //ex.set_num_threads(4);
    ex.input("input", in);
    ncnn::Mat output_x, output_y;

    ex.extract("output0", output_x);
    ex.extract("output1", output_y);


    keypoints.clear();
    for (int p = 0; p < output_x.h; p++)
    {
        KeyPoint kpt;
        // printf("j\n");
        const float* ptr_x = output_x.row(p);
        const float* ptr_y = output_y.row(p);
        float max_val = 0.f;
        int max_x = 0;
        int max_y = 0;

        for (int x = 0; x < output_x.w; x++)
        {
            // printf("x");
            float val = ptr_x[x];
            if (val > max_val)
            {
                max_val = val;
                max_x = x;}
        }
        kpt.x = max_x * w / (float)target_width + xc;
        kpt.maxval = max_val;

        max_val = 0.f;

        for (int y = 0; y < output_y.w; y++)
        {
            // printf("y");
            float val = ptr_y[y];
            if (val > max_val)
            {
                max_val = val;
                max_y = y;}
        }

        kpt.y =  max_y * h / (float)target_height + yc;
        kpt.maxval += max_val;
        kpt.maxval = 0.5f * kpt.maxval;
        keypoints.push_back(kpt);
    }
    return 0;
}

int PoseDet::detect(const cv::Mat& image)
{
    //TODO:add person detection
    cv::Mat bgr = image.clone();
    int img_w = bgr.cols;
    int img_h = bgr.rows;

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, \
        bgr.cols, bgr.rows, detector_size_width, detector_size_height);

    //数据预处理
    const float detect_mean_vals[3] = { 0.f, 0.f, 0.f };
    const float detect_norm_vals[3] = { 1 / 255.f, 1 / 255.f, 1 / 255.f };
    in.substract_mean_normalize(detect_mean_vals, detect_norm_vals);

    ncnn::Extractor ex = detectNet.create_extractor();
    //ex.set_num_threads(4);
    ex.input("data", in);
    ncnn::Mat out;
    ex.extract("output", out);

    for (int i = 0; i < out.h; i++)
    {
        //printf("==================================\n");
        float x1, y1, x2, y2, score, label;
        float pw, ph, cx, cy;
        const float* values = out.row(i);

        x1 = values[2] * img_w;
        y1 = values[3] * img_h;
        x2 = values[4] * img_w;
        y2 = values[5] * img_h;

        pw = x2 - x1;
        ph = y2 - y1;
        cx = x1 + 0.5 * pw;
        cy = y1 + 0.5 * ph;

        x1 = cx - 0.7 * pw;
        y1 = cy - 0.6 * ph;
        x2 = cx + 0.7 * pw;
        y2 = cy + 0.6 * ph;

        score = values[1];
        label = values[0];

        //处理坐标越界问题
        if (x1 < 0) x1 = 0;
        if (y1 < 0) y1 = 0;
        if (x2 < 0) x2 = 0;
        if (y2 < 0) y2 = 0;

        if (x1 > img_w) x1 = img_w;
        if (y1 > img_h) y1 = img_h;
        if (x2 > img_w) x2 = img_w;
        if (y2 > img_h) y2 = img_h;
        //截取人体ROI
        //printf("x1:%f y1:%f x2:%f y2:%f\n",x1,y1,x2,y2);
        cv::Mat roi;
        roi = bgr(cv::Rect(x1, y1, x2 - x1, y2 - y1)).clone();
        std::vector<KeyPoint> keypoints;
        detect_pose(roi, poseNet, keypoints, x1, y1);
        draw_pose(image, keypoints);
        cv::rectangle(image, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 0, 255), 2, 8, 0);
    }
    return 0;
}




