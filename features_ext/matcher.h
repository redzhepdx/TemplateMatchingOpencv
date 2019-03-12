#ifndef MATCHER_H
#define MATCHER_H

#include <iostream> 
#include <string>
#include <memory>
#include <algorithm>

#include "opencv2/core.hpp"
#include <opencv2/imgproc.hpp>
#include "opencv2/highgui.hpp"

#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"

#include "base_feature_op.h"

#include "../utils/json.hpp"
#include "../utils/file_utils.h"

#define CPU
#define SHOW_IMAGES 1

class Matcher :  public BaseFeatureOp {

public:
    Matcher(){}
    Matcher(std::unique_ptr<HyperparameterHolder> &&hyperparameters);

public:
    void findObjectInTheScene(const std::string object_file_name, const std::string scene_name, bool visualize);
    void testOverSet(std::string json_name, std::string rel_path);

private:
    std::unique_ptr<HyperparameterHolder> _hparameter_holder;

    cv::Ptr<cv::xfeatures2d::SURF> _feature_detector;
    cv::Ptr<cv::DescriptorMatcher> _feature_matcher;
    cv::Ptr<cv::BRISK> brisk_feature_detector;
};


#endif