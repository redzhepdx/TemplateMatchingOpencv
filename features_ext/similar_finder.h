#ifndef SIMILAR_FINDER_H
#define SIMILAR_FINDER_H

#include <iostream> 
#include <string>
#include <memory>

#include "opencv2/core.hpp"
#include <opencv2/imgproc.hpp>
#include "opencv2/highgui.hpp"

#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"

#include "base_feature_op.h"


using namespace cv;
using namespace cv::xfeatures2d;

class SimilarFinder : public BaseFeatureOp
{
public:
    SimilarFinder(){}
    SimilarFinder(std::unique_ptr<HyperparameterHolder> &&hyperparameters);

public:
    void read_images(std::string folder_name);
    void printOutput();
    void prepareKNN();

public:
    std::tuple<std::vector<int>, std::vector<float>> find_similar(std::string query_image_name);

public: 
    // Setters Getters
    std::vector<std::string> getImageNames(){ return this->image_names; }
    std::vector<cv::Mat> getImages() { return this->images; }

private:
    std::unique_ptr<HyperparameterHolder> _hparameter_holder;
    cv::Ptr<cv::xfeatures2d::SURF> _feature_detector;
    std::unique_ptr<cv::BOWKMeansTrainer> _bow_trainer;
    std::unique_ptr<cv::flann::Index> _flann_index;

    std::vector<std::string> image_names;
    std::vector<cv::Mat> images;
    std::vector<cv::Mat> descriptor_vec;

    cv::Mat vocabulary;
    int max_row;
};


#endif