#ifndef BASE_FEATURE_OP_H
#define BASE_FEATURE_OP_H

#include <iostream>
#include <vector>
#include <exception>
#include <memory>

#include "opencv2/core.hpp"
#include <opencv2/imgproc.hpp>
#include "opencv2/highgui.hpp"

#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"

#include "opencv2/calib3d/calib3d.hpp"


class HyperparameterHolder
{
  public:
    HyperparameterHolder(int _minHessian, float _ratioThreshold, int _K=4, float _th=0.65f) : minHessian(_minHessian), ratioThreshold(_ratioThreshold), K(_K), th(_th){}
    
    std::unique_ptr<HyperparameterHolder> copy(){
        return std::make_unique<HyperparameterHolder>(this->minHessian, this->ratioThreshold, this->K, this->th);
    }

  public:
    double minHessian;
    float ratioThreshold;
    float th;
    int K;
};

class BaseFeatureOp{
public:
    BaseFeatureOp(){}
    BaseFeatureOp(std::unique_ptr<HyperparameterHolder> &&hyperparameters);
};

// It searches for the right position, orientation and scale of the object in the scene based on the good_matches.
inline void localizeInImage(const std::vector<cv::DMatch>& good_matches,
                            const std::vector<cv::KeyPoint>& keypoints_object,
                            const std::vector<cv::KeyPoint>& keypoints_scene, 
                            const cv::Mat& img_object,
                            const cv::Mat& img_matches,
                            const cv::Mat& img_scene)
{
	//-- Localize the object
	std::vector<cv::Point2f> obj;
	std::vector<cv::Point2f> scene;

    for (int i = 0; i < good_matches.size(); i++) {
        //-- Get the keypoints from the good matches
        obj.push_back(keypoints_object[good_matches[i].queryIdx].pt);
        scene.push_back(keypoints_scene[good_matches[i].trainIdx].pt);
    }


    try{
        cv::Mat H = cv::findHomography(obj, scene, cv::RANSAC);
        //-- Get the corners from the image_1 ( the object to be "detected" )
        std::vector<cv::Point2f> obj_corners(4);
        obj_corners[0] = cv::Point(0, 0);
        obj_corners[1] = cv::Point(img_object.cols, 0);
        obj_corners[2] = cv::Point(img_object.cols, img_object.rows);
        obj_corners[3] = cv::Point(0, img_object.rows);
        std::vector<cv::Point2f> scene_corners(4);

        cv::perspectiveTransform(obj_corners, scene_corners, H);

        for(int i = 0; i < scene_corners.size(); ++i){
            std::cout << scene_corners[i].x << " " << scene_corners[i].y << std::endl;
        }

        // Draw lines between the corners (the mapped object in the scene - image_2 )
        cv::line(img_matches, scene_corners[0] + cv::Point2f(img_object.cols, 0),
                scene_corners[1] + cv::Point2f(img_object.cols, 0),
                cv::Scalar(255, 0, 0), 4);
        cv::line(img_matches, scene_corners[1] + cv::Point2f(img_object.cols, 0),
                scene_corners[2] + cv::Point2f(img_object.cols, 0),
                cv::Scalar(255, 0, 0), 4);
        cv::line(img_matches, scene_corners[2] + cv::Point2f(img_object.cols, 0),
                scene_corners[3] + cv::Point2f(img_object.cols, 0),
                cv::Scalar(255, 0, 0), 4);
        cv::line(img_matches, scene_corners[3] + cv::Point2f(img_object.cols, 0),
                scene_corners[0] + cv::Point2f(img_object.cols, 0),
                cv::Scalar(255, 0, 0), 4);
    }
    catch(cv::Exception& exp){
        std::cout << "error : " << exp.msg << std::endl;
    }
}

#endif