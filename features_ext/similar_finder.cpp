#include "similar_finder.h"
#include "../utils/file_utils.h"


SimilarFinder::SimilarFinder(std::unique_ptr<HyperparameterHolder> &&hyperparameters){
    this->_hparameter_holder = std::move(hyperparameters);
    this->_feature_detector  = cv::xfeatures2d::SURF::create(this->_hparameter_holder->minHessian);
    this->_bow_trainer       = std::make_unique<cv::BOWKMeansTrainer>(this->_hparameter_holder->K);
}

void SimilarFinder::prepareKNN(){
    // Detect keypoints using SURF Detector, compute the descriptors
    for(auto& img : this->images){
        cv::Mat desc;
        std::vector<cv::KeyPoint> keypoints;
        
        // Feature Extraction
        this->_feature_detector->detectAndCompute(img, cv::noArray(), keypoints, desc);

        // Save descriptors into the vector
        this->descriptor_vec.push_back(desc);
    }

    // Find Maximum Row
    std::cout << "------Finding largest descriptor matrix and initializing SuperMatrix...." << std::endl;
    this->max_row = (*std::max_element(this->descriptor_vec.begin(), this->descriptor_vec.end(), 
                                       [](cv::Mat lhs, cv::Mat rhs){ return lhs.rows < rhs.rows; })).rows;

    //Place descriptors into supermatrix row by row    
    cv::Mat features(cv::Size(this->max_row * descriptor_vec[0].cols, image_names.size()), CV_32F, cv::Scalar(0.0f));

    std::cout << "------Placing Descriptors into the SuperMatrix...." << std::endl;
    for(size_t i = 0; i < this->descriptor_vec.size() ; ++i)
    {
        cv::Mat flat_desc = this->descriptor_vec[i].reshape(1,1);
        cv::Mat dst = features(cv::Rect(0, i, flat_desc.cols, 1));
        flat_desc.copyTo(features(cv::Rect(0, i, flat_desc.cols, 1)));
    }

    std::cout << "------Super Matrix's Size : " << features.size() << std::endl;
    features.convertTo(features, CV_32F);

    // Add descriptors to BOW Trainer
    this->_bow_trainer->add(features);
    this->vocabulary = this->_bow_trainer->cluster();

    // KNN Similarity Searcher
    this->_flann_index = std::make_unique<cv::flann::Index>(
                                                        features,
                                                        cv::flann::KDTreeIndexParams(), //cv::flann::LshIndexParams(30, 20, 2),
                                                        cvflann::MANHATTAN
                                                    );
}

void SimilarFinder::read_images(std::string folder_name){
    this->image_names = getFilenamesFromFolder(folder_name);
    
    for(auto& image_name : image_names){
        this->images.push_back(cv::imread(image_name));    
    }
}

std::tuple<std::vector<int>, std::vector<float>> SimilarFinder::find_similar(std::string query_image_name){
    std::vector<int> indices(this->_hparameter_holder->K);
    std::vector<float> distances(this->_hparameter_holder->K);
    std::vector<cv::KeyPoint> query_keypoints;

    // Descriptor Matrices
    cv::Mat query_desc_holder, query_description(cv::Size(this->max_row * this->descriptor_vec[0].cols, 1), 
                                                 CV_32F, cv::Scalar(0.0f));

    // Read Image
    cv::Mat query_image = cv::imread(query_image_name);
    std::cout << "------Extracting Features of Query Image......" << std::endl;
    this->_feature_detector->detectAndCompute(query_image, cv::noArray(), query_keypoints, query_desc_holder);

    // Flatten the Query Descriptor 
    std::cout << "------Create Query Matrix...." << std::endl;
    query_desc_holder = query_desc_holder.reshape(1, 1);
    std::cout << "------Reshape...." << std::endl;
    query_desc_holder.convertTo(query_desc_holder, CV_32F);
    std::cout << "------Copy...." << std::endl;
    query_desc_holder.copyTo(query_description(cv::Rect(0, 0, query_desc_holder.cols, 1)));
    
    // KNN Search Over Image
    std::cout << "------Search query into the super matrix....." << std::endl << std::endl;
    this->_flann_index->knnSearch(query_description, indices, distances, this->_hparameter_holder->K);


    std::cout << "Results : " << std::endl;
    for(size_t i = 0; i < this->_hparameter_holder->K; i++)
    {        
        std::cout << "Closest " << i << "th Image's Name " << image_names[indices[i]] 
                  << " Distance : " << distances[i] << std::endl;
    }

    return std::make_tuple(indices, distances);
}

void SimilarFinder::printOutput(){

}
