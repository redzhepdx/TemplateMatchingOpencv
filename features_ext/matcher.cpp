#include "matcher.h"

Matcher::Matcher(std::unique_ptr<HyperparameterHolder> &&hyperparameters){
    this->_hparameter_holder = std::move(hyperparameters);
    this->_feature_detector  = cv::xfeatures2d::SURF::create(this->_hparameter_holder->minHessian);
    this->_feature_matcher   = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);

    int th = 60;
    int octaves = 4;
    float pattern_scales = 1.0f; 
    this->brisk_feature_detector = cv::BRISK::create(th, octaves, pattern_scales);
}

void Matcher::findObjectInTheScene(std::string object_file_name, std::string scene_file_name, bool visualize){
    cv::Mat img_object = cv::imread( object_file_name, cv::IMREAD_GRAYSCALE ); 
	cv::Mat img_scene  = cv::imread( scene_file_name, cv::IMREAD_GRAYSCALE );
    
    if( img_object.empty() || img_scene.empty() ){
        throw std::runtime_error("[ERROR] : Empty Image! Check the paths and images!");
    }

#ifndef GPU
    // Keypoints
    std::vector<cv::KeyPoint> keypoints_object, keypoints_scene;
    
	// Descriptors
    cv::Mat descriptors_object, descriptors_scene;

    //Output Image
    cv::Mat matched_image;

    // Matrix to store keypoint relations
    std::vector<std::vector<cv::DMatch>> matches;
    std::vector<cv::DMatch> good_matches;

    // Detect keypoints and compute descriptos
    this->_feature_detector->detectAndCompute(img_object, cv::noArray(), keypoints_object, descriptors_object);
    this->_feature_detector->detectAndCompute(img_scene, cv::noArray(), keypoints_scene, descriptors_scene);
    
    // Find Matches Between Images
    this->_feature_matcher->knnMatch(descriptors_object, descriptors_scene, matches, this->_hparameter_holder->K);
    
    // Match Filtering
    for(int i = 0; i < std::min(descriptors_scene.rows - 1 , (int)matches.size()); ++i){
        //std::cout << matches[i][0].distance << " " << matches[i][i].distance << std::endl;
        if( ((matches[i][0].distance < this->_hparameter_holder->th * (matches[i][1].distance))) && 
            ((int)matches[i].size() <= 2 && (int)matches[i].size() > 0)){
            good_matches.push_back(matches[i][0]);
        }   
    }

    // Visualize Matches and Localize Object into the Scene
    if(visualize){
        cv::drawMatches(img_object, keypoints_object, img_scene, keypoints_scene, good_matches, matched_image, 
                        cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    }

    // Draw Rectangle Around Best Match
    localizeInImage(good_matches, keypoints_object, keypoints_scene, img_object, matched_image, img_scene);

    cv::resize(matched_image, matched_image, cv::Size(), 0.5, 0.5);
    std::string unique_filename = generate_random_filename(10, ".jpg");

#if SHOW_IMAGES
    cv::imshow("localized", matched_image);
    cv::waitKey(0);
    cv::destroyAllWindows();
#else
    cv::imwrite("test_outputs/" + unique_filename, matched_image);
#endif

#else
    throw std::runtime_error("GPU Version is Not Ready!");
#endif
}

void Matcher::testOverSet(std::string json_name, std::string rel_path){
    nlohmann::json json_parser = read_json(json_name);
    for(auto& elem : json_parser){
        
        std::cout << elem << std::endl;
        std::string class_name = elem["label"];
        std::string crop_name  = elem["crop"].dump();
        std::string ff_name    = elem["fframe"].dump();

        // Remove quotes from image names
        crop_name.erase(std::remove(crop_name.begin(), crop_name.end(), '\"'),crop_name.end());
        ff_name.erase(std::remove(ff_name.begin(), ff_name.end(), '\"'),ff_name.end());

        crop_name = rel_path + "test_2/" + class_name + "/" + crop_name;
        ff_name   = rel_path + "test_aa_set/" + ff_name;
        
        this->findObjectInTheScene(crop_name, ff_name, true);

        std::cout << "\nAll Pairs are Checked!" << std::endl;
    }
}
