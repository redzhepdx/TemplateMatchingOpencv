#include <iostream>
#include <vector>
#include <algorithm>
#include <memory>

#include "opencv2/core.hpp"
#include <opencv2/imgproc.hpp>
#include "opencv2/highgui.hpp"

#ifdef HAVE_OPENCV_XFEATURES2D
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"

#include "feature_ext/similar_finder.h"
#include "feature_ext/matcher.h"
#include "utils/json.hpp"
#include "utils/file_utils.h"

using namespace cv;
using namespace cv::xfeatures2d;

#define JSON_DEBUG 0
#define HANDMADE_TEST 0
#define SIMILARITY_TEST 0
#define AUTO_ANNOTATION 1

int main(void)
{

#if JSON_DEBUG
    nlohmann::json json_parser = read_json("json_data/babayaga.json");
    for(auto& elem : json_parser){
        std::cout << elem << std::endl;
        std::cout << elem["crop"] << std::endl;
        getchar();
    }

#elif AUTO_ANNOTATION
    //Hyperparameter holder object
    std::unique_ptr<HyperparameterHolder> hyperparameters = std::make_unique<HyperparameterHolder>(100.0f, 0.6f, 2);
    std::unique_ptr<Matcher> matcher = std::make_unique<Matcher>(std::move(hyperparameters->copy()));
    std::string rel_path  = "/media/redzhep/e6ab57e2-2750-47a8-9a98-1e0791c275bd/home/redzhep/workspace-redzhep/AutomaticAnnotation/";
    std::string json_path = "json_data/babayaga.json";
    matcher->testOverSet(json_path, rel_path);

#elif HANDMADE_TEST
    //Hyperparameter holder object
    std::unique_ptr<HyperparameterHolder> hyperparameters = std::make_unique<HyperparameterHolder>(100.0f, 0.6f, 2);
    
    std::unique_ptr<Matcher> matcher = std::make_unique<Matcher>(std::move(hyperparameters->copy()));
    
    std::string folder_name = "test_images/";
    char decision = 'c';

    while(decision == 'c'){    
        std::string object_name;
        std::string scene_name;
        
        std::cout << "Enter Object Name : ";
        std::cin >> object_name;

        std::cout << "Enter Scene Name : ";
        std::cin >> scene_name;
        
        matcher->findObjectInTheScene(folder_name + object_name, 
                                      folder_name + scene_name, true);
    }

#elif SIMILARITY_TEST

    std::unique_ptr<SimilarFinder> similarFinder = std::make_unique<SimilarFinder>(std::move(hyperparameters->copy()));
    // Read Images to Vector
    std::cout << "------Reading Images...." << std::endl;
    similarFinder->read_images("data/*");
    
    std::vector<std::string> image_names = similarFinder->getImageNames();
    std::vector<cv::Mat> images = similarFinder->getImages();
    
    std::cout << "------KNN Training...." << std::endl;
    similarFinder->prepareKNN();

    std::tuple<std::vector<int>, std::vector<float>> res = similarFinder->find_similar("test_images/r1.png");

    std::cout << "\n-------Done..." << std::endl;
    getchar();
#endif
    return 0;
}
#else
int main(void)
{
    std::cout << "Something is wrong" std::endl;
    return 0;
}
#endif