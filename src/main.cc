#include "AutoZoom/AutoZoom.h"
#include <opencv4/opencv2/core/mat.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>

int main() {
    cv::Mat *ImgPtr = new cv::Mat;
    AutoZoom auto_zoom(ImgPtr);
    *ImgPtr = cv::imread("../images/auto_zoom/swan-3531856_1280.jpg");
    auto_zoom.adjusting();
    *ImgPtr = cv::imread("../images/auto_zoom/swan_blurred_1.jpg");
    auto_zoom.adjusting();
    *ImgPtr = cv::imread("../images/auto_zoom/swan_blurred_2.jpg");
    auto_zoom.adjusting();
}