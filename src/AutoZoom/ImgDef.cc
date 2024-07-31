#include "ImgDef.h"
#include "typedef.h"
#include <cstdint>
#include <sys/types.h>

IMG_TYPE ImgDef::checkImgType() const {
    cv::Mat *imgPtr   = getImgPtr();
    int      channels = imgPtr->channels();
    if (channels == 1) {
        return GRAY;
    } else if (channels == 3) {
        return BGR;
    } else {
        return UNKNOW;
    }
}

cv::Mat ImgDef::toGray(IMG_TYPE &img_type) {
    cv::Mat *imgPtr = this->getImgPtr();
    if (imgPtr == nullptr || imgPtr->empty()) {
        std::cerr << "Image pointer is nullptr or image is empty" << std::endl;
        exit(-1);
    }
    if (img_type == UNKNOW) {
        img_type = this->checkImgType();
    }

    cv::Mat gray_img;
    switch (img_type) {
        case GRAY:
            return *imgPtr;
        case BGR:
            cv::cvtColor(*imgPtr, gray_img, cv::COLOR_BGR2GRAY);
            break;
        default:
            std::cerr << "Unsupported image type: " << img_type << std::endl;
            exit(-1);
    }
    return gray_img;
}

float ImgDef::brenner() {
    cv::Mat img = this->toGray(this->img_type);

    // Calculates Brenner gradient
    float sum = 0.0f;
    for (int y = 1; y < img.rows - 1; ++y) {
        for (int x = 1; x < img.cols - 1; ++x) {
            float diff = (img.at<uint8_t>(y + 1, x) - img.at<uint8_t>(y - 1, x)) +
                         (img.at<uint8_t>(y, x + 1) - img.at<uint8_t>(y, x - 1));
            sum += diff * diff;
        }
    }
    return sum;
}