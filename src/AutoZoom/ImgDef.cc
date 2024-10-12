#include "ImgDef.h"
#include "typedef.h"
#include <cstdint>
#include <opencv2/opencv.hpp>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/core/base.hpp>
#include <opencv4/opencv2/core/hal/interface.h>
#include <opencv4/opencv2/core/mat.hpp>
#include <opencv4/opencv2/core/types.hpp>
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

float ImgDef::tenengrad() {
    cv::Mat img = this->toGray(this->img_type);
    float   sum = 0.0f;
    float   s_x = 0.0f;
    float   s_y = 0.0f;
    for (int x = 1; x < img.cols - 1; ++x) {
        for (int y = 1; y < img.rows - 1; ++y) {
            s_x = -1 * img.at<uint8_t>(y - 1, x - 1) + -2 * img.at<uint8_t>(y - 1, x) +
                  -1 * img.at<uint8_t>(y - 1, x + 1) + img.at<uint8_t>(y + 1, x - 1) +
                  2 * img.at<uint8_t>(y + 1, x) + img.at<uint8_t>(y + 1, x + 1);
            s_y = -1 * img.at<uint8_t>(y - 1, x - 1) + img.at<uint8_t>(y - 1, x + 1) +
                  -2 * img.at<uint8_t>(y, x - 1) + 2 * img.at<uint8_t>(y, x + 1) +
                  -1 * img.at<uint8_t>(y + 1, x - 1) + img.at<uint8_t>(y + 1, x + 1);
            if (this->four_dir) {
                float s_a = 0.0f;
                float s_b = 0.0f;

                s_a = img.at<uint8_t>(y - 1, x) + 2 * img.at<uint8_t>(y - 1, x + 1) +
                      -1 * img.at<uint8_t>(y, x - 1) + img.at<uint8_t>(y, x + 1) +
                      -2 * img.at<uint8_t>(y + 1, x - 1) + img.at<uint8_t>(y + 1, x);
                s_b = -2 * img.at<uint8_t>(y - 1, x - 1) + -1 * img.at<uint8_t>(y - 1, x) +
                      -1 * img.at<uint8_t>(y, x - 1) + img.at<uint8_t>(y, x + 1) +
                      img.at<uint8_t>(y + 1, x) + img.at<uint8_t>(y + 1, x + 1);
                sum += s_a * s_a + s_b * s_b;
            }
            sum += s_x * s_x + s_y * s_y;
        }
    }

    return sum;
}

float ImgDef::fft_method() {
    cv::Scalar sum        = 0.0f;
    cv::Mat    img        = this->toGray(this->img_type);
    cv::Mat    fft_result = this->do_fft(img);
    cv::Mat    roi = fft_result(cv::Rect(static_cast<uint16_t>(std::round(fft_result.cols * 0.25f)),
                                         static_cast<uint16_t>(std::round(fft_result.rows * 0.25f)),
                                         static_cast<uint16_t>(std::round(fft_result.cols * 0.5f)),
                                         static_cast<uint16_t>(std::round(fft_result.rows * 0.5f))));
    roi.setTo(0);
    sum = cv::sum(fft_result);
    return sum[ 0 ];
}

cv::Mat ImgDef::do_fft(const cv::Mat &img) const {
    cv::Mat amp;
    int     m = cv::getOptimalDFTSize(img.rows);
    int     n = cv::getOptimalDFTSize(img.cols);

    cv::Mat padded;
    cv::copyMakeBorder(img, padded, 0, m - img.rows, 0, n - img.cols, cv::BORDER_CONSTANT,
                       cv::Scalar::all(0));

    cv::Mat plans[] = {cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F)};
    cv::Mat complex_I;
    cv::merge(plans, 2, complex_I);
    cv::dft(complex_I, complex_I);
    cv::split(complex_I, plans);

    cv::magnitude(plans[ 0 ], plans[ 1 ], amp);
    this->fftshift(amp);
    return amp;
}

void ImgDef::fftshift(cv::Mat &fft_result) const {
    fft_result   = fft_result(cv::Rect(0, 0, fft_result.cols & -2, fft_result.rows & -2));
    uint16_t r_w = static_cast<uint16_t>(fft_result.cols * 0.5f);
    uint16_t r_h = static_cast<uint16_t>(fft_result.rows * 0.5f);

    cv::Mat q0(fft_result, cv::Rect(0, 0, r_w, r_h));     // ROI区域的左上
    cv::Mat q1(fft_result, cv::Rect(r_w, 0, r_w, r_h));   // ROI区域的右上
    cv::Mat q2(fft_result, cv::Rect(0, r_h, r_w, r_h));   // ROI区域的左下
    cv::Mat q3(fft_result, cv::Rect(r_w, r_h, r_w, r_h)); // ROI区域的右下

    // 交换象限（左上与右下进行交换）
    cv::Mat tmp;
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    // 交换象限（右上与左下进行交换）
    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);
}