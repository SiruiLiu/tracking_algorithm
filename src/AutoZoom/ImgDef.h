#ifndef IMGDEF_H
#define IMGDEF_H

#include "typedef.h"
#include <cstddef>
#include <iostream>
#include <opencv4/opencv2/core/mat.hpp>

class ImgDef {
public:
    ImgDef(cv::Mat *ImgPtr)
        : ImgPtr(ImgPtr) {
    }
    virtual ~ImgDef() {
        if (this->ImgPtr != NULL) {
            delete this->ImgPtr;
            this->ImgPtr = nullptr;
        }
    }

    // !============================== Protect functions =========================! //
protected:
    /**
     * @brief Check the input image type
     *
     * @return IMG_TYPE
     */
    IMG_TYPE checkImgType() const;
    /**
     * @brief Get the Img Type object
     *
     * @return IMG_TYPE
     */
    inline void getImgType() const {
        switch (this->img_type) {
            case GRAY:
                std::cout << "Image type is GRAY" << std::endl;
            case BGR:
                std::cout << "Image type is BGR" << std::endl;
                break;
            default:
                std::cout << "Image type is UNKNOW" << std::endl;
                break;
        }
    }
    /**
     * @brief Transform image type to gray
     *
     * @return cv::Mat
     */
    cv::Mat toGray(IMG_TYPE &img_type);
    /**
     * @brief Image definition function brenner
     *
     * @return float
     */
    float brenner();
    /**
     * @brief Use tenengrade function to measure
     *        the clarity of the picture
     *
     * @return float
     */
    float tenengrad();
    /**
     * @brief Use fft method to measure the clarity
     *        of the picture
     *
     * @return float
     */
    float fft_method();

    virtual cv::Mat *getImgPtr() const {
        return this->ImgPtr;
    }

private:
    // ?============================== Variables =========================? //
    cv::Mat *ImgPtr;
    IMG_TYPE img_type = UNKNOW;
    bool     four_dir = false;
    // !============================== Functions =========================! //
    cv::Mat do_fft(const cv::Mat &img) const;
    void    fftshift(cv::Mat &fft_result) const;
};

#endif