#ifndef IMGDEF_H
#define IMGDEF_H

#include "opencv4/opencv2/opencv.hpp"
#include "typedef.h"
#include <cstddef>
#include <iostream>

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

    virtual cv::Mat *getImgPtr() const {
        return this->ImgPtr;
    }

private:
    cv::Mat *ImgPtr;
    IMG_TYPE img_type;
};

#endif