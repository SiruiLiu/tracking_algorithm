#ifndef AUTOZOOM_H
#define AUTOZOOM_H

#include "ImgDef.h"

class AutoZoom : public ImgDef {
public:
    AutoZoom(cv::Mat *ImgPtr)
        : ImgDef(nullptr)
        , ImgPtr(ImgPtr) {
    }

    ~AutoZoom() override {
        if (this->ImgPtr != nullptr) {
            delete this->ImgPtr;
            this->ImgPtr = nullptr;
        }
    }

    void adjusting();

protected:
    virtual cv::Mat *getImgPtr() const override {
        return ImgPtr;
    }

private:
    cv::Mat *ImgPtr;
};

#endif