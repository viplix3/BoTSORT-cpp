#pragma once

#include <memory>
#include <string>

#include <opencv2/opencv.hpp>

#include "NvInfer.h"

static auto StreamDeleter = [](cudaStream_t *ptr) {
    if (ptr)
    {
        cudaStreamDestroy(*ptr);
        delete ptr;
    }
};

struct TRTDeleter
{
    template<typename T>
    void operator()(T *obj) const
    {
        if (obj) { obj->destroy(); }
    }
};


struct TRTDestroyer
{
    template<typename T>
    void operator()(T *obj) const
    {
        if (obj) { obj->destroy(); }
    }
};

template<typename T>
using TRTUniquePtr = std::unique_ptr<T, TRTDeleter>;

template<typename T>
inline TRTUniquePtr<T> infer_object(T *obj)
{
    if (!obj) { throw std::runtime_error("Failed to create object"); }
    return TRTUniquePtr<T>(obj);
}
