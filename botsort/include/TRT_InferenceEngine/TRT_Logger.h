#pragma once

#include <iostream>

#include <NvInfer.h>

class TRTLogger : public nvinfer1::ILogger
{
public:
    TRTLogger(Severity severity = Severity::kWARNING)
        : reportableSeverity(severity)
    {
    }

    void log(Severity severity, const char *msg) noexcept override
    {
        // suppress messages with severity enum value greater than the reportable
        if (severity > reportableSeverity)
            return;

        switch (severity)
        {
            case Severity::kINTERNAL_ERROR:
                std::cout << "INTERNAL_ERROR: ";
                break;
            case Severity::kERROR:
                std::cout << "ERROR: ";
                break;
            case Severity::kWARNING:
                std::cout << "WARNING: ";
                break;
            case Severity::kINFO:
                std::cout << "INFO: ";
                break;
            default:
                std::cout << "UNKNOWN: ";
                break;
        }
        std::cout << msg << std::endl;
        std::cout << std::flush;
    }

    Severity reportableSeverity;
};