#pragma once

#include <chrono>
#include <iostream>
#include <mutex>
#include <regex>
#include <string>
#include <thread>

#if PROFILE
#define PROFILE_SCOPE(name) Profiler timer##__LINE__(name)
#define PROFILE_FUNCTION() PROFILE_SCOPE(__PRETTY_FUNCTION__)
#else
#define PROFILE_SCOPE(name)
#define PROFILE_FUNCTION()
#endif


class ProfilerManager
{
public:
    static ProfilerManager &get_instance()
    {
        static ProfilerManager instance;
        return instance;
    }

    void add_profile(const std::string &name, long long duration)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto &data = profiles_[name];
        data.total_duration += duration;
        data.frame_count++;

        if (data.frame_count == 100)
        {
            std::string func_name = parse_function_name(name);
            std::cout << "[" << func_name << "] - Average over 100 frames: "
                      << (data.total_duration / 100.0 * 0.001) << " ms"
                      << " (" << 1000.0 / (data.total_duration / 100.0 * 0.001)
                      << " FPS)" << std::endl;
            data.total_duration = 0;
            data.frame_count = 0;
        }
    }

private:
    ProfilerManager()
    {
    }

    std::string parse_function_name(const std::string &function_name)
    {
        // Regex to match the function name and class (if any)
        std::regex name_regex("([\\w:]+)\\(");
        std::smatch matches;

        if (std::regex_search(function_name, matches, name_regex) &&
            matches.size() > 1)
            return matches[1].str();

        // Fallback in case the regex fails to match
        return function_name;
    }

    struct ProfileData
    {
        long long total_duration = 0;
        int frame_count = 0;
    };

    std::unordered_map<std::string, ProfileData> profiles_;
    std::mutex mutex_;
};


class Profiler
{
public:
    Profiler(const std::string &name)
        : _func_name(name),
          _start_time_point(std::chrono::high_resolution_clock::now())
    {
    }

    ~Profiler()
    {
        auto end_time_point = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                                end_time_point - _start_time_point)
                                .count();
        ProfilerManager::get_instance().add_profile(_func_name, duration);
    }

private:
    std::string _func_name;
    std::chrono::time_point<std::chrono::high_resolution_clock>
            _start_time_point;
};
