/*
# Copyright (c) 2012-2016, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef VW_TRACKER_RUNTIME_PERFORMANCE_LOGGER_HPP
#define VW_TRACKER_RUNTIME_PERFORMANCE_LOGGER_HPP

#ifndef _WIN32

#include <string>
#include <vector>
#include <fstream>
#include <iomanip>
#include <sys/stat.h>


//
// Running Performance Logger
//

class RuntimePerformanceLogger
{
public:
    enum
    {
        OBJECTNUM,
        PYRANDSAVE,
        OVERALL,
        NUM_PERF_ITEM,
    };

    const std::vector<std::string> item_name_;

    RuntimePerformanceLogger():
        item_name_ {"OBJECTNUM", "PYRANDSAVE", "OVERALL"},
        item_index_(-1),
        min_ {0},
        max_ {0},
        avg_ {0},
        sum_ {0},
        count_{0},
        perf_ {{0}}

    {}

    bool init(std::string perfLogPrefix);
    void log(void);

    static const unsigned int RUNNING_PERF_WINDOW_SIZE = 16*1024;   // 16K

    template<int PERF_TYPE_IDX>
    inline void addItem(double value)
    {
        min_[PERF_TYPE_IDX] = min_[PERF_TYPE_IDX] < value ? min_[PERF_TYPE_IDX] : value;
        max_[PERF_TYPE_IDX] = max_[PERF_TYPE_IDX] > value ? max_[PERF_TYPE_IDX] : value;
        sum_[PERF_TYPE_IDX] = sum_[PERF_TYPE_IDX] - perf_[item_index_][PERF_TYPE_IDX] + value;
        perf_[item_index_][PERF_TYPE_IDX] = value;

        count_[PERF_TYPE_IDX] = count_[PERF_TYPE_IDX] + 1 < RUNNING_PERF_WINDOW_SIZE ? count_[PERF_TYPE_IDX] + 1 : RUNNING_PERF_WINDOW_SIZE;
        avg_[PERF_TYPE_IDX] = sum_[PERF_TYPE_IDX] / count_[PERF_TYPE_IDX];
    }

    template<int PERF_TYPE_IDX>
    inline double getMin()
    {
        return min_[PERF_TYPE_IDX];
    }

    template<int PERF_TYPE_IDX>
    inline double getMax()
    {
        return max_[PERF_TYPE_IDX];
    }

    template<int PERF_TYPE_IDX>
    inline double getAvg()
    {
        return avg_[PERF_TYPE_IDX];
    }

    inline void newFrame()
    {
        item_index_++;
        item_index_ %= RUNNING_PERF_WINDOW_SIZE;
    }

private:
    int item_index_;
    double min_[NUM_PERF_ITEM];
    double max_[NUM_PERF_ITEM];
    double avg_[NUM_PERF_ITEM];
    double sum_[NUM_PERF_ITEM];
    unsigned int count_[NUM_PERF_ITEM];
    double perf_[RUNNING_PERF_WINDOW_SIZE][NUM_PERF_ITEM];
    std::ofstream sum_fid, record_fid;
};

#endif // _WIN32

#endif // VW_TRACKER_RUNTIME_PERFORMANCE_LOGGER_HPP
