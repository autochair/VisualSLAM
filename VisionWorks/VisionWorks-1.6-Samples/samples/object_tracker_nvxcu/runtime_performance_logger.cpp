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

#ifndef _WIN32

#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sys/stat.h>
#include "runtime_performance_logger.hpp"

bool RuntimePerformanceLogger::init(std::string perfLogPrefix)
{
    item_index_ = -1;
    std::fill(max_, max_+NUM_PERF_ITEM, 0);
    std::fill(avg_, avg_+NUM_PERF_ITEM, 0);
    std::fill(sum_, sum_+NUM_PERF_ITEM, 0);
    std::fill(count_, count_+NUM_PERF_ITEM, 0);
    std::fill(&perf_[0][0], &perf_[0][0]+RUNNING_PERF_WINDOW_SIZE*NUM_PERF_ITEM, 0);
    std::fill(min_, min_+NUM_PERF_ITEM, 1000);
    std::string perfSumLogFile;
    std::string perfRecordLogFile;

    if (perfLogPrefix.back() != '/')
    {
        perfLogPrefix += '/';
    }

    perfSumLogFile = perfLogPrefix + "perf_sum.log";
    perfRecordLogFile = perfLogPrefix + "perf_record.log";

    struct stat info;
    if (stat(perfLogPrefix.c_str(), &info) != 0) {
        const int dir_err = mkdir(perfLogPrefix.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        if (-1 == dir_err) {
            std::cerr << "Error creating directory!" << std::endl;
            return false;
        }
    }

    try
    {
       sum_fid.open(perfSumLogFile);
       record_fid.open(perfRecordLogFile);
    }
    catch (const std::ofstream::failure& e)
    {
        std::cerr << "Failed to open performance log file. " << std::endl;
        return false;
    }

    return true;
}

void RuntimePerformanceLogger::log(void)
{
    //
    //output summary
    //
    size_t dataColumnWidth = 16;
    size_t maxItemNameLength = 0;

    //std::cout << "********************************" << std::endl;
    for (auto str : item_name_)
    {
        maxItemNameLength = maxItemNameLength > str.length() ? maxItemNameLength : str.length();
    }

    sum_fid << std::setw(maxItemNameLength) << "--------"
            << std::setw(dataColumnWidth) << "avg_in_ms"
            << std::setw(dataColumnWidth) << "min_in_ms"
            << std::setw(dataColumnWidth) << "max_in_ms" << std::endl;

    for (int i = PYRANDSAVE; i < NUM_PERF_ITEM; i++)
    {
        sum_fid << std::setw(maxItemNameLength) << item_name_[i]
                << std::setw(dataColumnWidth) << avg_[i]
                << std::setw(dataColumnWidth) << min_[i]
                << std::setw(dataColumnWidth) << max_[i] << std::endl;
    }

    sum_fid.close();
    //
    //output record
    //
    record_fid << std::setw(dataColumnWidth) << "FRAME_IDX";
    for (int i = OBJECTNUM; i < NUM_PERF_ITEM; i++)
    {
        record_fid << std::setw(dataColumnWidth) << item_name_[i];
    }
    record_fid << std::endl;

    for (int i = 0, frame_idx = 0; i < item_index_; i++, frame_idx++)
    {
        record_fid << std::setw(dataColumnWidth) << frame_idx;

        //output module timing
        for (int j = OBJECTNUM; j < NUM_PERF_ITEM; j++)
        {
            record_fid << std::setw(dataColumnWidth) << perf_[i][j];
        }
        record_fid << std::endl;
    }
    record_fid.close();
}

#endif
