/*
# Copyright (c) 2012-2015, NVIDIA CORPORATION. All rights reserved.
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

#include "object_tracker_nvxcu.hpp"
#include "object_tracker_with_features_info_nvxcu.hpp"
#include "object_tracker_keypoint_nvxcu.hpp"

nvxcu::ObjectTrackerWithFeaturesInfo* nvxcuCreateKeypointObjectTrackerWithFeaturesInfo(const nvxcu::KeypointObjectTrackerParams& params)
{
    nvxcu_keypoint_tracker::ObjectTrackerPyrLKKeyHistory::Parameters default_parameters;

    default_parameters.cornerDetectorParameters_.maxCorners_ = params.max_corners;
    default_parameters.cornerDetectorParameters_.detectorCellSize_ = params.detector_cell_size;
    default_parameters.cornerDetectorParameters_.useFastDetector_ = params.use_fast_detector;

    default_parameters.cornerDetectorParameters_.fastType_ = params.fast_type;
    default_parameters.cornerDetectorParameters_.fastThreshold_ = params.fast_threshold;

    default_parameters.cornerDetectorParameters_.harrisK_ = params.harris_k;
    default_parameters.cornerDetectorParameters_.harrisThreshold_ = params.harris_threshold;

    default_parameters.cornerDetectorParameters_.bbDecreasingRatio_ = params.bb_decreasing_ratio;

    default_parameters.cornersFilteringParameters_.maxCornersInCell_ = params.max_corners_in_cell;
    default_parameters.cornersFilteringParameters_.xNumOfCells_ = params.x_num_of_cells;
    default_parameters.cornersFilteringParameters_.yNumOfCells_ = params.y_num_of_cells;

    default_parameters.pyrLKParameters_.maxIterationsNumber_ = params.lk_num_iters;
    default_parameters.pyrLKParameters_.numPyramidLevels_ = params.pyr_levels;
    default_parameters.pyrLKParameters_.windowSize_ = params.lk_win_size;

    return new nvxcu_keypoint_tracker::ObjectTrackerPyrLKKeyHistory(default_parameters);
}
