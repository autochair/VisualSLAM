/*
# Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
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

#ifndef NVXCU_FEATURE_TRACKER_HPP
#define NVXCU_FEATURE_TRACKER_HPP

#include <NVX/nvxcu.h>

namespace nvxcu
{
    class FeatureTracker
    {
    public:
        struct Params
        {
            // parameters for optical flow node
            uint32_t pyr_levels;
            uint32_t lk_num_iters;
            uint32_t lk_win_size;

            // common parameters for corner detector node
            uint32_t array_capacity;
            uint32_t detector_cell_size;
            bool use_harris_detector;

            // parameters for harris_track node
            float harris_k;
            float harris_thresh;

            // parameters for fast_track node
            uint32_t fast_type;
            uint32_t fast_thresh;

            Params();
        };

        static FeatureTracker * create(const Params& params = Params());

        virtual ~FeatureTracker() { }

        virtual void init(const nvxcu_image_t * firstFrame, const nvxcu_image_t * mask = nullptr) = 0;
        virtual void track(const nvxcu_image_t * newFrame, const nvxcu_image_t * mask = nullptr) = 0;

        // get list of tracked features on previous frame
        virtual const nvxcu_array_t * getPrevFeatures() const = 0;

        // get list of tracked features on current frame
        virtual const nvxcu_array_t * getCurrFeatures() const = 0;
    };

} // namespace nvxcu

#endif // NVXCU_FEATURE_TRACKER_HPP
