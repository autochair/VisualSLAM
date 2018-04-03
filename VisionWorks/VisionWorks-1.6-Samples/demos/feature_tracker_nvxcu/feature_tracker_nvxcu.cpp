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

#include "feature_tracker_nvxcu.hpp"

#include <climits>
#include <cfloat>
#include <cmath>
#include <sstream>
#include <iostream>
#include <iomanip>

#include <NVX/Utility.hpp>

//
// The feature_tracker_nvxcu.cpp contains the implementation of the  virtual void
// functions: track() and init()
//

namespace
{
    //
    // FeatureTracker based on Thin Layer Harris / Fast Track + Optical Flow PyrLK
    //

    class FeatureTrackerImpl :
            public nvxcu::FeatureTracker
    {
    public:
        explicit FeatureTrackerImpl(const Params& params);
        ~FeatureTrackerImpl();

        void init(const nvxcu_image_t * firstFrame, const nvxcu_image_t * mask);
        void track(const nvxcu_image_t * newFrame, const nvxcu_image_t * mask);

        const nvxcu_array_t * getPrevFeatures() const;
        const nvxcu_array_t * getCurrFeatures() const;

    private:
        void createDataObjects();

        void processFirstFrame(const nvxcu_image_t * frame, const nvxcu_image_t * mask);

        void release();

        Params params_;

        // Format for current frames
        nvxcu_df_image_e format_;
        uint32_t width_;
        uint32_t height_;

        // Pyramids for two successive frames
        nvxcu_pitch_linear_pyramid_t cu_prevPyr_;
        nvxcu_pitch_linear_pyramid_t cu_currPyr_;

        // Points to track for two successive frames
        nvxcu_plain_array_t cu_prevPts_;
        nvxcu_plain_array_t cu_currPts_;

        // Tracked points
        nvxcu_plain_array_t cu_kp_curr_list_;

        // Execution parameters
        nvxcu_stream_exec_target_t cu_exec_stream_target;

        // Border mode
        nvxcu_border_t cu_border_;

        nvxcu_tmp_buf_t cu_gauss_pyr_buf_;
        nvxcu_tmp_buf_t cu_keypoints_buf_;

        nvxcu_point2f_t * tmpArrayCPUData_;
        size_t * num_items_dev_ptr_;
    };

    nvxcu_plain_array_t createArrayPoint2F(uint32_t capacity)
    {
        void * dev_ptr = NULL;
        CUDA_SAFE_CALL( cudaMalloc(&dev_ptr, capacity * sizeof(nvxcu_point2f_t)) );

        uint32_t * num_items_dev_ptr = NULL;
        CUDA_SAFE_CALL( cudaMalloc((void **)&num_items_dev_ptr, sizeof(uint32_t)) );
        CUDA_SAFE_CALL( cudaMemset(num_items_dev_ptr, 0, sizeof(uint32_t)) );

        nvxcu_plain_array_t arr;
        arr.base.array_type = NVXCU_PLAIN_ARRAY;
        arr.base.item_type = NVXCU_TYPE_POINT2F;
        arr.base.capacity = capacity;
        arr.dev_ptr = dev_ptr;
        arr.num_items_dev_ptr = num_items_dev_ptr;

        return arr;
    }

    void releaseArray(nvxcu_plain_array_t *array) {
        CUDA_SAFE_CALL( cudaFree(array->num_items_dev_ptr) );
        array->num_items_dev_ptr = nullptr;

        CUDA_SAFE_CALL( cudaFree(array->dev_ptr) );
        array->dev_ptr = nullptr;
    }

    nvxcu_pitch_linear_image_t createImageU8(uint32_t width, uint32_t height)
    {
        void *dev_ptr = NULL;
        size_t pitch = 0;
        CUDA_SAFE_CALL( cudaMallocPitch(&dev_ptr, &pitch, width * sizeof(uint8_t), height) );

        nvxcu_pitch_linear_image_t image;
        image.base.image_type = NVXCU_PITCH_LINEAR_IMAGE;
        image.base.format = NVXCU_DF_IMAGE_U8;
        image.base.width = width;
        image.base.height = height;
        image.planes[0].dev_ptr = dev_ptr;
        image.planes[0].pitch_in_bytes = pitch;

        return image;
    }

    void releaseImageU8(nvxcu_pitch_linear_image_t *image) {
        CUDA_SAFE_CALL( cudaFree(image->planes[0].dev_ptr) );
        image->planes[0].dev_ptr = nullptr;
    }

    nvxcu_pitch_linear_pyramid_t createPyramidHalfScale(uint32_t width, uint32_t height, uint32_t num_levels)
    {
        ASSERT(num_levels > 0u);
        ASSERT(width > 0u);
        ASSERT(height > 0u);

        nvxcu_pitch_linear_pyramid_t pyr;
        pyr.base.pyramid_type = NVXCU_PITCH_LINEAR_PYRAMID;
        pyr.base.num_levels = num_levels;
        pyr.base.scale = NVXCU_SCALE_PYRAMID_HALF;

        pyr.levels = new nvxcu_pitch_linear_image_t[num_levels];

        uint32_t cur_width = width;
        uint32_t cur_height = height;
        float cur_scale = 1.0f;

        for (uint32_t i = 0; i < num_levels; ++i)
        {
            pyr.levels[i] = createImageU8(cur_width, cur_height);

            // Next level dimensions
            cur_scale *= pyr.base.scale;
            cur_width = (uint32_t)ceilf((float)width * cur_scale);
            cur_height = (uint32_t)ceilf((float)height * cur_scale);
        }

        return pyr;
    }

    void releasePyramid(nvxcu_pitch_linear_pyramid_t *pyramid) {
        for (uint32_t i = 0u; i < pyramid->base.num_levels; ++i)
        {
            releaseImageU8(&pyramid->levels[i]);
        }

        delete[] pyramid->levels;
        pyramid->levels = nullptr;
    }

    FeatureTrackerImpl::FeatureTrackerImpl(const Params& params) :
        params_(params),
        format_(NVXCU_DF_IMAGE_U8),
        width_(0), height_(0),
        cu_prevPyr_{}, cu_currPyr_{},
        cu_prevPts_{}, cu_currPts_{}, cu_kp_curr_list_{},
        cu_exec_stream_target{},
        cu_border_{},
        cu_gauss_pyr_buf_{}, cu_keypoints_buf_{}
    {
        tmpArrayCPUData_ = new nvxcu_point2f_t[params_.array_capacity];
        ASSERT(tmpArrayCPUData_);

        CUDA_SAFE_CALL( cudaMalloc((void **)&num_items_dev_ptr_, sizeof(size_t)) );

        {
            int currentDevice = -1;
            CUDA_SAFE_CALL( cudaGetDevice(&currentDevice) );

            cudaDeviceProp props = { };
            CUDA_SAFE_CALL( cudaGetDeviceProperties(&props, currentDevice) );

            cu_exec_stream_target.base.exec_target_type = NVXCU_STREAM_EXEC_TARGET;
            cu_exec_stream_target.dev_prop = props;
            CUDA_SAFE_CALL( cudaStreamCreate(&cu_exec_stream_target.stream) );
        }

        {
            cu_border_.mode = NVXCU_BORDER_MODE_UNDEFINED;
            cu_border_.constant_value.U8 = 0;
        }
    }

    FeatureTrackerImpl::~FeatureTrackerImpl()
    {
        release();
    }

    void FeatureTrackerImpl::init(const nvxcu_image_t * firstFrame, const nvxcu_image_t * mask)
    {
        format_ = firstFrame->format;
        width_ = firstFrame->width;
        height_ = firstFrame->height;

        // Check input format

        ASSERT(format_ == NVXCU_DF_IMAGE_RGBX);

        if (mask)
        {
            ASSERT(mask->format == NVXCU_DF_IMAGE_U8);
            ASSERT(mask->width == width_);
            ASSERT(mask->height == height_);
        }

        // Create data objects

        createDataObjects();

        // Process the first frame

        processFirstFrame(firstFrame, mask);
    }

    void FeatureTrackerImpl::track(const nvxcu_image_t * frame, const nvxcu_image_t * mask)
    {
        // Check input format and sizes

        ASSERT(format_ == frame->format);
        ASSERT(width_ == frame->width);
        ASSERT(height_ == frame->height);

        if (mask)
        {
            ASSERT(mask->format == NVXCU_DF_IMAGE_U8);
            ASSERT(mask->width == width_);
            ASSERT(mask->height == height_);
        }

        // Swap buffers

        std::swap(cu_currPts_, cu_prevPts_);
        std::swap(cu_currPyr_, cu_prevPyr_);

        // Main processing

        nvxcu_image_t * cu_frameGray = &cu_currPyr_.levels[0].base;

        NVXCU_SAFE_CALL( nvxcuColorConvert(frame, cu_frameGray,
                                           NVXCU_COLOR_SPACE_DEFAULT,
                                           NVXCU_CHANNEL_RANGE_FULL,
                                           &cu_exec_stream_target.base) );

        NVXCU_SAFE_CALL( nvxcuGaussianPyramid(&cu_currPyr_.base, &cu_gauss_pyr_buf_, &cu_border_,
                                              &cu_exec_stream_target.base) );

        // copy cu_prevPts_ into cu_kp_curr_list_ for an initial estimation
        {
            CUDA_SAFE_CALL( cudaMemcpyAsync(cu_kp_curr_list_.num_items_dev_ptr, cu_prevPts_.num_items_dev_ptr,
                                                  sizeof(uint32_t), cudaMemcpyDeviceToDevice,
                                                  cu_exec_stream_target.stream) );
            CUDA_SAFE_CALL( cudaMemcpyAsync(cu_kp_curr_list_.dev_ptr, cu_prevPts_.dev_ptr,
                                                  cu_prevPts_.base.capacity * sizeof(nvxcu_point2f_t),
                                                  cudaMemcpyDeviceToDevice, cu_exec_stream_target.stream) );
        }

        NVXCU_SAFE_CALL( nvxcuOpticalFlowPyrLK(&cu_prevPyr_.base, &cu_currPyr_.base,
                                               &cu_prevPts_.base, &cu_kp_curr_list_.base,
                                               NVXCU_TERM_CRITERIA_BOTH,
                                               0.01f, params_.lk_num_iters, params_.lk_win_size, 0,
                                               &cu_gauss_pyr_buf_, &cu_border_, &cu_exec_stream_target.base) );

        // Corner track node

        if (params_.use_harris_detector)
        {
            NVXCU_SAFE_CALL( nvxcuHarrisTrack(cu_frameGray, &cu_currPts_.base, mask,
                                              &cu_kp_curr_list_.base, params_.harris_k, params_.harris_thresh,
                                              params_.detector_cell_size, num_items_dev_ptr_, &cu_keypoints_buf_,
                                              &cu_exec_stream_target.base) );
        }
        else
        {
            NVXCU_SAFE_CALL( nvxcuFastTrack(cu_frameGray, &cu_currPts_.base, mask,
                                            &cu_kp_curr_list_.base, params_.fast_type, params_.fast_thresh,
                                            params_.detector_cell_size, num_items_dev_ptr_, &cu_keypoints_buf_,
                                            &cu_exec_stream_target.base) );
        }

        CUDA_SAFE_CALL( cudaStreamSynchronize(cu_exec_stream_target.stream) );
    }

    const nvxcu_array_t * FeatureTrackerImpl::getPrevFeatures() const
    {
        return &cu_prevPts_.base;
    }

    const nvxcu_array_t * FeatureTrackerImpl::getCurrFeatures() const
    {
        return &cu_kp_curr_list_.base;
    }

    void FeatureTrackerImpl::release()
    {
        format_ = NVXCU_DF_IMAGE_U8;
        width_ = height_ = 0u;

        //
        // Release CPU buffers
        //

        delete[] tmpArrayCPUData_;
        tmpArrayCPUData_ = nullptr;

        //
        // Release CUDA buffers
        //

        CUDA_SAFE_CALL( cudaFree(num_items_dev_ptr_) );
        num_items_dev_ptr_ = nullptr;

        //
        // Release NVXCU objects
        //

        for (nvxcu_tmp_buf_t * buf : { &cu_gauss_pyr_buf_, &cu_keypoints_buf_ })
        {
            if (buf->dev_ptr != nullptr) {
                CUDA_SAFE_CALL( cudaFree(buf->dev_ptr) );
                buf->dev_ptr = nullptr;
            }

            if (buf->host_ptr != nullptr) {
                CUDA_SAFE_CALL( cudaFreeHost(buf->host_ptr) );
                buf->host_ptr = nullptr;
            }
        }

        for (nvxcu_plain_array_t * array : { &cu_prevPts_, &cu_prevPts_, &cu_kp_curr_list_ })
        {
            releaseArray(array);
        }

        for (nvxcu_pitch_linear_pyramid_t * pyramid : { &cu_prevPyr_, &cu_currPyr_ })
        {
            releasePyramid(pyramid);
        }

        //
        // Release the CUDA stream
        //

        CUDA_SAFE_CALL( cudaStreamDestroy(cu_exec_stream_target.stream) );
        cu_exec_stream_target.stream = NULL;
    }

    void FeatureTrackerImpl::createDataObjects()
    {
        //
        // Image pyramids for two successive frames are necessary for the computation.
        //

        cu_prevPyr_ = createPyramidHalfScale(width_, height_, params_.pyr_levels);
        cu_currPyr_ = createPyramidHalfScale(width_, height_, params_.pyr_levels);

        //
        // Input points to track need to kept for two successive frames.
        //

        cu_currPts_ = createArrayPoint2F(params_.array_capacity);
        cu_prevPts_ = createArrayPoint2F(params_.array_capacity);

        //
        // Create the list of tracked points. This is the output of the frame processing
        //

        cu_kp_curr_list_ = createArrayPoint2F(params_.array_capacity);

        //
        // Pyramids
        //

        nvxcu_tmp_buf_size_t cu_gauss_pyr_buf_size = nvxcuGaussianPyramid_GetBufSize(
                    width_, height_, params_.pyr_levels,
                    NVXCU_SCALE_PYRAMID_HALF, &cu_border_, &cu_exec_stream_target.dev_prop);

        cu_gauss_pyr_buf_.host_ptr = nullptr;
        cu_gauss_pyr_buf_.dev_ptr = nullptr;
        if (cu_gauss_pyr_buf_size.host_buf_size != 0) {
            CUDA_SAFE_CALL( cudaMallocHost(&cu_gauss_pyr_buf_.host_ptr, cu_gauss_pyr_buf_size.host_buf_size) );
        }
        if (cu_gauss_pyr_buf_size.dev_buf_size != 0) {
            CUDA_SAFE_CALL( cudaMalloc(&cu_gauss_pyr_buf_.dev_ptr, cu_gauss_pyr_buf_size.dev_buf_size) );
        }

        //
        // Keypoints
        //

        nvxcu_tmp_buf_size_t cu_keypoints_buf_size = params_.use_harris_detector ?
                    nvxcuHarrisTrack_GetBufSize(width_, height_, format_,
                                                NVXCU_TYPE_POINT2F,
                                                params_.array_capacity,
                                                params_.detector_cell_size,
                                                &cu_exec_stream_target.dev_prop) :
                    nvxcuFastTrack_GetBufSize(width_, height_, format_,
                                              NVXCU_TYPE_POINT2F,
                                              params_.array_capacity,
                                              params_.detector_cell_size,
                                              &cu_exec_stream_target.dev_prop);

        cu_keypoints_buf_.host_ptr = nullptr;
        cu_keypoints_buf_.dev_ptr = nullptr;
        if (cu_keypoints_buf_size.host_buf_size != 0) {
            CUDA_SAFE_CALL( cudaMallocHost(&cu_keypoints_buf_.host_ptr, cu_keypoints_buf_size.host_buf_size) );
        }
        if (cu_keypoints_buf_size.dev_buf_size != 0) {
            CUDA_SAFE_CALL( cudaMalloc(&cu_keypoints_buf_.dev_ptr, cu_keypoints_buf_size.dev_buf_size) );
        }
    }

    //
    // The processFirstFrame() converts the first frame into grayscale,
    // builds initial Gaussian pyramid and detects initial keypoints.
    //

    void FeatureTrackerImpl::processFirstFrame(const nvxcu_image_t * frame, const nvxcu_image_t * mask)
    {
        nvxcu_pitch_linear_image_t cu_frameGray = cu_currPyr_.levels[0];

        NVXCU_SAFE_CALL( nvxcuColorConvert(frame, &cu_frameGray.base,
                                           NVXCU_COLOR_SPACE_DEFAULT,
                                           NVXCU_CHANNEL_RANGE_FULL,
                                           &cu_exec_stream_target.base) );

        NVXCU_SAFE_CALL( nvxcuGaussianPyramid(&cu_currPyr_.base, &cu_gauss_pyr_buf_, &cu_border_, &cu_exec_stream_target.base) );

        if (params_.use_harris_detector)
        {
            NVXCU_SAFE_CALL( nvxcuHarrisTrack(&cu_frameGray.base, &cu_currPts_.base, mask,
                                              nullptr, params_.harris_k, params_.harris_thresh,
                                              params_.detector_cell_size, num_items_dev_ptr_, &cu_keypoints_buf_,
                                              &cu_exec_stream_target.base) );
        }
        else
        {
            NVXCU_SAFE_CALL( nvxcuFastTrack(&cu_frameGray.base, &cu_currPts_.base, mask,
                                            nullptr, params_.fast_type, params_.fast_thresh,
                                            params_.detector_cell_size, num_items_dev_ptr_, &cu_keypoints_buf_,
                                            &cu_exec_stream_target.base) );
        }

        CUDA_SAFE_CALL( cudaStreamSynchronize(cu_exec_stream_target.stream) );
    }
}

nvxcu::FeatureTracker::Params::Params()
{
    // Parameters for optical flow node
    pyr_levels = 6u;
    lk_num_iters = 5u;
    lk_win_size = 10u;

    // Common parameters for corner detector node
    array_capacity = 2000u;
    detector_cell_size = 18u;
    use_harris_detector = true;

    // Parameters for harris_track node
    harris_k = 0.04f;
    harris_thresh = 100.0f;

    // Parameters for fast_track node
    fast_type = 9u;
    fast_thresh = 25u;
}

nvxcu::FeatureTracker * nvxcu::FeatureTracker::create(const Params& params)
{
    return new FeatureTrackerImpl(params);
}
