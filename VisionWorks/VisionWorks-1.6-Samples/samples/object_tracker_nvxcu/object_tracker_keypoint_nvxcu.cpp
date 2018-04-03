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

#include <algorithm>
#include <cfloat>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <cuda_runtime_api.h>

#include "object_tracker_nvxcu.hpp"
#include "object_tracker_with_features_info_nvxcu.hpp"
#include "object_tracker_keypoint_nvxcu.hpp"

#ifndef _WIN32
#include "runtime_performance_logger.hpp"
#include <NVX/nvx_timer.hpp>
#endif

using std::vector;

#define PRINT_ERROR(msg) \
    do { \
        std::ostringstream ostr_; \
        ostr_ << msg; \
        std::cout << ostr_.str(); \
    } while(0)

#define THROW_EXCEPTION(msg) \
    do { \
        std::ostringstream ostr_; \
        ostr_ << msg; \
        throw std::runtime_error(ostr_.str()); \
    } while(0)

#define NVXCU_SAFE_CALL(nvxcuOp) \
    do \
    { \
        nvxcu_error_status_e stat = (nvxcuOp); \
        if (stat != NVXCU_SUCCESS) \
        { \
            THROW_EXCEPTION(#nvxcuOp << " failure [status = " << stat << "]" << " in file " << __FILE__ << " line " << __LINE__); \
        } \
    } while (0)

#define ASSERT(cond) \
    do \
    { \
        bool stat = (cond); \
        if (!stat) \
        { \
            THROW_EXCEPTION(#cond << " failure in file " << __FILE__ << " line " << __LINE__); \
        } \
    } while (0)

#define CUDA_SAFE_CALL(cudaOp) \
    do \
    { \
        cudaError_t err = (cudaOp); \
        if (err != cudaSuccess) \
        { \
            THROW_EXCEPTION(#cudaOp << " failure [CUDA error = " << err << "]" << " in file " << __FILE__ << " line " << __LINE__);  \
        } \
    } while (0)

#define CUDA_ERROR_PRINT(cudaOp) \
    do \
    { \
        cudaError_t err = (cudaOp); \
        if (err != cudaSuccess) \
        { \
            PRINT_ERROR(#cudaOp << " failure [CUDA error = " << err << "]" << " in file " << __FILE__ << " line " << __LINE__);  \
        } \
    } while (0)


namespace
{
inline nvxcu_point2f_t create_point2f(float x, float y)
{
    nvxcu_point2f_t res = {x,y};
    return res;
}
inline nvxcu_point2f_t create_point2f()
{
    nvxcu_point2f_t res = {0,0};
    return res;
}

struct rectanglef_t
{
    float x;
    float y;
    float width;
    float height;

    bool isValid() const
    {
        return (width >= 0.0f) && (height >=0.0f);
    }

    rectanglef_t()
        :x(0.0f),
        y(0.0f),
        width(0.0f),
        height(0.0f)
    {
    }
    rectanglef_t(float x1, float y1, float width1, float height1)
        :x(x1),
        y(y1),
        width(width1),
        height(height1)
    {
    }
    rectanglef_t(nvxcu_point2f_t tl, nvxcu_point2f_t br)
    {
        x = std::min(tl.x, br.x);
        y = std::min(tl.y, br.y);
        float end_x = std::max(tl.x, br.x);
        float end_y = std::max(tl.y, br.y);

        width = end_x - x;
        height = end_y - y;
    }
    nvxcu_point2f_t tl() const
    {
        return create_point2f(x, y);
    }
    nvxcu_point2f_t br() const
    {
        return create_point2f(x + width, y + height);
    }

    static rectanglef_t intersect(const rectanglef_t& rect1, const rectanglef_t& rect2)
    {
        float tl_x = std::max(rect1.x, rect2.x);
        float tl_y = std::max(rect1.y, rect2.y);
        float br_x = std::min(rect1.x + rect1.width, rect2.x + rect2.width);
        float br_y = std::min(rect1.y + rect1.height, rect2.y + rect2.height);
        if ( (tl_x > br_x) || (tl_y > br_y) )
            return rectanglef_t();
        return rectanglef_t(tl_x, tl_y, br_x - tl_x, br_y - tl_y);
    }
};

inline std::string to_str(const rectanglef_t& val)
{
    std::stringstream strstr;
    strstr << "(" << val.x << ", " << val.y << "; " << val.width << "x" << val.height << ")";
    return strstr.str();
}

inline double getDistance(const nvxcu_point2f_t& pt1, const nvxcu_point2f_t& pt2)
{
    double diff_x = pt2.x - pt1.x;
    double diff_y = pt2.y - pt1.y;
    double val = diff_x*diff_x + diff_y*diff_y;
    return sqrt(val);
}

inline double getDistance(const nvxcu_keypointf_t& pt1, const nvxcu_keypointf_t& pt2)
{
    double diff_x = pt2.x - pt1.x;
    double diff_y = pt2.y - pt1.y;
    double val = diff_x*diff_x + diff_y*diff_y;
    return sqrt(val);
}

inline bool isPointInsideRect(const nvxcu_keypointf_t& p, const rectanglef_t& rect)
{
    if (p.x < rect.tl().x)
        return false;
    if (p.y < rect.tl().y)
        return false;

    if (p.x >= rect.br().x)
        return false;
    if (p.y >= rect.br().y)
        return false;

    return true;
}

inline bool isPointInsideRect(const nvxcu_point2f_t& p, const nvxcu_rectangle_t& rect)
{
    if (p.x < rect.start_x)
        return false;
    if (p.y < rect.start_y)
        return false;

    if (p.x >= rect.end_x)
        return false;
    if (p.y >= rect.end_y)
        return false;

    return true;
}

nvxcu_point2f_t getCenterPoint(const std::vector<nvxcu_point2f_t>& points, const std::vector<float>& weights)
{
    if (points.empty())
        return create_point2f();

    ASSERT(weights.size() == points.size());
    int32_t N = points.size();

    double sum_weights = 0;
    double sum_x = 0;
    double sum_y = 0;
    for (int32_t n = 0; n < N; n++)
    {
        double w = weights[n];
        sum_x += points[n].x * w;
        sum_y += points[n].y * w;
        sum_weights += w;
    }
    double coeff = 1.0 / sum_weights;
    nvxcu_point2f_t result = create_point2f(sum_x * coeff, sum_y * coeff);
    return result;
}

template<typename T>
inline T getMedian(vector<T> &vec)
{
    size_t index = vec.size() / 2;
    nth_element(vec.begin(), vec.begin()+index, vec.end());
    return vec[index];
}

nvxcu_rectangle_t increaseRectangleByBorder(const nvxcu_rectangle_t& rect, int32_t border, int32_t width, int32_t height)
{
    int32_t start_x = (int32_t)rect.start_x - (int32_t)border;
    int32_t start_y = (int32_t)rect.start_y - (int32_t)border;
    int32_t end_x = (int32_t)rect.end_x + (int32_t)border;
    int32_t end_y = (int32_t)rect.end_y + (int32_t)border;
    if (start_x < 0)
        start_x = 0;
    if (start_y < 0)
        start_y = 0;
    if (end_x > width)
        end_x = width;
    if (end_y > height)
        end_y = height;

    return {(uint32_t)start_x, (uint32_t)start_y, (uint32_t)end_x, (uint32_t)end_y};
}

nvxcu_rectangle_t decreaseRectangleByCoeff(const nvxcu_rectangle_t& rect, float ratio,  int32_t width, int32_t height)
{
    uint32_t rect_width = rect.end_x - rect.start_x;
    uint32_t  rect_height = rect.end_y - rect.start_y;

    uint32_t  decreased_rect_width = (uint32_t ) (ratio * rect_width);
    uint32_t  decreased_rect_height = (uint32_t ) (ratio * rect_height);

    uint32_t  border_x = (rect_width - decreased_rect_width) / 2;
    uint32_t  border_y = (rect_height - decreased_rect_height) / 2;

    int32_t  start_x = (int32_t )rect.start_x + (int32_t )border_x;
    int32_t  start_y = (int32_t )rect.start_y + (int32_t )border_y;

    int32_t  end_x = (int32_t )rect.end_x - (int32_t )border_x;
    int32_t  end_y = (int32_t )rect.end_y - (int32_t )border_y;

    if (start_x < 0)
        start_x = 0;
    if (start_y < 0)
        start_y = 0;
    if (end_x > width)
        end_x = width;
    if (end_y > height)
        end_y = height;

    return {(uint32_t )start_x, (uint32_t )start_y, (uint32_t )end_x, (uint32_t )end_y};
}

size_t allocateBufferForArraysKeypointF(uint32_t num_of_arrays, uint32_t capacity, void **buf_ptr)
{
    size_t size = num_of_arrays * sizeof(uint32_t) + num_of_arrays * capacity * sizeof(nvxcu_keypointf_t);
    CUDA_SAFE_CALL(cudaMalloc(buf_ptr, size));
    return size;
}

std::vector<nvxcu_plain_array_t> createArraysKeyPointFFromBuffer(uint32_t num_of_arrays, uint32_t capacity, void *buf_ptr)
{
    std::vector<nvxcu_plain_array_t> arrays_v;

    for (uint32_t i = 0; i < num_of_arrays; i++)
    {
        uint32_t* num_items_dev_ptr = (uint32_t*)buf_ptr + i;
        nvxcu_keypointf_t* dev_ptr = (nvxcu_keypointf_t*)((uint32_t*)buf_ptr + num_of_arrays) + i * capacity;

        nvxcu_plain_array_t arr;
        arr.base.array_type = NVXCU_PLAIN_ARRAY;
        arr.base.item_type = NVXCU_TYPE_KEYPOINTF;
        arr.base.capacity = capacity;
        arr.dev_ptr = dev_ptr;
        arr.num_items_dev_ptr = num_items_dev_ptr;
        arrays_v.push_back(arr);
    }

    return arrays_v;
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
    CUDA_ERROR_PRINT( cudaFree(image->planes[0].dev_ptr) );
    image->planes[0].dev_ptr = nullptr;
}

nvxcu_pitch_linear_pyramid_t createPyramidHalfScale(uint32_t width, uint32_t height, uint32_t num_levels)
{
    nvxcu_pitch_linear_pyramid_t pyr;
    pyr.base.pyramid_type = NVXCU_PITCH_LINEAR_PYRAMID;
    pyr.base.num_levels = num_levels;
    pyr.base.scale = NVXCU_SCALE_PYRAMID_HALF;

    pyr.levels = (nvxcu_pitch_linear_image_t *)calloc(num_levels, sizeof(nvxcu_pitch_linear_image_t));

    uint32_t cur_width = width;
    uint32_t cur_height = height;
    float cur_scale = 1.0f;

    for (uint32_t i = 0; i < num_levels; i++)
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

    free(pyramid->levels);
    pyramid->levels = nullptr;
}

} // Anonymous namespace

namespace nvxcu_keypoint_tracker
{
class MatChar
{
public:
    MatChar()
        :width_(0),
        height_(0)
    {
    }
    MatChar(int32_t  width, int32_t  height)
        :width_(width),
        height_(height),
        data_(width*height, 0)
    {

        ASSERT(this->isValid());

    }
    MatChar(const MatChar& other)
        :width_(other.width_),
        height_(other.height_)
    {
        ASSERT(other.total() == total());
        ASSERT(other.isValid());

        data_.insert(data_.begin(), other.data_.begin(), other.data_.end());
    }
    inline int32_t  total() const
    {
        return width_ * height_;
    }
    bool isValid() const
    {
        return ((int32_t )data_.size() == total());
    }
    inline void setToZero()
    {
        ASSERT(isValid());
        int32_t  N = total();
        std::fill_n(data_.begin(), N, 0);
    }
    int8_t & at(int32_t  i, int32_t  j)
    {
        return data_ [width_*i + j];
    }
    int8_t & at(const nvxcu_keypointf_t& pt)
    {
        int32_t  x = (int32_t )(pt.x + 0.5);
        int32_t  y = (int32_t )(pt.y + 0.5);
        return data_ [width_*y + x];
    }

    uint32_t  width_;
    uint32_t  height_;
    std::vector<int8_t> data_;
};

class OutlierRemoval
{
public:
    typedef ObjectTrackerPyrLKKeyHistory::OutlierRemovalParameters Parameters;
    OutlierRemoval()
    {
    }
    explicit OutlierRemoval(const Parameters& parameters):
        parameters_(parameters)
    {
    }
    std::vector<float> process(std::vector<nvxcu_point2f_t>& points, const vector<float>& src_weights);
private:
    Parameters parameters_;

    static inline double getAverageDistance(const std::vector<nvxcu_point2f_t>& points, const nvxcu_point2f_t& center, const std::vector<float>& weights);
    static inline bool removeOutliers(const std::vector<nvxcu_point2f_t>& points, const nvxcu_point2f_t& center, double coeff_of_mean_distance, double max_mean_distance_to_stop,
                                      std::vector<float>& weights);
    static inline double getMaxDistance(const std::vector<nvxcu_point2f_t>& points, const nvxcu_point2f_t& center, const std::vector<float>& weights);
    double getMaxWeightedDistance(const std::vector<nvxcu_point2f_t>& points, const nvxcu_point2f_t& center, const std::vector<float>& weights);
};

class OpticalFlowBuilder
{
public:
    typedef ObjectTrackerPyrLKKeyHistory::PyramidalLucasKanadeParameters PyrLKParameters;
    typedef ObjectTrackerPyrLKKeyHistory::CornerDetectorParameters CornerDetectorParameters;


    OpticalFlowBuilder(const PyrLKParameters& pyr_lk_params, const CornerDetectorParameters& corner_detector_params);
    ~OpticalFlowBuilder();
    nvxcu_error_status_e buildPyramidAndSave(const nvxcu_pitch_linear_image_t* frame,
                                             const std::list< std::unique_ptr<nvxcu_keypoint_tracker::ObjectTrackerPyrLKKeyHistory::TrackedObjectImpl> >& objects);

    nvxcu_error_status_e processObject(const nvxcu_rectangle_t& obj_rect,
                            vector<nvxcu_keypointf_t>& res_points0,
                            vector<nvxcu_keypointf_t>& res_points1,
                            bool& is_object_lost);

private:
    PyrLKParameters pyrLKParameters_;
    CornerDetectorParameters cornerDetectorParameters_;
    nvxcu_error_status_e setObjectsMask (const std::list< std::unique_ptr<nvxcu_keypoint_tracker::ObjectTrackerPyrLKKeyHistory::TrackedObjectImpl> >& objects);
    nvxcu_error_status_e createDataObjects(uint32_t  width, uint32_t  height);
    nvxcu_error_status_e allocateBuffers();
    nvxcu_error_status_e processFirstFrame(const nvxcu_pitch_linear_image_t* frame);
    nvxcu_error_status_e initBuilder(const nvxcu_pitch_linear_image_t* firstFrame);

    void releaseIntObjects();

    bool initialized_builder_;
    nvxcu_df_image_e format_;
    uint32_t  width_;
    uint32_t  height_;

    nvxcu_pitch_linear_image_t mask_;

    nvxcu_pitch_linear_pyramid_t prev_pyr_;
    nvxcu_pitch_linear_pyramid_t curr_pyr_;

    nvxcu_plain_array_t points0_;
    nvxcu_plain_array_t points1_;
    nvxcu_plain_array_t pointsFb_;
    size_t *num_corners_dev_ptr_;

    nvxcu_tmp_buf_size_t fast_harris_track_buf_size_;
    nvxcu_tmp_buf_size_t gauss_pyr_buf_size_;
    nvxcu_tmp_buf_size_t opt_flow_pyr_lk_buf_size_;
    nvxcu_tmp_buf_size_t final_buf_size_;
    nvxcu_tmp_buf_t tmp_buf_;


    nvxcu_stream_exec_target_t exec_target_;
    nvxcu_border_t border_;

    void *points_dev_ptr_;
    uint8_t *points_cpu_ptr_;
    uint32_t num_of_points_buffers_;
};

class ScaleLogarithmicAverager
{
public:
    typedef ObjectTrackerPyrLKKeyHistory::AdvancedScaleEstimatorParameters Parameters;
    explicit ScaleLogarithmicAverager(const Parameters& params):
        parameters_(params)
    {
    }

    bool process(const std::vector<nvxcu_keypointf_t>& points0,
                 const std::vector<nvxcu_keypointf_t>& points1,
                 const std::vector<float>& weights,
                 nvxcu_point2f_t& motion_scale);
private:
    typedef std::pair<float, int32_t > WeightWithIndex;
    static bool getNQuantilesFromSortedWeightVector(const std::vector<WeightWithIndex>& sorted_weights_with_indexes,
                                                    const int32_t  desirable_num_quantiles,
                                                    const int32_t  minimal_number_of_points,
                                                    std::vector<int32_t >& first_elements_of_quantiles);

    static rectanglef_t getBoundingBoxForPointsWithWeightNotLessThanThreshold(const std::vector<nvxcu_point2f_t>& points,
                                                                              const std::vector<WeightWithIndex>& sorted_weights_with_indexes,
                                                                              int32_t  first_element,
                                                                              int32_t  min_number_of_points);
    static nvxcu_point2f_t getMedianRatioForQuantile(const std::vector<nvxcu_keypointf_t>& points0,
                                                 const std::vector<nvxcu_keypointf_t>& points1,
                                                 const std::vector<WeightWithIndex>& sorted_weights_with_indexes,
                                                 int32_t  first_element);
    static void makeLogarithmicAveragingOfScales(const std::vector<nvxcu_point2f_t >& scales_xy,
                                                 std::vector<WeightWithIndex>& sorted_weights_with_indexes,
                                                 const std::vector<int32_t >& first_elements_of_quantiles,
                                                 double& scale_x,
                                                 double& scale_y);

    Parameters parameters_;
    static bool compareWeightsWithIndexes(const WeightWithIndex& a, const WeightWithIndex& b)
    {
        return a.first < b.first;
    }
    static void fillVectorOfWeightsWithIndexes(const std::vector<float>& weights, std::vector<WeightWithIndex>& res);
};
} //namespace nvxcu_keypoint_tracker
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

double nvxcu_keypoint_tracker::OutlierRemoval::getAverageDistance(const std::vector<nvxcu_point2f_t>& points, const nvxcu_point2f_t& center, const std::vector<float>& weights)
{
    if (points.empty())
        return 0;

    ASSERT(weights.size() == points.size());
    int32_t  N = points.size();

    double sum = 0;
    double sum_weights = 0;
    for (int32_t  n = 0; n < N; n++)
    {
        const nvxcu_point2f_t& pt = points[n];
        sum += getDistance(create_point2f(pt.x, pt.y), center) * weights[n];
        sum_weights += weights[n];
    }

    return sum / sum_weights;
}

bool nvxcu_keypoint_tracker::OutlierRemoval::removeOutliers(const std::vector<nvxcu_point2f_t>& points, const nvxcu_point2f_t& center, double coeff_of_mean_distance, double max_mean_distance_to_stop,
                                                      std::vector<float>& weights)
{
    if (points.empty())
    {
        weights = vector<float>();
        return false;
    }

    double mean_distance = getAverageDistance(points, center, weights);
    double distance_to_compare = mean_distance * coeff_of_mean_distance;

    if (mean_distance <= max_mean_distance_to_stop)
    {
        return false;
    }

    ASSERT(weights.size() == points.size());
    int32_t  N = points.size();

    vector<float> res_weights = weights;
    bool were_outliers_removed = false;
    for (int32_t  n = 0; n < N; n++)
    {
        if (weights[n] == 0)
            continue;

        const nvxcu_point2f_t& pt = points[n];
        double cur_distance = getDistance(pt, center);
        if (cur_distance > distance_to_compare)
        {
            res_weights[n] = 0;
            were_outliers_removed = true;
        }
    }
    weights = res_weights;
    return were_outliers_removed;
}

std::vector<float> nvxcu_keypoint_tracker::OutlierRemoval::process(std::vector<nvxcu_point2f_t>& points,
                                                                  const vector<float>& src_weights)
{
    int32_t  num_iterations = parameters_.maxIterationsNumber_;
    double coeff_of_mean_distance = parameters_.relativeDistanceToRemoveOutlier_;
    double max_mean_distance_to_stop = parameters_.maxMeanDistanceToStop_;

    ASSERT(points.size() == src_weights.size());

    vector<float> weights = src_weights;
    for(int32_t  n = 0; n < num_iterations; n++)
    {
        nvxcu_point2f_t center = getCenterPoint(points, weights);


        vector<float> next_weights = weights;

        bool were_outliers_removed = removeOutliers(points, center, coeff_of_mean_distance, max_mean_distance_to_stop,
                                                    next_weights);

        weights = next_weights;
        if (!were_outliers_removed)
        {
            break;
        }
    }
    return weights;
}

/////////////////////////////////////////////////////////////////////////////////////////

nvxcu_keypoint_tracker::ObjectTrackerPyrLKKeyHistory::TrackedObjectImpl::TrackedObjectImpl(uint64_t id, const nvxcu_rectangle_t& loc) :
    id_(id), loc_(loc), status_(TRACKED)
{
}

nvxcu_keypoint_tracker::ObjectTrackerPyrLKKeyHistory::TrackedObjectImpl::~TrackedObjectImpl()
{
}

uint32_t  nvxcu_keypoint_tracker::ObjectTrackerPyrLKKeyHistory::TrackedObjectImpl::getID() const
{
    return id_;
}

nvxcu_rectangle_t nvxcu_keypoint_tracker::ObjectTrackerPyrLKKeyHistory::TrackedObjectImpl::getLocation() const
{
    return loc_;
}

nvxcu::ObjectTracker::ObjectStatus nvxcu_keypoint_tracker::ObjectTrackerPyrLKKeyHistory::TrackedObjectImpl::getStatus() const
{
    return status_;
}
const nvxcu_keypoint_tracker::ObjectTrackerPyrLKKeyHistory::FeaturePointsVector&
nvxcu_keypoint_tracker::ObjectTrackerPyrLKKeyHistory::TrackedObjectImpl::getFeaturePointSet() const
{
    return features_info_;
}

nvxcu_keypoint_tracker::ObjectTrackerPyrLKKeyHistory::ObjectTrackerPyrLKKeyHistory(const Parameters& params):
    parameters_(params)
{
#ifndef _WIN32
    char const *perfLoggerResPath = getenv("PERF_LOGGER_RESULT_PATH");
    if (perfLoggerResPath)
    {
        perfLogger_ = std::make_shared<RuntimePerformanceLogger>();
        if (!perfLogger_->init(perfLoggerResPath))
        {
            perfLogger_.reset();
        }
    }
#endif

    outlierRemoval_.reset(new nvxcu_keypoint_tracker::OutlierRemoval(parameters_.outlierRemovalParameters_));
    opticalFlowBuilder_.reset(new OpticalFlowBuilder(parameters_.pyrLKParameters_, parameters_.cornerDetectorParameters_));

    initialized_tracker_ = false;
    format_ = NVXCU_DF_IMAGE_RGB;
    width_ = 0;
    height_ = 0;

    maxObjectId_ = 0;

    initializeWeightsLookupTableForPoints();
}

nvxcu_keypoint_tracker::ObjectTrackerPyrLKKeyHistory::~ObjectTrackerPyrLKKeyHistory()
{

#ifndef _WIN32
    if(perfLogger_)
    {
        perfLogger_->log();
    }
#endif
    removeAllObjects();
    releaseInternalObjects();
}

nvxcu::ObjectTrackerWithFeaturesInfo::TrackedObject* nvxcu_keypoint_tracker::ObjectTrackerPyrLKKeyHistory::addObject(const nvxcu_rectangle_t& rect)
{
    if(rect.end_x < rect.start_x || rect.end_y < rect.start_y)
        return NULL;

    objects_.push_back(std::unique_ptr<TrackedObjectImpl>(new TrackedObjectImpl(maxObjectId_++, rect)));
    return objects_.back().get();
}

nvxcu_error_status_e nvxcu_keypoint_tracker::ObjectTrackerPyrLKKeyHistory::removeObject(ObjectTracker::TrackedObject* obj)
{
    for (std::list< std::unique_ptr<TrackedObjectImpl> >::iterator it = objects_.begin(); it != objects_.end(); ++it)
    {
        const std::unique_ptr<TrackedObjectImpl>& obj_u_p  = *it;
        if (obj_u_p.get() == obj)
        {
            objects_.erase(it);
            return NVXCU_SUCCESS;
        }
    }

    return NVXCU_FAILURE;
}

nvxcu_error_status_e nvxcu_keypoint_tracker::ObjectTrackerPyrLKKeyHistory::removeAllObjects()
{
    objects_.clear();
    return NVXCU_SUCCESS;
}

void nvxcu_keypoint_tracker::ObjectTrackerPyrLKKeyHistory::updateObject(TrackedObjectImpl& obj, const nvxcu_point2f_t& motion_center, const nvxcu_point2f_t& motion_scale)
{

    nvxcu_point2f_t res = motion_center;

    ASSERT( (motion_scale.x > 0) && (motion_scale.y > 0));
    // update object rectangle


    double scale_x = motion_scale.x;
    double scale_y = motion_scale.y;
    nvxcu_coordinates2d_t obj_size = calcObjectSize(obj);
    uint32_t  obj_width = obj_size.x;
    uint32_t  obj_height = obj_size.y;

    nvxcu_rectangle_t& obj_rect = obj.loc_;


    int32_t  newX = round(obj_rect.start_x + res.x - 0.5 * (scale_x - 1.0) * obj_width);
    int32_t  newY = round(obj_rect.start_y + res.y - 0.5 * (scale_y - 1.0) * obj_height);
    obj_width = round(scale_x * obj_width);
    obj_height = round(scale_y * obj_height);

    if (newX >= 0)
    {
        obj_rect.start_x = newX;
    }
    else
    {
        obj_rect.start_x = 0;
        obj_width = (uint32_t )(obj_width + newX);
    }

    if (newY >= 0)
    {
        obj_rect.start_y = newY;
    }
    else
    {
        obj_rect.start_y = 0;
        obj_height = (uint32_t )(obj_height + newY);
    }

    obj_rect.end_x = obj_rect.start_x + obj_width;
    obj_rect.end_y = obj_rect.start_y + obj_height;

}

nvxcu_coordinates2d_t nvxcu_keypoint_tracker::ObjectTrackerPyrLKKeyHistory::calcObjectSize(TrackedObjectImpl& obj)
{
    const nvxcu_rectangle_t& obj_rect = obj.loc_;

    uint32_t  obj_width = obj_rect.end_x - obj_rect.start_x;
    uint32_t  obj_height = obj_rect.end_y - obj_rect.start_y;

    const nvxcu_coordinates2d_t size = {obj_width, obj_height};

    return size;
}

nvxcu_error_status_e nvxcu_keypoint_tracker::ObjectTrackerPyrLKKeyHistory::process(const nvxcu_pitch_linear_image_t* frame)
{
#ifndef _WIN32
    if (perfLogger_)
    {
        perfLogger_->newFrame();
    }

#endif

    if (!initialized_tracker_)
    {
        NVXCU_SAFE_CALL(initTracker(frame));
        return NVXCU_SUCCESS;
    }

    nextHistoryNumberFrames_->setToZero();//TODO: move to a separate function

    nvxcu_df_image_e format = frame->base.format;
    uint32_t  width = frame->base.width;
    uint32_t  height = frame->base.height;

    if ( (format != format_)
         ||
         (width != width_)
         ||
         (height != height_) )
    {
        std::cout << "nvxcu_keypoint_tracker::ObjectTrackerPyrLKKeyHistory::process: Error: wrong format/width/height of the input frame" << std::endl;
        return NVXCU_FAILURE;
    }


    NVXCU_SAFE_CALL(opticalFlowBuilder_->buildPyramidAndSave(frame, objects_));


    // Process all objects
    size_t ind = 0;
    for (std::list< std::unique_ptr<TrackedObjectImpl> >::iterator it = objects_.begin(); it != objects_.end(); ++it, ++ind)
    {
        std::unique_ptr <TrackedObjectImpl>& obj_u_p = *it;
        TrackedObjectImpl& obj = *(obj_u_p.get());


        if (obj.status_ != TRACKED)
            continue;

        vector<nvxcu_keypointf_t> points0;
        vector<nvxcu_keypointf_t> points1;

        bool is_object_lost = false;
        NVXCU_SAFE_CALL(opticalFlowBuilder_->processObject(obj.loc_, points0, points1, is_object_lost));


        if(is_object_lost)
        {
            obj.status_ = LOST;
            continue;
        }

        nvxcu_point2f_t motion_center, motion_scale;
        bool is_ok = estimateMotionByPointsAndUpdateHistory(obj.loc_, points0, points1, motion_center, motion_scale, obj.features_info_);
        if(! is_ok)
        {
            obj.status_ = LOST;
            continue;
        }
        updateObject(obj, motion_center, motion_scale);
    }

    nextHistoryNumberFrames_.swap(historyNumberFrames_);

#ifndef _WIN32

    if (perfLogger_)
    {
       perfLogger_->addItem<RuntimePerformanceLogger::OBJECTNUM>(objects_.size());
    }
#endif

    return NVXCU_SUCCESS;
}

nvxcu_error_status_e nvxcu_keypoint_tracker::ObjectTrackerPyrLKKeyHistory::initTracker(const nvxcu_pitch_linear_image_t* first_frame)
{
    ASSERT(initialized_tracker_ == false);// this method should be called only once

    if( (first_frame->base.format != NVXCU_DF_IMAGE_RGB) &&
            (first_frame->base.format != NVXCU_DF_IMAGE_RGBX) &&
            (first_frame->base.format != NVXCU_DF_IMAGE_U8) )
    {
        std::cout << "ATTENTION: in the function '" << __FUNCTION__ << "' format has wrong value '" <<
                     first_frame->base.format << "'" << std::endl;
        return NVXCU_FAILURE;
    }

    // Create data objects
    releaseInternalObjects();

    format_ = first_frame->base.format;
    width_ = first_frame->base.width;
    height_ = first_frame->base.height;

    NVXCU_SAFE_CALL( createTrackerDataObjects() );

    // Process first frame
    NVXCU_SAFE_CALL( opticalFlowBuilder_->buildPyramidAndSave(first_frame, objects_) );

    initialized_tracker_ = true;

    return NVXCU_SUCCESS;
}

void nvxcu_keypoint_tracker::ObjectTrackerPyrLKKeyHistory::releaseInternalObjects()
{
    format_ = NVXCU_DF_IMAGE_RGB;
    width_ = 0;
    height_ = 0;

    historyNumberFrames_.reset();
    nextHistoryNumberFrames_.reset();
}

nvxcu_error_status_e nvxcu_keypoint_tracker::ObjectTrackerPyrLKKeyHistory::createTrackerDataObjects()
{
    ASSERT( (width_ > 0) && (height_ > 0));

    historyNumberFrames_.reset(new MatChar(width_, height_));
    nextHistoryNumberFrames_.reset(new MatChar(width_, height_));

    return NVXCU_SUCCESS;
}

void nvxcu_keypoint_tracker::ObjectTrackerPyrLKKeyHistory::getHistoryIndexForPoints(const std::vector<nvxcu_keypointf_t>& points0,
                                                                                 const std::vector<uint8_t>& status,
                                                                                 std::vector<uint8_t>& history_index) const
{
    ASSERT(historyNumberFrames_->width_ == width_);
    ASSERT(historyNumberFrames_->height_ == height_);

    int32_t  N = points0.size();

    rectanglef_t frame_rect(0, 0, width_, height_);

    history_index.resize(N);
    for(int32_t  n = 0; n < N; n++)
    {
        if (status[n] == 0)
        {
            history_index[n] = 0;
            continue;
        }
        const nvxcu_keypointf_t& pt = points0[n];

        if (!isPointInsideRect(pt,frame_rect))
        {
            history_index[n] = 0;
            continue;
        }
        history_index[n] = historyNumberFrames_->at(pt) + 1;
    }
}
void nvxcu_keypoint_tracker::ObjectTrackerPyrLKKeyHistory::updateHistoryIndexForPoints(const std::vector<nvxcu_keypointf_t>& points0,
                                                                                    const std::vector<nvxcu_keypointf_t>& points1,
                                                                                    const std::vector<uint8_t>& status,
                                                                                    int32_t  radius_to_search_keypoint)
{
    ASSERT(historyNumberFrames_->width_ == width_);
    ASSERT(historyNumberFrames_->height_ == height_);

    ASSERT(nextHistoryNumberFrames_->width_ == width_);
    ASSERT(nextHistoryNumberFrames_->height_ == height_);

    int32_t  N = points0.size();
    const int32_t & radius = radius_to_search_keypoint;

    rectanglef_t frame_rect(0, 0, width_, height_);

    for(int32_t  n = 0; n < N; n++)
    {
        if (status[n] == 0)
        {
            continue;
        }
        const nvxcu_keypointf_t& pt0 = points0[n];
        const nvxcu_keypointf_t& pt1 = points1[n];

        if (!isPointInsideRect(pt0, frame_rect))
        {
            continue;
        }
        if (!isPointInsideRect(pt1,frame_rect))
        {
            continue;
        }

        nvxcu_point2f_t pt1_int = create_point2f(roundf(pt1.x), roundf(pt1.y));


        nvxcu_point2f_t region_tl = create_point2f(pt1_int.x - radius, pt1_int.y - radius);
        nvxcu_point2f_t region_br = create_point2f(pt1_int.x + radius+1, pt1_int.y + radius+1);

        rectanglef_t region(region_tl, region_br);
        region = rectanglef_t::intersect(region, frame_rect);

        int32_t  cur_history_index = historyNumberFrames_->at(pt0);
        uint8_t next_history_index = (uint8_t)std::min( 255, cur_history_index + 1);

        //start filling
        for (int32_t  y = region.tl().y; y < region.br().y; y++)
        {
            for(int32_t  x = region.tl().x; x < region.br().x; x++)
            {
                nvxcu_keypointf_t pp = {(float)x, (float)y};
                (void)pp;//to avoid warnings if ASSERT-s are turned off
                ASSERT(isPointInsideRect(pp,frame_rect));

                uint8_t cur_val = nextHistoryNumberFrames_->at(y,x);
                nextHistoryNumberFrames_->at(y,x) = std::max(cur_val, next_history_index);
            }
        }
        //end filling
    }
}

void nvxcu_keypoint_tracker::ObjectTrackerPyrLKKeyHistory::initializeWeightsLookupTableForPoints()
{
    weightsLookupTable_.clear();
    weightsLookupTable_.push_back(0);//the 0-th index MUST be zero

    float coeff1 = parameters_.historyWeightingParameters_.coefficientOfWeightDecreasing_;
    float cur_coeff = 1;
    for(int32_t  n = 0; n < parameters_.historyWeightingParameters_.maxHistoryLength_; n++)
    {
        cur_coeff *= coeff1;
        float weight = 1 - cur_coeff;
        ASSERT( (0 <= weight) && (weight <= 1));
        weightsLookupTable_.push_back(weight);
    }
    weightsLookupTable_.push_back(1);//the last index MUST be 1
}

float nvxcu_keypoint_tracker::ObjectTrackerPyrLKKeyHistory::getWeightForHistoryIndex(uint8_t history_index) const
{
    ASSERT( !weightsLookupTable_.empty() );

    size_t num_weights = weightsLookupTable_.size();
    if (history_index >= num_weights)
    {
        return weightsLookupTable_.back();
    }
    return weightsLookupTable_[history_index];
}
void nvxcu_keypoint_tracker::ObjectTrackerPyrLKKeyHistory::getWeightsForPoints(const std::vector<nvxcu_keypointf_t>& points0,
                                                                            const std::vector<uint8_t>& status,
                                                                            std::vector<float>& weights) const
{
    ASSERT(points0.size() == status.size());

    vector<uint8_t> history_index;
    getHistoryIndexForPoints(points0, status, history_index);

    int32_t  N = points0.size();
    weights.resize(N);

    for(int32_t  n = 0; n < N; n++)
    {
        weights[n] = getWeightForHistoryIndex(history_index[n]);
    }
}

void nvxcu_keypoint_tracker::ScaleLogarithmicAverager::fillVectorOfWeightsWithIndexes(const std::vector<float>& weights, std::vector<WeightWithIndex>& res)
{
    res.resize(weights.size());
    for (size_t n = 0; n < weights.size(); n++)
    {
        res[n] = WeightWithIndex(weights[n], int32_t(n));
    }
}

bool nvxcu_keypoint_tracker::ScaleLogarithmicAverager::process(const std::vector<nvxcu_keypointf_t>& points0,
                                                         const std::vector<nvxcu_keypointf_t>& points1,
                                                         const std::vector<float>& weights,
                                                         nvxcu_point2f_t& motion_scale)
{
    ASSERT(points0.size() == points1.size());
    ASSERT(points0.size() == weights.size());

    if (points0.size() <= 0)
    {
        return false;
    }

    const int32_t  num_quantiles = parameters_.numQuantiles_;
    const int32_t  min_number_of_points = parameters_.minNumberOfPoints_;


    vector<WeightWithIndex> sorted_weights_with_indexes;
    fillVectorOfWeightsWithIndexes(weights, sorted_weights_with_indexes);

    std::sort(sorted_weights_with_indexes.begin(), sorted_weights_with_indexes.end(), compareWeightsWithIndexes);

    vector<int32_t > first_elements_of_quantiles;
    bool is_ok = getNQuantilesFromSortedWeightVector(sorted_weights_with_indexes, num_quantiles, min_number_of_points,
                                                     first_elements_of_quantiles);

    if (!is_ok)
    {
        return false;
    }


    int32_t  actual_num_quantiles = first_elements_of_quantiles.size();

    vector<nvxcu_point2f_t> scales_xy;
    for(int32_t  n = 0; n < actual_num_quantiles; n++)
    {
        nvxcu_point2f_t scale_xy = getMedianRatioForQuantile(points0, points1, sorted_weights_with_indexes, first_elements_of_quantiles[n]);
        scales_xy.push_back(scale_xy);
    }

    double scale_x = 0;
    double scale_y = 0;
    makeLogarithmicAveragingOfScales(scales_xy, sorted_weights_with_indexes, first_elements_of_quantiles, scale_x, scale_y);

    motion_scale.x = scale_x;
    motion_scale.y = scale_y;
    return true;
}

bool nvxcu_keypoint_tracker::ScaleLogarithmicAverager::getNQuantilesFromSortedWeightVector(const std::vector<WeightWithIndex>& sorted_weights_with_indexes,
                                                                                     const int32_t  desirable_num_quantiles,
                                                                                     const int32_t  minimal_number_of_points,
                                                                                     std::vector<int32_t >& first_elements_of_quantiles)
{
    ASSERT(desirable_num_quantiles > 0);
    ASSERT(minimal_number_of_points > 0);

    const int32_t  N = sorted_weights_with_indexes.size();
    ASSERT(N > 0);

    int32_t  first_index = 0;
    bool is_first_index_found = false;
    for(int32_t  i = 0; i < N; i++)
    {
        first_index = i;
        if (sorted_weights_with_indexes[i].first > 0)
        {
            is_first_index_found = true;
            break;
        }
    }

    if (! is_first_index_found)
    {
        return false;
    }

    int32_t  last_index = N - minimal_number_of_points;

    if (first_index > last_index)
    {
        return false;
    }

    int32_t  num_elements_to_use = last_index - first_index + 1;

    int32_t  num_quantiles = desirable_num_quantiles;

    float step = (float)num_elements_to_use / (float)num_quantiles;
    if (step < 1)
    {
        step = 1;
        num_quantiles = num_elements_to_use;
    }

    first_elements_of_quantiles.clear();
    for(int32_t  n = 0; n < num_quantiles; n++)
    {
        int32_t  index = first_index + (int32_t )floorf(n * step);
        ASSERT(index <= last_index);
        float cur_weight = sorted_weights_with_indexes[index].first;
        while ( (index > 0) && (sorted_weights_with_indexes[index-1].first == cur_weight) )
        {
            index--;
        }
        if ( (!first_elements_of_quantiles.empty())
             &&
             (first_elements_of_quantiles.back() == index) )
        {
            continue;
        }
        first_elements_of_quantiles.push_back(index);
    }
    return true;
}

rectanglef_t nvxcu_keypoint_tracker::ScaleLogarithmicAverager::getBoundingBoxForPointsWithWeightNotLessThanThreshold(const std::vector<nvxcu_point2f_t>& points,
                                                                                                               const std::vector<WeightWithIndex>& sorted_weights_with_indexes,
                                                                                                               int32_t  first_element,
                                                                                                               int32_t  min_number_of_points)
{
    int32_t  N = sorted_weights_with_indexes.size();
    ASSERT(points.size() == (size_t)N);

    nvxcu_point2f_t tl = {FLT_MAX, FLT_MAX};
    nvxcu_point2f_t br = {-FLT_MAX, -FLT_MAX};

    int32_t  number_of_points = 0;
    for(int32_t  n = first_element; n < N; n++)
    {
        int32_t  index = sorted_weights_with_indexes[n].second;

        const nvxcu_point2f_t& pt = points[index];
        tl.x = std::min(tl.x, pt.x);
        tl.y = std::min(tl.y, pt.y);
        br.x = std::max(br.x, pt.x);
        br.y = std::max(br.y, pt.y);

        number_of_points++;
    }
    (void)min_number_of_points;//to avoid warnings if ASSERT-s are turned off.
    ASSERT(number_of_points >= min_number_of_points);//data in arrays should be such that it is true
    return rectanglef_t(tl, br);
}

nvxcu_point2f_t nvxcu_keypoint_tracker::ScaleLogarithmicAverager::getMedianRatioForQuantile(const std::vector<nvxcu_keypointf_t>& points0,
                                                                              const std::vector<nvxcu_keypointf_t>& points1,
                                                                              const std::vector<WeightWithIndex>& sorted_weights_with_indexes,
                                                                              int32_t  first_element)
{
    int32_t  N = sorted_weights_with_indexes.size();
    ASSERT(points0.size() == (size_t)N);
    ASSERT(points1.size() == (size_t)N);
    ASSERT(first_element >= 0);
    ASSERT(first_element < N);

    int32_t  num_elements_to_work = N - first_element;
    int32_t  num_pairs = num_elements_to_work * (num_elements_to_work - 1) / 2;

    std::vector<float> scales_x;
    std::vector<float> scales_y;
    scales_x.reserve(num_pairs);
    scales_y.reserve(num_pairs);

    static const float MIN_DIFF = 1e-3;
    //begin filling scales arrays
    for (int32_t  i = 0; i < num_elements_to_work-1; i++)
    {
        int32_t  index_i = first_element + i;
        const nvxcu_keypointf_t& pt0_i = points0[index_i];
        const nvxcu_keypointf_t& pt1_i = points1[index_i];
        for (int32_t  j = i+1; j < num_elements_to_work; j++)
        {
            int32_t  index_j = first_element + j;
            const nvxcu_keypointf_t& pt0_j = points0[index_j];
            const nvxcu_keypointf_t& pt1_j = points1[index_j];

            float diff_0 =  sqrt((pt0_i.x - pt0_j.x)*(pt0_i.x - pt0_j.x) + (pt0_i.y - pt0_j.y)*(pt0_i.y - pt0_j.y));

            if (diff_0 > MIN_DIFF)
            {
                float diff_1 = sqrt((pt1_i.x - pt1_j.x)*(pt1_i.x - pt1_j.x) + (pt1_i.y - pt1_j.y)*(pt1_i.y - pt1_j.y));
                float cur_scale = diff_1/diff_0;
                scales_x.push_back(cur_scale);
                scales_y.push_back(cur_scale);
            }
        }
    }
    //end filling scales arrays

    float res_scale_x = 1;
    if (!scales_x.empty())
        res_scale_x = getMedian(scales_x);

    float res_scale_y = 1;
    if (!scales_y.empty())
        res_scale_y = getMedian(scales_y);

    nvxcu_point2f_t res_scale_xy = {res_scale_x, res_scale_y};
    return res_scale_xy;
}

void nvxcu_keypoint_tracker::ScaleLogarithmicAverager::makeLogarithmicAveragingOfScales(const std::vector<nvxcu_point2f_t >& scales_xy,
                                                                                  std::vector<WeightWithIndex>& sorted_weights_with_indexes,
                                                                                  const std::vector<int32_t >& first_elements_of_quantiles,
                                                                                  double& scale_x,
                                                                                  double& scale_y)
{
    int32_t  N = first_elements_of_quantiles.size();
    ASSERT(scales_xy.size() == (size_t)N);
    ASSERT(N > 0);

    double sum_log_w = 0;
    double sum_log_h = 0;
    double sum_weights = 0;
    for(int32_t  n = 0; n < N; n++)
    {
        int32_t  cur_index = first_elements_of_quantiles[n];
        double cur_weight = sorted_weights_with_indexes[cur_index].first;
        double log_w_diff = log(scales_xy[n].x);
        double log_h_diff = log(scales_xy[n].y);

        sum_log_w += log_w_diff * cur_weight;
        sum_log_h += log_h_diff * cur_weight;

        sum_weights += cur_weight;
    }
    double avg_log_w = sum_log_w / sum_weights;
    double avg_log_h = sum_log_h / sum_weights;

    scale_x = exp(avg_log_w);
    scale_y = exp(avg_log_h);

    if (!std::isfinite(scale_x))
    {
        scale_x = 1;
    }
    if (!std::isfinite(scale_y))
    {
        scale_y = 1;
    }
}

bool nvxcu_keypoint_tracker::ObjectTrackerPyrLKKeyHistory::estimateScaleByPoints(const std::vector<nvxcu_keypointf_t>& points0,
                                                                              const std::vector<nvxcu_keypointf_t>& points1,
                                                                              const std::vector<float>& weights,
                                                                              nvxcu_point2f_t& motion_scale)
{

    ScaleLogarithmicAverager averager(parameters_.advancedScaleEstimatorParameters_);
    bool is_ok = averager.process(points0, points1, weights, motion_scale);
    if (!is_ok)
    {
        motion_scale.x = 1;
        motion_scale.y = 1;
    }

    return true;
}


uint32_t nvxcu_keypoint_tracker::ObjectTrackerPyrLKKeyHistory::getCellIndexForPoint(const nvxcu_rectangle_t& obj_rect, nvxcu_keypointf_t kp, size_t x_num_of_cells, size_t y_num_of_cells)
{
     float grid_x_step = (obj_rect.end_x - obj_rect.start_x) / (float)x_num_of_cells;
     uint32_t  cell_x = (uint32_t )((kp.x  - obj_rect.start_x) / grid_x_step);
     ASSERT(cell_x < x_num_of_cells);
     float grid_y_step = (obj_rect.end_y - obj_rect.start_y) / (float)y_num_of_cells;
     uint32_t  cell_y = (uint32_t )((kp.y  - obj_rect.start_y) / grid_y_step);
     ASSERT(cell_y < y_num_of_cells);
     uint32_t  cell_ind = cell_x + cell_y * x_num_of_cells;
     return cell_ind;
}

void nvxcu_keypoint_tracker::ObjectTrackerPyrLKKeyHistory::filterKeypoints(const nvxcu_rectangle_t& obj_rect,
                                                                        size_t x_num_of_cells,
                                                                        size_t y_num_of_cells,
                                                                        size_t max_corners_in_cell,
                                                                        const std::vector<nvxcu_keypointf_t>& points0,
                                                                        const std::vector<nvxcu_keypointf_t>& points1,
                                                                        std::vector<nvxcu_keypointf_t>& filterered_points0,
                                                                        std::vector<nvxcu_keypointf_t>& filterered_points1)
{
    vector<vector<nvxcu_keypointf_t>> kps0_cells(x_num_of_cells * y_num_of_cells);
    vector<vector<nvxcu_keypointf_t>> kps1_cells(x_num_of_cells * y_num_of_cells);

    for (size_t n = 0; n < points0.size(); n++)
    {
        uint32_t ind = getCellIndexForPoint(obj_rect, points0[n], x_num_of_cells, y_num_of_cells);
        ASSERT(ind < kps0_cells.size());

        if(  kps0_cells[ind].size() < max_corners_in_cell)
        {
            kps0_cells[ind].push_back(points0[n]);
            kps1_cells[ind].push_back(points1[n]);
        }
    }

    for (size_t i = 0; i < kps0_cells.size(); i++)
    {
        filterered_points0.insert(filterered_points0.end(), kps0_cells[i].begin(), kps0_cells[i].end());
        filterered_points1.insert(filterered_points1.end(), kps1_cells[i].begin(), kps1_cells[i].end());
    }

}

static void convert_vectors_to_feature_points_info(const std::vector<nvxcu_keypointf_t>& points0,
                                                   const std::vector<nvxcu_keypointf_t>& points1,
                                                   const std::vector<float>& weights,
                                                   nvxcu_keypoint_tracker::ObjectTrackerPyrLKKeyHistory::FeaturePointsVector& features_info)
{
    ASSERT(points0.size() == points1.size());
    ASSERT(points0.size() == weights.size());

    size_t N = points0.size();
    features_info.data.resize(N);
    for(size_t n = 0; n < N; n++)
    {
        features_info.data[n] = {points0[n], points1[n], weights[n]};
    }
}
bool nvxcu_keypoint_tracker::ObjectTrackerPyrLKKeyHistory::estimateMotionByPointsAndUpdateHistory(const nvxcu_rectangle_t& obj_rect,
                                                                                               const std::vector<nvxcu_keypointf_t>& points0,
                                                                                               const std::vector<nvxcu_keypointf_t>& points1,
                                                                                               nvxcu_point2f_t& motion_center,
                                                                                               nvxcu_point2f_t& motion_scale,
                                                                                               FeaturePointsVector& features_info)
{
    std::vector<nvxcu_keypointf_t> new_points0;
    std::vector<nvxcu_keypointf_t> new_points1;
    std::vector<uint8_t> new_status;

    nvxcu_rectangle_t obj_rect_decreased = decreaseRectangleByCoeff(obj_rect, parameters_.cornerDetectorParameters_.bbDecreasingRatio_, width_, height_);
    filterKeypoints(obj_rect_decreased, parameters_.cornersFilteringParameters_.xNumOfCells_, parameters_.cornersFilteringParameters_.yNumOfCells_,
                    parameters_.cornersFilteringParameters_.maxCornersInCell_, points0, points1, new_points0, new_points1);

    new_status.resize(new_points0.size(), 255);

    ASSERT(new_points0.size() == new_points1.size());
    ASSERT(new_points0.size() == new_status.size());
    int32_t  N = new_points0.size();

    vector<float> weights(N, 0);
    vector<nvxcu_point2f_t> motion(N);
    for(int32_t  n = 0; n < N; n++)
    {
        motion[n].x = new_points1[n].x - new_points0[n].x;
        motion[n].y = new_points1[n].y - new_points0[n].y;
    }
    getWeightsForPoints(new_points0, new_status, weights);


    if (parameters_.algorithmicParameters_.shouldUseOutlierRemoval_)
    {
        weights = outlierRemoval_->process(motion, weights);
        //TODO: insert threshold: number of points
    }

    convert_vectors_to_feature_points_info(new_points0, new_points1, weights, features_info);

    ASSERT(weights.size() == (size_t)N);
    vector<uint8_t> updated_status = new_status;

    if (parameters_.algorithmicParameters_.shouldRemoveOutliersAtAll_)
    {
        for (int32_t  n = 0; n < N; n++)
        {
            if (weights[n] == 0)
            {
                updated_status[n] = 0;
            }
        }
    }
    updateHistoryIndexForPoints(new_points0, new_points1, updated_status, parameters_.historyWeightingParameters_.radiusToSearchKeypoint_);


    float sum_weights = 0;
    for (size_t i = 0; i < weights.size(); i++)
    {
        sum_weights += weights[i];
    }
    if (!sum_weights)
    {
        return false;
    }

    nvxcu_point2f_t res_motion = getCenterPoint(motion, weights);
    motion_center.x = res_motion.x;
    motion_center.y = res_motion.y;

    estimateScaleByPoints(new_points0, new_points1, weights, motion_scale);

    return true;
}



nvxcu::ObjectTracker* nvxcu::createPyrLKKeyHistory(const nvxcu_keypoint_tracker::ObjectTrackerPyrLKKeyHistory::Parameters& params)
{
    return new nvxcu_keypoint_tracker::ObjectTrackerPyrLKKeyHistory(params);
}



nvxcu_keypoint_tracker::OpticalFlowBuilder::OpticalFlowBuilder(const PyrLKParameters& pyr_lk_params,
                                                               const CornerDetectorParameters& corner_detector_params)
{
    pyrLKParameters_ = pyr_lk_params;
    cornerDetectorParameters_ = corner_detector_params;
    initialized_builder_ = false;

    width_ = 0;
    height_ = 0;

    mask_ = { };

    prev_pyr_ = { };
    curr_pyr_ = { };

    points0_ = { };
    points1_ = { };
    pointsFb_ = { };
    num_corners_dev_ptr_ = nullptr;

    fast_harris_track_buf_size_ = { };
    gauss_pyr_buf_size_ = { };
    opt_flow_pyr_lk_buf_size_ = { };
    final_buf_size_ = { };
    tmp_buf_ = { };

    exec_target_ = { };
    border_ = { };

    points_dev_ptr_ = NULL;
    points_cpu_ptr_ = NULL;
    num_of_points_buffers_ = 3;
}

nvxcu_keypoint_tracker::OpticalFlowBuilder::~OpticalFlowBuilder()
{
    // clean up
    CUDA_ERROR_PRINT(cudaFree(points_dev_ptr_) );
    CUDA_ERROR_PRINT(cudaFree(num_corners_dev_ptr_) );

    releasePyramid(&prev_pyr_);
    releasePyramid(&curr_pyr_);

    releaseImageU8(&mask_);

    if (exec_target_.stream)
        CUDA_ERROR_PRINT( cudaStreamDestroy(exec_target_.stream) );

    CUDA_ERROR_PRINT( cudaFree(tmp_buf_.dev_ptr) );
    CUDA_ERROR_PRINT( cudaFreeHost(tmp_buf_.host_ptr) );

    delete [] points_cpu_ptr_;
}

nvxcu_error_status_e nvxcu_keypoint_tracker::OpticalFlowBuilder::processFirstFrame(const nvxcu_pitch_linear_image_t* firstFrame)
{

    if(firstFrame->base.format!= NVXCU_DF_IMAGE_U8)
    {
        NVXCU_SAFE_CALL( nvxcuColorConvert(&firstFrame->base, &prev_pyr_.levels[0].base, NVXCU_COLOR_SPACE_DEFAULT, NVXCU_CHANNEL_RANGE_FULL, &exec_target_.base) );
    }
    else
    {
        CUDA_SAFE_CALL( cudaMemcpy2DAsync(prev_pyr_.levels[0].planes[0].dev_ptr, prev_pyr_.levels[0].planes[0].pitch_in_bytes,
                firstFrame->planes[0].dev_ptr, firstFrame->planes[0].pitch_in_bytes, firstFrame->base.width * sizeof(uint8_t), firstFrame->base.height,
                cudaMemcpyDeviceToDevice,
                exec_target_.stream) );
    }



    NVXCU_SAFE_CALL( nvxcuGaussianPyramid(&prev_pyr_.base,
                         &tmp_buf_,
                         &border_,
                         &exec_target_.base) );



    return NVXCU_SUCCESS;
}

nvxcu_error_status_e nvxcu_keypoint_tracker::OpticalFlowBuilder::initBuilder(const nvxcu_pitch_linear_image_t* firstFrame)
{
    ASSERT(initialized_builder_ == false);//since this method should be called only once for initialization

    if ( (firstFrame->base.format != NVXCU_DF_IMAGE_U8) && (firstFrame->base.format != NVXCU_DF_IMAGE_RGB) && (firstFrame->base.format != NVXCU_DF_IMAGE_RGBX) )
    {
        std::cout << "nvxcu_keypoint_tracker::OpticalFlowBuilder::initBuilder: ERROR: the class OpticalFlowBuilder can work only with input frames "
                 "which have format NVXCU_DF_IMAGE_U8 or NVXCU_DF_IMAGE_RGB or NVXCU_DF_IMAGE_RGBX, "
                 "the received format is " << firstFrame->base.format << std::endl;
        return NVXCU_FAILURE;
    }

    // Create data objects, allocate buffers
    NVXCU_SAFE_CALL(createDataObjects(firstFrame->base.width, firstFrame->base.height));
    NVXCU_SAFE_CALL(allocateBuffers());
    NVXCU_SAFE_CALL(processFirstFrame(firstFrame));


    initialized_builder_ = true;
    return NVXCU_SUCCESS;
}

nvxcu_error_status_e nvxcu_keypoint_tracker::OpticalFlowBuilder::createDataObjects(uint32_t width, uint32_t height)
{

    // create data objects
    width_ = width;
    height_ = height;

    prev_pyr_ = createPyramidHalfScale(width_, height_, pyrLKParameters_.numPyramidLevels_);
    curr_pyr_ = createPyramidHalfScale(width_, height_, pyrLKParameters_.numPyramidLevels_);

    mask_ = createImageU8(width_, height_);

    size_t points_dev_ptr_size = allocateBufferForArraysKeypointF(num_of_points_buffers_, cornerDetectorParameters_.maxCorners_, &points_dev_ptr_);
    points_cpu_ptr_ = new uint8_t[points_dev_ptr_size];

    std::vector<nvxcu_plain_array_t> points_v = createArraysKeyPointFFromBuffer(num_of_points_buffers_, cornerDetectorParameters_.maxCorners_, points_dev_ptr_);

    points0_ = points_v[0];
    points1_ = points_v[1];
    pointsFb_ = points_v[2];

    num_corners_dev_ptr_ = NULL;
    CUDA_SAFE_CALL( cudaMalloc((void **)&num_corners_dev_ptr_, sizeof(size_t)) );

    exec_target_.base.exec_target_type = NVXCU_STREAM_EXEC_TARGET;
    cudaStreamCreate(&exec_target_.stream);
    cudaGetDeviceProperties(&exec_target_.dev_prop, 0);
    border_.mode = NVXCU_BORDER_MODE_REPLICATE;

    return NVXCU_SUCCESS;
}

nvxcu_error_status_e nvxcu_keypoint_tracker::OpticalFlowBuilder::allocateBuffers()
{
    fast_harris_track_buf_size_ = cornerDetectorParameters_.useFastDetector_ ?
                nvxcuFastTrack_GetBufSize(width_, height_, NVXCU_DF_IMAGE_U8,
                                          NVXCU_TYPE_KEYPOINTF, cornerDetectorParameters_.maxCorners_,
                                          cornerDetectorParameters_.detectorCellSize_,
                                          &exec_target_.dev_prop) :
                nvxcuHarrisTrack_GetBufSize(width_, height_, NVXCU_DF_IMAGE_U8,
                                            NVXCU_TYPE_KEYPOINTF, cornerDetectorParameters_.maxCorners_,
                                            cornerDetectorParameters_.detectorCellSize_,
                                            &exec_target_.dev_prop);

    gauss_pyr_buf_size_ = nvxcuGaussianPyramid_GetBufSize(width_, height_,
                                                          pyrLKParameters_.numPyramidLevels_, NVXCU_SCALE_PYRAMID_HALF,
                                                          &border_, &exec_target_.dev_prop);


    opt_flow_pyr_lk_buf_size_ = nvxcuOpticalFlowPyrLK_GetBufSize(width_, height_,
                                                                 pyrLKParameters_.numPyramidLevels_, NVXCU_SCALE_PYRAMID_HALF,
                                                                 NVXCU_TYPE_KEYPOINTF, cornerDetectorParameters_.maxCorners_,
                                                                 pyrLKParameters_.windowSize_ /*window_dimension*/,
                                                                 0 /*calcError*/,
                                                                 &border_,
                                                                 &exec_target_.dev_prop);

    final_buf_size_.dev_buf_size = fast_harris_track_buf_size_.dev_buf_size;
    final_buf_size_.dev_buf_size =
            final_buf_size_.dev_buf_size > gauss_pyr_buf_size_.dev_buf_size ?
                final_buf_size_.dev_buf_size :
                gauss_pyr_buf_size_.dev_buf_size;
    final_buf_size_.dev_buf_size =
            final_buf_size_.dev_buf_size > opt_flow_pyr_lk_buf_size_.dev_buf_size ?
                final_buf_size_.dev_buf_size :
                opt_flow_pyr_lk_buf_size_.dev_buf_size;
    final_buf_size_.host_buf_size = fast_harris_track_buf_size_.host_buf_size;
    final_buf_size_.host_buf_size =
            final_buf_size_.host_buf_size > gauss_pyr_buf_size_.host_buf_size ?
                final_buf_size_.host_buf_size :
                gauss_pyr_buf_size_.host_buf_size;
    final_buf_size_.host_buf_size =
            final_buf_size_.host_buf_size > opt_flow_pyr_lk_buf_size_.host_buf_size ?
                final_buf_size_.host_buf_size :
                opt_flow_pyr_lk_buf_size_.host_buf_size;

    tmp_buf_ = {NULL, NULL};
    if (final_buf_size_.dev_buf_size > 0)
    {
        CUDA_SAFE_CALL( cudaMalloc(&tmp_buf_.dev_ptr, final_buf_size_.dev_buf_size) );
    }
    if (final_buf_size_.host_buf_size > 0)
    {
         CUDA_SAFE_CALL( cudaMallocHost(&tmp_buf_.host_ptr, final_buf_size_.host_buf_size) );
    }

    return NVXCU_SUCCESS;
}

nvxcu_error_status_e nvxcu_keypoint_tracker::OpticalFlowBuilder::setObjectsMask(const std::list<std::unique_ptr<nvxcu_keypoint_tracker::ObjectTrackerPyrLKKeyHistory::TrackedObjectImpl>>& objects)
{

    const int border_for_subimage = 16;

    cudaMemset2DAsync(mask_.planes[0].dev_ptr, mask_.planes[0].pitch_in_bytes, 0, mask_.base.width * sizeof(uint8_t), mask_.base.height, exec_target_.stream);

    for (std::list<std::unique_ptr<nvxcu_keypoint_tracker::ObjectTrackerPyrLKKeyHistory::TrackedObjectImpl>>::const_iterator it = objects.begin(); it != objects.end(); ++it)
    {
        const std::unique_ptr<nvxcu_keypoint_tracker::ObjectTrackerPyrLKKeyHistory::TrackedObjectImpl>& obj_u_p = *it;
        const nvxcu_keypoint_tracker::ObjectTrackerPyrLKKeyHistory::TrackedObjectImpl& obj = *(obj_u_p.get());
        if (obj.status_ != nvxcu_keypoint_tracker::ObjectTrackerPyrLKKeyHistory::TRACKED)
            continue;

        nvxcu_rectangle_t obj_rect = increaseRectangleByBorder(obj.loc_, border_for_subimage, width_, height_);

        CUDA_SAFE_CALL( cudaMemset2DAsync((uint8_t*)(mask_.planes[0].dev_ptr) + obj_rect.start_y * mask_.planes[0].pitch_in_bytes + obj_rect.start_x * sizeof(uint8_t),
                mask_.planes[0].pitch_in_bytes, 255,
                     (obj_rect.end_x - obj_rect.start_x) * sizeof(uint8_t), obj_rect.end_y - obj_rect.start_y, exec_target_.stream) );
    }

    return NVXCU_SUCCESS;
}

nvxcu_error_status_e nvxcu_keypoint_tracker::OpticalFlowBuilder::buildPyramidAndSave(const nvxcu_pitch_linear_image_t* frame,
                                                                                     const std::list<std::unique_ptr<nvxcu_keypoint_tracker::ObjectTrackerPyrLKKeyHistory::TrackedObjectImpl>>& objects)
{
    if (initialized_builder_ == false)
    {
        NVXCU_SAFE_CALL(initBuilder(frame));
        return NVXCU_SUCCESS;
    }


    if ( (frame->base.width != width_)
         ||
         (frame->base.height != height_) )
    {
        std::cout << "nvxcu_keypoint_tracker::OpticalFlowBuilder::buildPyramidAndSave: Error: wrong width/height of the input frame" << std::endl;
        return NVXCU_FAILURE;
    }

    NVXCU_SAFE_CALL( setObjectsMask(objects) );


    if(frame->base.format!= NVXCU_DF_IMAGE_U8)
    {
        NVXCU_SAFE_CALL( nvxcuColorConvert(&frame->base, &curr_pyr_.levels[0].base, NVXCU_COLOR_SPACE_DEFAULT, NVXCU_CHANNEL_RANGE_FULL, &exec_target_.base) );
    }
    else
    {
        CUDA_SAFE_CALL( cudaMemcpy2DAsync( curr_pyr_.levels[0].planes[0].dev_ptr, curr_pyr_.levels[0].planes[0].pitch_in_bytes,
                frame->planes[0].dev_ptr, frame->planes[0].pitch_in_bytes, frame->base.width * sizeof(uint8_t), frame->base.height,
                cudaMemcpyDeviceToDevice,
                exec_target_.stream) );
    }

    NVXCU_SAFE_CALL( nvxcuGaussianPyramid(&curr_pyr_.base,
                         &tmp_buf_,
                         &border_,
                         &exec_target_.base) );



    if (cornerDetectorParameters_.useFastDetector_)
    {
        NVXCU_SAFE_CALL( nvxcuFastTrack(&curr_pyr_.levels[0].base, &points0_.base, &mask_.base, NULL,
                       cornerDetectorParameters_.fastType_, cornerDetectorParameters_.fastThreshold_, cornerDetectorParameters_.detectorCellSize_,
                       num_corners_dev_ptr_, &tmp_buf_, &exec_target_.base) );


    }
    else
    {
        NVXCU_SAFE_CALL( nvxcuHarrisTrack(&curr_pyr_.levels[0].base, &points0_.base, &mask_.base, NULL, cornerDetectorParameters_.harrisK_,
                         cornerDetectorParameters_.harrisThreshold_, cornerDetectorParameters_.detectorCellSize_,
                         num_corners_dev_ptr_, &tmp_buf_, &exec_target_.base) );
    }

    CUDA_SAFE_CALL( cudaMemcpyAsync(points1_.dev_ptr, points0_.dev_ptr,
                    points0_.base.capacity * sizeof(nvxcu_keypointf_t),
                    cudaMemcpyDeviceToDevice,
                    exec_target_.stream) );

    CUDA_SAFE_CALL( cudaMemcpyAsync(points1_.num_items_dev_ptr, points0_.num_items_dev_ptr,
                    sizeof(uint32_t),
                    cudaMemcpyDeviceToDevice,
                    exec_target_.stream) );


    NVXCU_SAFE_CALL( nvxcuOpticalFlowPyrLK(&prev_pyr_.base,
                          &curr_pyr_.base,
                          &points0_.base,
                          &points1_.base,
                          NVXCU_TERM_CRITERIA_BOTH,
                          0.01f /*epsilon*/,
                          pyrLKParameters_.maxIterationsNumber_ /*num_iterations*/,
                          pyrLKParameters_.windowSize_ /*window_dimension*/,
                          0 /*calcError*/,
                          &tmp_buf_,
                          &border_,
                          &exec_target_.base) );



    CUDA_SAFE_CALL (cudaMemcpyAsync(pointsFb_.dev_ptr, points1_.dev_ptr,
                    points1_.base.capacity * sizeof(nvxcu_keypointf_t),
                    cudaMemcpyDeviceToDevice,
                    exec_target_.stream) );

    CUDA_SAFE_CALL( cudaMemcpyAsync(pointsFb_.num_items_dev_ptr, points1_.num_items_dev_ptr,
                    sizeof(uint32_t),
                    cudaMemcpyDeviceToDevice,
                    exec_target_.stream) );


    NVXCU_SAFE_CALL( nvxcuOpticalFlowPyrLK(&curr_pyr_.base,
                          &prev_pyr_.base,
                          &points1_.base,
                          &pointsFb_.base,
                          NVXCU_TERM_CRITERIA_BOTH,
                          0.01f /*epsilon*/,
                          pyrLKParameters_.maxIterationsNumber_ /*num_iterations*/,
                          pyrLKParameters_.windowSize_ /*window_dimension*/,
                          0 /*calcError*/,
                          &tmp_buf_,
                          &border_,
                          &exec_target_.base) );



    std::swap(curr_pyr_, prev_pyr_);


    CUDA_SAFE_CALL( cudaMemcpyAsync(points_cpu_ptr_, points_dev_ptr_,
                                    num_of_points_buffers_ * sizeof(uint32_t) +
                                    num_of_points_buffers_ * cornerDetectorParameters_.maxCorners_ * sizeof(nvxcu_keypointf_t),
                                    cudaMemcpyDeviceToHost, exec_target_.stream) );

    CUDA_SAFE_CALL( cudaStreamSynchronize(exec_target_.stream) );


    return NVXCU_SUCCESS;
}

bool nvxcu_keypointf_t_strength_cmp(const nvxcu_keypointf_t& left, const nvxcu_keypointf_t& right)
{
    return left.strength > right.strength;
}

nvxcu_error_status_e nvxcu_keypoint_tracker::OpticalFlowBuilder::processObject(const nvxcu_rectangle_t& obj_rect,
                                                              vector<nvxcu_keypointf_t>& res_points0,
                                                              vector<nvxcu_keypointf_t>& res_points1,
                                                              bool& is_object_lost)
{
    is_object_lost = true;

    uint32_t points0_num_items = ((uint32_t*)points_cpu_ptr_)[0];
    uint32_t points1_num_items = ((uint32_t*)points_cpu_ptr_)[1];
    uint32_t pointsFb_num_items = ((uint32_t*)points_cpu_ptr_)[2];

    ASSERT(points0_num_items == points1_num_items);
    ASSERT(points0_num_items == pointsFb_num_items);

    nvxcu_keypointf_t* points_cpu = (nvxcu_keypointf_t*)((uint32_t*)points_cpu_ptr_ + num_of_points_buffers_);

    nvxcu_rectangle_t obj_rect_decreased = decreaseRectangleByCoeff(obj_rect, cornerDetectorParameters_.bbDecreasingRatio_, width_, height_);

    if (points0_num_items > 0)
    {
        res_points0.clear();
        res_points1.clear();

        size_t pts_inside_rect = 0;
        for(size_t n = 0; n < points0_num_items; n++)
        {

            nvxcu_keypointf_t kkp0 = points_cpu[n];
            nvxcu_keypointf_t kkp1 = points_cpu[cornerDetectorParameters_.maxCorners_ + n];
            nvxcu_keypointf_t kkp_fb = points_cpu[2 * cornerDetectorParameters_.maxCorners_ + n];;



            if (kkp0.x >= obj_rect_decreased.start_x && kkp0.x < obj_rect_decreased.end_x &&
                    kkp0.y >= obj_rect_decreased.start_y && kkp0.y < obj_rect_decreased.end_y)
            {
                if ((kkp1.tracking_status != 255) || (kkp_fb.tracking_status != 255))
                {
                    continue;
                }

                double cur_motion = getDistance(kkp0, kkp1);
                if (cur_motion > pyrLKParameters_.maxPossibleKeypointMotion_)
                {
                    continue;
                }

                double fb_dist = getDistance(kkp0, kkp_fb);
                if (fb_dist > pyrLKParameters_.maxBackwardCheckingError_)
                {
                    continue;
                }
                res_points0.push_back(kkp0);
                res_points1.push_back(kkp1);
                pts_inside_rect++;
            }
        }


        for (size_t i = 0; i < res_points0.size(); i++)
        {
            ASSERT(res_points0[i].strength == res_points1[i].strength);
        }

        std::stable_sort(res_points0.begin(), res_points0.end(), nvxcu_keypointf_t_strength_cmp);
        std::stable_sort(res_points1.begin(), res_points1.end(), nvxcu_keypointf_t_strength_cmp);

        for (size_t i = 0; i < res_points0.size(); i++)
        {
            ASSERT(res_points0[i].strength == res_points1[i].strength);
        }


        ASSERT(res_points0.size() == res_points1.size());
        int32_t  res_count = res_points0.size();

        int32_t  min_num_of_points_with_found_motion = (-1) + floor(pts_inside_rect * pyrLKParameters_.minRatioOfPointsWithFoundMotion_);
        min_num_of_points_with_found_motion = std::max(min_num_of_points_with_found_motion, pyrLKParameters_.minNumberOfPointsWithFoundMotion_);
        if (res_count < min_num_of_points_with_found_motion)
        {
            return NVXCU_SUCCESS;
        }

        is_object_lost = false;
    }

    return NVXCU_SUCCESS;
}



std::string nvxcu_keypoint_tracker::ObjectTrackerPyrLKKeyHistory::CornerDetectorParameters::toString() const
{
    std::stringstream strstr;
    strstr << " CornerDetectorParameters:" << std::endl
        << "    maxCorners_ = " << maxCorners_ << std::endl
        << "    detectorCellSize_ = " << detectorCellSize_ << std::endl
        << "    useFastDetector_ = " << std::boolalpha << useFastDetector_ << std::noboolalpha << std::endl
        << "    fastThreshold_ = " << fastThreshold_ << std::endl
        << "    fastType_ = " << fastType_ << std::endl
        << "    harrisK_ = " << harrisK_ << std::endl
        << "    harrisThreshold_ = " << harrisThreshold_ << std::endl
        << "    bbDecreasingRatio_ = " << bbDecreasingRatio_ << std::endl;

    return strstr.str();
}

std::string nvxcu_keypoint_tracker::ObjectTrackerPyrLKKeyHistory::CornersFilteringParameters::toString() const
{
    std::stringstream strstr;

    strstr << "  CornersFilteringParameters:" << std::endl
        << "    maxCornersInCell_ = " << maxCornersInCell_ << std::endl
        << "    xNumOfCells_ = " << xNumOfCells_ << std::endl
        << "    yNumOfCells_ = " << yNumOfCells_ << std::endl;
    return strstr.str();
}

std::string nvxcu_keypoint_tracker::ObjectTrackerPyrLKKeyHistory::PyramidalLucasKanadeParameters::toString() const
{
    std::stringstream strstr;
    strstr << "  PyramidlLucasKanadeParameters:" << std::endl
        << "    windowSize_ = " << windowSize_ << std::endl
        << "    numPyramidLevels_ = " << numPyramidLevels_ << std::endl
        << "    maxIterationsNumber_ = " << maxIterationsNumber_ << std::endl
        << "    maxPossibleKeypointMotion_ = " << maxPossibleKeypointMotion_ << std::endl
        << "    minRatioOfPointsWithFoundMotion_ = " << minRatioOfPointsWithFoundMotion_ << std::endl
        << "    minNumberOfPointsWithFoundMotion_ = " << minNumberOfPointsWithFoundMotion_ << std::endl
        << "    shouldUseBackwardChecking_ = " << shouldUseBackwardChecking_ << std::endl
        << "    maxBackwardCheckingError_ = " << maxBackwardCheckingError_ << std::endl;
    return strstr.str();
}
std::string nvxcu_keypoint_tracker::ObjectTrackerPyrLKKeyHistory::HistoryWeightingParameters::toString() const
{
    std::stringstream strstr;
    strstr << "  HistoryWeightingParameters:" << std::endl
        << "    coefficientOfWeightDecreasing_ = " << coefficientOfWeightDecreasing_ << std::endl
        << "    maxHistoryLength_ = " << maxHistoryLength_ << std::endl
        << "    radiusToSearchKeypoint_ = " << radiusToSearchKeypoint_ << std::endl;
    return strstr.str();
}
std::string nvxcu_keypoint_tracker::ObjectTrackerPyrLKKeyHistory::OutlierRemovalParameters::toString() const
{
    std::stringstream strstr;
    strstr << "  OutlierRemovalParameters:" << std::endl
        << "    relativeDistanceToRemoveOutlier_ = " << relativeDistanceToRemoveOutlier_ << std::endl
        << "    maxIterationsNumber_ = " << maxIterationsNumber_ << std::endl
        << "    maxMeanDistanceToStop_ = " << maxMeanDistanceToStop_ << std::endl;
    return strstr.str();
}
std::string nvxcu_keypoint_tracker::ObjectTrackerPyrLKKeyHistory::AdvancedScaleEstimatorParameters::toString() const
{
    std::stringstream strstr;
    strstr << "  AdvancedScaleEstimatorParameters:" << std::endl
        << "    numQuantiles_ = " << numQuantiles_ << std::endl
        << "    minNumberOfPoints_ = " << minNumberOfPoints_ << std::endl;
    return strstr.str();
}
std::string nvxcu_keypoint_tracker::ObjectTrackerPyrLKKeyHistory::AlgorithmicParameters::toString() const
{
    std::stringstream strstr;
    strstr << "  AlgorithmicParameters:" << std::endl
        << "    shouldUseOutlierRemoval_ = " << shouldUseOutlierRemoval_ << std::endl
        << "    shouldRemoveOutliersAtAll_ = " << shouldRemoveOutliersAtAll_ << std::endl;
    return strstr.str();
}
std::string nvxcu_keypoint_tracker::ObjectTrackerPyrLKKeyHistory::Parameters::toString() const
{
    std::stringstream strstr;
    strstr << "Parameters:" << std::endl;
    strstr << cornerDetectorParameters_.toString();
    strstr << pyrLKParameters_.toString();
    strstr << historyWeightingParameters_.toString();
    strstr << outlierRemovalParameters_.toString();
    strstr << advancedScaleEstimatorParameters_.toString();
    strstr << algorithmicParameters_.toString();
    return strstr.str();
}

size_t nvxcu_keypoint_tracker::ObjectTrackerPyrLKKeyHistory::FeaturePointsVector::getSize() const
{
    return data.size();
}

const nvxcu::ObjectTrackerWithFeaturesInfo::FeaturePoint&
nvxcu_keypoint_tracker::ObjectTrackerPyrLKKeyHistory::FeaturePointsVector::getFeaturePoint(size_t index) const
{
    if (index >= data.size())
    {
        THROW_EXCEPTION("Cannot get from FeaturePointSet the element with index " << index);
    }
    return data[index];
}
