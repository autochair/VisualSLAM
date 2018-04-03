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

#ifndef NVXCU_KEYPOINT_TRACKER_HPP
#define NVXCU_KEYPOINT_TRACKER_HPP

#include <vector>
#include <list>
#include <numeric>
#include <cassert>
#include <string>
#include <math.h>
#include <memory>

#include <NVX/nvxcu.h>
#include "object_tracker_nvxcu.hpp"
#include "object_tracker_with_features_info_nvxcu.hpp"

#ifndef _WIN32
#include "runtime_performance_logger.hpp"
#endif


namespace nvxcu_keypoint_tracker
{

class OutlierRemoval;
class MatChar;
class OpticalFlowBuilder;

class ObjectTrackerPyrLKKeyHistory : public nvxcu::ObjectTrackerWithFeaturesInfo
{
public:
    struct CornerDetectorParameters
    {
        size_t maxCorners_;
        uint32_t detectorCellSize_;

        bool useFastDetector_;

        uint32_t fastThreshold_;
        uint32_t fastType_;

        float harrisK_;
        float harrisThreshold_;

        float bbDecreasingRatio_;

        CornerDetectorParameters()
        {
            maxCorners_ = 5000;
            detectorCellSize_ = 3;

            useFastDetector_ = true;

            fastThreshold_ = 40;
            fastType_ = 9;

            harrisK_ = 0.04f;
            harrisThreshold_ = 100.0f;

            bbDecreasingRatio_ = 1.0f;
        }
        std::string toString() const;
    };

    struct CornersFilteringParameters
    {
        size_t maxCornersInCell_;
        size_t xNumOfCells_;
        size_t yNumOfCells_;

        CornersFilteringParameters()
        {
            maxCornersInCell_ = 5;
            xNumOfCells_ = 2;
            yNumOfCells_ = 2;
        }
        std::string toString() const;
    };

    struct PyramidalLucasKanadeParameters
    {
        int32_t windowSize_;
        uint32_t numPyramidLevels_;
        uint32_t maxIterationsNumber_;

        float maxPossibleKeypointMotion_; //all keypoints with motion grreater than this value will be removed
        float minRatioOfPointsWithFoundMotion_; // if ratio of points with found motion is less than this threshold, object is counted as lost
        int32_t minNumberOfPointsWithFoundMotion_; // if number of points with found motion is less than this threshold, object is counted as lost

        bool shouldUseBackwardChecking_;
        float maxBackwardCheckingError_;

        PyramidalLucasKanadeParameters()
        {
            windowSize_ = 10;
            numPyramidLevels_ = 6;
            maxIterationsNumber_ = 20;

            maxPossibleKeypointMotion_ = 190;//TODO: 90? earlier was 40...
            minRatioOfPointsWithFoundMotion_ = 0.1;//TODO: tune this parameter -- maybe 0.3? 0.5 ?
            minNumberOfPointsWithFoundMotion_ = 3;//TODO: tune this parameter

            shouldUseBackwardChecking_ = true;
            maxBackwardCheckingError_ = 1.5;
        }
        std::string toString() const;
    };

    struct HistoryWeightingParameters
    {
        double coefficientOfWeightDecreasing_;
        int32_t maxHistoryLength_;
        int32_t radiusToSearchKeypoint_;

        HistoryWeightingParameters():
            coefficientOfWeightDecreasing_(0.8),
            maxHistoryLength_(10),
            radiusToSearchKeypoint_(1)
        {
        }
        std::string toString() const;
    };

    struct OutlierRemovalParameters
    {
        float   relativeDistanceToRemoveOutlier_;
        uint32_t     maxIterationsNumber_;
        float    maxMeanDistanceToStop_;

        OutlierRemovalParameters():
            relativeDistanceToRemoveOutlier_(1.5),
            maxIterationsNumber_(10),
            maxMeanDistanceToStop_(1.25) //TODO: earlier 0.25 was used
        {
        }
        std::string toString() const;
    };

    struct AdvancedScaleEstimatorParameters
    {
        int32_t numQuantiles_;
        int32_t minNumberOfPoints_;


        AdvancedScaleEstimatorParameters():
            numQuantiles_(1),//TODO: eralier was 5
            minNumberOfPoints_(3)
        {
        }
        std::string toString() const;
    };

    struct AlgorithmicParameters
    {
        bool shouldUseOutlierRemoval_;
        bool shouldRemoveOutliersAtAll_;

        AlgorithmicParameters():
            shouldUseOutlierRemoval_(true),
            shouldRemoveOutliersAtAll_(false)
        {
        }
        std::string toString() const;
    };

    struct Parameters
    {
        CornersFilteringParameters cornersFilteringParameters_;
        CornerDetectorParameters cornerDetectorParameters_;
        PyramidalLucasKanadeParameters pyrLKParameters_;
        HistoryWeightingParameters historyWeightingParameters_;
        OutlierRemovalParameters outlierRemovalParameters_;
        AdvancedScaleEstimatorParameters advancedScaleEstimatorParameters_;
        AlgorithmicParameters algorithmicParameters_;

        Parameters() {}
        std::string toString() const;
    };

    explicit ObjectTrackerPyrLKKeyHistory(const Parameters& parameters);
    ~ObjectTrackerPyrLKKeyHistory();

    ObjectTrackerWithFeaturesInfo::TrackedObject* addObject(const nvxcu_rectangle_t& rect);
    nvxcu_error_status_e removeObject(ObjectTracker::TrackedObject* obj);
    nvxcu_error_status_e removeAllObjects();
    nvxcu_error_status_e process(const nvxcu_pitch_linear_image_t* frame);

    class FeaturePointsVector: public ObjectTrackerWithFeaturesInfo::FeaturePointSet
    {
    public:
        FeaturePointsVector() = default;
        ~FeaturePointsVector() = default;

        virtual size_t getSize() const;
        virtual const FeaturePoint& getFeaturePoint(size_t index) const;

        std::vector<FeaturePoint> data;
    };

    class TrackedObjectImpl : public ObjectTrackerWithFeaturesInfo::TrackedObject
    {
    public:
        TrackedObjectImpl(uint64_t id, const nvxcu_rectangle_t& loc);
        ~TrackedObjectImpl();

        uint32_t getID() const;
        nvxcu_rectangle_t getLocation() const;
        ObjectStatus getStatus() const;
        const FeaturePointsVector& getFeaturePointSet() const;

        uint64_t id_;
        nvxcu_rectangle_t loc_;
        ObjectStatus status_;
        FeaturePointsVector features_info_;
    };


private:
    static const int32_t DEFAULT_VALUE_HISTORY_NUMBER_FRAMES = 0;

    nvxcu_error_status_e initTracker(const nvxcu_pitch_linear_image_t* firstFrame);

    nvxcu_error_status_e createTrackerDataObjects();

    void updateObject(TrackedObjectImpl& obj, const nvxcu_point2f_t& motion_center, const nvxcu_point2f_t& motion_scale);

    nvxcu_coordinates2d_t calcObjectSize(TrackedObjectImpl& obj);

    static uint32_t getCellIndexForPoint(const nvxcu_rectangle_t& obj_rect, nvxcu_keypointf_t kp,
                                          size_t x_num_of_cells, size_t y_num_of_cells);

    static void filterKeypoints(const nvxcu_rectangle_t& obj_rect,
                                size_t x_num_of_cells,
                                size_t y_num_of_cells,
                                size_t max_corners_in_cell,
                                const std::vector<nvxcu_keypointf_t>& points0,
                                const std::vector<nvxcu_keypointf_t>& points1,
                                std::vector<nvxcu_keypointf_t>& filterered_points0,
                                std::vector<nvxcu_keypointf_t>& filterered_points1);

    bool estimateMotionByPointsAndUpdateHistory(const nvxcu_rectangle_t& obj_rect,
                                                const std::vector<nvxcu_keypointf_t>& points0,
                                                const std::vector<nvxcu_keypointf_t>& points1,
                                                nvxcu_point2f_t& motion_center,
                                                nvxcu_point2f_t& motion_scale,
                                                FeaturePointsVector& features_info);

    bool estimateScaleByPoints(const std::vector<nvxcu_keypointf_t>& points0,
                               const std::vector<nvxcu_keypointf_t>& points1,
                               const std::vector<float>& weights,
                               nvxcu_point2f_t& motion_scale);

    void releaseInternalObjects();

    Parameters parameters_;
    std::unique_ptr<OutlierRemoval> outlierRemoval_;

    bool initialized_tracker_;
    nvxcu_df_image_e format_;
    uint32_t width_;
    uint32_t height_;

    std::list<std::unique_ptr<TrackedObjectImpl> > objects_;

    uint64_t maxObjectId_;

    std::unique_ptr<MatChar> historyNumberFrames_;
    std::unique_ptr<MatChar> nextHistoryNumberFrames_;
    std::unique_ptr<OpticalFlowBuilder> opticalFlowBuilder_;

    std::vector<float> weightsLookupTable_;
    void initializeWeightsLookupTableForPoints();
    float getWeightForHistoryIndex(uint8_t history_index) const;

    void getHistoryIndexForPoints(const std::vector<nvxcu_keypointf_t>& points0,
                                  const std::vector<uint8_t>& status,
                                  std::vector<uint8_t>& historyIndex) const;

    void getWeightsForPoints(const std::vector<nvxcu_keypointf_t>& points0,
                             const std::vector<uint8_t>& status,
                             std::vector<float>& weight) const;

    void updateHistoryIndexForPoints(const std::vector<nvxcu_keypointf_t>& points0,
                                     const std::vector<nvxcu_keypointf_t>& points1,
                                     const std::vector<uint8_t>& status,
                                     int32_t radius_to_search_keypoint);
#ifndef _WIN32
    std::shared_ptr<RuntimePerformanceLogger> perfLogger_;
#endif

};

} // namespace nvxcu_keypoint_tracker

namespace nvxcu
{

ObjectTracker* createPyrLKKeyHistory(const nvxcu_keypoint_tracker::ObjectTrackerPyrLKKeyHistory::Parameters& params);

} // namespace nvxcu

#endif // NVXCU_KEYPOINT_TRACKER_HPP
