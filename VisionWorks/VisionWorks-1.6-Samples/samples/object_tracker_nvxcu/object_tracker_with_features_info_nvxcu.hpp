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

/**
 * \file
 * \brief NVIDIA VisionWorks CUDA Layer Object Tracker With Features Info API
 */

#ifndef NVXCU_TRACKING_WITH_FEATURES_INFO_HPP
#define NVXCU_TRACKING_WITH_FEATURES_INFO_HPP

#include <NVX/nvxcu.h>
#include "object_tracker_nvxcu.hpp"

//----------------------------------------------------------------------------
// Generic Interface for Object Trackers With Features Info based on the
// the VisionWorks CUDA API
//----------------------------------------------------------------------------

namespace nvxcu {

/**
 * \ingroup nvxcu_tracking_object_tracker_with_features_info
 * \brief   Object Tracker Interface, which allows
 *          to get for each object its features info.
 */
class ObjectTrackerWithFeaturesInfo: public ObjectTracker
{
public:
    /**
     * \brief Structure to store info for one feature point
     */
    struct FeaturePoint
    {
        nvxcu_keypointf_t point_on_previous_frame; /**< \brief Info on this feature point on the previous frame. */
        nvxcu_keypointf_t point_on_current_frame;  /**< \brief Info on this feature point on the current frame. */
        float weight;  /**< \brief Tracking weight of this feature point (depends on the tracking history). */
    };

    /**
     * \brief Class to get info for all feature points of an object
     */
    class FeaturePointSet
    {
    public:
        /**
         * \brief Gets the number of feature points.
         */
        virtual size_t getSize() const = 0;

        /**
         * \brief Gets a constant reference to the feature point with the pointed index.
         *        If the index is not less than the number of feature points,
         *        an exception will be thrown.
         */
        virtual const FeaturePoint& getFeaturePoint(size_t index) const = 0;

    protected:
        /**
         * \brief Destructor is hidden; do not delete the objects created by ObjectTrackerWithFeaturesInfo.
         */
        virtual ~FeaturePointSet() {}
    };

    /**
     * \brief An interface for getting information on ID, location, speed,
     *          status, and feature points info of the tracked object.
     */
    class TrackedObject: public ObjectTracker::TrackedObject
    {
    public:
        /**
         * \brief Gets the feature points info.
         */
        virtual const FeaturePointSet& getFeaturePointSet() const = 0;
    protected:
        /**
         * \brief Destructor is hidden; do not delete the objects created by ObjectTrackerWithFeaturesInfo.
         */
        virtual ~TrackedObject() {}
    };

    /**
     * \brief Adds an object to be tracked.
     *
     * \param [in] rect  Specifies a reference to a rectangle representing the
     *                      object's location on the last processed frame.
     *
     * \note Returned pointer should not be released with the `delete` operator.
     *          Use removeObject() method instead.
     *
     * \return An \ref NVXCUObjectTrackerWithFeaturesInfo::TrackedObject interface for the added object.
     */
    virtual TrackedObject* addObject(const nvxcu_rectangle_t& rect) = 0;

};

} // namespace nvxcu

/**
 * \ingroup nvxcu_tracking_object_tracker_with_features_info
 * \brief Creates lightweight Keypoint Object Tracker which allows to get for each object
 *        its feature points info.
 *
 * \param [in] params   Specifies parameters of Keypoint Object Tracker.
 *
 * \return A pointer to a Keypoint Object Tracker Algorithm implementation.
 */
nvxcu::ObjectTrackerWithFeaturesInfo* nvxcuCreateKeypointObjectTrackerWithFeaturesInfo(
        const nvxcu::KeypointObjectTrackerParams& params = nvxcu::KeypointObjectTrackerParams());

#endif // NVXCU_TRACKING_WITH_FEATURES_INFO_HPP
