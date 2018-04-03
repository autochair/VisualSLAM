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
 * \brief NVIDIA VisionWorks CUDA Layer Object Tracker API
 */

#ifndef NVXCU_TRACKING_HPP
#define NVXCU_TRACKING_HPP

#include <NVX/nvxcu.h>


//----------------------------------------------------------------------------
// Generic Interface for Object Trackers based on the VisionWorks CUDA API
//----------------------------------------------------------------------------

namespace nvxcu {

/**
 * \ingroup nvxcu_algorithms
 * \brief   Object Tracker Interface.
 *
 */
class ObjectTracker
{
public:
    /**
     * \brief Defines tracking status values.
     */
    enum ObjectStatus
    {
        TRACKED = 0,        /**< \brief Indicates the object is tracked. */
        TEMPORARY_LOST = 1, /**< \brief Indicates the object is temporarily not tracked. */
        LOST = 2            /**< \brief Indicates the object is not tracked. */
    };

    /**
     * \brief An interface for getting information on ID, location, speed, and
     *          status of the tracked object.
     */
    class TrackedObject
    {
    public:
        /**
         * \brief   Gets the ID of the object.
         *
         * \return The ID of the object.
         */
        virtual uint32_t getID() const = 0;

        /**
         * \brief   Gets the location of the object.
         *
         * \return A rectangle representing the object's location
         *            on the frame.
         */
        virtual nvxcu_rectangle_t getLocation() const = 0;

        /**
         * \brief   Gets tracking status of the object.
         *
         * \return A \ref NVXCUObjectTracker::ObjectStatus enumerator.
         */
        virtual ObjectStatus getStatus() const = 0;

    protected:
        /**
         * \brief Destructor is hidden; do not delete the objects created by ObjectTracker.
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
     * \return An \ref NVXCUObjectTracker::TrackedObject interface for the added object.
     */
    virtual TrackedObject* addObject(const nvxcu_rectangle_t& rect) = 0;

    /**
     * \brief Stops tracking the specified object.
     *
     * \param [in] objectHandler Specifies a pointer to the handler of an object
     *                              to be removed.
     *
     * \return A `vx_status` enumerator.
     */
    virtual nvxcu_error_status_e removeObject(TrackedObject* objectHandler) = 0;

    /**
     * \brief Resets tracked objects.
     *
     * \return A `vx_status` enumerator.
     */
    virtual nvxcu_error_status_e removeAllObjects() = 0;

    /**
     * \brief Tracks objects.
     *
     * \param [in] frame Specifies the input frame (8-bit grayscale).
     *
     * \return A `vx_status` enumerator.
     */
    virtual nvxcu_error_status_e process(const nvxcu_pitch_linear_image_t* frame) = 0;

    /**
     * \brief Destructor.
     */
    virtual ~ObjectTracker() {}
};


/**
 * \ingroup nvxcu_algorithms
 * \brief Defines Keypoint Object Tracker parameters.
 */

struct KeypointObjectTrackerParams
{
    /**
     * \brief The number of levels desired.
     *           This must be a non-zero value.
     *           Default value is 6.
     */
    uint32_t pyr_levels;

    /**
     * \brief The number of iterations.
     *           Default value is 20.
     */
    uint32_t lk_num_iters;

    /**
     * \brief The size of the window on which to perform the algorithm.
     *           Default value is 10.
     */
    uint32_t lk_win_size;

    /**
     * \brief Maximal number of FAST / Harris corners per image.
     *           This must be a non-zero value.
     *           Default value is 5000.
     */
    uint32_t max_corners;

    /**
     * \brief Specifies the size of cells for cell-based non-max suppression.
     *           Default value is 3.
     */
    uint32_t detector_cell_size;

    /**
     * \brief Specifies whether to use FAST or Harris corner detector.
     *           Default value is true.
     */
    bool use_fast_detector;

    /**
     * \brief  Specifies the number of neighborhoods to test.
     *            Supported values : 9, 10, 11, 12.
     *            Default value is 9.
     */
    uint32_t fast_type;

    /** \brief   Specifies the threshold difference between
     *              intensity of the central pixel and pixels of
     *              a circle around this pixel.
     *              Default value is 40.
     */
    uint32_t fast_threshold;

    /** \brief  Specifies the Harris K parameter.
     *             Default value is 0.04.
     */
    float harris_k;

    /**
     * \brief  Specifies the Harris threshold.
     *            Default value is 100.
     */
    float harris_threshold;

    /**
     * \brief Maximal number of FAST / Harris corners per cell in bounding box after
     *           filtering for scale / motion-center evaluation.
     *           This must be a non-zero value.
     *           Default value is 5.
     */
    uint32_t max_corners_in_cell;


    /**
     * \brief Number of cells along the x-axis to split bounding box for corners filtering.
     *           This must be a non-zero value.
     *           Default value is 2.
     */
    uint32_t x_num_of_cells;


    /**
     * \brief Number of cells along the y-axis to split bounding box for corners filtering.
     *          This must be a non-zero value.
     *          Default value is 2.
     */
    uint32_t y_num_of_cells;

    /**
     * \brief Specifies the ratio between decreased and initial bounding box of object.
     *           Must be in the range [0:1].
     *           Default value is 1 - means bounding box remains the same.
     */
    float bb_decreasing_ratio;

    /**
     * \brief Default constructor.
     */
    KeypointObjectTrackerParams()
    {
        pyr_levels = 6;
        lk_num_iters = 20;
        lk_win_size = 10;

        max_corners = 5000;
        detector_cell_size = 3;

        use_fast_detector = true;

        fast_type = 9;
        fast_threshold = 40;

        harris_k = 0.04f;
        harris_threshold = 100.0f;

        bb_decreasing_ratio = 1.0f;

        max_corners_in_cell = 5;
        x_num_of_cells = 2;
        y_num_of_cells = 2;
    }
};

} // namespace nvxcu

/**
 * \ingroup nvxcu_algorithms
 * \brief Creates lightweight Keypoint Object Tracker.
 *
 * \param [in] params   Specifies parameters of Keypoint Object Tracker.
 *
 * \return A pointer to a Keypoint Object Tracker Algorithm implementation.
 */
nvxcu::ObjectTracker* nvxcuCreateKeypointObjectTracker(const nvxcu::KeypointObjectTrackerParams& params = nvxcu::KeypointObjectTrackerParams());

#endif
