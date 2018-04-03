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

#include <iostream>
#include <sstream>
#include <iomanip>
#include <string>
#include <memory>

#include "feature_tracker_nvxcu.hpp"
#include <NVX/ConfigParser.hpp>
#include <NVX/SyncTimer.hpp>

#include <NVX/FrameSource.hpp>
#include <NVX/Render.hpp>

//
// Process events
//

struct EventData
{
    EventData() :
        shouldStop(false), pause(false)
    {
    }

    bool shouldStop;
    bool pause;
};

static void eventCallback(void * eventData, char key, uint32_t, uint32_t)
{
    EventData * data = static_cast<EventData *>(eventData);

    if (key == 27)
    {
        data->shouldStop = true;
    }
    else if (key == 32)
    {
        data->pause = !data->pause;
    }
}

static void displayState(nvxio::Render * renderer,
                         const nvxio::FrameSource::Parameters & sourceParams,
                         nvxcu::FeatureTracker::Params & config)
{
    std::ostringstream txt;

    txt << std::fixed << std::setprecision(1);

    nvxio::Render::TextBoxStyle style = {{255, 255, 255, 255}, {0, 0, 0, 127}, {10, 10}};

    txt << "Source size: " << sourceParams.frameWidth << 'x' << sourceParams.frameHeight << std::endl;
    txt << "Detector: " << (config.use_harris_detector ? "Harris" : "FAST") << std::endl;

    txt << std::setprecision(6);
    txt.unsetf(std::ios_base::floatfield);
    txt << "LIMITED TO " << nvxio::Application::get().getFPSLimit() << " FPS FOR DISPLAY" << std::endl;

    txt << "Space - pause/resume" << std::endl;
    txt << "Esc - close the demo";
    renderer->putTextViewport(txt.str(), style);
}

static bool read(const std::string & nf, nvxcu::FeatureTracker::Params &config, std::string &message)
{
    std::unique_ptr<nvxio::ConfigParser> ftparser(nvxio::createConfigParser());

    ftparser->addParameter("pyr_levels", nvxio::OptionHandler::unsignedInteger(&config.pyr_levels,
             nvxio::ranges::atLeast(1u) & nvxio::ranges::atMost(8u)));
    ftparser->addParameter("lk_win_size", nvxio::OptionHandler::unsignedInteger(&config.lk_win_size,
             nvxio::ranges::atLeast(3u) & nvxio::ranges::atMost(32u)));
    ftparser->addParameter("lk_num_iters", nvxio::OptionHandler::unsignedInteger(&config.lk_num_iters,
             nvxio::ranges::atLeast(1u) & nvxio::ranges::atMost(100u)));

    ftparser->addParameter("array_capacity", nvxio::OptionHandler::unsignedInteger(&config.array_capacity,
             nvxio::ranges::atLeast(1u)));
    ftparser->addParameter("detector_cell_size", nvxio::OptionHandler::unsignedInteger(&config.detector_cell_size,
             nvxio::ranges::atLeast(1u)));
    ftparser->addParameter("detector", nvxio::OptionHandler::oneOf(&config.use_harris_detector, {
        {"harris", true},
        {"fast", false}
    }));

    ftparser->addParameter("harris_k", nvxio::OptionHandler::real(&config.harris_k,
             nvxio::ranges::moreThan(0.0f)));
    ftparser->addParameter("harris_thresh", nvxio::OptionHandler::real(&config.harris_thresh,
             nvxio::ranges::moreThan(0.0f)));

    ftparser->addParameter("fast_type", nvxio::OptionHandler::unsignedInteger(&config.fast_type,
             nvxio::ranges::atLeast(9u) & nvxio::ranges::atMost(12u)));
    ftparser->addParameter("fast_thresh", nvxio::OptionHandler::unsignedInteger(&config.fast_thresh,
             nvxio::ranges::lessThan(255u)));

    message = ftparser->parse(nf);

    return message.empty();
}

static nvxcu_pitch_linear_image_t createImageRGBX(uint32_t width, uint32_t height)
{
    void * dev_ptr = NULL;
    size_t pitch = 0;
    NVXIO_CUDA_SAFE_CALL( cudaMallocPitch(&dev_ptr, &pitch, width * sizeof(uint8_t) * 4, height) );

    nvxcu_pitch_linear_image_t image;
    image.base.image_type = NVXCU_PITCH_LINEAR_IMAGE;
    image.base.format = NVXCU_DF_IMAGE_RGBX;
    image.base.width = width;
    image.base.height = height;
    image.planes[0].dev_ptr = dev_ptr;
    image.planes[0].pitch_in_bytes = pitch;

    return image;
}

static void releaseImage(nvxcu_pitch_linear_image_t * image)
{
    NVXIO_CUDA_SAFE_CALL( cudaFree(image->planes[0].dev_ptr) );
    image->planes[0].dev_ptr = nullptr;
}

//
// main - Application entry point
//
// The main function call of the feature tracker demo creates the object of
// type Application(defined in NVXIO library).
//

int main(int argc, char* argv[])
{
    try
    {
        nvxio::Application &app = nvxio::Application::get();

        //
        // Parse command line arguments.The input video filename is read into
        // sourceURI and the configuration parameters are read into configFile
        //

        std::string sourceUri = app.findSampleFilePath("cars.mp4");
        std::string configFile = app.findSampleFilePath("feature_tracker_demo_config.ini");

        app.setDescription("This demo demonstrates Feature Tracker algorithm");
        app.addOption('s', "source", "Source URI", nvxio::OptionHandler::string(&sourceUri));
        app.addOption('c', "config", "Config file path", nvxio::OptionHandler::string(&configFile));

#if defined USE_OPENCV || defined USE_GSTREAMER
        std::string maskFile;
        app.addOption('m', "mask", "Optional mask", nvxio::OptionHandler::string(&maskFile));
#endif

        app.init(argc, argv);

        //
        // Read and check input parameters
        //

        nvxcu::FeatureTracker::Params params;
        std::string error;
        if (!read(configFile, params, error))
        {
            std::cout<<error;
            return nvxio::Application::APP_EXIT_CODE_INVALID_VALUE;
        }

        //
        // Create a NVIDIAIO-based frame source
        //

        std::unique_ptr<nvxio::FrameSource> source(
            nvxio::createDefaultFrameSource(sourceUri));

        if (!source || !source->open())
        {
            std::cerr << "Error: Can't open source URI " << sourceUri << std::endl;
            return nvxio::Application::APP_EXIT_CODE_NO_RESOURCE;
        }

        if (source->getSourceType() == nvxio::FrameSource::SINGLE_IMAGE_SOURCE)
        {
            std::cerr << "Error: Can't work on a single image." << std::endl;
            return nvxio::Application::APP_EXIT_CODE_INVALID_FORMAT;
        }

        nvxio::FrameSource::Parameters sourceParams = source->getConfiguration();

        //
        // Create a NVIDIAIO-based render
        //

        std::unique_ptr<nvxio::Render> renderer(nvxio::createDefaultRender(
            "Feature Tracker Demo", sourceParams.frameWidth, sourceParams.frameHeight));

        if (!renderer)
        {
            std::cerr << "Error: Can't create a renderer" << std::endl;
            return nvxio::Application::APP_EXIT_CODE_NO_RENDER;
        }

        EventData eventData;
        renderer->setOnKeyboardEventCallback(eventCallback, &eventData);

        //
        // Create OpenVX Images to hold frames from video source
        //

        nvxcu_pitch_linear_image_t prevFrame = createImageRGBX(sourceParams.frameWidth, sourceParams.frameHeight);
        nvxcu_pitch_linear_image_t frame = createImageRGBX(sourceParams.frameWidth, sourceParams.frameHeight);

        //
        // Load optional mask image if needed. To be used later by tracker
        //

        nvxcu_image_t * mask = nullptr;
        nvxcu_pitch_linear_image_t pl_mask = { };

#if defined USE_OPENCV || defined USE_GSTREAMER
        if (!maskFile.empty())
        {
            pl_mask = nvxio::loadImageFromFile(maskFile, NVXCU_DF_IMAGE_U8);

            if (pl_mask.base.width != sourceParams.frameWidth ||
                    pl_mask.base.height != sourceParams.frameHeight)
            {
                std::cerr << "Error: The mask must have the same size as the input source." << std::endl;
                return nvxio::Application::APP_EXIT_CODE_INVALID_DIMENSIONS;
            }

            mask = &pl_mask.base;
        }
#endif

        //
        // Create nvxcu::FeatureTracker instance
        //

        std::unique_ptr<nvxcu::FeatureTracker> tracker(nvxcu::FeatureTracker::create(params));

        //
        // The first frame is read to initialize the tracker (tracker->init()).
        // See the FeatureTrackerPyrLK::init() call in the file feature_tracker.cpp for further details
        //

        nvxio::FrameSource::FrameStatus frameStatus =
                nvxio::FrameSource::TIMEOUT;

        do
        {
            frameStatus = source->fetch(frame);
        } while (frameStatus == nvxio::FrameSource::TIMEOUT);

        if (frameStatus == nvxio::FrameSource::CLOSED)
        {
            std::cerr << "Error: Source has no frames" << std::endl;
            return nvxio::Application::APP_EXIT_CODE_NO_FRAMESOURCE;
        }

        tracker->init(&frame.base, mask);

        std::swap(frame, prevFrame);

        //
        // Run the main processing loop in which we read subsequent frames and
        // then pass them to tracker->track().
        // See the FeatureTracker::track() call in the feature_tracker.cpp for
        // details. The tracker gets the featureList from the previous
        // frame and the CurrentFrame and draws the arrows between them.
        //

        while (!eventData.shouldStop)
        {
            if (!eventData.pause)
            {
                frameStatus = source->fetch(frame);

                if (frameStatus == nvxio::FrameSource::TIMEOUT)
                {
                    continue;
                }
                if (frameStatus == nvxio::FrameSource::CLOSED)
                {
                    if (!source->open())
                    {
                        std::cerr << "Error: Failed to reopen the source" << std::endl;
                        break;
                    }
                    continue;
                }

                //
                // Process
                //


                tracker->track(&frame.base, mask);

            }

            //
            // Show the previous frame
            //

            renderer->putImage(prevFrame);

            //
            // Draw arrows & state
            //

            nvxio::Render::FeatureStyle featureStyle = { { 255, 0, 0, 255 }, 4.0f };
            nvxio::Render::LineStyle arrowStyle = {{0, 255, 0, 255}, 1};

            const nvxcu_array_t * old_points = tracker->getPrevFeatures();
            const nvxcu_array_t * new_points = tracker->getCurrFeatures();

            renderer->putArrows(*(const nvxcu_plain_array_t *)old_points,
                                *(const nvxcu_plain_array_t *)new_points, arrowStyle);
            renderer->putFeatures(*(const nvxcu_plain_array_t *)old_points, featureStyle);

            //
            // Add a delay to limit display frame rate (default=30ms)
            //

            displayState(renderer.get(), sourceParams, params);

            if (!renderer->flush())
            {
                eventData.shouldStop = true;
            }

            if (!eventData.pause)
            {
                std::swap(frame, prevFrame);
            }
        }

        //
        // Release all objects
        //

        if (mask)
            releaseImage(&pl_mask);

        releaseImage(&frame);
        releaseImage(&prevFrame);
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return nvxio::Application::APP_EXIT_CODE_ERROR;
    }

    return nvxio::Application::APP_EXIT_CODE_SUCCESS;
}
