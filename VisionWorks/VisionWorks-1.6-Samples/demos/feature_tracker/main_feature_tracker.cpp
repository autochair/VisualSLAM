/*
# Copyright (c) 2014-2016, NVIDIA CORPORATION. All rights reserved.
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

#include <NVX/nvx.h>
#include <NVX/nvx_timer.hpp>

#include "feature_tracker.hpp"
#include <NVX/Application.hpp>
#include <NVX/ConfigParser.hpp>
#include <OVX/FrameSourceOVX.hpp>
#include <OVX/RenderOVX.hpp>
#include <NVX/SyncTimer.hpp>
#include <OVX/UtilityOVX.hpp>

//
// Process events
//

struct EventData
{
    EventData(): shouldStop(false), pause(false) {}

    bool shouldStop;
    bool pause;
};

static void eventCallback(void* eventData, vx_char key, vx_uint32, vx_uint32)
{
    EventData* data = static_cast<EventData*>(eventData);

    if (key == 27)
    {
        data->shouldStop = true;
    }
    else if (key == 32)
    {
        data->pause = !data->pause;
    }
}

static void displayState(ovxio::Render *renderer,
                         const ovxio::FrameSource::Parameters &sourceParams,
                         nvx::FeatureTracker::Params &config,
                         double proc_ms, double total_ms)
{
    std::ostringstream txt;

    txt << std::fixed << std::setprecision(1);

    ovxio::Render::TextBoxStyle style = {{255, 255, 255, 255}, {0, 0, 0, 127}, {10, 10}};

    txt << "Source size: " << sourceParams.frameWidth << 'x' << sourceParams.frameHeight << std::endl;
    txt << "Detector: " << (config.use_harris_detector ? "Harris" : "FAST") << std::endl;
    txt << "Algorithm: " << proc_ms << " ms / " << 1000.0 / proc_ms << " FPS" << std::endl;
    txt << "Display: " << total_ms  << " ms / " << 1000.0 / total_ms << " FPS" << std::endl;

    txt << std::setprecision(6);
    txt.unsetf(std::ios_base::floatfield);
    txt << "LIMITED TO " << nvxio::Application::get().getFPSLimit() << " FPS FOR DISPLAY" << std::endl;

    txt << "Space - pause/resume" << std::endl;
    txt << "Esc - close the demo";
    renderer->putTextViewport(txt.str(), style);
}

static bool read(const std::string & nf, nvx::FeatureTracker::Params &config, std::string &message)
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
        ovxio::printVersionInfo();

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
        // Create OpenVX context
        //

        ovxio::ContextGuard context;
        vxDirective(context, VX_DIRECTIVE_ENABLE_PERFORMANCE);

        //
        // Messages generated by the OpenVX framework will be processed by
        // ovxio::stdoutLogCallback
        //

        vxRegisterLogCallback(context, &ovxio::stdoutLogCallback, vx_false_e);

        //
        // Read and check input parameters
        //

        nvx::FeatureTracker::Params params;
        std::string error;
        if (!read(configFile, params, error))
        {
            std::cout<<error;
            return nvxio::Application::APP_EXIT_CODE_INVALID_VALUE;
        }

        //
        // Create a OVXIO-based frame source
        //

        std::unique_ptr<ovxio::FrameSource> source(
            ovxio::createDefaultFrameSource(context, sourceUri));

        if (!source || !source->open())
        {
            std::cerr << "Error: Can't open source URI " << sourceUri << std::endl;
            return nvxio::Application::APP_EXIT_CODE_NO_RESOURCE;
        }

        if (source->getSourceType() == ovxio::FrameSource::SINGLE_IMAGE_SOURCE)
        {
            std::cerr << "Error: Can't work on a single image." << std::endl;
            return nvxio::Application::APP_EXIT_CODE_INVALID_FORMAT;
        }

        ovxio::FrameSource::Parameters sourceParams = source->getConfiguration();

        //
        // Create a OVXIO-based render
        //

        std::unique_ptr<ovxio::Render> renderer(ovxio::createDefaultRender(
            context, "Feature Tracker Demo", sourceParams.frameWidth, sourceParams.frameHeight));

        if (!renderer)
        {
            std::cerr << "Error: Can't create a renderer" << std::endl;
            return nvxio::Application::APP_EXIT_CODE_NO_RENDER;
        }

        EventData eventData;
        renderer->setOnKeyboardEventCallback(eventCallback, &eventData);

        //
        // Create OpenVX Image to hold frames from video source
        //

        vx_image frameExemplar = vxCreateImage(context,
            sourceParams.frameWidth, sourceParams.frameHeight, VX_DF_IMAGE_RGBX);
        NVXIO_CHECK_REFERENCE(frameExemplar);
        vx_delay frame_delay = vxCreateDelay(context, (vx_reference)frameExemplar, 2);
        NVXIO_CHECK_REFERENCE(frame_delay);
        vxReleaseImage(&frameExemplar);

        //
        // Frame delay object is created to hold previous and current frames
        // from video source in the frame_delay variable
        //

        vx_image prevFrame = (vx_image)vxGetReferenceFromDelay(frame_delay, -1);
        vx_image frame = (vx_image)vxGetReferenceFromDelay(frame_delay, 0);

        //
        // Load optional mask image if needed. To be used later by tracker
        //

        vx_image mask = NULL;

#if defined USE_OPENCV || defined USE_GSTREAMER
        if (!maskFile.empty())
        {
            mask = ovxio::loadImageFromFile(context, maskFile, VX_DF_IMAGE_U8);

            vx_uint32 mask_width = 0, mask_height = 0;
            NVXIO_SAFE_CALL( vxQueryImage(mask, VX_IMAGE_ATTRIBUTE_WIDTH, &mask_width, sizeof(mask_width)) );
            NVXIO_SAFE_CALL( vxQueryImage(mask, VX_IMAGE_ATTRIBUTE_HEIGHT, &mask_height, sizeof(mask_height)) );

            if (mask_width != sourceParams.frameWidth || mask_height != sourceParams.frameHeight)
            {
                std::cerr << "Error: The mask must have the same size as the input source." << std::endl;
                return nvxio::Application::APP_EXIT_CODE_INVALID_DIMENSIONS;
            }
        }
#endif

        //
        // Create FeatureTracker instance
        //

        std::unique_ptr<nvx::FeatureTracker> tracker(nvx::FeatureTracker::create(context, params));

        ovxio::FrameSource::FrameStatus frameStatus;

        //
        // The first frame is read to initialize the tracker (tracker->init())
        // and immediately "age" the delay. See the FeatureTrackerPyrLK::init()
        // call in the file feature_tracker.cpp for further details
        //

        do
        {
            frameStatus = source->fetch(frame);
        } while (frameStatus == ovxio::FrameSource::TIMEOUT);

        if (frameStatus == ovxio::FrameSource::CLOSED)
        {
            std::cerr << "Error: Source has no frames" << std::endl;
            return nvxio::Application::APP_EXIT_CODE_NO_FRAMESOURCE;
        }

        tracker->init(frame, mask);

        vxAgeDelay(frame_delay);

        //
        // Run the main processing loop in which we read subsequent frames and
        // then pass them to tracker->track() and then aging the delay again.
        // See the FeatureTracker::track() call in the feature_tracker.cpp for
        // details. The aging mechanism allows the algorithm to access current
        // and previous frame. The tracker gets the featureList from the prevoius
        // frame and the CurrentFrame and draws the arrows between them
        //

        std::unique_ptr<nvxio::SyncTimer> syncTimer = nvxio::createSyncTimer();
        syncTimer->arm(1. / app.getFPSLimit());

        nvx::Timer totalTimer;
        totalTimer.tic();
        double proc_ms = 0;
        while (!eventData.shouldStop)
        {
            if (!eventData.pause)
            {
                frameStatus = source->fetch(frame);

                if (frameStatus == ovxio::FrameSource::TIMEOUT) {
                    continue;
                }
                if (frameStatus == ovxio::FrameSource::CLOSED) {
                    if (!source->open()) {
                        std::cerr << "Error: Failed to reopen the source" << std::endl;
                        break;
                    }
                    continue;
                }

                //
                // Process
                //

                nvx::Timer procTimer;
                procTimer.tic();

                tracker->track(frame, mask);

                proc_ms = procTimer.toc();

                //
                // Print performance results
                //

                tracker->printPerfs();
            }

            //
            // Show the previous frame
            //

            renderer->putImage(prevFrame);

            //
            // Draw arrows & state
            //

            ovxio::Render::FeatureStyle featureStyle = { { 255, 0, 0, 255 }, 4.0f };
            ovxio::Render::LineStyle arrowStyle = {{0, 255, 0, 255}, 1};

            vx_array old_points = tracker->getPrevFeatures();
            vx_array new_points = tracker->getCurrFeatures();

            renderer->putArrows(old_points, new_points, arrowStyle);
            renderer->putFeatures(old_points, featureStyle);

            double total_ms = totalTimer.toc();

            std::cout << "Display Time : " << total_ms << " ms" << std::endl << std::endl;

            //
            // Add a delay to limit display frame rate (default=30ms)
            //

            syncTimer->synchronize();

            total_ms = totalTimer.toc();

            totalTimer.tic();

            displayState(renderer.get(), sourceParams, params, proc_ms, total_ms);

            if (!renderer->flush())
            {
                eventData.shouldStop = true;
            }

            if (!eventData.pause)
            {
                vxAgeDelay(frame_delay);
            }
        }

        //
        // Release all objects
        //

        vxReleaseImage(&mask);
        vxReleaseDelay(&frame_delay);
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return nvxio::Application::APP_EXIT_CODE_ERROR;
    }

    return nvxio::Application::APP_EXIT_CODE_SUCCESS;
}
