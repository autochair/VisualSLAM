/*
# Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
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

#include <NVX/Application.hpp>
#include <NVX/ConfigParser.hpp>
#include <OVX/FrameSourceOVX.hpp>
#include <OVX/RenderOVX.hpp>
#include <NVX/SyncTimer.hpp>
#include <OVX/UtilityOVX.hpp>

#include "stereo_matching.hpp"
#include "color_disparity_graph.hpp"

//
// Utility functions
//

static void displayState(ovxio::Render *renderer,
                         const ovxio::FrameSource::Parameters &sourceParams,
                         double proc_ms, double total_ms)
{
    std::ostringstream txt;

    txt << std::fixed << std::setprecision(1);

    ovxio::Render::TextBoxStyle style = {{255, 255, 255, 255}, {0, 0, 0, 127}, {10, 10}};

    txt << "Source size: " << sourceParams.frameWidth << 'x' << sourceParams.frameHeight / 2 << std::endl;
    txt << "Algorithm: " << proc_ms << " ms / " << 1000.0 / proc_ms << " FPS" << std::endl;
    txt << "Display: " << total_ms  << " ms / " << 1000.0 / total_ms << " FPS" << std::endl;

    txt << std::setprecision(6);
    txt.unsetf(std::ios_base::floatfield);
    txt << "LIMITED TO " << nvxio::Application::get().getFPSLimit() << " FPS FOR DISPLAY" << std::endl;
    txt << "S - switch Frame / Disparity / Color output" << std::endl;
    txt << "Space - pause/resume" << std::endl;
    txt << "Esc - close the demo" << std::endl;
    renderer->putTextViewport(txt.str(), style);
}

static bool read(const std::string &nf, StereoMatching::StereoMatchingParams &config, std::string &message)
{
    std::unique_ptr<nvxio::ConfigParser> parser(nvxio::createConfigParser());
    parser->addParameter("min_disparity",
                         nvxio::OptionHandler::integer(
                             &config.min_disparity,
                             nvxio::ranges::atLeast(0) & nvxio::ranges::atMost(256)));
    parser->addParameter("max_disparity",
                         nvxio::OptionHandler::integer(
                             &config.max_disparity,
                             nvxio::ranges::atLeast(0) & nvxio::ranges::atMost(256)));
    parser->addParameter("P1",
                         nvxio::OptionHandler::integer(
                             &config.P1,
                             nvxio::ranges::atLeast(0) & nvxio::ranges::atMost(256)));
    parser->addParameter("P2",
                         nvxio::OptionHandler::integer(
                             &config.P2,
                             nvxio::ranges::atLeast(0) & nvxio::ranges::atMost(256)));
    parser->addParameter("sad",
                         nvxio::OptionHandler::integer(
                             &config.sad,
                             nvxio::ranges::atLeast(0) & nvxio::ranges::atMost(31)));
    parser->addParameter("bt_clip_value",
                         nvxio::OptionHandler::integer(
                             &config.bt_clip_value,
                             nvxio::ranges::atLeast(15) & nvxio::ranges::atMost(95)));
    parser->addParameter("max_diff",
                         nvxio::OptionHandler::integer(
                             &config.max_diff));
    parser->addParameter("uniqueness_ratio",
                         nvxio::OptionHandler::integer(
                             &config.uniqueness_ratio,
                             nvxio::ranges::atLeast(0) & nvxio::ranges::atMost(100)));
    parser->addParameter("scanlines_mask",
                         nvxio::OptionHandler::integer(
                             &config.scanlines_mask,
                             nvxio::ranges::atLeast(0) & nvxio::ranges::atMost(256)));
    parser->addParameter("flags",
                         nvxio::OptionHandler::integer(
                             &config.flags,
                             nvxio::ranges::atLeast(0) & nvxio::ranges::atMost(3)));
    parser->addParameter("ct_win_size",
                         nvxio::OptionHandler::integer(
                             &config.ct_win_size,
                             nvxio::ranges::atLeast(0) & nvxio::ranges::atMost(5)));
    parser->addParameter("hc_win_size",
                         nvxio::OptionHandler::integer(
                             &config.hc_win_size,
                             nvxio::ranges::atLeast(0) & nvxio::ranges::atMost(5)));

    message = parser->parse(nf);

    return message.empty();
}

//
// Process events
//

enum OUTPUT_IMAGE
{
    ORIG_FRAME,
    ORIG_DISPARITY,
    COLOR_OUTPUT
};

struct EventData
{
    EventData() : shouldStop(false), outputImg(COLOR_OUTPUT), pause(false) {}

    bool shouldStop;
    OUTPUT_IMAGE outputImg;
    bool pause;
};

static void eventCallback(void* eventData, vx_char key, vx_uint32, vx_uint32)
{
    EventData* data = static_cast<EventData*>(eventData);

    if (key == 27)
    {
        data->shouldStop = true;
    }
    else if (key == 's')
    {
        switch (data->outputImg)
        {
        case ORIG_FRAME:
            data->outputImg = ORIG_DISPARITY;
            break;

        case ORIG_DISPARITY:
            data->outputImg = COLOR_OUTPUT;
            break;

        case COLOR_OUTPUT:
            data->outputImg = ORIG_FRAME;
            break;
        }
    }
    else if (key == 32)
    {
        data->pause = !data->pause;
    }
}

//
// main - Application entry point
//
// The main function call of stereo_matching demo creates the object of type
// Application (defined in NVXIO library). Command line arguments are parsed
// and the input video filename is read into sourceURI and the configuration
// parameters are read into the configFile. The input video stream is expected
// to be in the top-bottom layout and expected to be already undistorted and
// rectified.
//

int main(int argc, char* argv[])
{
    try
    {
        nvxio::Application &app = nvxio::Application::get();
        ovxio::printVersionInfo();

        //
        // Parse command line arguments
        //

        std::string sourceUri  = app.findSampleFilePath("left_right.mp4");

        std::string configFile = app.findSampleFilePath("stereo_matching_demo_config.ini");
        StereoMatching::StereoMatchingParams params;
        StereoMatching::ImplementationType implementationType = StereoMatching::HIGH_LEVEL_API;

        app.setDescription("This demo demonstrates Stereo Matching algorithm");
        app.addOption('s', "source", "Source URI", nvxio::OptionHandler::string(&sourceUri));
        app.addOption('c', "config", "Config file path", nvxio::OptionHandler::string(&configFile));
        app.addOption('t', "type", "Implementation type",
                      nvxio::OptionHandler::oneOf(&implementationType,
                                                  {
                                                      {"hl", StereoMatching::HIGH_LEVEL_API},
                                                      {"ll", StereoMatching::LOW_LEVEL_API},
                                                      {"pyr", StereoMatching::LOW_LEVEL_API_PYRAMIDAL}
                                                  }));

        app.init(argc, argv);

        //
        // Read and check input parameters
        //

        std::string error;
        if (!read(configFile, params, error))
        {
            std::cerr << error;
            return nvxio::Application::APP_EXIT_CODE_INVALID_VALUE;
        }

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
        // Create a NVXIO-based frame source
        //

        std::unique_ptr<ovxio::FrameSource> source(
            ovxio::createDefaultFrameSource(context, sourceUri));

        if (!source || !source->open())
        {
            std::cerr << "Error: Can't open source URI " << sourceUri << std::endl;
            return nvxio::Application::APP_EXIT_CODE_NO_FRAMESOURCE;
        }

        ovxio::FrameSource::Parameters sourceParams = source->getConfiguration();

        if (sourceParams.frameHeight % 2 != 0)
        {
            std::cerr << "\"" << sourceUri.c_str()
                      << "\" has odd height (" << sourceParams.frameHeight
                      << "). This demo requires the source's height to be even." << std::endl;
            return nvxio::Application::APP_EXIT_CODE_INVALID_DIMENSIONS;
        }

        //
        // Create a NVXIO-based renderer
        //

        std::unique_ptr<ovxio::Render> renderer(ovxio::createDefaultRender(context,
            "Stereo Matching Demo", sourceParams.frameWidth, sourceParams.frameHeight / 2));

        if (!renderer)
        {
            std::cerr << "Error: Can't create a renderer" << std::endl;
            return nvxio::Application::APP_EXIT_CODE_NO_RENDER;
        }

        //
        // Application recieves the keyboard events via the eventCallback()
        // function registered via the renderer object
        //

        EventData eventData;
        renderer->setOnKeyboardEventCallback(eventCallback, &eventData);

        //
        // Create OpenVX Image to hold frames from the video source. Since the
        // input stream consists of the left and right frames in the top-bottom
        // layout, they are separated out in the vx_image left and vx_image right
        // with the help of the appropriate vx_rectangle_t objects passed to
        // vxCreateImageFromROI() (where ROI stands for region Of interest)
        //

        vx_image top_bottom = vxCreateImage
            (context, sourceParams.frameWidth, sourceParams.frameHeight, VX_DF_IMAGE_RGBX);
        NVXIO_CHECK_REFERENCE(top_bottom);

        vx_rectangle_t left_rect { 0, 0, sourceParams.frameWidth, sourceParams.frameHeight / 2 };
        vx_image left  = vxCreateImageFromROI(top_bottom, &left_rect);
        NVXIO_CHECK_REFERENCE(left);
        vx_rectangle_t right_rect { 0, sourceParams.frameHeight / 2, sourceParams.frameWidth, sourceParams.frameHeight };
        vx_image right = vxCreateImageFromROI(top_bottom, &right_rect);
        NVXIO_CHECK_REFERENCE(right);

        //
        // Disparity vx_image object is created that holds U8 (byte-wide)
        // output of the semi-global matching (SGM) algorithm
        //

        vx_image disparity = vxCreateImage
            (context, sourceParams.frameWidth, sourceParams.frameHeight / 2, VX_DF_IMAGE_U8);
        NVXIO_CHECK_REFERENCE(disparity);

        //
        // color_output vx_image object is created to hold the output color images
        // This is done by a separate auxiliary pipeline which performs the linear
        // conversion of the disparity values into HSV colorspace for improved
        // visualization so that the far and near objects are color-coded according
        // to their distance from the camera
        //

        vx_image color_output = vxCreateImage
            (context, sourceParams.frameWidth, sourceParams.frameHeight / 2, VX_DF_IMAGE_RGB);

        //
        // Create StereoMatching instance
        // The class StereoMatching compose the primary pipeline that performs
        // the SGM algorithm. The stereo object of this class is created by
        // calling StereoMatching::createStereoMatching(), passing the config
        // file parameters (to be used by the SGM pipeline), the desired
        // implementation type and the previously created left, right and
        // disparity vx_image objects
        //

        std::unique_ptr<StereoMatching> stereo(
            StereoMatching::createStereoMatching(
                context, params,
                implementationType,
                left, right, disparity));

        //
        // The output of the SGM pipeline (disparity vx_image) is then passed to the
        // auxiliary pipeline, managed by ColorDisparityGraph class. The result of the
        // auxiliary pipeline is stored in the color_output vx_image
        //

        ColorDisparityGraph color_disp_graph(context, disparity, color_output, params.max_disparity);
        bool color_disp_update = true;

        //
        // Run processing loop
        //
        // The main processing loop simply reads the input frames using fetch()
        // and passing the control to the StereoMatching::run() function. The rendering
        // code in the main loop decides whether to display orignal (unprocessed)
        // image, plain disparity (U8) or the colored disparity based on user input
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
                ovxio::FrameSource::FrameStatus frameStatus;

                do
                {
                    frameStatus = source->fetch(top_bottom);
                }
                while(frameStatus == ovxio::FrameSource::TIMEOUT);

                if (frameStatus == ovxio::FrameSource::CLOSED)
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

                nvx::Timer procTimer;
                procTimer.tic();

                stereo->run();

                proc_ms = procTimer.toc();

                //
                // Print performance results
                //

                stereo->printPerfs();

                color_disp_update = true;
            }

            switch (eventData.outputImg)
            {
            case ORIG_FRAME:
                renderer->putImage(left);
                break;

            case ORIG_DISPARITY:
                renderer->putImage(disparity);
                break;

            case COLOR_OUTPUT:
                if (color_disp_update)
                {
                    color_disp_graph.process();
                    color_disp_graph.printPerfs();
                    color_disp_update = false;
                }
                renderer->putImage(color_output);
                break;
            }

            double total_ms = totalTimer.toc();

            std::cout << "Display Time : " << total_ms << " ms" << std::endl << std::endl;

            syncTimer->synchronize();

            total_ms = totalTimer.toc();

            totalTimer.tic();

            displayState(renderer.get(), sourceParams, proc_ms, total_ms);

            if (!renderer->flush())
            {
                eventData.shouldStop = true;
            }
        }

        //
        // Release all objects
        //

        vxReleaseImage(&top_bottom);
        vxReleaseImage(&left);
        vxReleaseImage(&right);
        vxReleaseImage(&disparity);
        vxReleaseImage(&color_output);
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return nvxio::Application::APP_EXIT_CODE_ERROR;
    }

    return nvxio::Application::APP_EXIT_CODE_SUCCESS;
}
