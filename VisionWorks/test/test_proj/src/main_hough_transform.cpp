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

#include <cmath>
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


namespace {

//
// Utility
//

struct HoughTransformDemoParams
{
    vx_uint32   switchPeriod;
    vx_float32  scaleFactor;
    vx_enum     scaleType;
    vx_int32    CannyLowerThresh;
    vx_int32    CannyUpperThresh;
    vx_float32  dp;
    vx_float32  minDist;
    vx_uint32   minRadius;
    vx_uint32   maxRadius;
    vx_uint32   accThreshold;
    vx_uint32   circlesCapacity;
    vx_float32  rho;
    vx_float32  theta;
    vx_uint32   votesThreshold;
    vx_uint32   minLineLength;
    vx_uint32   maxLineGap;
    vx_uint32   linesCapacity;

    HoughTransformDemoParams()
        : switchPeriod(400),
          scaleFactor(.5f),
          scaleType(VX_INTERPOLATION_TYPE_BILINEAR),
          CannyLowerThresh(230),
          CannyUpperThresh(250),
          dp(2.f),
          minDist(10.f),
          minRadius(1),
          maxRadius(25),
          accThreshold(110),
          circlesCapacity(300),
          rho(1.f),
          theta(1.f),
          votesThreshold(100),
          minLineLength(25),
          maxLineGap(2),
          linesCapacity(300)
     {}
};

bool checkParams(vx_int32& CannyLowerThresh, vx_int32& CannyUpperThresh,
                        vx_uint32& minRadius, vx_uint32& maxRadius, std::string & error)
{
    if (CannyLowerThresh > CannyUpperThresh)
    {
        error = "Inconsistent values of lower and upper Canny thresholds";
    }

    if (minRadius > maxRadius)
    {
        error = "Inconsistent minimum and maximum circle radius values";
    }

    return error.empty();
}


bool read(const std::string &configFile, HoughTransformDemoParams &config, std::string &error)
{
    const std::unique_ptr<nvxio::ConfigParser> parser(nvxio::createConfigParser());

    parser->addParameter("switchPeriod",
                         nvxio::OptionHandler::unsignedInteger(
                             &config.switchPeriod));


    parser->addParameter("scaleFactor",
                         nvxio::OptionHandler::real(
                             &config.scaleFactor,
                             nvxio::ranges::moreThan(0.f) & nvxio::ranges::atMost(1.f)));
    parser->addParameter("scaleType",
                         nvxio::OptionHandler::oneOf(
                             &config.scaleType,
                             {
                                 {"nearest",  VX_INTERPOLATION_TYPE_NEAREST_NEIGHBOR},
                                 {"bilinear", VX_INTERPOLATION_TYPE_BILINEAR},
                                 {"area",     VX_INTERPOLATION_TYPE_AREA},
                             }));
    parser->addParameter("CannyLowerThresh",
                         nvxio::OptionHandler::integer(
                             &config.CannyLowerThresh,
                             nvxio::ranges::moreThan(0)));
    parser->addParameter("CannyUpperThresh",
                         nvxio::OptionHandler::integer(
                             &config.CannyUpperThresh,
                             nvxio::ranges::moreThan(0)));
    parser->addParameter("dp",
                         nvxio::OptionHandler::real(
                             &config.dp,
                             nvxio::ranges::atLeast(1.f)));
    parser->addParameter("minDist",
                         nvxio::OptionHandler::real(
                             &config.minDist,
                             nvxio::ranges::moreThan(0.f)));
    parser->addParameter("minRadius",
                         nvxio::OptionHandler::unsignedInteger(
                             &config.minRadius));
    parser->addParameter("maxRadius",
                         nvxio::OptionHandler::unsignedInteger(
                             &config.maxRadius,
                             nvxio::ranges::moreThan(0u)));
    parser->addParameter("accThreshold",
                         nvxio::OptionHandler::unsignedInteger(
                             &config.accThreshold,
                             nvxio::ranges::moreThan(0u)));
    parser->addParameter("circlesCapacity",
                         nvxio::OptionHandler::unsignedInteger(
                             &config.circlesCapacity,
                             nvxio::ranges::moreThan(0u) & nvxio::ranges::atMost(1000u)));
    parser->addParameter("rho",
                         nvxio::OptionHandler::real(
                             &config.rho,
                             nvxio::ranges::moreThan(0.f)));
    parser->addParameter("theta",
                         nvxio::OptionHandler::real(
                             &config.theta,
                             nvxio::ranges::moreThan(0.f) & nvxio::ranges::atMost(180.f)));
    parser->addParameter("votesThreshold",
                         nvxio::OptionHandler::unsignedInteger(
                             &config.votesThreshold,
                             nvxio::ranges::moreThan(0u)));
    parser->addParameter("minLineLength",
                         nvxio::OptionHandler::unsignedInteger(
                             &config.minLineLength,
                             nvxio::ranges::moreThan(0u)));
    parser->addParameter("maxLineGap",
                         nvxio::OptionHandler::unsignedInteger(
                             &config.maxLineGap));
    parser->addParameter("linesCapacity",
                         nvxio::OptionHandler::unsignedInteger(
                             &config.linesCapacity,
                             nvxio::ranges::moreThan(0u) & nvxio::ranges::atMost(1000u)));

    error = parser->parse(configFile);

    if (!error.empty())
    {
        return false;
    }

    return checkParams(config.CannyLowerThresh, config.CannyUpperThresh, config.minRadius, config.maxRadius, error);
}

//
// Process events
//

struct EventData
{
    EventData(): showSource(true), stop(false), pause(false) {}

    bool showSource;
    bool stop;
    bool pause;
};


void keyboardEventCallback(void* eventData, vx_char key, vx_uint32 /*x*/, vx_uint32 /*y*/)
{
    EventData* data = static_cast<EventData*>(eventData);

    if (key == 27) // escape
    {
        data->stop = true;
    }
    else if (key == 'm')
    {
        data->showSource = !data->showSource;
    }
    else if (key == 32) // space
    {
        data->pause = !data->pause;
    }
}

}


//
// main - Application entry point
//
// The main function call of hough_transform demo creates the object of type
// Application (defined in NVXIO library). Command line arguments are parsed
// and the input video filename is read into sourceURI and the configuration
// parameters are read into the configFile
//

int main(int argc, char** argv)
{
    try
    {
        nvxio::Application &app = nvxio::Application::get();
        ovxio::printVersionInfo();

        //
        // Parse command line arguments
        //

        std::string sourceUri = app.findSampleFilePath("signs.avi");

        std::string configFile = app.findSampleFilePath("hough_transform_demo_config.ini");
        HoughTransformDemoParams params;

        app.setDescription("This demo demonstrates circles and lines detection via Hough transform");
        app.addOption('s', "source", "Input URI", nvxio::OptionHandler::string(&sourceUri));
        app.addOption('c', "config", "Config file path", nvxio::OptionHandler::string(&configFile));

        app.init(argc, argv);

        //
        // Read and check input parameters
        //

        std::string error;
        if (!read(configFile, params, error))
        {
            std::cerr << error << std::endl;
            return nvxio::Application::APP_EXIT_CODE_INVALID_VALUE;
        }

        params.theta *= ovxio::PI_F / 180.0f; // convert to radians

        //
        // NVXIO-based renderer object and frame source are instantiated
        // and attached to the OpenVX context object. NVXIO ContextGuard
        // object is used to automatically manage the OpenVX context
        // creation and destruction.
        //

        ovxio::ContextGuard context;
        vxDirective(context, VX_DIRECTIVE_ENABLE_PERFORMANCE);

        //
        // Messages generated by the OpenVX framework will be given
        // to ovxio::stdoutLogCallback
        //
        vxRegisterLogCallback(context, &ovxio::stdoutLogCallback, vx_false_e);

        //
        // Create a NVXIO-based frame source
        //

        std::unique_ptr<ovxio::FrameSource> frameSource(
            ovxio::createDefaultFrameSource(context, sourceUri));

        if (!frameSource || !frameSource->open())
        {
            std::cerr << "Error: Can't open source URI " << sourceUri << std::endl;
            return nvxio::Application::APP_EXIT_CODE_NO_RESOURCE;
        }

        ovxio::FrameSource::Parameters frameConfig = frameSource->getConfiguration();

        if ((frameConfig.frameWidth * params.scaleFactor < 16) ||
            (frameConfig.frameHeight * params.scaleFactor < 16))
        {
            std::cerr << "Error: Scale factor is too small" << std::endl;
            return nvxio::Application::APP_EXIT_CODE_INVALID_VALUE;
        }

        //
        // Create a NVXIO-based render
        //

        std::unique_ptr<ovxio::Render> render(
            ovxio::createDefaultRender(context, "Hough Transform Demo", frameConfig.frameWidth, frameConfig.frameHeight));

        if (!render)
        {
            std::cerr << "Error: Cannot create render!" << std::endl;
            return nvxio::Application::APP_EXIT_CODE_NO_RENDER;
        }

        //
        // The application recieves the keyboard events via the
        // keyboardEventCallback() function registered via the renderer object
        //

        EventData eventData;
        render->setOnKeyboardEventCallback(keyboardEventCallback, &eventData);

        //
        // Create OpenVX objects
        //
        // frame and edges vx_image objects are created to hold the frames from
        // the video source and the output of the Canny edge detector respectively
        //

        vx_image frame = vxCreateImage(context, frameConfig.frameWidth, frameConfig.frameHeight, VX_DF_IMAGE_RGBX);
        NVXIO_CHECK_REFERENCE(frame);

        vx_image edges = vxCreateImage(context, frameConfig.frameWidth, frameConfig.frameHeight, VX_DF_IMAGE_U8);
        NVXIO_CHECK_REFERENCE(edges);

        //
        // Similary the lines and circles vx_array objects are created to hold
        // the output from Hough line-segment detector and Hough circle detector
        //

        vx_array circles = vxCreateArray(context, NVX_TYPE_POINT3F, params.circlesCapacity);
        NVXIO_CHECK_REFERENCE(circles);

        vx_array lines = vxCreateArray(context, NVX_TYPE_POINT4F, params.linesCapacity);
        NVXIO_CHECK_REFERENCE(lines);

        //
        // Create OpenVX Threshold to hold Canny thresholds
        //

        vx_threshold CannyThreshold = vxCreateThreshold(context, VX_THRESHOLD_TYPE_RANGE, VX_TYPE_INT32);
        NVXIO_CHECK_REFERENCE(CannyThreshold);
        NVXIO_SAFE_CALL( vxSetThresholdAttribute(CannyThreshold, VX_THRESHOLD_ATTRIBUTE_THRESHOLD_LOWER,
                                                 &params.CannyLowerThresh, sizeof(params.CannyLowerThresh)) );
        NVXIO_SAFE_CALL( vxSetThresholdAttribute(CannyThreshold, VX_THRESHOLD_ATTRIBUTE_THRESHOLD_UPPER,
                                                 &params.CannyUpperThresh, sizeof(params.CannyUpperThresh)) );

        //
        // vxCreateGraph() instantiates the pipeline
        //

        vx_graph graph = vxCreateGraph(context);
        NVXIO_CHECK_REFERENCE(graph);

        //
        // Virtual images for internal processing
        //

        vx_image virt_U8 = vxCreateVirtualImage(graph, 0, 0, VX_DF_IMAGE_U8);
        NVXIO_CHECK_REFERENCE(virt_U8);

        vx_image virt_scaled = vxCreateVirtualImage(graph, static_cast<vx_uint32>(frameConfig.frameWidth * params.scaleFactor),
                                                    static_cast<vx_uint32>(frameConfig.frameHeight * params.scaleFactor), VX_DF_IMAGE_U8);
        NVXIO_CHECK_REFERENCE(virt_scaled);

        vx_image virt_blurred = vxCreateVirtualImage(graph, 0, 0, VX_DF_IMAGE_U8);
        NVXIO_CHECK_REFERENCE(virt_blurred);

        vx_image virt_equalized = vxCreateVirtualImage(graph, 0, 0, VX_DF_IMAGE_U8);
        NVXIO_CHECK_REFERENCE(virt_equalized);

        vx_image virt_edges = vxCreateVirtualImage(graph, 0, 0, VX_DF_IMAGE_U8);
        NVXIO_CHECK_REFERENCE(virt_edges);

        vx_image virt_dx = vxCreateVirtualImage(graph, 0, 0, VX_DF_IMAGE_S16);
        NVXIO_CHECK_REFERENCE(virt_dx);

        vx_image virt_dy = vxCreateVirtualImage(graph, 0, 0, VX_DF_IMAGE_S16);
        NVXIO_CHECK_REFERENCE(virt_dy);

        //
        // Node creation
        //
        // The frame is converted to grayscale by converting from RGB to YUV and
        // extracting the Y channel and scaled down (scale factor is defined in
        // params.scaleType and defaults to 0.5). Frame is then blurred with
        // a median3x3 filter and has histogram equalized. Output is passed to
        // the Canny edge detector and Sobel 3x3 filter. The output is then passed
        // through nvxHoughSegmentsNode() and nvxHoughCirclesNode() to detect line
        // segments and circles. The detected line segments and circles are scaled
        // up back by params.scaleType
        //


        vx_node cvtNode = vxColorConvertNode(graph, frame, virt_U8);
        NVXIO_CHECK_REFERENCE(cvtNode);

        vx_node scaleDownNode = vxScaleImageNode(graph, virt_U8, virt_scaled, params.scaleType);
        NVXIO_CHECK_REFERENCE(scaleDownNode);

        vx_node median3x3Node = vxMedian3x3Node(graph, virt_scaled, virt_blurred);
        NVXIO_CHECK_REFERENCE(median3x3Node);

        vx_node equalizeHistNode = vxEqualizeHistNode(graph, virt_blurred, virt_equalized);
        NVXIO_CHECK_REFERENCE(equalizeHistNode);

        vx_node CannyNode = vxCannyEdgeDetectorNode(graph, virt_equalized, CannyThreshold, 3, VX_NORM_L1, virt_edges);
        NVXIO_CHECK_REFERENCE(CannyNode);

        vx_node scaleUpNode = vxScaleImageNode(graph, virt_edges, edges, params.scaleType);
        NVXIO_CHECK_REFERENCE(scaleUpNode);

        vx_node Sobel3x3Node = vxSobel3x3Node(graph, virt_equalized, virt_dx, virt_dy);
        NVXIO_CHECK_REFERENCE(Sobel3x3Node);

        vx_node HoughCirclesNode = nvxHoughCirclesNode(graph, virt_edges, virt_dx, virt_dy,
                                                       circles, nullptr,  params.dp, params.minDist,
                                                       params.minRadius, params.maxRadius, params.accThreshold);
        NVXIO_CHECK_REFERENCE(HoughCirclesNode);

        vx_node HoughSegmentsNode = nvxHoughSegmentsNode(graph, virt_edges, lines, params.rho, params.theta,
                                                         params.votesThreshold, params.minLineLength,
                                                         params.maxLineGap, nullptr);
        NVXIO_CHECK_REFERENCE(HoughSegmentsNode);

        //
        // Release virtual images (the graph will hold references internally)
        //

        vxReleaseImage(&virt_U8);
        vxReleaseImage(&virt_scaled);
        vxReleaseImage(&virt_blurred);
        vxReleaseImage(&virt_equalized);
        vxReleaseImage(&virt_edges);
        vxReleaseImage(&virt_dx);
        vxReleaseImage(&virt_dy);

        //
        // Release Threshold object (the graph will hold references internally)
        //

        vxReleaseThreshold(&CannyThreshold);

        //
        // Ensure highest graph optimization level
        //

        const char* option = "-O3";
        NVXIO_SAFE_CALL( vxSetGraphAttribute(graph, NVX_GRAPH_VERIFY_OPTIONS, option, strlen(option)) );

        //
        // Verify the graph
        //

        vx_status verify_status = vxVerifyGraph(graph);
        if (verify_status != VX_SUCCESS)
        {
            std::cerr << "Error: Graph verification failed. See the NVX LOG for explanation." << std::endl;
            return nvxio::Application::APP_EXIT_CODE_INVALID_GRAPH;
        }

        //
        // Main loop
        //

        vx_int32 numFrames = 0;

        std::unique_ptr<nvxio::SyncTimer> syncTimer = nvxio::createSyncTimer();
        syncTimer->arm(1. / app.getFPSLimit());

        double proc_ms = 0;
        nvx::Timer totalTimer;
        totalTimer.tic();

        while (!eventData.stop)
        {
            if (!eventData.pause)
            {
                //
                // The main processing loop simply reads the input frames using
                // the source->fetch(). The rendering code in the main loop decides
                // whether to display orignal (unprocessed) source image or edges
                // overlayed with detected lines and circles based on user input
                //

                ovxio::FrameSource::FrameStatus frameStatus = frameSource->fetch(frame);

                if (frameStatus == ovxio::FrameSource::TIMEOUT)
                    continue;

                if (frameStatus == ovxio::FrameSource::CLOSED)
                {
                    if (!frameSource->open())
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

                NVXIO_SAFE_CALL( vxProcessGraph(graph) );

                proc_ms = procTimer.toc();


                //
                // Scale detected circles
                //

                vx_size num_circles = 0;
                NVXIO_SAFE_CALL( vxQueryArray(circles, VX_ARRAY_ATTRIBUTE_NUMITEMS, &num_circles, sizeof(num_circles)) );
                std::cout << "Found " << num_circles << " circles" << std::endl;

                if (num_circles > 0)
                {
                    vx_map_id map_id;
                    void *ptr;
                    vx_size stride;
                    NVXIO_SAFE_CALL( vxMapArrayRange(circles, 0, num_circles, &map_id, &stride, &ptr, VX_READ_AND_WRITE, VX_MEMORY_TYPE_HOST, 0) );

                    for (vx_size i = 0; i < num_circles; ++i)
                    {
                        nvx_point3f_t *c = (nvx_point3f_t *)vxFormatArrayPointer(ptr, i, stride);
                        c->x /= params.scaleFactor;
                        c->y /= params.scaleFactor;
                        c->z /= params.scaleFactor;
                    }

                    NVXIO_SAFE_CALL( vxUnmapArrayRange(circles, map_id) );
                }

                //
                // Scale detected lines (convert it to array with start and end coordinates)
                //

                vx_size lines_count = 0;
                NVXIO_SAFE_CALL( vxQueryArray(lines, VX_ARRAY_ATTRIBUTE_NUMITEMS, &lines_count, sizeof(lines_count)) );
                std::cout << "Found " << lines_count << " lines" << std::endl;

                if (lines_count > 0)
                {
                    vx_map_id map_id;
                    vx_size stride;
                    void *ptr;
                    NVXIO_SAFE_CALL( vxMapArrayRange(lines, 0, lines_count, &map_id, &stride, &ptr, VX_READ_AND_WRITE, VX_MEMORY_TYPE_HOST, 0) );

                    for (vx_size i = 0; i < lines_count; ++i)
                    {
                        nvx_point4f_t *coord = (nvx_point4f_t *)vxFormatArrayPointer(ptr, i, stride);

                        coord->x /= params.scaleFactor;
                        coord->y /= params.scaleFactor;
                        coord->z /= params.scaleFactor;
                        coord->w /= params.scaleFactor;
                    }

                    NVXIO_SAFE_CALL( vxUnmapArrayRange(lines, map_id) );
                }

                //
                // switch image/edges view every switchPeriod-th frame
                //

                if (params.switchPeriod > 0)
                {
                    numFrames++;
                    if (numFrames % params.switchPeriod == 0)
                    {
                        eventData.showSource = !eventData.showSource;
                    }
                }

                //
                // Show performance statistics
                //

                vx_perf_t perf;

                NVXIO_SAFE_CALL( vxQueryGraph(graph, VX_GRAPH_ATTRIBUTE_PERFORMANCE, &perf, sizeof(perf)) );
                std::cout << "Graph Time : " << perf.tmp / 1000000.0 << " ms" << std::endl;

                NVXIO_SAFE_CALL( vxQueryNode(cvtNode, VX_NODE_ATTRIBUTE_PERFORMANCE, &perf, sizeof(perf)) );
                std::cout << "\t Color Convert Time : " << perf.tmp / 1000000.0 << " ms" << std::endl;

                NVXIO_SAFE_CALL( vxQueryNode(scaleDownNode, VX_NODE_ATTRIBUTE_PERFORMANCE, &perf, sizeof(perf)) );
                std::cout << "\t Scale Down Time : " << perf.tmp / 1000000.0 << " ms" << std::endl;

                NVXIO_SAFE_CALL( vxQueryNode(median3x3Node, VX_NODE_ATTRIBUTE_PERFORMANCE, &perf, sizeof(perf)) );
                std::cout << "\t Median3x3 Time : " << perf.tmp / 1000000.0 << " ms" << std::endl;

                NVXIO_SAFE_CALL( vxQueryNode(equalizeHistNode, VX_NODE_ATTRIBUTE_PERFORMANCE, &perf, sizeof(perf)) );
                std::cout << "\t Equalize Hist Time : " << perf.tmp / 1000000.0 << " ms" << std::endl;

                NVXIO_SAFE_CALL( vxQueryNode(CannyNode, VX_NODE_ATTRIBUTE_PERFORMANCE, &perf, sizeof(perf)) );
                std::cout << "\t Canny Time : " << perf.tmp / 1000000.0 << " ms" << std::endl;

                NVXIO_SAFE_CALL( vxQueryNode(scaleUpNode, VX_NODE_ATTRIBUTE_PERFORMANCE, &perf, sizeof(perf)) );
                std::cout << "\t Scale Up Time : " << perf.tmp / 1000000.0 << " ms" << std::endl;

                NVXIO_SAFE_CALL( vxQueryNode(Sobel3x3Node, VX_NODE_ATTRIBUTE_PERFORMANCE, &perf, sizeof(perf)) );
                std::cout << "\t Sobel 3x3 Time : " << perf.tmp / 1000000.0 << " ms" << std::endl;

                NVXIO_SAFE_CALL( vxQueryNode(HoughCirclesNode, VX_NODE_ATTRIBUTE_PERFORMANCE, &perf, sizeof(perf)) );
                std::cout << "\t Hough Circles Time : " << perf.tmp / 1000000.0 << " ms" << std::endl;

                NVXIO_SAFE_CALL( vxQueryNode(HoughSegmentsNode, VX_NODE_ATTRIBUTE_PERFORMANCE, &perf, sizeof(perf)) );
                std::cout << "\t Hough Segments Time : " << perf.tmp / 1000000.0 << " ms" << std::endl;
            }

            double total_ms = totalTimer.toc();

            std::cout << "Display Time : " << total_ms << " ms" << std::endl << std::endl;

            syncTimer->synchronize();

            total_ms = totalTimer.toc();

            totalTimer.tic();

            //
            // Show original image or detected edges
            //

            if (eventData.showSource)
            {
                render->putImage(frame);
            }
            else
            {
                render->putImage(edges);
            }

            //
            // Draw detected circles
            //

            ovxio::Render::CircleStyle circleStyle = {
                { 255u, 0u, 255u, 255u},
                2
            };

            render->putCircles(circles, circleStyle);

            //
            // Draw detected lines
            //

            ovxio::Render::LineStyle lineStyle = {
                { 0u, 255u, 255u, 255u},
                2
            };

            render->putLines(lines, lineStyle);

            //
            // Display information and performance metrics
            //

            std::ostringstream msg;
            msg << std::fixed << std::setprecision(1);

            msg << "Resolution: " << frameConfig.frameWidth << 'x' << frameConfig.frameHeight << std::endl;
            msg << "Algorithm: " << proc_ms << " ms / " << 1000.0 / proc_ms << " FPS" << std::endl;
            msg << "Display: " << total_ms  << " ms / " << 1000.0 / total_ms << " FPS" << std::endl;

            msg << std::setprecision(6);
            msg.unsetf(std::ios_base::floatfield);
            msg << "LIMITED TO " << app.getFPSLimit() << " FPS FOR DISPLAY" << std::endl;
            msg << "M - switch Source/Edges" << std::endl;
            msg << "Space - pause/resume" << std::endl;
            msg << "Esc - close the demo";

            ovxio::Render::TextBoxStyle textStyle = {
                {255u, 255u, 255u, 255u}, // color
                {0u,   0u,   0u,   127u}, // bgcolor
                {10u, 10u} // origin
            };

            render->putTextViewport(msg.str(), textStyle);

            //
            // Flush all rendering commands
            //

            if (!render->flush())
            {
                eventData.stop = true;
            }
        }

        //
        // Release all objects
        //

        vxReleaseNode(&cvtNode);
        vxReleaseNode(&scaleDownNode);
        vxReleaseNode(&median3x3Node);
        vxReleaseNode(&equalizeHistNode);
        vxReleaseNode(&CannyNode);
        vxReleaseNode(&scaleUpNode);
        vxReleaseNode(&Sobel3x3Node);
        vxReleaseNode(&HoughCirclesNode);
        vxReleaseNode(&HoughSegmentsNode);

        vxReleaseGraph(&graph);

        vxReleaseImage(&frame);
        vxReleaseImage(&edges);
        vxReleaseArray(&circles);
        vxReleaseArray(&lines);
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return nvxio::Application::APP_EXIT_CODE_ERROR;
    }

    return nvxio::Application::APP_EXIT_CODE_SUCCESS;
}
