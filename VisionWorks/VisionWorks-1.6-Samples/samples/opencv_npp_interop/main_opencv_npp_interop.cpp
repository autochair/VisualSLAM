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

/*
 * Disclaimer: This sample was written and tested against OpenCV 2.4.13. It might need changes to be compatible with other OpenCV versions.
 */

#include <iostream>
#include "NVX/Application.hpp"

#if !defined USE_OPENCV || !defined USE_NPP

int main(int, char**)
{
#ifndef USE_OPENCV
    std::cout << "NVXIO and samples were built without OpenCV support." << std::endl;
    std::cout << "Install OpenCV (2.4.13) and rebuild the sample." << std::endl;
#endif
#ifndef USE_NPP
    std::cout << "The sample was built without CUDA NPP support." << std::endl;
    std::cout << "Install CUDA NPP library and rebuild the sample." << std::endl;
#endif

    return nvxio::Application::APP_EXIT_CODE_ERROR;
}

#else

#include <string>
#include <iomanip>
#include <memory>

#include <NVX/nvx.h>
#include <NVX/nvx_timer.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "alpha_comp_node.hpp"

#include "OVX/RenderOVX.hpp"
#include "NVX/SyncTimer.hpp"
#include "OVX/UtilityOVX.hpp"

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

static void displayState(ovxio::Render *renderer, const cv::Size & size, double proc_ms, double total_ms)
{
    std::ostringstream txt;

    txt << std::fixed << std::setprecision(1);

    txt << "Source size: " << size.width << 'x' << size.height << std::endl;
    txt << "Algorithm: " << proc_ms << " ms / " << 1000.0 / proc_ms << " FPS" << std::endl;
    txt << "Display: " << total_ms  << " ms / " << 1000.0 / total_ms << " FPS" << std::endl;

    txt << std::setprecision(6);
    txt.unsetf(std::ios_base::floatfield);
    txt << "LIMITED TO " << nvxio::Application::get().getFPSLimit() << " FPS FOR DISPLAY" << std::endl;
    txt << "Space - pause/resume" << std::endl;
    txt << "Esc - close the demo";

    ovxio::Render::TextBoxStyle style = {{255, 255, 255, 255}, {0, 0, 0, 127}, {10, 10}};
    renderer->putTextViewport(txt.str(), style);
}

static void VX_CALLBACK myLogCallback(vx_context /*context*/, vx_reference /*ref*/, vx_status /*status*/, const vx_char string[])
{
    std::cout << "VisionWorks LOG : " << string << std::endl;
}

//
// main - Application entry point
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

        std::string fileName1 = app.findSampleFilePath("lena.jpg");
        std::string fileName2 = app.findSampleFilePath("baboon.jpg");

        app.setDescription("This sample accepts as input two images and performs alpha blending of them");
        app.addOption(0, "img1", "First image", nvxio::OptionHandler::string(&fileName1));
        app.addOption(0, "img2", "Second image", nvxio::OptionHandler::string(&fileName2));
        app.init(argc, argv);

        //
        // Load input images
        //

        if (fileName1 == fileName2)
        {
            std::cerr << "Error: Please, use different files for img1 and img2" << std::endl;
            return nvxio::Application::APP_EXIT_CODE_INVALID_VALUE;
        }

        cv::Mat cv_src1 = cv::imread(fileName1, cv::IMREAD_GRAYSCALE);
        cv::Mat cv_src2 = cv::imread(fileName2, cv::IMREAD_GRAYSCALE);

        if (cv_src1.empty())
        {
            std::cerr << "Error: Can't load input image " << fileName1 << std::endl;
            return nvxio::Application::APP_EXIT_CODE_NO_RESOURCE;
        }

        if (cv_src2.empty())
        {
            std::cerr << "Error: Can't load input image " << fileName2 << std::endl;
            return nvxio::Application::APP_EXIT_CODE_NO_RESOURCE;
        }

        if (cv_src1.size() != cv_src2.size())
        {
            std::cerr << "Error: Input images must have the same size." << std::endl;
            return nvxio::Application::APP_EXIT_CODE_INVALID_DIMENSIONS;
        }

        //
        // Create OpenVX context
        //

        ovxio::ContextGuard context;
        vxRegisterLogCallback(context, &myLogCallback, vx_false_e);
        vxDirective(context, VX_DIRECTIVE_ENABLE_PERFORMANCE);

        std::unique_ptr<ovxio::Render> renderer(ovxio::createDefaultRender(context, "OpenCV NPP Interop Sample",
                                                                           3 * cv_src1.cols, cv_src1.rows));

        if (!renderer) {
            std::cerr << "Error: Can't create a renderer." << std::endl;
            return nvxio::Application::APP_EXIT_CODE_NO_RENDER;
        }

        EventData eventData;
        renderer->setOnKeyboardEventCallback(eventCallback, &eventData);

        //
        // Import input images into OpenVX context
        //

        vx_rectangle_t rect = {
            0u, 0u,
            vx_uint32(cv_src1.cols), vx_uint32(cv_src1.rows)
        };

        vx_imagepatch_addressing_t src1_addr;
        src1_addr.dim_x = cv_src1.cols;
        src1_addr.dim_y = cv_src1.rows;
        src1_addr.stride_x = sizeof(vx_uint8);
        src1_addr.stride_y = static_cast<vx_int32>(cv_src1.step);

        void *src1_ptrs[] = {
            cv_src1.data
        };

        vx_image src1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &src1_addr, src1_ptrs, VX_MEMORY_TYPE_HOST);
        NVXIO_CHECK_REFERENCE(src1);

        vx_imagepatch_addressing_t src2_addr;
        src2_addr.dim_x = cv_src2.cols;
        src2_addr.dim_y = cv_src2.rows;
        src2_addr.stride_x = sizeof(vx_uint8);
        src2_addr.stride_y = static_cast<vx_int32>(cv_src2.step);

        void *src2_ptrs[] = {
            cv_src2.data
        };

        vx_image src2 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &src2_addr, src2_ptrs, VX_MEMORY_TYPE_HOST);
        NVXIO_CHECK_REFERENCE(src2);

        //
        // Create output image
        //

        vx_uint32 width = 0, height = 0;
        NVXIO_SAFE_CALL( vxQueryImage(src1, VX_IMAGE_ATTRIBUTE_WIDTH, &width, sizeof(width)) );
        NVXIO_SAFE_CALL( vxQueryImage(src1, VX_IMAGE_ATTRIBUTE_HEIGHT, &height, sizeof(height)) );

        vx_image dst = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
        NVXIO_CHECK_REFERENCE(dst);

        vx_image demoImg = vxCreateImage(context, 3 * width, height, VX_DF_IMAGE_U8);
        NVXIO_CHECK_REFERENCE(demoImg);

        vx_rectangle_t leftRect;
        NVXIO_SAFE_CALL( vxGetValidRegionImage(dst, &leftRect) );

        vx_rectangle_t middleRect;
        middleRect.start_x = leftRect.end_x;
        middleRect.start_y = leftRect.start_y;
        middleRect.end_x = 2 * leftRect.end_x;
        middleRect.end_y = leftRect.end_y;

        vx_rectangle_t rightRect;
        rightRect.start_x = middleRect.end_x;
        rightRect.start_y = leftRect.start_y;
        rightRect.end_x = 3 * leftRect.end_x;
        rightRect.end_y = leftRect.end_y;

        vx_image leftRoi = vxCreateImageFromROI(demoImg, &leftRect);
        NVXIO_CHECK_REFERENCE(leftRoi);
        vx_image middleRoi = vxCreateImageFromROI(demoImg, &middleRect);
        NVXIO_CHECK_REFERENCE(middleRoi);
        vx_image rightRoi = vxCreateImageFromROI(demoImg, &rightRect);
        NVXIO_CHECK_REFERENCE(rightRoi);

        //
        // Notify the framework that we are going to access imported images
        //

        vx_map_id src1_map_id;
        void *src1_ptr;
        NVXIO_SAFE_CALL( vxMapImagePatch(src1, &rect, 0, &src1_map_id, &src1_addr, &src1_ptr, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0) );

        vx_map_id src2_map_id;
        void *src2_ptr;
        NVXIO_SAFE_CALL( vxMapImagePatch(src2, &rect, 0, &src2_map_id, &src2_addr, &src2_ptr, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0) );

        NVXIO_SAFE_CALL( vxUnmapImagePatch(src1, src1_map_id) );
        NVXIO_SAFE_CALL( vxUnmapImagePatch(src2, src2_map_id) );

        //
        // Create scalars
        //

        vx_uint8 alpha1 = 255;
        vx_uint8 alpha2 = 255 - alpha1;
        vx_enum alphaOp = static_cast<vx_enum>(NPPI_OP_ALPHA_PLUS);

        vx_scalar s_alpha1 = vxCreateScalar(context, VX_TYPE_UINT8, &alpha1);
        NVXIO_CHECK_REFERENCE(s_alpha1);

        vx_scalar s_alpha2 = vxCreateScalar(context, VX_TYPE_UINT8, &alpha2);
        NVXIO_CHECK_REFERENCE(s_alpha2);

        vx_scalar s_alphaOp = vxCreateScalar(context, VX_TYPE_ENUM, &alphaOp);
        NVXIO_CHECK_REFERENCE(s_alphaOp);

        //
        // Register user defined kernels
        //

        registerAlphaCompKernel(context);

        //
        // Create a processing graph
        //

        vx_graph graph = vxCreateGraph(context);
        NVXIO_CHECK_REFERENCE(graph);

        //
        // Virtual images for internal processing
        //

        vx_image src1_blurred = vxCreateVirtualImage(graph, width, height, VX_DF_IMAGE_U8);
        NVXIO_CHECK_REFERENCE(src1_blurred);

        vx_image src2_blurred = vxCreateVirtualImage(graph, width, height, VX_DF_IMAGE_U8);
        NVXIO_CHECK_REFERENCE(src2_blurred);

        //
        // Gaussian blurring nodes
        //

        vx_node blur1_node = vxGaussian3x3Node(graph, src1, src1_blurred);
        NVXIO_CHECK_REFERENCE(blur1_node);

        vx_node blur2_node = vxGaussian3x3Node(graph, src2, src2_blurred);
        NVXIO_CHECK_REFERENCE(blur2_node);

        //
        // Alpha channel (semi-transparency) node
        //

        vx_node alphaComp_node = alphaCompNode(graph, src1_blurred, s_alpha1, src2_blurred, s_alpha2, dst, s_alphaOp);
        NVXIO_CHECK_REFERENCE(alphaComp_node);

        //
        // Ensure highest graph optimization level
        //

        const char* option = "-O3";
        NVXIO_SAFE_CALL( vxSetGraphAttribute(graph, NVX_GRAPH_VERIFY_OPTIONS, option, strlen(option)) );

        //
        // Verify the graph
        //

        vx_status status = vxVerifyGraph(graph);

        if (status != VX_SUCCESS)
        {
            std::cerr << "Error: Graph verification failed (see LOG)" << std::endl;
            return nvxio::Application::APP_EXIT_CODE_INVALID_GRAPH;
        }

        //
        // Processing loop
        //

        bool alpha1Increase = false;

        std::unique_ptr<nvxio::SyncTimer> syncTimer = nvxio::createSyncTimer();
        syncTimer->arm(1. / app.getFPSLimit());

        nvx::Timer totalTimer;
        totalTimer.tic();
        const vx_uint8 alpha1Step = 5;
        double proc_ms = 0;
        while(!eventData.shouldStop)
        {
            if (!eventData.pause)
            {
                //
                // Change alpha values
                //

                if (alpha1Increase)
                {
                    alpha1 += alpha1Step;

                    if (alpha1 >= 255 - alpha1Step)
                    {
                        alpha1Increase = false;
                    }
                }
                else
                {
                    alpha1 -= alpha1Step;

                    if (alpha1 <= alpha1Step)
                    {
                        alpha1Increase = true;
                    }
                }

                alpha2 = 255 - alpha1;

                NVXIO_SAFE_CALL( vxCopyScalar(s_alpha1, &alpha1, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST) );
                NVXIO_SAFE_CALL( vxCopyScalar(s_alpha2, &alpha2, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST) );

                //
                // Process
                //

                nvx::Timer procTimer;
                procTimer.tic();
                status = vxProcessGraph(graph);
                proc_ms = procTimer.toc();

                if (status != VX_SUCCESS)
                {
                    std::cerr << "Graph processing failed (see LOG)" << std::endl;

                    NVXIO_SAFE_CALL( vxQueryNode(blur1_node, VX_NODE_ATTRIBUTE_STATUS, &status, sizeof(status)) );
                    std::cout << "\t Gaussian Blur 1 Status : " << (status == VX_SUCCESS ? "SUCCESS" : "FAILED") << std::endl;

                    NVXIO_SAFE_CALL( vxQueryNode(blur2_node, VX_NODE_ATTRIBUTE_STATUS, &status, sizeof(status)) );
                    std::cout << "\t Gaussian Blur 2 Status : " << (status == VX_SUCCESS ? "SUCCESS" : "FAILED") << std::endl;

                    NVXIO_SAFE_CALL( vxQueryNode(alphaComp_node, VX_NODE_ATTRIBUTE_STATUS, &status, sizeof(status)) );
                    std::cout << "\t Alpha Comp Status : " << (status == VX_SUCCESS ? "SUCCESS" : "FAILED") << std::endl;

                    break;
                }

                //
                // Report performance
                //

                vx_perf_t perf;

                NVXIO_SAFE_CALL( vxQueryGraph(graph, VX_GRAPH_ATTRIBUTE_PERFORMANCE, &perf, sizeof(perf)) );
                std::cout << "Graph Time : " << perf.tmp / 1000000.0 << " ms" << std::endl;

                NVXIO_SAFE_CALL( vxQueryNode(blur1_node, VX_NODE_ATTRIBUTE_PERFORMANCE, &perf, sizeof(perf)) );
                std::cout << "\t Gaussian Blur 1 Time : " << perf.tmp / 1000000.0 << " ms" << std::endl;

                NVXIO_SAFE_CALL( vxQueryNode(blur2_node, VX_NODE_ATTRIBUTE_PERFORMANCE, &perf, sizeof(perf)) );
                std::cout << "\t Gaussian Blur 2 Time : " << perf.tmp / 1000000.0 << " ms" << std::endl;

                NVXIO_SAFE_CALL( vxQueryNode(alphaComp_node, VX_NODE_ATTRIBUTE_PERFORMANCE, &perf, sizeof(perf)) );
                std::cout << "\t Alpha Comp Time : " << perf.tmp / 1000000.0 << " ms" << std::endl;

                NVXIO_SAFE_CALL( nvxuCopyImage(context, src1, leftRoi) );
                NVXIO_SAFE_CALL( nvxuCopyImage(context, dst, middleRoi) );
                NVXIO_SAFE_CALL( nvxuCopyImage(context, src2, rightRoi) );
            }

            //
            // Show results
            //

            renderer->putImage(demoImg);

            double total_ms = totalTimer.toc();

            std::cout << "Display Time : " << total_ms << " ms" << std::endl << std::endl;

            syncTimer->synchronize();

            total_ms = totalTimer.toc();

            totalTimer.tic();

            displayState(renderer.get(), cv_src2.size(), proc_ms, total_ms);

            if (!renderer->flush())
            {
                eventData.shouldStop = true;
            }
        }

        //
        // Release all objects
        //

        renderer->close();
        vxReleaseImage(&demoImg);
        vxReleaseImage(&leftRoi);
        vxReleaseImage(&middleRoi);
        vxReleaseImage(&rightRoi);
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return nvxio::Application::APP_EXIT_CODE_ERROR;
    }

    return nvxio::Application::APP_EXIT_CODE_SUCCESS;
}

#endif // USE_OPENCV
