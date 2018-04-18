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
#include <unistd.h>

#include <NVX/nvx.h>
#include <NVX/nvx_timer.hpp>

#include <NVX/Application.hpp>
#include <NVX/ConfigParser.hpp>
#include <OVX/FrameSourceOVX.hpp>
#include <OVX/RenderOVX.hpp>
#include <NVX/SyncTimer.hpp>
#include <OVX/UtilityOVX.hpp>

#include <NVX/nvx_opencv_interop.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>


namespace {

  //
  //Utility functions
  //

  struct ProcessParams
  {
    vx_uint32 sourceWidth;
    vx_uint32 sourceHeight;
    vx_uint32 fps;
    vx_int32  uUpperThreshold;
    vx_int32  uLowerThreshold;
    vx_int32  vUpperThreshold;
    vx_int32  vLowerThreshold;    

    ProcessParams()
      : sourceWidth(1280),
        sourceHeight(720),
        fps(120),
        uUpperThreshold(127),
        uLowerThreshold(127),
        vUpperThreshold(127),
        vLowerThreshold(127)
    {}
  };

  bool checkParams(vx_uint32& sourceWidth, vx_uint32& sourceHeight, vx_uint32& fps, std::string & error)
  {
    if((sourceWidth == 2592 && !(sourceHeight == 1944 || sourceHeight == 1458))
       ||
       (sourceWidth == 1280 && sourceHeight != 720))
      {
        error = "Invalid source resolution!";
      }
    if(sourceWidth == 2592 && fps > 30)
      {
        error = "Invalid fps";
      }
    return error.empty();
  }

  bool read(const std::string &configFile, ProcessParams &config, std::string &error)
  {
    const std::unique_ptr<nvxio::ConfigParser> parser(nvxio::createConfigParser());

    parser->addParameter("sourceWidth",
                         nvxio::OptionHandler::unsignedInteger(&config.sourceWidth));
    parser->addParameter("sourceHeight",
                         nvxio::OptionHandler::unsignedInteger(&config.sourceHeight));
    parser->addParameter("fps",
                         nvxio::OptionHandler::unsignedInteger(&config.fps,
                                                               nvxio::ranges::atLeast(10u) & nvxio::ranges::atMost(120u)));
    parser->addParameter("uUpperThreshold",
                         nvxio::OptionHandler::integer(&config.uUpperThreshold,
                                                       nvxio::ranges::atLeast(0) & nvxio::ranges::atMost(255)));
    parser->addParameter("uLowerThreshold",
                         nvxio::OptionHandler::integer(&config.uLowerThreshold,
                                                       nvxio::ranges::atLeast(0) & nvxio::ranges::atMost(255)));
    parser->addParameter("vUpperThreshold",
                         nvxio::OptionHandler::integer(&config.vUpperThreshold,
                                                       nvxio::ranges::atLeast(0) & nvxio::ranges::atMost(255)));
    parser->addParameter("vLowerThreshold",
                         nvxio::OptionHandler::integer(&config.vLowerThreshold,
                                                       nvxio::ranges::atLeast(0) & nvxio::ranges::atMost(255)));    
    error = parser->parse(configFile);

    if(!error.empty())
      {
        return false;
      }

    return checkParams(config.sourceWidth, config.sourceHeight, config.fps, error);
                                                               
  }
  
  struct EventData
  {
    EventData(): alive(true), pause(false), showSource(false), pixelQuery(false), x(0), y(0) {}

    bool alive;
    bool pause;
    bool showSource;
    bool pixelQuery;
    vx_uint32 x;
    vx_uint32 y;
    
  };

  static void keyboardEventCallback(void* context, vx_char key, vx_uint32 x, vx_uint32 y)
  {
    EventData* eventData = static_cast<EventData*>(context);
    if(key == 27) //escape
      {
        eventData->alive = false;      
      }
    else if (key == 32) //space
      {
        eventData->pause = !eventData->pause;
      }
    else if (key=='m')
      {
        eventData->showSource = !eventData->showSource;
      }
    else if (key='q')
      {
        eventData->x = x;
        eventData->y = y;
        eventData->pixelQuery = true;
      }
  }
}; //namespace

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
      std::string sourceUri = "device:///nvcamera";
      std::string configFile = app.findSampleFilePath("configs/color_detector_config.ini");
      ProcessParams params;

      app.setDescription("Color Detector");
      app.addOption('c', "config", "Config file path", nvxio::OptionHandler::string(&configFile));
      app.addOption('s', "source", "Video Input Source", nvxio::OptionHandler::string(&sourceUri));
      
      app.init(argc, argv);

      std::string error;
      if (!read(configFile, params, error))
        {
          std::cerr << error << std::endl;
          return nvxio::Application::APP_EXIT_CODE_INVALID_VALUE;
        }


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

      // Create NVXIO frame source
      std::unique_ptr<ovxio::FrameSource> frameSource(ovxio::createDefaultFrameSource(context, sourceUri));

      if( !frameSource)
        {
          std::cerr << "Error: Can't open source URI " << sourceUri << std::endl;
          return nvxio::Application::APP_EXIT_CODE_NO_RESOURCE;
        }

      // set frame source resolution BEFORE opening the source!
      ovxio::FrameSource::Parameters frameConfig;
      frameConfig.frameWidth = params.sourceWidth;
      frameConfig.frameHeight = params.sourceHeight;
      frameConfig.fps = params.fps;

      if(!frameSource->setConfiguration(frameConfig))
        {
          std::cout << "Error: cannot set the frame source to the specified configuration!" << std::endl;
          return nvxio::Application::APP_EXIT_CODE_NO_RESOURCE;
        }

      if(!frameSource->open())
        {
          std::cerr << "Error: Can't open source URI " << sourceUri << std::endl;
          return nvxio::Application::APP_EXIT_CODE_NO_RESOURCE;
        }

      //after frame source is opened, retrieve all of its configuration settings
      frameConfig = frameSource->getConfiguration();

      // Create NVXIO render
      std::unique_ptr<ovxio::Render> render
        (ovxio::createDefaultRender(context, "Color Detector", frameConfig.frameWidth, frameConfig.frameHeight));

      if(!render){
        std::cerr << "Error: Cannot create render!" << std::endl;
        return nvxio::Application::APP_EXIT_CODE_NO_RENDER;
      }

      //set up keyboard event handling
      EventData eventData;
      render->setOnKeyboardEventCallback(keyboardEventCallback, &eventData);

      //image object to hold frames from video stream
      vx_image frame = vxCreateImage(context, frameConfig.frameWidth, frameConfig.frameHeight, frameConfig.format);
      NVXIO_CHECK_REFERENCE(frame);

      //image to hold output
      vx_image output = vxCreateImage(context, frameConfig.frameWidth, frameConfig.frameHeight, VX_DF_IMAGE_RGB);
      NVXIO_CHECK_REFERENCE(output);

      //create graph
      vx_graph inGraph = vxCreateGraph(context);
      NVXIO_CHECK_REFERENCE(inGraph);

      vx_graph outGraph = vxCreateGraph(context);
      NVXIO_CHECK_REFERENCE(outGraph);

      vx_image inGray = vxCreateImage(context, frameConfig.frameWidth,frameConfig.frameHeight,VX_DF_IMAGE_U8);
      NVXIO_CHECK_REFERENCE(inGray);

      vx_image outGray = vxCreateImage(context, frameConfig.frameWidth, frameConfig.frameHeight, VX_DF_IMAGE_U8);
      NVXIO_CHECK_REFERENCE(outGray);

      //
      //virtual images for processing pipeline
      //



      //
      //graph node creation
      //

      vx_node convertIn = vxColorConvertNode(inGraph, frame, inGray);

      vx_node convertOut = vxColorConvertNode(outGraph, outGray, output);
      //vx_node convertOut1 = vxColorConvertNode(inGraph, inGray, output);
      


      //
      //release virtual images (the graph will hold referenced internally)
      //



      //
      // Ensure highest graph optimization level
      //

      const char* option = "-O3";
      NVXIO_SAFE_CALL( vxSetGraphAttribute(inGraph, NVX_GRAPH_VERIFY_OPTIONS, option, strlen(option)) );
      NVXIO_SAFE_CALL( vxSetGraphAttribute(outGraph, NVX_GRAPH_VERIFY_OPTIONS, option, strlen(option)) );

      //
      //verify the graph
      //

      vx_status verify_status = vxVerifyGraph(inGraph);
      if (verify_status != VX_SUCCESS)
        {
          std::cerr << "Error: inGraph verification failed. See the NVX LOG for explanation." << std::endl;
          return nvxio::Application::APP_EXIT_CODE_INVALID_GRAPH;
        }

      verify_status = vxVerifyGraph(outGraph);
      if (verify_status != VX_SUCCESS)
        {
          std::cerr << "Error: outGraph verification failed. See the NVX LOG for explanation." << std::endl;
          return nvxio::Application::APP_EXIT_CODE_INVALID_GRAPH;
        }      
      


      //
      //main loop
      //
      ovxio::FrameSource::FrameStatus frameStatus = frameSource->fetch(frame);

      std::unique_ptr<nvxio::SyncTimer> syncTimer = nvxio::createSyncTimer();
      syncTimer->arm(1. / app.getFPSLimit());

      double proc_ms = 0;
      nvx::Timer totalTimer;
      totalTimer.tic();      

      while(eventData.alive)
        {
          if(!eventData.pause)
            {
              frameStatus = frameSource->fetch(frame);
       
              if(frameStatus == ovxio::FrameSource::TIMEOUT)
                {
                  continue;
                }
          
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
              // process graph
              //

              nvx::Timer procTimer;
              procTimer.tic();
              

              NVXIO_SAFE_CALL(vxProcessGraph(inGraph));
              {
                //convert vx_image to cv::mat
                static vx_uint32 plane_index = 0;
                static vx_rectangle_t rect = {
                  0u, 0u,
                  frameConfig.frameWidth, frameConfig.frameHeight
                };
              
                nvx_cv::VXImageToCVMatMapper map(inGray, plane_index, &rect, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
                cv::Mat cv_img = map.getMat();

                //convert cv::mat to vx_image
                vx_image conCV = nvx_cv::createVXImageFromCVMat(context, cv_img);
                nvxuCopyImage(context, conCV, outGray);
              
                NVXIO_SAFE_CALL(vxProcessGraph(outGraph));

                vxReleaseImage(&conCV);
                cv_img.release();
                } //nvx_cv::ImageToMatMapper destructor will unmap internally 
              
              proc_ms = procTimer.toc();
              }

          double total_ms = totalTimer.toc();

          std::cout << "Display Time : " << total_ms << " ms" << std::endl << std::endl;

          syncTimer->synchronize();

          total_ms = totalTimer.toc();

          totalTimer.tic();

          if (eventData.showSource)
            {
              render->putImage(frame);
            }
          else
            {
              render->putImage(output);
            }

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



          if(!render->flush())
            {
              std::cout << "Finished!" << std::endl;
              break;
            }
        }

      //
      //release all objects
      //

      
      
    }
  catch (const std::exception& e)
    {
      std::cerr << "Error: " << e.what() << std::endl;
      return nvxio::Application::APP_EXIT_CODE_ERROR;
    }



  
  return nvxio::Application::APP_EXIT_CODE_SUCCESS;
}
