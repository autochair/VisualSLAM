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
#include <unistd.h>

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
  //Utility functions
  //

  struct ProcessParams
  {
    vx_uint32 sourceWidth;
    vx_uint32 sourceHeight;
    vx_uint32 fps;
    vx_int32  binaryThreshold;

    ProcessParams()
      : sourceWidth(1280),
        sourceHeight(720),
        fps(120),
        binaryThreshold(127)
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
    parser->addParameter("binaryThreshold",
                         nvxio::OptionHandler::integer(&config.binaryThreshold,
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
    EventData(): alive(true), pause(false){}

    bool alive;
    bool pause;
  };

  static void keyboardEventCallback(void* context, vx_char key, vx_uint32 /*x*/, vx_uint32 /*y*/)
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
      // vx_image frame = vxCreateImage(context, frameConfig.frameWidth, frameConfig.frameHeight, VX_DF_IMAGE_RGBX);
      NVXIO_CHECK_REFERENCE(frame);

      //image to hold output
      vx_image output = vxCreateImage(context, frameConfig.frameWidth, frameConfig.frameHeight, VX_DF_IMAGE_RGB);
      NVXIO_CHECK_REFERENCE(output);

      //thresholds
      vx_threshold binaryThreshold = vxCreateThreshold(context,VX_THRESHOLD_TYPE_BINARY,VX_TYPE_UINT8);
      NVXIO_CHECK_REFERENCE(binaryThreshold);
      NVXIO_SAFE_CALL( vxSetThresholdAttribute(binaryThreshold, VX_THRESHOLD_THRESHOLD_VALUE,
                                               &params.binaryThreshold,sizeof(params.binaryThreshold)));

      //create graph
      vx_graph graph = vxCreateGraph(context);
      NVXIO_CHECK_REFERENCE(graph);

      //
      //virtual images for processing pipeline
      //
      
      vx_image inputRGB = vxCreateVirtualImage(graph, 0, 0, VX_DF_IMAGE_RGB);
      NVXIO_CHECK_REFERENCE(inputRGB);
      
      vx_image green = vxCreateVirtualImage(graph, 0, 0, VX_DF_IMAGE_U8);
      NVXIO_CHECK_REFERENCE(green);

      vx_image binary = vxCreateVirtualImage(graph, 0,0,VX_DF_IMAGE_U8);
      NVXIO_CHECK_REFERENCE(binary);

      vx_image eroded = vxCreateVirtualImage(graph, 0,0,VX_DF_IMAGE_U8);
      NVXIO_CHECK_REFERENCE(eroded);

      vx_image dilated = vxCreateVirtualImage(graph, 0,0,VX_DF_IMAGE_U8);
      NVXIO_CHECK_REFERENCE(dilated);

      vx_pixel_value_t bl = {U8:0};
      vx_image black = vxCreateUniformImage(context, frameConfig.frameWidth, frameConfig.frameHeight, VX_DF_IMAGE_U8, &bl);

      //
      //graph node creation
      //
      
      vx_node toRGBNode = vxColorConvertNode(graph, frame, inputRGB);
      NVXIO_CHECK_REFERENCE(toRGBNode);
      
      vx_node greenExtractNode = vxChannelExtractNode(graph, inputRGB, VX_CHANNEL_G, green);
      NVXIO_CHECK_REFERENCE(greenExtractNode);

      vx_node recombineNode = vxChannelCombineNode(graph, black, green, black, NULL, output);
      NVXIO_CHECK_REFERENCE(recombineNode);

      
      /*
      vx_node binNode = vxThresholdNode(graph, green, binaryThreshold, binary);
      NVXIO_CHECK_REFERENCE(binNode);

      vx_node erodeNode = vxErode3x3Node(graph, binary, eroded);
      NVXIO_CHECK_REFERENCE(erodeNode);

      vx_node dilateNode = vxDilate3x3Node(graph, eroded, output);
      NVXIO_CHECK_REFERENCE(dilateNode);
      */


      //
      //realse virtual images (the graph will hold referenced internally)
      //

      //      vxReleaseImage(&grayscale);
      vxReleaseImage(&inputRGB);
      vxReleaseImage(&green);
      vxReleaseImage(&binary);
      vxReleaseImage(&eroded);
      vxReleaseImage(&dilated);
      vxReleaseImage(&black);

      //
      // Ensure highest graph optimization level
      //

      const char* option = "-O3";
      NVXIO_SAFE_CALL( vxSetGraphAttribute(graph, NVX_GRAPH_VERIFY_OPTIONS, option, strlen(option)) );

      //
      //verify the graph
      //

      vx_status verify_status = vxVerifyGraph(graph);
      if (verify_status != VX_SUCCESS)
        {
          std::cerr << "Error: Graph verification failed. See the NVX LOG for explanation." << std::endl;
          return nvxio::Application::APP_EXIT_CODE_INVALID_GRAPH;
        }
      
      //
      //main loop
      //
      ovxio::FrameSource::FrameStatus frameStatus = frameSource->fetch(frame);      
      while(eventData.alive)
        {
          if(!eventData.pause)
            {
              frameStatus = frameSource->fetch(frame);
            }

          if(frameStatus == ovxio::FrameSource::TIMEOUT)
            {
              continue;
            }
          
          if (frameStatus == ovxio::FrameSource::CLOSED)
            {
              std::cout << "Finished!" << std::endl;
              break;
            }

          //
          // process graph
          //

          NVXIO_SAFE_CALL(vxProcessGraph(graph));

          //          std::cout << vxuChannelExtract(context, frame, VX_CHANNEL_G, output) << std::endl;

          render->putImage(output);
          if(!render->flush())
            {
              std::cout << "Finished!" << std::endl;
              break;
            }
        }

      //
      //release all objects
      //

      vxReleaseNode(&toRGBNode);
      vxReleaseNode(&recombineNode);
      vxReleaseNode(&greenExtractNode);
      /*      vxReleaseNode(&binNode);
      vxReleaseNode(&erodeNode);
      vxReleaseNode(&dilateNode);
      */
      
    }
  catch (const std::exception& e)
    {
      std::cerr << "Error: " << e.what() << std::endl;
      return nvxio::Application::APP_EXIT_CODE_ERROR;
    }



  
  return nvxio::Application::APP_EXIT_CODE_SUCCESS;
}
