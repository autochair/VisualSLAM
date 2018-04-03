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

#include <stdlib.h>

#include <NVX/nvx.h>
#include <NVX/nvxcu.h>
#include <NVX/nvx_timer.hpp>

#include <NVX/ConfigParser.hpp>
#include <NVX/FrameSource.hpp>
#include <NVX/Render.hpp>
#include <NVX/SyncTimer.hpp>

#include "object_tracker_nvxcu.hpp"
#include "object_tracker_with_features_info_nvxcu.hpp"

struct Scalar
{
    uint8_t values[3];
};

typedef std::vector<nvxcu::ObjectTrackerWithFeaturesInfo::FeaturePoint> FeaturePointsVector;
typedef std::vector<nvxcu::ObjectTrackerWithFeaturesInfo::TrackedObject*> TrackedObjectPointersVector;

nvxcu_pitch_linear_image_t createImage(uint32_t width, uint32_t height, nvxcu_df_image_e format)
{
    NVXIO_ASSERT(format == NVXCU_DF_IMAGE_U8 ||
                 format == NVXCU_DF_IMAGE_RGB ||
                 format == NVXCU_DF_IMAGE_RGBX);

    nvxcu_pitch_linear_image_t image = { };

    image.base.format = format;
    image.base.image_type = NVXCU_PITCH_LINEAR_IMAGE;
    image.base.width = width;
    image.base.height = height;

    size_t pitch = 0ul;
    uint32_t cn = format == NVXCU_DF_IMAGE_U8 ? 1u :
                  format == NVXCU_DF_IMAGE_RGB ? 3u : 4u;

    NVXIO_CUDA_SAFE_CALL( cudaMallocPitch(&image.planes[0].dev_ptr, &pitch,
                          image.base.width * cn, image.base.height) );
    image.planes[0].pitch_in_bytes = static_cast<uint32_t>(pitch);

    return image;
}

void releaseImage(nvxcu_pitch_linear_image_t * image)
{
    NVXIO_CUDA_SAFE_CALL( cudaFree(image->planes[0].dev_ptr) );
}

nvxcu_plain_array_t createArray(uint32_t capacity, nvxcu_array_item_type_e itemType)
{
    NVXIO_ASSERT(itemType == NVXCU_TYPE_POINT4F ||
                 itemType == NVXCU_TYPE_POINT3F);

    nvxcu_plain_array_t array = { };
    array.base.array_type = NVXCU_PLAIN_ARRAY;
    array.base.capacity = capacity;
    array.base.item_type = itemType;

    size_t elemSize = itemType == NVXCU_TYPE_POINT4F ? sizeof(nvxcu_point4f_t) :
                                                       sizeof(nvxcu_point3f_t);

    size_t arraySize = elemSize * array.base.capacity;
    NVXIO_CUDA_SAFE_CALL( cudaMalloc(&array.dev_ptr, arraySize + sizeof(uint32_t)) );
    array.num_items_dev_ptr = reinterpret_cast<uint32_t *>(static_cast<uint8_t *>(array.dev_ptr) + arraySize);

    return array;
}

void releaseArray(nvxcu_plain_array_t * array)
{
    NVXIO_CUDA_SAFE_CALL( cudaFree(array->dev_ptr) );
}

struct FeaturePointsVisualizationStyle
{
    float min_weight;
    float max_weight;
    uint8_t color[4];
    unsigned char radius;
    unsigned char thickness;
};

static const int MAX_NUMBER_OF_ELEMENTS_IN_ARRAYS = 500;

struct EventHandlerData
{
    bool isPressed;
    bool done;
    bool readNextFrame;
    bool pause;
    bool shouldShowFeatures;

    Scalar curColor;

    nvxcu_coordinates2d_t objectTL;
    nvxcu_coordinates2d_t objectBR;
    nvxcu_coordinates2d_t currentPoint;

    nvxcu::ObjectTrackerWithFeaturesInfo* tracker;
    TrackedObjectPointersVector objects;
    std::vector<Scalar> colors;

    uint32_t frameWidth;
    uint32_t frameHeight;
    nvxcu_pitch_linear_image_t frame;

    EventHandlerData() :
        isPressed(false), done(false), readNextFrame(true),
        pause(false), shouldShowFeatures(false), tracker(nullptr),
        frameWidth(0), frameHeight(0), frame { }
    {
        curColor.values[0] = curColor.values[1] = curColor.values[2] = 0;

        objectTL.x = objectTL.y = 0;
        objectBR.x = objectBR.y = 0;
        currentPoint.x = currentPoint.y = 0;
    }
};

static bool isPointContained(const nvxcu_rectangle_t & rect, const nvxcu_coordinates2d_t & pt)
{
    return (pt.x >= rect.start_x) && (rect.end_x >= pt.x) &&
           (pt.y >= rect.start_y) && (rect.end_y >= pt.y);
}

static void makeValidRect(nvxcu_rectangle_t &rect)
{
    if (rect.end_x < rect.start_x)
    {
        uint32_t tmp = rect.end_x;
        rect.end_x = rect.start_x;
        rect.start_x = tmp;
    }
    if (rect.end_y < rect.start_y)
    {
        uint32_t tmp = rect.end_y;
        rect.end_y = rect.start_y;
        rect.start_y = tmp;
    }
}

static void keyHandler(void* eventData, char key, uint32_t, uint32_t)
{
    EventHandlerData* data = static_cast<EventHandlerData*>(eventData);

    switch (key)
    {
        case 27:
            data->done = true;
            break;
        case 'S':
        case 's':
            if (data->objects.empty())
                data->readNextFrame = true;
            break;
        case 'C':
        case 'c':
            data->tracker->removeAllObjects();
            data->objects.clear();
            data->colors.clear();
            break;
        case 32:
            data->pause = !data->pause;
            break;
        case 'v':
        case 'V':
            data->shouldShowFeatures = !data->shouldShowFeatures;
            break;
    }
}

static void mouseHandler(void* context, nvxio::Render::MouseButtonEvent event, uint32_t x, uint32_t y)
{
    EventHandlerData* data = static_cast<EventHandlerData*>(context);

    nvxcu_rectangle_t frameRoi = {0, 0, data->frameWidth, data->frameHeight};
    nvxcu_coordinates2d_t curPt = {x, y};

    if (event == nvxio::Render::LeftButtonDown && !data->isPressed)
    {
        if (isPointContained(frameRoi, curPt))
        {
            data->objectTL = curPt;
            data->objectBR = {0, 0};
            data->currentPoint = curPt;

            data->isPressed = true;
            data->curColor.values[0] = rand() % 256;
            data->curColor.values[1] = rand() % 256;
            data->curColor.values[2] = rand() % 256;
        }
    }
    else if (event == nvxio::Render::MouseMove && data->isPressed)
    {
        if (isPointContained(frameRoi, curPt))
        {
            data->currentPoint = curPt;
        }
    }
    else if (event == nvxio::Render::LeftButtonUp && data->isPressed)
    {
        if (isPointContained(frameRoi, curPt))
        {
            data->objectBR = curPt;
        }
        else
        {
            data->objectBR = data->currentPoint;
        }

        data->isPressed = false;

        nvxcu_rectangle_t rect = {
            uint32_t(data->objectTL.x), uint32_t(data->objectTL.y),
            uint32_t(data->objectBR.x), uint32_t(data->objectBR.y)
        };
        makeValidRect(rect);

        data->objects.push_back(data->tracker->addObject(rect));
        data->colors.push_back(data->curColor);
    }
}

static void displayState(nvxio::Render *renderer, uint32_t frameWidth, uint32_t frameHeight,
                         const std::string& mode, bool pause, bool useFastDetector)
{
    std::ostringstream txt;
    std::ostringstream consoleLog;

    txt << std::fixed << std::setprecision(1);
    consoleLog << std::fixed << std::setprecision(1);

    txt << "Source size: " << frameWidth << 'x' << frameHeight << std::endl;
    txt << "Detector: " << (useFastDetector ? "FAST" : "Harris") << std::endl;
    txt << "Mode: " << mode << std::endl;

    if (pause)
    {
        txt << "PAUSE" << std::endl;
    }
    else
    {

        txt << consoleLog.str();

        txt << std::setprecision(6);
        txt.unsetf(std::ios_base::floatfield);
        txt << "LIMITED TO " << nvxio::Application::get().getFPSLimit() << " FPS FOR DISPLAY" << std::endl;
    }

    txt << "C - clear objects" << std::endl;
    txt << "V - toggle key points visualization" << std::endl;
    txt << "Space - pause/resume" << std::endl;
    txt << "Esc - close the demo";

    nvxio::Render::TextBoxStyle style = {{255, 255, 255, 255}, {0, 0, 0, 255}, {10, 10}};
    renderer->putTextViewport(txt.str(), style);
}

static bool read(const std::string &nf,  nvxcu::KeypointObjectTrackerParams &config, std::string &message)
{
    std::unique_ptr<nvxio::ConfigParser> ftparser(nvxio::createConfigParser());

    ftparser->addParameter("pyr_levels",nvxio::OptionHandler::unsignedInteger(&config.pyr_levels,
             nvxio::ranges::atLeast(1u) & nvxio::ranges::atMost(8u)));
    ftparser->addParameter("lk_num_iters",nvxio::OptionHandler::unsignedInteger(&config.lk_num_iters,
             nvxio::ranges::atLeast(1u) & nvxio::ranges::atMost(100u)));
    ftparser->addParameter("lk_win_size",nvxio::OptionHandler::unsignedInteger(&config.lk_win_size,
             nvxio::ranges::atLeast(3u) & nvxio::ranges::atMost(32u)));
    ftparser->addParameter("max_corners",nvxio::OptionHandler::unsignedInteger(&config.max_corners,
             nvxio::ranges::atLeast(0u)));
    ftparser->addParameter("detector_cell_size",nvxio::OptionHandler::unsignedInteger(&config.detector_cell_size,
             nvxio::ranges::atLeast(1u) & nvxio::ranges::atMost(719u)));

    ftparser->addParameter("detector", nvxio::OptionHandler::oneOf(&config.use_fast_detector, {
        {"fast", true},
        {"harris", false}
    }));
    ftparser->addParameter("fast_type",nvxio::OptionHandler::unsignedInteger(&config.fast_type,
             nvxio::ranges::atLeast(9u) & nvxio::ranges::atMost(12u)));
    ftparser->addParameter("fast_threshold",nvxio::OptionHandler::unsignedInteger(&config.fast_threshold,
             nvxio::ranges::atLeast(1u) & nvxio::ranges::atMost(254u)));

    ftparser->addParameter("harris_k", nvxio::OptionHandler::real(&config.harris_k,
             nvxio::ranges::moreThan(0.0f)));
    ftparser->addParameter("harris_threshold", nvxio::OptionHandler::real(&config.harris_threshold,
             nvxio::ranges::moreThan(0.0f)));

    ftparser->addParameter("bb_decreasing_ratio",nvxio::OptionHandler::real(&config.bb_decreasing_ratio,
             nvxio::ranges::atLeast(0.f) & nvxio::ranges::atMost(1.f)));
    ftparser->addParameter("max_corners_in_cell",nvxio::OptionHandler::unsignedInteger(&config.max_corners_in_cell,
             nvxio::ranges::atLeast(0u)));
    ftparser->addParameter("x_num_of_cells",nvxio::OptionHandler::unsignedInteger(&config.x_num_of_cells,
             nvxio::ranges::atLeast(0u)));
    ftparser->addParameter("y_num_of_cells",nvxio::OptionHandler::unsignedInteger(&config.y_num_of_cells,
             nvxio::ranges::atLeast(0u)));

    message = ftparser->parse(nf);
    return message.empty();
}

static nvxcu_rectangle_t getBoundingRectangle(const std::vector< nvxcu::ObjectTrackerWithFeaturesInfo::FeaturePoint>& object_feature_points)
{
    nvxcu_rectangle_t result =
    {
        std::numeric_limits<uint32_t>::max(),
        std::numeric_limits<uint32_t>::max(),
        0u,
        0u
    };

    for (const auto& fp: object_feature_points)
    {
        uint32_t x = (uint32_t)fp.point_on_current_frame.x;
        uint32_t y = (uint32_t)fp.point_on_current_frame.y;

        result.start_x = std::min(result.start_x, x);
        result.start_y = std::min(result.start_y, y);
        result.end_x = std::max(result.end_x, x);
        result.end_y = std::max(result.end_y, y);
    }

    return result;
}

static void drawBoundingBoxes(nvxio::Render& renderer, const std::vector<FeaturePointsVector>& objects_features_info)
{
    nvxio::Render::DetectedObjectStyle bounding_box_style =
    {
        std::string(),
        {0, 255, 255, 100},
        1,
        0,
        true
    };

    for (const auto& obj_features: objects_features_info)
    {
        if (obj_features.empty())
        {
            continue;
        }

        nvxcu_rectangle_t bounding_box = getBoundingRectangle(obj_features);
        renderer.putObjectLocation(bounding_box, bounding_box_style);
    }
}

static void drawLines(nvxio::Render& renderer, const std::vector<FeaturePointsVector>& objects_features_info,
                      nvxcu_plain_array_t & lines_array)
{
    std::vector<nvxcu_point4f_t> lines;

    for (const auto& obj_features: objects_features_info)
    {
        for (const auto& fp: obj_features)
        {
            const nvxcu_keypointf_t& kp0 = fp.point_on_previous_frame;
            const nvxcu_keypointf_t& kp1 = fp.point_on_current_frame;
            lines.push_back({kp0.x, kp0.y, kp1.x, kp1.y});
        }
    }

    nvxio::Render::LineStyle line_style = { {255, 0, 0, 110}, 1};

    uint32_t numItems = static_cast<uint32_t>(lines.size());
    size_t arraySize = numItems * sizeof(nvxcu_point4f_t);

    NVXIO_ASSERT(numItems <= lines_array.base.capacity);
    NVXIO_ASSERT(lines_array.base.item_type == NVXCU_TYPE_POINT4F);

    if (!lines.empty())
        NVXIO_CUDA_SAFE_CALL( cudaMemcpy(lines_array.dev_ptr, &lines[0], arraySize, cudaMemcpyHostToDevice) );

    NVXIO_CUDA_SAFE_CALL( cudaMemcpy(lines_array.num_items_dev_ptr, &numItems, sizeof(numItems), cudaMemcpyHostToDevice) );

    renderer.putLines(lines_array, line_style);
}

static std::vector< std::vector<nvxcu_point3f_t> >
prepareCirclesForFeaturePointsDrawing(const std::vector<FeaturePointsVector>& objects_features_info,
                                      const std::vector<FeaturePointsVisualizationStyle>& features_styles)
{
    size_t num_styles = features_styles.size();

    std::vector< std::vector<nvxcu_point3f_t> > circles_vector_for_each_style(num_styles);

    for(const auto& fp_info : objects_features_info)
    {
        for (const auto& fp: fp_info)
        {
            auto iter = std::find_if(features_styles.begin(), features_styles.end(),
                                     [&](const FeaturePointsVisualizationStyle& cur_style)
                                     {
                                         return (fp.weight >= cur_style.min_weight && fp.weight <= cur_style.max_weight);
                                     });

            if (iter != features_styles.end())
            {
                int32_t index_found_style = std::distance(features_styles.begin(), iter);
                const auto& kp = fp.point_on_current_frame;
                circles_vector_for_each_style[index_found_style].push_back({kp.x, kp.y, (float)features_styles[index_found_style].radius});
            }
        }
    }

    return circles_vector_for_each_style;
}

static void drawCircles(nvxio::Render& renderer, const std::vector<FeaturePointsVector>& objects_features_info,
                        nvxcu_plain_array_t & circles_array)
{
    static const std::vector<FeaturePointsVisualizationStyle> features_visualization_styles =
        {
            FeaturePointsVisualizationStyle{0.00, 0.30, {0, 0, 255, 100}, 2, 1},
            FeaturePointsVisualizationStyle{0.30, 0.60, {0, 255, 0, 100}, 2, 1},
            FeaturePointsVisualizationStyle{0.60, 0.85, {255, 0, 0, 100}, 2, 1},
            FeaturePointsVisualizationStyle{0.85, 1.00, {255, 0, 0, 255}, 2, 2}
        };
    size_t num_feature_styles = features_visualization_styles.size();

    std::vector< std::vector<nvxcu_point3f_t> > circles_vector_for_each_style =
            prepareCirclesForFeaturePointsDrawing(objects_features_info,
                                                  features_visualization_styles);

    NVXIO_ASSERT(circles_vector_for_each_style.size() == num_feature_styles);
    NVXIO_ASSERT(circles_array.base.item_type == NVXCU_TYPE_POINT3F);

    for (size_t n = 0; n < num_feature_styles; n++)
    {
        const auto& circles = circles_vector_for_each_style[n];
        const auto& cur_style = features_visualization_styles[n];

        const auto& color = cur_style.color;
        nvxio::Render::CircleStyle circle_style = { {color[0], color[1], color[2], color[3]}, cur_style.thickness};

        uint32_t numItems = static_cast<uint32_t>(circles.size());

        if (!circles.empty())
        {
            size_t arraySize = numItems * sizeof(nvxcu_point3f_t);

            NVXIO_ASSERT(numItems <= circles_array.base.capacity);
            NVXIO_CUDA_SAFE_CALL( cudaMemcpy(circles_array.dev_ptr, &circles[0], arraySize, cudaMemcpyHostToDevice) );
        }

        NVXIO_CUDA_SAFE_CALL( cudaMemcpy(circles_array.num_items_dev_ptr, &numItems, sizeof(numItems), cudaMemcpyHostToDevice) );

        renderer.putCircles(circles_array, circle_style);
    }
}

static FeaturePointsVector convertFeaturePointsInfoToVector(const  nvxcu::ObjectTrackerWithFeaturesInfo::FeaturePointSet& src)
{
    size_t N = src.getSize();
    std::vector< nvxcu::ObjectTrackerWithFeaturesInfo::FeaturePoint> dst(N);
    for (size_t n = 0; n < N; n++)
    {
        dst[n] = src.getFeaturePoint(n);
    }

    return dst;
}

static void drawFeaturePoints(nvxio::Render* renderer, const TrackedObjectPointersVector& objects,
                              nvxcu_plain_array_t & lines_array, nvxcu_plain_array_t & circles_array)
{
    std::vector<FeaturePointsVector> objects_features_info;
    for (const auto& tr_obj : objects)
    {
        NVXIO_ASSERT(tr_obj != nullptr);
        const  nvxcu::ObjectTrackerWithFeaturesInfo::FeaturePointSet& obj_fp_info = tr_obj->getFeaturePointSet();
        objects_features_info.push_back( convertFeaturePointsInfoToVector(obj_fp_info) );
    }

    drawBoundingBoxes(*renderer, objects_features_info);
    drawLines        (*renderer, objects_features_info, lines_array);
    drawCircles      (*renderer, objects_features_info, circles_array);
}


int main(int argc, char* argv[])
{
    srand(0);

    try
    {
        nvxio::Application &app = nvxio::Application::get();

        std::string configFile = app.findSampleFilePath("object_tracker_nvxcu_sample_config.ini");
        std::string defaultSourceUri = app.findSampleFilePath("cars.mp4");
        std::string sourceUri = defaultSourceUri;

        app.setDescription("This demo demonstrates Object Tracker algorithm");
        app.addOption('s', "source", "Input URI", nvxio::OptionHandler::string(&sourceUri));
        app.init(argc, argv);

        nvxcu::KeypointObjectTrackerParams params;

        std::string msg;
        if (!read(configFile, params, msg))
        {
            std::cout << msg << std::endl;
            return nvxio::Application::APP_EXIT_CODE_INVALID_VALUE;
        }

        bool trackPreconfiguredObject = sourceUri == defaultSourceUri;

        // define preconfigured object to track
        nvxcu_rectangle_t _initialObjectRect = {670, 327, 710, 363};

        std::unique_ptr<nvxio::FrameSource> source(nvxio::createDefaultFrameSource(sourceUri));

        if (!source || !source->open())
        {
            std::cerr << "Can't open source URI " << sourceUri << std::endl;
            return nvxio::Application::APP_EXIT_CODE_NO_RESOURCE;
        }

        if (source->getSourceType() == nvxio::FrameSource::SINGLE_IMAGE_SOURCE)
        {
            std::cerr << "Can't work on a single image." << std::endl;
            return nvxio::Application::APP_EXIT_CODE_INVALID_FORMAT;
        }

        nvxio::FrameSource::Parameters sourceParams = source->getConfiguration();

        nvxcu_pitch_linear_image_t grayScaleFrame = createImage(sourceParams.frameWidth, sourceParams.frameHeight,
                                                                NVXCU_DF_IMAGE_U8);

        nvxcu_plain_array_t lines_array = createArray(MAX_NUMBER_OF_ELEMENTS_IN_ARRAYS, NVXCU_TYPE_POINT4F);
        nvxcu_plain_array_t circles_array = createArray(MAX_NUMBER_OF_ELEMENTS_IN_ARRAYS, NVXCU_TYPE_POINT3F);

        std::unique_ptr<nvxio::Render> renderer(nvxio::createDefaultRender("Object Tracker Sample",
            sourceParams.frameWidth, sourceParams.frameHeight, sourceParams.format));

        if (!renderer)
        {
            std::cerr << "Can't create a renderer." << std::endl;
            return nvxio::Application::APP_EXIT_CODE_NO_RENDER;
        }

        EventHandlerData eventData;
        eventData.frameWidth = sourceParams.frameWidth;
        eventData.frameHeight = sourceParams.frameHeight;
        eventData.frame = createImage(sourceParams.frameWidth, sourceParams.frameHeight, sourceParams.format);

        std::string mode = "keypoint";

        std::unique_ptr<nvxcu::ObjectTrackerWithFeaturesInfo> tracker(
            nvxcuCreateKeypointObjectTrackerWithFeaturesInfo(params));

        if (!tracker)
        {
            std::cerr << "Error: Can't initialize object tracker algorithm." << std::endl;
            return nvxio::Application::APP_EXIT_CODE_CAN_NOT_CREATE;
        }

        eventData.tracker = tracker.get();

        renderer->setOnKeyboardEventCallback(keyHandler, &eventData);
        renderer->setOnMouseEventCallback(mouseHandler, &eventData);

        if (trackPreconfiguredObject)
        {
            // Set a preconfigured object to track
            nvxcu_rectangle_t initialObjectRect = _initialObjectRect;

            tracker->addObject(initialObjectRect);
            eventData.objects.push_back(eventData.tracker->addObject(initialObjectRect));
            eventData.colors.push_back({{0, 255, 0}});
        }


        std::unique_ptr<nvxio::SyncTimer> syncTimer = nvxio::createSyncTimer();
        syncTimer->arm(1. / app.getFPSLimit());

        int32_t deviceID = -1;
        NVXIO_CUDA_SAFE_CALL( cudaGetDevice(&deviceID) );

        nvxcu_stream_exec_target_t exec_target = { };
        exec_target.base.exec_target_type = NVXCU_STREAM_EXEC_TARGET;
        exec_target.stream = nullptr;
        NVXIO_CUDA_SAFE_CALL( cudaGetDeviceProperties(&exec_target.dev_prop, deviceID) );

        while (!eventData.done)
        {
            syncTimer->synchronize();

            if (eventData.readNextFrame)
            {
                nvxio::FrameSource::FrameStatus status = source->fetch(eventData.frame);

                NVXIO_ASSERT( nvxcuColorConvert(&eventData.frame.base, &grayScaleFrame.base,
                                                NVXCU_COLOR_SPACE_DEFAULT, NVXCU_CHANNEL_RANGE_FULL,
                                                &exec_target.base) == NVXCU_SUCCESS );
                NVXIO_CUDA_SAFE_CALL( cudaStreamSynchronize(exec_target.stream) );

                if (status == nvxio::FrameSource::TIMEOUT) continue;
                if (status == nvxio::FrameSource::CLOSED)
                {
                    eventData.objects.clear();
                    eventData.colors.clear();
                    tracker->removeAllObjects();

                    if (!source->open())
                    {
                        std::cerr << "Failed to reopen the source" << std::endl;
                        break;
                    }
                    continue;
                }
            }

            renderer->putImage(eventData.frame);

            //
            // Manage Manual Object Selection
            //

            if (eventData.isPressed)
            {
                nvxcu_rectangle_t rect =
                {
                    uint32_t(eventData.objectTL.x), uint32_t(eventData.objectTL.y),
                    uint32_t(eventData.currentPoint.x), uint32_t(eventData.currentPoint.y)
                };

                makeValidRect(rect);
                nvxio::Render::DetectedObjectStyle style =
                {
                    "", {eventData.curColor.values[0], eventData.curColor.values[1],
                    eventData.curColor.values[2], 255}, 2, false
                };

                renderer->putObjectLocation(nvxcu_rectangle_t{rect.start_x, rect.start_y, rect.end_x, rect.end_y}, style);
            }

            if (eventData.objects.empty())
            {
                eventData.readNextFrame = false;

                nvxio::Render::TextBoxStyle style = {{255, 255, 255, 255}, {0, 0, 0, 255}, {10, 10}};
                renderer->putTextViewport("Objects were lost\n"
                                          "Please select a bounding box to track\n"
                                          "S - skip current frame\n"
                                          "Esc - close the demo",
                                          style);
            }
            else
            {
                eventData.readNextFrame = !eventData.isPressed && !eventData.pause;

                //
                // Perform Object Tracking
                //

                if (!eventData.pause)
                    tracker->process(&grayScaleFrame);

                //
                // Filter tracked objects
                //

                auto colorIt = eventData.colors.begin();
                for (auto it = eventData.objects.begin(); it != eventData.objects.end();)
                {
                    nvxcu::ObjectTrackerWithFeaturesInfo::TrackedObject* obj = *it;

                    nvxcu::ObjectTracker::ObjectStatus status = obj->getStatus();
                    nvxcu_rectangle_t rect = obj->getLocation();
                    unsigned int area = (rect.end_x - rect.start_x) * (rect.end_y - rect.start_y);
                    unsigned int frameArea = eventData.frameWidth*eventData.frameHeight;

                    if (status !=  nvxcu::ObjectTracker::LOST && area < frameArea / 3)
                    {
                        ++it;
                        ++colorIt;
                    }
                    else
                    {
                        it = eventData.objects.erase(it);
                        colorIt = eventData.colors.erase(colorIt);
                        tracker->removeObject(obj);
                    }
                }

                //
                // Rendering
                //

                for (size_t i = 0; i < eventData.objects.size(); i++)
                {
                    nvxcu_rectangle_t rect = eventData.objects[i]->getLocation();

                    const Scalar &c = eventData.colors[i];
                    nvxio::Render::DetectedObjectStyle style = {
                        "", {c.values[0], c.values[1], c.values[2], 255}, 2, false
                    };
                    renderer->putObjectLocation(nvxcu_rectangle_t{rect.start_x, rect.start_y, rect.end_x, rect.end_y}, style);
                }

                if (eventData.shouldShowFeatures)
                {
                    drawFeaturePoints(renderer.get(), eventData.objects,
                                      lines_array, circles_array);
                }

                displayState(renderer.get(), eventData.frameWidth, eventData.frameHeight,
                             mode, eventData.pause, params.use_fast_detector);

            }

            if (!renderer->flush())
            {
                eventData.done = true;
            }
        }

        releaseImage(&grayScaleFrame);
        releaseImage(&eventData.frame);

        releaseArray(&lines_array);
        releaseArray(&circles_array);
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return nvxio::Application::APP_EXIT_CODE_ERROR;
    }

    return nvxio::Application::APP_EXIT_CODE_SUCCESS;
}
