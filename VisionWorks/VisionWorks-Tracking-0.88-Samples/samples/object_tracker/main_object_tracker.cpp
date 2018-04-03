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
#include <NVX/nvx_timer.hpp>

#include <NVX/Application.hpp>
#include <NVX/ConfigParser.hpp>
#include <OVX/FrameSourceOVX.hpp>
#include <OVX/RenderOVX.hpp>
#include <NVX/SyncTimer.hpp>
#include <OVX/UtilityOVX.hpp>

#include "NVX/tracking/tracking.hpp"
#include "NVX/tracking/tracking_with_features_info.hpp"

struct Scalar
{
    vx_uint8 values[3];
};

typedef std::vector<ObjectTrackerWithFeaturesInfo::FeaturePoint> FeaturePointsVector;
typedef std::vector<ObjectTrackerWithFeaturesInfo::TrackedObject*> TrackedObjectPointersVector;

struct FeaturePointsVisualizationStyle
{
    float min_weight;
    float max_weight;
    vx_uint8 color[4];
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

    vx_coordinates2d_t objectTL;
    vx_coordinates2d_t objectBR;
    vx_coordinates2d_t currentPoint;

    ObjectTrackerWithFeaturesInfo* tracker;
    TrackedObjectPointersVector objects;
    std::vector<Scalar> colors;

    vx_uint32 frameWidth;
    vx_uint32 frameHeight;
    vx_image frame;

    EventHandlerData() : isPressed{false}, done{false}, readNextFrame{true}, pause{false}, shouldShowFeatures{false},
                         objectTL{0,0}, objectBR{0,0}, currentPoint{0,0}, tracker{NULL},
                         frameWidth{0}, frameHeight{0}, frame{NULL}
    {
    }
};

static bool isPointContained(vx_rectangle_t rect, vx_coordinates2d_t pt)
{
    return (( (pt.x >= rect.start_x) && (rect.end_x >= pt.x)) && ((pt.y >= rect.start_y) && (rect.end_y >= pt.y)));
}

static void makeValidRect(vx_rectangle_t &rect)
{
    if(rect.end_x < rect.start_x)
    {
        vx_uint32 tmp = rect.end_x;
        rect.end_x = rect.start_x;
        rect.start_x = tmp;
    }
    if(rect.end_y < rect.start_y)
    {
        vx_uint32 tmp = rect.end_y;
        rect.end_y = rect.start_y;
        rect.start_y = tmp;
    }
}

static void keyHandler(void* eventData, vx_char key, vx_uint32, vx_uint32)
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

static void mouseHandler(void* context, ovxio::Render::MouseButtonEvent event, vx_uint32 x, vx_uint32 y)
{
    EventHandlerData* data = static_cast<EventHandlerData*>(context);

    vx_rectangle_t frameRoi = {0, 0, data->frameWidth, data->frameHeight};
    vx_coordinates2d_t curPt = {x, y};

    if (event == ovxio::Render::LeftButtonDown && !data->isPressed)
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
    else if (event == ovxio::Render::MouseMove && data->isPressed)
    {
        if (isPointContained(frameRoi, curPt))
        {
            data->currentPoint = curPt;
        }
    }
    else if (event == ovxio::Render::LeftButtonUp && data->isPressed)
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

        vx_rectangle_t rect = {
            vx_uint32(data->objectTL.x), vx_uint32(data->objectTL.y),
            vx_uint32(data->objectBR.x), vx_uint32(data->objectBR.y)
        };

        makeValidRect(rect);

        data->objects.push_back(data->tracker->addObject(rect));
        data->colors.push_back(data->curColor);
    }
}

static void displayState(ovxio::Render *renderer, const vx_uint32 &frameWidth, const vx_uint32 &frameHeight,
                         double proc_ms, double total_ms, const std::string& mode, bool pause)
{
    std::ostringstream txt;
    std::ostringstream consoleLog;

    txt << std::fixed << std::setprecision(1);
    consoleLog << std::fixed << std::setprecision(1);

    txt << "Source size: " << frameWidth << 'x' << frameHeight << std::endl;
    txt << "Mode: " << mode << std::endl;

    if (pause)
    {
        txt << "PAUSE" << std::endl;
    }
    else
    {
        consoleLog << "Algorithm: " << proc_ms << " ms / " << 1000.0 / proc_ms << " FPS" << std::endl;
        consoleLog << "Display: " << total_ms  << " ms / " << 1000.0 / total_ms << " FPS" << std::endl;

        txt << consoleLog.str();

        txt << std::setprecision(6);
        txt.unsetf(std::ios_base::floatfield);
        txt << "LIMITED TO " << nvxio::Application::get().getFPSLimit() << " FPS FOR DISPLAY" << std::endl;
    }

    txt << "C - clear objects" << std::endl;
    txt << "V - toggle key points visualization" << std::endl;
    txt << "Space - pause/resume" << std::endl;
    txt << "Esc - close the demo";

    ovxio::Render::TextBoxStyle style = {{255, 255, 255, 255}, {0, 0, 0, 255}, {10, 10}};
    renderer->putTextViewport(txt.str(), style);
}

static bool read(const std::string &nf, KeypointObjectTrackerParams &config, std::string &message)
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
    ftparser->addParameter("strength_threshold",nvxio::OptionHandler::real(&config.strength_threshold,
             nvxio::ranges::atLeast(1.f) & nvxio::ranges::atMost(254.f)));
    ftparser->addParameter("fast_type",nvxio::OptionHandler::unsignedInteger(&config.fast_type,
             nvxio::ranges::atLeast(9u) & nvxio::ranges::atMost(12u)));
    ftparser->addParameter("detector_cell_size",nvxio::OptionHandler::unsignedInteger(&config.detector_cell_size,
             nvxio::ranges::atLeast(1u) & nvxio::ranges::atMost(719u)));
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

static vx_rectangle_t getBoundingRectangle(const std::vector<ObjectTrackerWithFeaturesInfo::FeaturePoint>& object_feature_points)
{
    vx_rectangle_t result = {std::numeric_limits<vx_uint32>::max(), std::numeric_limits<vx_uint32>::max(), 0, 0};
    for (const auto& fp: object_feature_points)
    {
        vx_uint32 x = (vx_uint32)fp.point_on_current_frame.x;
        vx_uint32 y = (vx_uint32)fp.point_on_current_frame.y;

        result.start_x = std::min(result.start_x, x);
        result.start_y = std::min(result.start_y, y);
        result.end_x = std::max(result.end_x, x);
        result.end_y = std::max(result.end_y, y);
    }

    return result;
}

static void drawBoundingBoxes(ovxio::Render& renderer, const std::vector<FeaturePointsVector>& objects_features_info)
{
    ovxio::Render::DetectedObjectStyle bounding_box_style
        = {
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

        vx_rectangle_t bounding_box = getBoundingRectangle(obj_features);
        renderer.putObjectLocation(bounding_box, bounding_box_style);
    }
}

static void drawLines(vx_context context, ovxio::Render& renderer, const std::vector<FeaturePointsVector>& objects_features_info)
{
    std::vector<nvx_point4f_t> lines;

    for (const auto& obj_features: objects_features_info)
    {
        for (const auto& fp: obj_features)
        {
            const nvx_keypointf_t& kp0 = fp.point_on_previous_frame;
            const nvx_keypointf_t& kp1 = fp.point_on_current_frame;
            lines.push_back({kp0.x, kp0.y, kp1.x, kp1.y});
        }
    }

    ovxio::Render::LineStyle line_style = { {255, 0, 0, 110}, 1};

    vx_array lines_array = vxCreateArray(context, NVX_TYPE_POINT4F, MAX_NUMBER_OF_ELEMENTS_IN_ARRAYS);
    NVXIO_CHECK_REFERENCE(lines_array);

    if (!lines.empty())
    {
        NVXIO_SAFE_CALL( vxAddArrayItems(lines_array, lines.size(), lines.data(), sizeof(lines[0])) );
    }
    renderer.putLines(lines_array, line_style);

    vxReleaseArray(&lines_array);
}

static std::vector< std::vector<nvx_point3f_t> >
prepareCirclesForFeaturePointsDrawing(const std::vector<FeaturePointsVector>& objects_features_info,
                                      const std::vector<FeaturePointsVisualizationStyle>& features_styles)
{
    size_t num_styles = features_styles.size();

    std::vector< std::vector<nvx_point3f_t> > circles_vector_for_each_style(num_styles);

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
                vx_int32 index_found_style = std::distance(features_styles.begin(), iter);
                const auto& kp = fp.point_on_current_frame;
                circles_vector_for_each_style[index_found_style].push_back({kp.x, kp.y, (float)features_styles[index_found_style].radius});
            }
        }
    }
    return circles_vector_for_each_style;
}

static void drawCircles(vx_context context, ovxio::Render& renderer, const std::vector<FeaturePointsVector>& objects_features_info)
{
    static const std::vector<FeaturePointsVisualizationStyle> features_visualization_styles =
        {
            FeaturePointsVisualizationStyle{0.00, 0.30, {0, 0, 255, 100}, 2, 1},
            FeaturePointsVisualizationStyle{0.30, 0.60, {0, 255, 0, 100}, 2, 1},
            FeaturePointsVisualizationStyle{0.60, 0.85, {255, 0, 0, 100}, 2, 1},
            FeaturePointsVisualizationStyle{0.85, 1.00, {255, 0, 0, 255}, 2, 2}
        };
    size_t num_feature_styles = features_visualization_styles.size();

    vx_array circles_array = vxCreateArray(context, NVX_TYPE_POINT3F, MAX_NUMBER_OF_ELEMENTS_IN_ARRAYS);

    std::vector< std::vector<nvx_point3f_t> > circles_vector_for_each_style
        = prepareCirclesForFeaturePointsDrawing(objects_features_info,
                                                features_visualization_styles);

    assert(circles_vector_for_each_style.size() == num_feature_styles);

    for (size_t n = 0; n < num_feature_styles; n++)
    {
        const auto& circles = circles_vector_for_each_style[n];
        const auto& cur_style = features_visualization_styles[n];

        const auto& color = cur_style.color;
        ovxio::Render::CircleStyle circle_style = { {color[0], color[1], color[2], color[3]}, cur_style.thickness};

        NVXIO_SAFE_CALL( vxTruncateArray(circles_array, 0) );

        if (!circles.empty())
        {
            NVXIO_SAFE_CALL( vxAddArrayItems(circles_array, circles.size(), circles.data(), sizeof(circles[0])) );
        }

        renderer.putCircles(circles_array, circle_style);
    }

    vxReleaseArray(&circles_array);
}

static FeaturePointsVector convertFeaturePointsInfoToVector(const ObjectTrackerWithFeaturesInfo::FeaturePointSet& src)
{
    size_t N = src.getSize();
    std::vector<ObjectTrackerWithFeaturesInfo::FeaturePoint> dst(N);
    for (size_t n = 0; n < N; n++)
    {
        dst[n] = src.getFeaturePoint(n);
    }

    return dst;
}

static void drawFeaturePoints(vx_context context, ovxio::Render* renderer, const TrackedObjectPointersVector& objects)
{
    std::vector<FeaturePointsVector> objects_features_info;
    for (const auto& tr_obj : objects)
    {
        assert(tr_obj != NULL);
        const ObjectTrackerWithFeaturesInfo::FeaturePointSet& obj_fp_info = tr_obj->getFeaturePointSet();
        objects_features_info.push_back( convertFeaturePointsInfoToVector(obj_fp_info) );
    }

    drawBoundingBoxes(*renderer, objects_features_info);
    drawLines        (context, *renderer, objects_features_info);
    drawCircles      (context, *renderer, objects_features_info);
}


int main(int argc, char* argv[])
{
    srand(0);
    try
    {
        nvxio::Application &app = nvxio::Application::get();

        std::string configFile = app.findSampleFilePath("tracking/tracking_config.ini");
        std::string defaultSourceUri = app.findSampleFilePath("tracking/cars.mp4");
        std::string sourceUri = defaultSourceUri;

        app.setDescription("This demo demonstrates Object Tracker algorithm");
        app.addOption('s', "source", "Input URI", nvxio::OptionHandler::string(&sourceUri));
        app.init(argc, argv);

        KeypointObjectTrackerParams params;

        std::string msg;
        if (!read(configFile, params, msg))
        {
            std::cout << msg << std::endl;
            return nvxio::Application::APP_EXIT_CODE_INVALID_VALUE;
        }

        nvx_module_version_t trackingVersion;
        nvxTrackingGetVersion(&trackingVersion);
        std::cout << "VisionWorks Object Tracker version: " << trackingVersion.major << "." << trackingVersion.minor
                  << "." << trackingVersion.patch << trackingVersion.suffix << std::endl << std::endl;

        bool trackPreconfiguredObject = sourceUri == defaultSourceUri;

        //define preconfigured object to track
        vx_rectangle_t _initialObjectRect = {670, 327, 710, 363};

        // OpenVX context
        ovxio::ContextGuard context;
        vxDirective((vx_reference)(vx_context)context, NVX_DIRECTIVE_PERFORMANCE_DISABLE);

        vxRegisterLogCallback(context, &ovxio::stdoutLogCallback, vx_false_e);

        std::unique_ptr<ovxio::FrameSource> source(ovxio::createDefaultFrameSource(context, sourceUri));

        if (!source || !source->open()) {
            std::cerr << "Can't open source URI " << sourceUri << std::endl;
            return nvxio::Application::APP_EXIT_CODE_NO_RESOURCE;
        }

        if (source->getSourceType() == ovxio::FrameSource::SINGLE_IMAGE_SOURCE)
        {
            std::cerr << "Can't work on a single image." << std::endl;
            return nvxio::Application::APP_EXIT_CODE_INVALID_FORMAT;
        }

        ovxio::FrameSource::Parameters sourceParams = source->getConfiguration();

        std::unique_ptr<ovxio::Render> renderer(ovxio::createDefaultRender(context, "Object Tracker Sample",
            sourceParams.frameWidth, sourceParams.frameHeight, sourceParams.format));

        if (!renderer) {
            std::cerr << "Can't create a renderer." << std::endl;
            return nvxio::Application::APP_EXIT_CODE_NO_RENDER;
        }

        EventHandlerData eventData;
        eventData.frameWidth = sourceParams.frameWidth;
        eventData.frameHeight = sourceParams.frameHeight;
        eventData.frame = vxCreateImage(context, sourceParams.frameWidth, sourceParams.frameHeight, sourceParams.format);
        NVXIO_CHECK_REFERENCE(eventData.frame);

        std::string mode = "keypoint";
        std::unique_ptr<ObjectTrackerWithFeaturesInfo> tracker(nvxTrackingCreateKeypointObjectTrackerWithFeaturesInfo(context, params));

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
            //set an preconfigured object to track
            vx_rectangle_t initialObjectRect = _initialObjectRect;
            eventData.objects.push_back(eventData.tracker->addObject(initialObjectRect));
            eventData.colors.push_back({{0, 255, 0}});
        }

        nvx::Timer totalTimer;
        totalTimer.tic();

        std::unique_ptr<nvxio::SyncTimer> syncTimer = nvxio::createSyncTimer();
        syncTimer->arm(1. / app.getFPSLimit());

        while (!eventData.done)
        {
            double total_ms_nolim = totalTimer.toc();
            syncTimer->synchronize();
            double total_ms = totalTimer.toc();
            totalTimer.tic();

            if (eventData.readNextFrame)
            {
                ovxio::FrameSource::FrameStatus status = source->fetch(eventData.frame);


                if (status == ovxio::FrameSource::TIMEOUT) continue;
                if (status == ovxio::FrameSource::CLOSED) {
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

            if (eventData.isPressed) {
                vx_rectangle_t rect = {
                    vx_uint32(eventData.objectTL.x), vx_uint32(eventData.objectTL.y),
                    vx_uint32(eventData.currentPoint.x), vx_uint32(eventData.currentPoint.y)
                };
                makeValidRect(rect);
                ovxio::Render::DetectedObjectStyle style = {
                    "", {eventData.curColor.values[0], eventData.curColor.values[1],
                    eventData.curColor.values[2], 255}, 2, false
                };
                renderer->putObjectLocation(rect, style);
            }

            if (eventData.objects.empty())
            {
                eventData.readNextFrame = false;
                std::ostringstream txt;

                txt << std::fixed << std::setprecision(1);

                if (eventData.pause)
                {
                    txt << "PAUSE" << std::endl;
                }

                txt << "Objects were lost"<< std::endl;
                txt << "Please select a bounding box to track" << std::endl;
                txt << "S - skip current frame" << std::endl;
                txt << "V - toggle key points visualization" << std::endl;
                txt << "Space - pause/resume" << std::endl;

                ovxio::Render::TextBoxStyle style = {{255, 255, 255, 255}, {0, 0, 0, 255}, {10, 10}};
                renderer->putTextViewport(txt.str(), style);
            }
            else
            {
                eventData.readNextFrame = !eventData.isPressed && !eventData.pause;
                nvx::Timer procTimer;
                procTimer.tic();

                if (!eventData.pause)
                    tracker->process(eventData.frame);
                const double proc_ms = procTimer.toc();

                auto colorIt = eventData.colors.begin();
                for (auto it = eventData.objects.begin(); it != eventData.objects.end();)
                {
                    ObjectTrackerWithFeaturesInfo::TrackedObject* obj = *it;

                    ObjectTracker::ObjectStatus status = obj->getStatus();
                    vx_rectangle_t rect = obj->getLocation();
                    unsigned int area = (rect.end_x - rect.start_x) * (rect.end_y - rect.start_y);
                    unsigned int frameArea = eventData.frameWidth*eventData.frameHeight;

                    if (status != ObjectTracker::LOST && area < frameArea / 3)
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

                for (size_t i = 0; i < eventData.objects.size(); i++)
                {
                    vx_rectangle_t rect = eventData.objects[i]->getLocation();

                    const Scalar &c = eventData.colors[i];
                    ovxio::Render::DetectedObjectStyle style = {
                        "", {c.values[0], c.values[1], c.values[2], 255}, 2, false
                    };
                    renderer->putObjectLocation(rect, style);
                }

                if (eventData.shouldShowFeatures)
                {
                    drawFeaturePoints(context, renderer.get(), eventData.objects);
                }

                displayState(renderer.get(), eventData.frameWidth, eventData.frameHeight, proc_ms, total_ms, mode, eventData.pause);

                if (!eventData.pause)
                {
                    std::cout << "Graph Time : " << proc_ms << " ms" << std::endl;
                }
                std::cout << "Display Time : " << total_ms_nolim << " ms" << std::endl << std::endl;
            }


            if (!renderer->flush())
            {
                eventData.done = true;
            }
        }

        vxReleaseImage(&eventData.frame);
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return nvxio::Application::APP_EXIT_CODE_ERROR;
    }

    return nvxio::Application::APP_EXIT_CODE_SUCCESS;
}
