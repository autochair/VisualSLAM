Copyright (c) 2014-2016, NVIDIA CORPORATION.  All rights reserved.

CUDA Layer Object Tracker Sample App
@brief CUDA Layer Object Tracker Sample user guide.

## Introduction ##

`nvx_sample_object_tracker_nvxcu` demonstrates a simple tracking approach based on pyramidal
Optical Flow with advanced tracker for non-rigid objects.

`nvx_sample_object_tracker_nvxcu` is installed in the following directory:

    /usr/share/visionworks/sources/demos/object_tracker_nvxcu/

For the steps to build sample applications, see see: nvx_samples_and_demos. The
specific section name depends on your OS.

## Executing the CUDA Layer Object Tracker Sample ##

    ./nvx_sample_object_tracker_nvxcu [options]

### Command Line Options ###

This topic provides a list of supported options and the values they consume.

#### \-s, \--source ####
- Parameter: [inputUri]
- Description: Specifies the input URI. Accepted parameters include a video (in .avi format), an image sequence (.png, .jpg, .jpeg, .bmp, .tiff) or camera to grab frames.
- Usage:
  - \--source=/path/to/video.avi for video
  - \--source=/path/to/image_%04d_sequence.png for image sequence
  - \--source=device://camera0 for the first camera
  - \--source=device://camera1 for the second camera.

#### -h, \--help ####
- Parameter: true
- Description: Prints the help message.

### Operational Keys ###
- Use the mouse to select objects to track.
- Use `C` to clear the objects.
- Use `V` to toggle visualization of the key points.
- Use `S` to skip the current frame (if there are no objects that are currently tracking).
- Use `Space` to pause/resume.
- Use `ESC` to close the sample.

## Input Data ##
- config `*.ini` file (default URI: `/path_to_vw_samples_and_demos_sources/VisionWorks-${NVX_VERSION}-Samples/data/object_tracker_nvxcu_sample_config.ini`)
   It has the following structure:
   - **pyr_levels**
     - Parameter: [integer in range [1..8]]
     - Description: Number of levels for Gaussian pyramid in Lucas-Kanade Optical Flow algorithm.

   - **lk_num_iters**
    - Parameter: [integer > 0]
    - Description: Number of iterations in Lucas-Kanade
    Optical Flow algorithm.

   - **lk_win_size**
    - Parameter: [integer in range [3..32]]
    - Description: Window size in Lucas-Kanade Optical Flow algorithm.

   - **detector_cell_size**
    - Parameter: [integer < the smaller one of input image dimensions]
    - Description: Specifies the size of cells for cell-based non-max suppression.

   - **max_corners**
    - Parameter: [non-zero, positive integer]
    - Description: Total number of detected corners in the image.

   - **detector**
    - Parameter: [harris or fast]
    - Description: The corner detector algorithm. Default is "fast".

   - **fast_type**
    - Parameter: [integer in range [9..12]]
    - Description: Specifies the number of neighborhoods to test.

   - **fast_threshold**
    - Parameter: [integer value less than 255]
    - Description: Specifies the threshold difference between intensity of the central pixel and pixels of a circle around this pixel.

   - **harris_k**
    - Parameter: [floating point value greater than zero]
    - Description: The Harris corner detector "k" parameter. Default is 0.04.

   - **harris_threshold**
    - Parameter: [floating point value greater than zero]
    - Description: The Harris corner detector threshold. Default is 100.0.

   - **bb_decreasing_ratio**
    - Parameter: [floating-point  in range [0.0..1.0]]
    - Description: Specifies the ratio between decreased and initial bounding box of object.

   - **max_corners_in_cell**
    - Parameter: [non-zero, positive integer]
    - Description: Total number of detected corners per cell in object's bounding box after filtering.

   - **x_num_of_cells**
    - Parameter: [non-zero, positive integer]
    - Description: Number of cells along the x-axis to split bounding box for corners filtering.

   - **y_num_of_cells**
    - Parameter: [non-zero, positive integer]
    - Description: Number of cells along the y-axis to split bounding box for corners filtering.

- video sequence with "tracking" use-case (default URI: `/path_to_vw_samples_and_demos_sources/VisionWorks-${NVX_VERSION}-Samples/data/cars.mp4`)

### Tracker Algorithm Overview
Tracker uses the FAST algorithm to find corners and passes them through the Optical Flow to the next frame.
Corners it finds are filtered by motion distance, direction and Optical Flow precision to remove
outliers. With the rest corners, named "key points" below, the algorithm tries to map each key point with
key points tracked from the previous frames to set the weights of the new key points.
Then using these key points for each object, the bounding box center is found, and then the bounding box
scale is estimated.

### FAST Algorithm
Features from accelerated segment test (FAST) is a corner detection method
originally developed by Edward Rosten and Tom Drummond. For more information, see:

http://www.edwardrosten.com/work/rosten_2006_machine.pdf

### Optical Flow
Optical flow is applied with the classical differential Lucas-Kanade method used both ways to compare
each point before starting and after comeback. If the difference for a corner is too big, this corner is
ignored as an outlier. For more information, see:

http://cseweb.ucsd.edu/classes/sp02/cse252/lucaskanade81.pdf
http://robots.stanford.edu/cs223b04/algo_tracking.pdf

### Tracking of Key Points
Each corner that has not been filtered out as an outlier is mapped to the tracked key points from the
previous frame. If for a corner the difference to the closest key point from the previous frame is less
than a threshold, then the algorithm supposes that it is the same key point, updates its position (using
optical flow results), and increases its weight.

### Key Points Weighting
Weight for each key point is determined by the number of frames this point has been tracked before the
current frame.

### Bounding Box Estimation
Object bounding box center is determined as the mass center for all key points with their weight as mass.
Bounding box scale estimation is found as follows:
* the distance between each two key points on the current frame is calculated,
* the distance between each two key points on the previous frame is calculated,
* the ratio between these distances is calculated for each pair of key points,
* the scale estimation is the median in the array of all the ratios.

### FAST corners filtering
To reduce time of bounding box estimation, the procedure of preliminary corners filtering could be applied to
each bounding box. It is done as follows: the object's bounding box is split into several cells. All FAST corners
in cells are sorted in descending order of corners strength. Then for each cell, the number of corners are reduced,
dropping ones with the lower strength.

### Loose Criterion
Tracker loses object in the following case: If the suitable key points count becomes too small, then the
object's status becomes "LOST" and will not be tracked further.

### Tracker Use Cases
This key point tracker is sharpened for short-term tracking cars on the road.

Expected use cases:

- To track cars going towards the camera
- To track car being followed by thecamera
- To track cars overtaken by the camera

It is also a good way to track rigid shapes (e.g., car front or back side).
@note When selecting a rect for track, select it as close as possible to
avoid picking up corners on the background. Also, for this point, parameter
`bb_decreasing_ratio` can be used. It is responsible for the ratio between the
selected rect and the rect used for tracking. Set it to less than 1 to not
search for key points close to rect borders (so that background points
are not tracked). Note that setting `bb_decreasing_ratio` to zero is quite
meaningless since no key points can be found in the zero-sized bounding
box.

