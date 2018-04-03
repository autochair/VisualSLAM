Copyright (c) 2014-2016, NVIDIA CORPORATION.  All rights reserved.

Video Playback Sample App
@brief Video Playback Sample user guide.

## Introduction ##

`nvx_sample_player` demonstrates basic image and video I/O facilities and camera access.

`nvx_sample_player` is installed in the following directory:

    /usr/share/visionworks/sources/samples/player

For the steps to build sample applications, see the see: nvx_samples_and_demos section for your OS.

## Executing the Player Sample ##

    ./nvx_sample_player [options]

### Command Line Options ###

This topic provides a list of supported options and the values they consume.

#### \-s, \--source ####
- Parameter: [inputUri]
- Description: Specifies the input URI. Accepted parameters include a video (in .avi format), an image or an image sequence (.png, .jpg, .jpeg, .bmp, .tiff), camera to grab frames.
- Usage:

    - `--source=/path/to/video.avi` for video
    - `--source=/path/to/image.png` for image
    - `--source=/path/to/image_%04d_sequence.png` for image sequence
    - `--source="device:///nvcamera?index=0"` for the GStreamer NVIDIA camera (Jetson TX1 only).

#### \-h, \--help ####
- Description: Prints the help message.

### Operational Keys ###
- Use `Space` to pause/resume the sample.
- Use `ESC` to close the sample.

