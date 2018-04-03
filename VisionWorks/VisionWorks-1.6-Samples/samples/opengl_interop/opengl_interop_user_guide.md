Copyright (c) 2015-2016, NVIDIA CORPORATION.  All rights reserved.

OpenGL Interop Sample App
@brief OpenGL Interop Sample user guide.

## Introduction ##

`nvx_sample_opengl_interop` shows the interoperability of VisionWorks with OpenGL.

`nvx_sample_opengl_interop` is installed in the following directory:

    /usr/share/visionworks/sources/samples/opengl_interop

For the steps to build sample applications, see the see: nvx_samples_and_demos section for your OS.

## Executing the OpenGL Interop Sample ##

    ./nvx_sample_opengl_interop [options]

### Command Line Options ###

This topic provides a list of supported options and the values they consume.

#### \-s, \--source ####
- Parameter: [inputUri]
- Description: Specifies the input URI. Accepted parameters include a video (in .avi format), an image or an image sequence (.png, .jpg, .jpeg, .bmp, .tiff), camera to grab frames.
- Usage:

    - `--source=/path/to/video.avi` for video
    - `--source=/path/to/image.png` for image
    - `--source=/path/to/image_%04d_sequence.png` for image sequence sequence
    - `--source="device:///nvcamera?index=0"` for the GStreamer NVIDIA camera (Jetson TX1 only).

#### \-h, \--help ####
- Description: Prints the help message.

### Operational Keys ###
- Use `ESC` to close the sample.

