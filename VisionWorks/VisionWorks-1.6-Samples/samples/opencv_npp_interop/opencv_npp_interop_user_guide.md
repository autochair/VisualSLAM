Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

OpenCV and NPP Interop Sample App
@brief OpenCV and NPP Interop Sample user guide.

## Introduction ##

`nvx_sample_opencv_npp_interop` shows the interoperability of VisionWorks with other libraries,
such as OpenCV and NPP.

@note This sample was written and tested against OpenCV 3.1 and 2.4.13. It might need changes to be compatible with other OpenCV versions. For instructions on how to build OpenCV for Tegra platforms, please refer to <a href="http://docs.opencv.org/master/d6/d15/tutorial_building_tegra_cuda.html" target="_blank">Building OpenCV for Tegra with CUDA</a>.

This sample accepts 2 images as input, blurs them, and performs alpha blending between them.

The sample uses OpenCV library for loading the input images and displaying the result image.
The images loaded by OpenCV are imported into the VisionWorks framework using the `vxCreateImageFromHandle` function.
For alpha blending, the NPP library is used. The alpha blending operation is implemented as User Defined Kernel.
For blurring standard `Gaussian3x3` kernel is used.

The full pipeline is implemented as the following graph:

       (image1)        (image2)
          |               |
    [Gaussian3x3]   [Gaussian3x3]
          |               |
          +-------+-------+
                  |
             [AlphaComp]
                  |
              (output)

For detailed information about User Defined Kernels, see see: group_user_kernels.

`nvx_sample_opencv_npp_interop` is installed in the following directory:

    /usr/share/visionworks/sources/samples/opencv_npp_interop

For the steps to build sample applications, see the see: nvx_samples_and_demos_user_guides section for your OS.

## Executing the OpenCV and NPP Interoperability Sample ##

    ./nvx_sample_opencv_npp_interop [options]

### Command Line Options ###

The `[--img1]` and `[--img2]` options specify the 2 images to perform alpha blending.
The input images must be of the same size. Input images can be ommitted; in that case,
the demo will use images from default data set.

#### \--img1 ####

- Parameter: [image]
- Description: Specifies the first image to perform alpha blending.
- Usage:

  `./nvx_sample_opencv_npp_interop --img1=PATH_TO_IMG1 --img2=PATH_TO_IMG2`

#### \--img2 ####

- Parameter: [image]
- Description: Specifies the second image to perform alpha blending.
- Usage:

  `./nvx_sample_opencv_npp_interop --img1=PATH_TO_IMG1 --img2=PATH_TO_IMG2`

#### \-h, \--help ####
- Description: Prints the help message.

### Operational Keys ###
- Use `Space` to pause/resume the sample.
- Use `ESC` to close the sample.

