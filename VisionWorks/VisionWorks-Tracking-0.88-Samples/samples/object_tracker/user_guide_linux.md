Copyright (c) 2015, NVIDIA CORPORATION.  All rights reserved.

Linux: Building and Running Samples
@brief Shows how to build samples using native compilation on Linux.

## Native Compilation of Sample Application ##

Sources for samples are provided in the `libvisionworks-tracking-dev` package.
After package installation, source code and make files are located at:

    /usr/share/visionworks-tracking/sources

The directory is write protected;
copy its contents to a directory with write access, such as your home directory.
Execute the following commands:

    $ /usr/share/visionworks-tracking/sources/install-samples.sh ~/
    $ cd ~/VisionWorks-Tracking-<ver>-Samples/
    $ make -j4 # add dbg=1 to make debug build

Where `<ver>` is the version of the VisionWorks Object Tracker sample.

## Running Samples ##

**Applies to:** Jetson devices only.

1. Start the X window manager:

        $ export DISPLAY=:0
        $ X -ac &
        $ blackbox

2. Navigate to the samples directory:

        $ cd ~/VisionWorks-Tracking-<ver>-Samples/sources/bin/<arch>/linux/release

    Where `<ver>` is the version of the VisionWorks Object Tracker sample and `arch` is
    platform architecture.

3. Execute the following command:

        $ ./nvx_sample_object_tracker

