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

#if defined USE_OPENCV && defined USE_NPP

#include "alpha_comp_node.hpp"

//
// Define user kernel
//

#define KERNEL_ALPHA_COMP_NAME "example.nvx.alpha_comp"

// Kernel implementation
static vx_status VX_CALLBACK alphaComp_kernel(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    if (num != 6)
        return VX_FAILURE;

    vx_image src1 = (vx_image)parameters[0];
    vx_scalar s_alpha1 = (vx_scalar)parameters[1];
    vx_image src2 = (vx_image)parameters[2];
    vx_scalar s_alpha2 = (vx_scalar)parameters[3];
    vx_image dst = (vx_image)parameters[4];
    vx_scalar s_alphaOp = (vx_scalar)parameters[5];

    vx_uint8 alpha1 = 0;
    vx_uint8 alpha2 = 0;
    vx_enum alphaOp = 0;

    vx_status status = VX_SUCCESS;

    // Get scalars values

    vxCopyScalar(s_alpha1, &alpha1, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyScalar(s_alpha2, &alpha2, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyScalar(s_alphaOp, &alphaOp, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);

    // Get CUDA stream, which is used for current node

    cudaStream_t stream = NULL;
    vxQueryNode(node, NVX_NODE_CUDA_STREAM, &stream, sizeof(stream));

    // Use this stream for NPP launch
    nppSetStream(stream);

    // Map OpenVX data objects into CUDA device memory

    vx_rectangle_t rect = {};
    vxGetValidRegionImage(src1, &rect);

    vx_map_id src1_map_id;
    vx_uint8* src1_ptr;
    vx_imagepatch_addressing_t src1_addr;
    status = vxMapImagePatch(src1, &rect, 0, &src1_map_id, &src1_addr, (void **)&src1_ptr, VX_READ_ONLY, NVX_MEMORY_TYPE_CUDA, 0);

    if (status != VX_SUCCESS)
    {
        vxAddLogEntry((vx_reference)src1, status, "[%s:%u] Failed to access \'src1\' in AlphaComp Kernel", __FUNCTION__, __LINE__);
        return status;
    }

    vx_map_id src2_map_id;
    vx_uint8* src2_ptr;
    vx_imagepatch_addressing_t src2_addr;
    vxMapImagePatch(src2, &rect, 0, &src2_map_id, &src2_addr, (void **)&src2_ptr, VX_READ_ONLY, NVX_MEMORY_TYPE_CUDA, 0);

    if (status != VX_SUCCESS)
    {
        vxAddLogEntry((vx_reference)src2, status, "[%s:%u] Failed to access \'src2\' in AlphaComp Kernel", __FUNCTION__, __LINE__);
        vxUnmapImagePatch(src1, src1_map_id);
        return status;
    }

    vx_map_id dst_map_id;
    vx_uint8* dst_ptr;
    vx_imagepatch_addressing_t dst_addr;
    status = vxMapImagePatch(dst, &rect, 0, &dst_map_id, &dst_addr, (void **)&dst_ptr, VX_WRITE_ONLY, NVX_MEMORY_TYPE_CUDA, 0);

    if (status != VX_SUCCESS)
    {
        vxAddLogEntry((vx_reference)src2, status, "[%s:%u] Failed to access \'dst\' in AlphaComp Kernel", __FUNCTION__, __LINE__);
        vxUnmapImagePatch(src1, src1_map_id);
        vxUnmapImagePatch(src2, src2_map_id);
        return status;
    }

    // Call NPP function

    NppiSize oSizeROI;
    oSizeROI.width = src1_addr.dim_x;
    oSizeROI.height = src1_addr.dim_y;

    NppStatus npp_status = nppiAlphaCompC_8u_C1R(src1_ptr, src1_addr.stride_y, alpha1,
                                                 src2_ptr, src2_addr.stride_y, alpha2,
                                                 dst_ptr, dst_addr.stride_y,
                                                 oSizeROI,
                                                 static_cast<NppiAlphaOp>(alphaOp));
    if (npp_status != NPP_SUCCESS)
    {
        vxAddLogEntry((vx_reference)node, VX_FAILURE, "[%s:%u] nppiAlphaCompC_8u_C1R error", __FUNCTION__, __LINE__);
        status = VX_FAILURE;
    }

    // Unmap OpenVX data objects from CUDA device memory

    vxUnmapImagePatch(src1, src1_map_id);
    vxUnmapImagePatch(src2, src2_map_id);
    vxUnmapImagePatch(dst, dst_map_id);

    return status;
}

// Parameter validator
static vx_status VX_CALLBACK alphaComp_validate(vx_node, const vx_reference parameters[],
                                                vx_uint32 num_params, vx_meta_format metas[])
{
    if (num_params != 6) return VX_ERROR_INVALID_PARAMETERS;

    vx_image src1 = (vx_image)parameters[0];
    vx_scalar alpha1 = (vx_scalar)parameters[1];
    vx_image src2 = (vx_image)parameters[2];
    vx_scalar alpha2 = (vx_scalar)parameters[3];
    vx_scalar alphaOp = (vx_scalar)parameters[5];

    vx_df_image src1_format = 0;
    vxQueryImage(src1, VX_IMAGE_ATTRIBUTE_FORMAT, &src1_format, sizeof(src1_format));

    vx_uint32 src1_width = 0, src1_height = 0;
    vxQueryImage(src1, VX_IMAGE_ATTRIBUTE_WIDTH, &src1_width, sizeof(src1_width));
    vxQueryImage(src1, VX_IMAGE_ATTRIBUTE_HEIGHT, &src1_height, sizeof(src1_height));

    vx_enum alpha1_type = 0;
    vxQueryScalar(alpha1, VX_SCALAR_ATTRIBUTE_TYPE, &alpha1_type, sizeof(alpha1_type));

    vx_df_image src2_format = 0;
    vxQueryImage(src2, VX_IMAGE_ATTRIBUTE_FORMAT, &src2_format, sizeof(src2_format));

    vx_uint32 src2_width = 0, src2_height = 0;
    vxQueryImage(src2, VX_IMAGE_ATTRIBUTE_WIDTH, &src2_width, sizeof(src2_width));
    vxQueryImage(src2, VX_IMAGE_ATTRIBUTE_HEIGHT, &src2_height, sizeof(src2_height));

    vx_enum alpha2_type = 0;
    vxQueryScalar(alpha2, VX_SCALAR_ATTRIBUTE_TYPE, &alpha2_type, sizeof(alpha2_type));

    vx_enum alphaOp_type = 0;
    vxQueryScalar(alphaOp, VX_SCALAR_ATTRIBUTE_TYPE, &alphaOp_type, sizeof(alphaOp_type));

    vx_status status = VX_SUCCESS;

    if (src1_format != VX_DF_IMAGE_U8)
    {
        status = VX_ERROR_INVALID_FORMAT;
        vxAddLogEntry((vx_reference)src1, status, "[%s:%u] Invalid format for \'src1\' in AlphaComp Kernel, it should be VX_DF_IMAGE_U8", __FUNCTION__, __LINE__);
    }

    if (alpha1_type != VX_TYPE_UINT8)
    {
        status = VX_ERROR_INVALID_TYPE;
        vxAddLogEntry((vx_reference)alpha1, status, "[%s:%u] Invalid format for \'alpha1\' in AlphaComp Kernel, it should be VX_TYPE_UINT8", __FUNCTION__, __LINE__);
    }

    if (src2_format != src1_format || src2_height != src1_height || src2_width != src1_width)
    {
        status = VX_ERROR_INVALID_PARAMETERS;
        vxAddLogEntry((vx_reference)src2, status, "[%s:%u] \'src1\' and \'src2\' have different size/format in AlphaComp Kernel", __FUNCTION__, __LINE__);
    }

    if (alpha2_type != VX_TYPE_UINT8)
    {
        status = VX_ERROR_INVALID_TYPE;
        vxAddLogEntry((vx_reference)alpha2, status, "[%s:%u] Invalid format for \'alpha2\' in AlphaComp Kernel, it should be VX_TYPE_UINT8", __FUNCTION__, __LINE__);
    }

    if (alphaOp_type != VX_TYPE_ENUM)
    {
        status = VX_ERROR_INVALID_TYPE;
        vxAddLogEntry((vx_reference)alphaOp, status, "[%s:%u] Invalid format for \'alphaOp\' in AlphaComp Kernel, it should be VX_TYPE_ENUM", __FUNCTION__, __LINE__);
    }

    vx_meta_format dst_meta = metas[4];

    vxSetMetaFormatAttribute(dst_meta, VX_IMAGE_ATTRIBUTE_FORMAT, &src1_format, sizeof(src1_format));
    vxSetMetaFormatAttribute(dst_meta, VX_IMAGE_ATTRIBUTE_WIDTH, &src1_width, sizeof(src1_width));
    vxSetMetaFormatAttribute(dst_meta, VX_IMAGE_ATTRIBUTE_HEIGHT, &src1_height, sizeof(src1_height));

    return status;
}

// Register user defined kernel in OpenVX context
vx_status registerAlphaCompKernel(vx_context context)
{
    vx_status status = VX_SUCCESS;

    vx_enum id;
    status = vxAllocateUserKernelId(context, &id);
    if (status != VX_SUCCESS)
    {
        vxAddLogEntry((vx_reference)context, status, "Failed to allocate an ID for the AlphaComp kernel");
        return status;
    }

    vx_kernel kernel = vxAddUserKernel(context, "gpu:" KERNEL_ALPHA_COMP_NAME, id,
                                       alphaComp_kernel,
                                       6,    // numParams
                                       alphaComp_validate,
                                       NULL, // init
                                       NULL  // deinit
                                       );

    status = vxGetStatus((vx_reference)kernel);
    if (status != VX_SUCCESS)
    {
        vxAddLogEntry((vx_reference)context, status, "Failed to create AlphaComp Kernel");
        return status;
    }

    status |= vxAddParameterToKernel(kernel, 0, VX_INPUT , VX_TYPE_IMAGE , VX_PARAMETER_STATE_REQUIRED); // src1
    status |= vxAddParameterToKernel(kernel, 1, VX_INPUT , VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED); // alpha1
    status |= vxAddParameterToKernel(kernel, 2, VX_INPUT , VX_TYPE_IMAGE , VX_PARAMETER_STATE_REQUIRED); // src2
    status |= vxAddParameterToKernel(kernel, 3, VX_INPUT , VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED); // alpha2
    status |= vxAddParameterToKernel(kernel, 4, VX_OUTPUT, VX_TYPE_IMAGE , VX_PARAMETER_STATE_REQUIRED); // dst
    status |= vxAddParameterToKernel(kernel, 5, VX_INPUT , VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED); // alphaOp

    if (status != VX_SUCCESS)
    {
        vxReleaseKernel(&kernel);
        vxAddLogEntry((vx_reference)context, status, "Failed to initialize AlphaComp Kernel parameters");
        return VX_FAILURE;
    }

    status = vxFinalizeKernel(kernel);

    if (status != VX_SUCCESS)
    {
        vxReleaseKernel(&kernel);
        vxAddLogEntry((vx_reference)context, status, "Failed to finalize AlphaComp Kernel");
        return VX_FAILURE;
    }

    return status;
}

// Create AlphaComp node
vx_node alphaCompNode(vx_graph graph, vx_image src1, vx_scalar alpha1, vx_image src2, vx_scalar alpha2, vx_image dst, vx_scalar alphaOp)
{
    vx_node node = NULL;

    vx_kernel kernel = vxGetKernelByName(vxGetContext((vx_reference)graph), KERNEL_ALPHA_COMP_NAME);

    if (vxGetStatus((vx_reference)kernel) == VX_SUCCESS)
    {
        node = vxCreateGenericNode(graph, kernel);
        vxReleaseKernel(&kernel);

        if (vxGetStatus((vx_reference)node) == VX_SUCCESS)
        {
            vxSetParameterByIndex(node, 0, (vx_reference)src1);
            vxSetParameterByIndex(node, 1, (vx_reference)alpha1);
            vxSetParameterByIndex(node, 2, (vx_reference)src2);
            vxSetParameterByIndex(node, 3, (vx_reference)alpha2);
            vxSetParameterByIndex(node, 4, (vx_reference)dst);
            vxSetParameterByIndex(node, 5, (vx_reference)alphaOp);
        }
    }

    return node;
}

#endif // defined USE_OPENCV && defined USE_NPP
