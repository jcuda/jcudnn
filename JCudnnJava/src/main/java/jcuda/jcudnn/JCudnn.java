/*
 * JCudnn - Java bindings for cuDNN, the NVIDIA CUDA
 * Deep Neural Network library, to be used with JCuda
 *
 * Copyright (c) 2015-2015 Marco Hutter - http://www.jcuda.org
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use,
 * copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following
 * conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */
package jcuda.jcudnn;

import jcuda.CudaException;
import jcuda.LibUtils;
import jcuda.LogLevel;
import jcuda.Pointer;
import jcuda.runtime.cudaStream_t;

/**
 * Java bindings for cuDNN, the NVIDIA CUDA
 * Deep Neural Network library.
 */
public class JCudnn
{
    public static final int CUDNN_MAJOR      = 4;
    public static final int CUDNN_MINOR      = 0;
    public static final int CUDNN_PATCHLEVEL = 4;

    public static final int CUDNN_VERSION    =
        (CUDNN_MAJOR * 1000 + CUDNN_MINOR * 100 + CUDNN_PATCHLEVEL);
    
    /**
     * Maximum supported number of tensor dimensions 
     */
    public static final int CUDNN_DIM_MAX = 8;
    
    /**
     * Minimum epsilon allowed to be used in the Batch Normalization formula
     */
    public static final double CUDNN_BN_MIN_EPSILON = 1e-5;
    
    /**
     * The flag that indicates whether the native library has been
     * loaded
     */
    private static boolean initialized = false;

    /**
     * Whether a CudaException should be thrown if a method is about
     * to return a result code that is not
     * cudnnStatus.CUDNN_STATUS_SUCCESS
     */
    private static boolean exceptionsEnabled = false;

    /* Private constructor to prevent instantiation */
    private JCudnn()
    {
    }

    // Initialize the native library.
    static
    {
        initialize();
    }

    /**
     * Initializes the native library. Note that this method
     * does not have to be called explicitly, since it will
     * be called automatically when this class is loaded.
     */
    public static void initialize()
    {
        if (!initialized)
        {
            LibUtils.loadLibrary("JCudnn");
            initialized = true;
        }
    }


    /**
     * Enables or disables exceptions. By default, the methods of this class
     * only set the {@link cudnnStatus} from the native methods.
     * If exceptions are enabled, a CudaException with a detailed error
     * message will be thrown if a method is about to set a result code
     * that is not cudnnStatus.CUDNN_STATUS_SUCCESS
     *
     * @param enabled Whether exceptions are enabled
     */
    public static void setExceptionsEnabled(boolean enabled)
    {
        exceptionsEnabled = enabled;
    }

    /**
     * If the given result is not cudnnStatus.CUDNN_STATUS_SUCCESS
     * and exceptions have been enabled, this method will throw a
     * CudaException with an error message that corresponds to the
     * given result code. Otherwise, the given result is simply
     * returned.
     *
     * @param result The result to check
     * @return The result that was given as the parameter
     * @throws CudaException If exceptions have been enabled and
     * the given result code is not cudnnStatus.CUDNN_STATUS_SUCCESS
     */
    private static int checkResult(int result)
    {
        if (exceptionsEnabled && result !=
            cudnnStatus.CUDNN_STATUS_SUCCESS)
        {
            throw new CudaException(cudnnStatus.stringFor(result));
        }
        return result;
    }

    /**
     * Set the specified log level for the JCudnn library.<br />
     * <br />
     * Currently supported log levels:
     * <br />
     * LOG_QUIET: Never print anything <br />
     * LOG_ERROR: Print error messages <br />
     * LOG_TRACE: Print a trace of all native function calls <br />
     *
     * @param logLevel The log level to use.
     */
    public static void setLogLevel(LogLevel logLevel)
    {
        setLogLevelNative(logLevel.ordinal());
    }

    private static native void setLogLevelNative(int logLevel);
    
    
    public static long cudnnGetVersion()
    {
        return cudnnGetVersionNative();
    }
    private static native long cudnnGetVersionNative();


    // human-readable error messages
    public static String cudnnGetErrorString(
        int status)
    {
        return cudnnGetErrorStringNative(status);
    }
    private static native String cudnnGetErrorStringNative(
        int status);


    public static int cudnnCreate(
        cudnnHandle handle)
    {
        return checkResult(cudnnCreateNative(handle));
    }
    private static native int cudnnCreateNative(
        cudnnHandle handle);


    public static int cudnnDestroy(
        cudnnHandle handle)
    {
        return checkResult(cudnnDestroyNative(handle));
    }
    private static native int cudnnDestroyNative(
        cudnnHandle handle);


    public static int cudnnSetStream(
        cudnnHandle handle, 
        cudaStream_t streamId)
    {
        return checkResult(cudnnSetStreamNative(handle, streamId));
    }
    private static native int cudnnSetStreamNative(
        cudnnHandle handle, 
        cudaStream_t streamId);


    public static int cudnnGetStream(
        cudnnHandle handle, 
        cudaStream_t streamId)
    {
        return checkResult(cudnnGetStreamNative(handle, streamId));
    }
    private static native int cudnnGetStreamNative(
        cudnnHandle handle, 
        cudaStream_t streamId);


    /** Create an instance of a generic Tensor descriptor */
    public static int cudnnCreateTensorDescriptor(
        cudnnTensorDescriptor tensorDesc)
    {
        return checkResult(cudnnCreateTensorDescriptorNative(tensorDesc));
    }
    private static native int cudnnCreateTensorDescriptorNative(
        cudnnTensorDescriptor tensorDesc);


    public static int cudnnSetTensor4dDescriptor(
        cudnnTensorDescriptor tensorDesc, 
        int format, 
        int dataType, // image data type
        int n, // number of inputs (batch size)
        int c, // number of input feature maps
        int h, // height of input section
        int w)// width of input section
    {
        return checkResult(cudnnSetTensor4dDescriptorNative(tensorDesc, format, dataType, n, c, h, w));
    }
    private static native int cudnnSetTensor4dDescriptorNative(
        cudnnTensorDescriptor tensorDesc, 
        int format, 
        int dataType, // image data type
        int n, // number of inputs (batch size)
        int c, // number of input feature maps
        int h, // height of input section
        int w);// width of input section


    public static int cudnnSetTensor4dDescriptorEx(
        cudnnTensorDescriptor tensorDesc, 
        int dataType, // image data type
        int n, // number of inputs (batch size)
        int c, // number of input feature maps
        int h, // height of input section
        int w, // width of input section
        int nStride, 
        int cStride, 
        int hStride, 
        int wStride)
    {
        return checkResult(cudnnSetTensor4dDescriptorExNative(tensorDesc, dataType, n, c, h, w, nStride, cStride, hStride, wStride));
    }
    private static native int cudnnSetTensor4dDescriptorExNative(
        cudnnTensorDescriptor tensorDesc, 
        int dataType, // image data type
        int n, // number of inputs (batch size)
        int c, // number of input feature maps
        int h, // height of input section
        int w, // width of input section
        int nStride, 
        int cStride, 
        int hStride, 
        int wStride);


    public static int cudnnGetTensor4dDescriptor(
        cudnnTensorDescriptor tensorDesc, 
        int[] dataType, // image data type
        Pointer n, // number of inputs (batch size)
        Pointer c, // number of input feature maps
        Pointer h, // height of input section
        Pointer w, // width of input section
        Pointer nStride, 
        Pointer cStride, 
        Pointer hStride, 
        Pointer wStride)
    {
        return checkResult(cudnnGetTensor4dDescriptorNative(tensorDesc, dataType, n, c, h, w, nStride, cStride, hStride, wStride));
    }
    private static native int cudnnGetTensor4dDescriptorNative(
        cudnnTensorDescriptor tensorDesc, 
        int[] dataType, // image data type
        Pointer n, // number of inputs (batch size)
        Pointer c, // number of input feature maps
        Pointer h, // height of input section
        Pointer w, // width of input section
        Pointer nStride, 
        Pointer cStride, 
        Pointer hStride, 
        Pointer wStride);


    public static int cudnnSetTensorNdDescriptor(
        cudnnTensorDescriptor tensorDesc, 
        int dataType, 
        int nbDims, 
        int[] dimA, 
        int[] strideA)
    {
        return checkResult(cudnnSetTensorNdDescriptorNative(tensorDesc, dataType, nbDims, dimA, strideA));
    }
    private static native int cudnnSetTensorNdDescriptorNative(
        cudnnTensorDescriptor tensorDesc, 
        int dataType, 
        int nbDims, 
        int[] dimA, 
        int[] strideA);


    public static int cudnnGetTensorNdDescriptor(
        cudnnTensorDescriptor tensorDesc, 
        int nbDimsRequested, 
        int[] dataType, 
        int[] nbDims, 
        int[] dimA, 
        int[] strideA)
    {
        return checkResult(cudnnGetTensorNdDescriptorNative(tensorDesc, nbDimsRequested, dataType, nbDims, dimA, strideA));
    }
    private static native int cudnnGetTensorNdDescriptorNative(
        cudnnTensorDescriptor tensorDesc, 
        int nbDimsRequested, 
        int[] dataType, 
        int[] nbDims, 
        int[] dimA, 
        int[] strideA);


    /**
     * <pre>
     * PixelOffset( n, c, h, w ) = n *input_stride + c * feature_stride + h * h_stride + w * w_stride
    
       1)Example of all images in row major order one batch of features after the other (with an optional padding on row)
       input_stride :  c x h x h_stride
       feature_stride : h x h_stride
       h_stride  :  >= w  ( h_stride = w if no padding)
       w_stride  : 1
    
    
       2)Example of all images in row major with features maps interleaved
       input_stride :  c x h x h_stride
       feature_stride : 1
       h_stride  :  w x c
       w_stride  : c
    
       3)Example of all images in column major order one batch of features after the other (with optional padding on column)
       input_stride :  c x w x w_stride
       feature_stride : w x w_stride
       h_stride  :  1
       w_stride  :  >= h
    
     * </pre>
     */
    /** Destroy an instance of Tensor4d descriptor */
    public static int cudnnDestroyTensorDescriptor(
        cudnnTensorDescriptor tensorDesc)
    {
        return checkResult(cudnnDestroyTensorDescriptorNative(tensorDesc));
    }
    private static native int cudnnDestroyTensorDescriptorNative(
        cudnnTensorDescriptor tensorDesc);


    /** Tensor layout conversion helper (y = alpha * x + beta * y) */
    public static int cudnnTransformTensor(
        cudnnHandle handle, 
        Pointer alpha, 
        cudnnTensorDescriptor xDesc, 
        Pointer x, 
        Pointer beta, 
        cudnnTensorDescriptor yDesc, 
        Pointer y)
    {
        return checkResult(cudnnTransformTensorNative(handle, alpha, xDesc, x, beta, yDesc, y));
    }
    private static native int cudnnTransformTensorNative(
        cudnnHandle handle, 
        Pointer alpha, 
        cudnnTensorDescriptor xDesc, 
        Pointer x, 
        Pointer beta, 
        cudnnTensorDescriptor yDesc, 
        Pointer y);


    /** Tensor Bias addition : y = alpha * b + beta * y  */
    public static int cudnnAddTensor(
        cudnnHandle handle, 
        Pointer alpha, 
        cudnnTensorDescriptor bDesc, 
        Pointer b, 
        Pointer beta, 
        cudnnTensorDescriptor yDesc, 
        Pointer y)
    {
        return checkResult(cudnnAddTensorNative(handle, alpha, bDesc, b, beta, yDesc, y));
    }
    private static native int cudnnAddTensorNative(
        cudnnHandle handle, 
        Pointer alpha, 
        cudnnTensorDescriptor bDesc, 
        Pointer b, 
        Pointer beta, 
        cudnnTensorDescriptor yDesc, 
        Pointer y);


    /**
     * <pre>
     * cudnnAddTensor_v3 is now mapped to cudnnAddTensor
       and will be removed at the same time as cudnnAddTensor_v2
       Use cudnnAddTensor instead
     * </pre>
     */
    public static int cudnnAddTensor_v3(
        cudnnHandle handle, 
        Pointer alpha, 
        cudnnTensorDescriptor bDesc, 
        Pointer b, 
        Pointer beta, 
        cudnnTensorDescriptor yDesc, 
        Pointer y)
    {
        return checkResult(cudnnAddTensor_v3Native(handle, alpha, bDesc, b, beta, yDesc, y));
    }
    private static native int cudnnAddTensor_v3Native(
        cudnnHandle handle, 
        Pointer alpha, 
        cudnnTensorDescriptor bDesc, 
        Pointer b, 
        Pointer beta, 
        cudnnTensorDescriptor yDesc, 
        Pointer y);


    /** Set all values of a tensor to a given value : y[i] = value[0] */
    public static int cudnnSetTensor(
        cudnnHandle handle, 
        cudnnTensorDescriptor yDesc, 
        Pointer y, 
        Pointer valuePtr)
    {
        return checkResult(cudnnSetTensorNative(handle, yDesc, y, valuePtr));
    }
    private static native int cudnnSetTensorNative(
        cudnnHandle handle, 
        cudnnTensorDescriptor yDesc, 
        Pointer y, 
        Pointer valuePtr);


    /** Scale all values of a tensor by a given factor : y[i] = alpha * y[i] */
    public static int cudnnScaleTensor(
        cudnnHandle handle, 
        cudnnTensorDescriptor yDesc, 
        Pointer y, 
        Pointer alpha)
    {
        return checkResult(cudnnScaleTensorNative(handle, yDesc, y, alpha));
    }
    private static native int cudnnScaleTensorNative(
        cudnnHandle handle, 
        cudnnTensorDescriptor yDesc, 
        Pointer y, 
        Pointer alpha);


    /** Create an instance of FilterStruct */
    public static int cudnnCreateFilterDescriptor(
        cudnnFilterDescriptor filterDesc)
    {
        return checkResult(cudnnCreateFilterDescriptorNative(filterDesc));
    }
    private static native int cudnnCreateFilterDescriptorNative(
        cudnnFilterDescriptor filterDesc);


    public static int cudnnSetFilter4dDescriptor(
        cudnnFilterDescriptor filterDesc, 
        int dataType, // image data type
        int k, // number of output feature maps
        int c, // number of input feature maps
        int h, // height of each input filter
        int w)// width of  each input filter
    {
        return checkResult(cudnnSetFilter4dDescriptorNative(filterDesc, dataType, k, c, h, w));
    }
    private static native int cudnnSetFilter4dDescriptorNative(
        cudnnFilterDescriptor filterDesc, 
        int dataType, // image data type
        int k, // number of output feature maps
        int c, // number of input feature maps
        int h, // height of each input filter
        int w);// width of  each input filter


    public static int cudnnSetFilter4dDescriptor_v4(
        cudnnFilterDescriptor filterDesc, 
        int dataType, // image data type
        int format, 
        int k, // number of output feature maps
        int c, // number of input feature maps
        int h, // height of each input filter
        int w)// width of  each input filter
    {
        return checkResult(cudnnSetFilter4dDescriptor_v4Native(filterDesc, dataType, format, k, c, h, w));
    }
    private static native int cudnnSetFilter4dDescriptor_v4Native(
        cudnnFilterDescriptor filterDesc, 
        int dataType, // image data type
        int format, 
        int k, // number of output feature maps
        int c, // number of input feature maps
        int h, // height of each input filter
        int w);// width of  each input filter


    public static int cudnnGetFilter4dDescriptor(
        cudnnFilterDescriptor filterDesc, 
        int[] dataType, // image data type
        Pointer k, // number of output feature maps
        Pointer c, // number of input feature maps
        Pointer h, // height of each input filter
        Pointer w)// width of  each input filter
    {
        return checkResult(cudnnGetFilter4dDescriptorNative(filterDesc, dataType, k, c, h, w));
    }
    private static native int cudnnGetFilter4dDescriptorNative(
        cudnnFilterDescriptor filterDesc, 
        int[] dataType, // image data type
        Pointer k, // number of output feature maps
        Pointer c, // number of input feature maps
        Pointer h, // height of each input filter
        Pointer w);// width of  each input filter


    public static int cudnnGetFilter4dDescriptor_v4(
        cudnnFilterDescriptor filterDesc, 
        int[] dataType, // image data type
        int[] format, 
        Pointer k, // number of output feature maps
        Pointer c, // number of input feature maps
        Pointer h, // height of each input filter
        Pointer w)// width of  each input filter
    {
        return checkResult(cudnnGetFilter4dDescriptor_v4Native(filterDesc, dataType, format, k, c, h, w));
    }
    private static native int cudnnGetFilter4dDescriptor_v4Native(
        cudnnFilterDescriptor filterDesc, 
        int[] dataType, // image data type
        int[] format, 
        Pointer k, // number of output feature maps
        Pointer c, // number of input feature maps
        Pointer h, // height of each input filter
        Pointer w);// width of  each input filter


    public static int cudnnSetFilterNdDescriptor(
        cudnnFilterDescriptor filterDesc, 
        int dataType, // image data type
        int nbDims, 
        int[] filterDimA)
    {
        return checkResult(cudnnSetFilterNdDescriptorNative(filterDesc, dataType, nbDims, filterDimA));
    }
    private static native int cudnnSetFilterNdDescriptorNative(
        cudnnFilterDescriptor filterDesc, 
        int dataType, // image data type
        int nbDims, 
        int[] filterDimA);


    public static int cudnnSetFilterNdDescriptor_v4(
        cudnnFilterDescriptor filterDesc, 
        int dataType, // image data type
        int format, 
        int nbDims, 
        int[] filterDimA)
    {
        return checkResult(cudnnSetFilterNdDescriptor_v4Native(filterDesc, dataType, format, nbDims, filterDimA));
    }
    private static native int cudnnSetFilterNdDescriptor_v4Native(
        cudnnFilterDescriptor filterDesc, 
        int dataType, // image data type
        int format, 
        int nbDims, 
        int[] filterDimA);


    public static int cudnnGetFilterNdDescriptor(
        cudnnFilterDescriptor filterDesc, 
        int nbDimsRequested, 
        int[] dataType, // image data type
        Pointer nbDims, 
        int[] filterDimA)
    {
        return checkResult(cudnnGetFilterNdDescriptorNative(filterDesc, nbDimsRequested, dataType, nbDims, filterDimA));
    }
    private static native int cudnnGetFilterNdDescriptorNative(
        cudnnFilterDescriptor filterDesc, 
        int nbDimsRequested, 
        int[] dataType, // image data type
        Pointer nbDims, 
        int[] filterDimA);


    public static int cudnnGetFilterNdDescriptor_v4(
        cudnnFilterDescriptor filterDesc, 
        int nbDimsRequested, 
        int[] dataType, // image data type
        int[] format, 
        Pointer nbDims, 
        int[] filterDimA)
    {
        return checkResult(cudnnGetFilterNdDescriptor_v4Native(filterDesc, nbDimsRequested, dataType, format, nbDims, filterDimA));
    }
    private static native int cudnnGetFilterNdDescriptor_v4Native(
        cudnnFilterDescriptor filterDesc, 
        int nbDimsRequested, 
        int[] dataType, // image data type
        int[] format, 
        Pointer nbDims, 
        int[] filterDimA);


    public static int cudnnDestroyFilterDescriptor(
        cudnnFilterDescriptor filterDesc)
    {
        return checkResult(cudnnDestroyFilterDescriptorNative(filterDesc));
    }
    private static native int cudnnDestroyFilterDescriptorNative(
        cudnnFilterDescriptor filterDesc);


    /** Create an instance of convolution descriptor */
    public static int cudnnCreateConvolutionDescriptor(
        cudnnConvolutionDescriptor convDesc)
    {
        return checkResult(cudnnCreateConvolutionDescriptorNative(convDesc));
    }
    private static native int cudnnCreateConvolutionDescriptorNative(
        cudnnConvolutionDescriptor convDesc);


    public static int cudnnSetConvolution2dDescriptor(
        cudnnConvolutionDescriptor convDesc, 
        int pad_h, // zero-padding height
        int pad_w, // zero-padding width
        int u, // vertical filter stride
        int v, // horizontal filter stride
        int upscalex, // upscale the input in x-direction
        int upscaley, // upscale the input in y-direction
        int mode)
    {
        return checkResult(cudnnSetConvolution2dDescriptorNative(convDesc, pad_h, pad_w, u, v, upscalex, upscaley, mode));
    }
    private static native int cudnnSetConvolution2dDescriptorNative(
        cudnnConvolutionDescriptor convDesc, 
        int pad_h, // zero-padding height
        int pad_w, // zero-padding width
        int u, // vertical filter stride
        int v, // horizontal filter stride
        int upscalex, // upscale the input in x-direction
        int upscaley, // upscale the input in y-direction
        int mode);


    public static int cudnnGetConvolution2dDescriptor(
        cudnnConvolutionDescriptor convDesc, 
        Pointer pad_h, // zero-padding height
        Pointer pad_w, // zero-padding width
        Pointer u, // vertical filter stride
        Pointer v, // horizontal filter stride
        Pointer upscalex, // upscale the input in x-direction
        Pointer upscaley, // upscale the input in y-direction
        int[] mode)
    {
        return checkResult(cudnnGetConvolution2dDescriptorNative(convDesc, pad_h, pad_w, u, v, upscalex, upscaley, mode));
    }
    private static native int cudnnGetConvolution2dDescriptorNative(
        cudnnConvolutionDescriptor convDesc, 
        Pointer pad_h, // zero-padding height
        Pointer pad_w, // zero-padding width
        Pointer u, // vertical filter stride
        Pointer v, // horizontal filter stride
        Pointer upscalex, // upscale the input in x-direction
        Pointer upscaley, // upscale the input in y-direction
        int[] mode);


    /** Helper function to return the dimensions of the output tensor given a convolution descriptor */
    public static int cudnnGetConvolution2dForwardOutputDim(
        cudnnConvolutionDescriptor convDesc, 
        cudnnTensorDescriptor inputTensorDesc, 
        cudnnFilterDescriptor filterDesc, 
        Pointer n, 
        Pointer c, 
        Pointer h, 
        Pointer w)
    {
        return checkResult(cudnnGetConvolution2dForwardOutputDimNative(convDesc, inputTensorDesc, filterDesc, n, c, h, w));
    }
    private static native int cudnnGetConvolution2dForwardOutputDimNative(
        cudnnConvolutionDescriptor convDesc, 
        cudnnTensorDescriptor inputTensorDesc, 
        cudnnFilterDescriptor filterDesc, 
        Pointer n, 
        Pointer c, 
        Pointer h, 
        Pointer w);


    public static int cudnnSetConvolutionNdDescriptor(
        cudnnConvolutionDescriptor convDesc, 
        int arrayLength, /** nbDims-2 size */
        int[] padA, 
        int[] filterStrideA, 
        int[] upscaleA, 
        int mode, 
        int dataType)// convolution data type
    {
        return checkResult(cudnnSetConvolutionNdDescriptorNative(convDesc, arrayLength, padA, filterStrideA, upscaleA, mode, dataType));
    }
    private static native int cudnnSetConvolutionNdDescriptorNative(
        cudnnConvolutionDescriptor convDesc, 
        int arrayLength, /** nbDims-2 size */
        int[] padA, 
        int[] filterStrideA, 
        int[] upscaleA, 
        int mode, 
        int dataType);// convolution data type


    public static int cudnnGetConvolutionNdDescriptor(
        cudnnConvolutionDescriptor convDesc, 
        int arrayLengthRequested, 
        int[] arrayLength, 
        int[] padA, 
        int[] strideA, 
        int[] upscaleA, 
        int[] mode, 
        int[] dataType)// convolution data type
    {
        return checkResult(cudnnGetConvolutionNdDescriptorNative(convDesc, arrayLengthRequested, arrayLength, padA, strideA, upscaleA, mode, dataType));
    }
    private static native int cudnnGetConvolutionNdDescriptorNative(
        cudnnConvolutionDescriptor convDesc, 
        int arrayLengthRequested, 
        int[] arrayLength, 
        int[] padA, 
        int[] strideA, 
        int[] upscaleA, 
        int[] mode, 
        int[] dataType);// convolution data type


    /** cudnnSetConvolutionNdDescriptor_v3 is now mapped to cudnnSetConvolutionNdDescriptor
       and will be removed at the same time than cudnnSetConvolutionNdDescriptor_v2
       Use cudnnSetConvolutionNdDescriptor instead */
    public static int cudnnSetConvolutionNdDescriptor_v3(
        cudnnConvolutionDescriptor convDesc, 
        int arrayLength, /** nbDims-2 size */
        int[] padA, 
        int[] filterStrideA, 
        int[] upscaleA, 
        int mode, 
        int dataType)// convolution data type
    {
        return checkResult(cudnnSetConvolutionNdDescriptor_v3Native(convDesc, arrayLength, padA, filterStrideA, upscaleA, mode, dataType));
    }
    private static native int cudnnSetConvolutionNdDescriptor_v3Native(
        cudnnConvolutionDescriptor convDesc, 
        int arrayLength, /** nbDims-2 size */
        int[] padA, 
        int[] filterStrideA, 
        int[] upscaleA, 
        int mode, 
        int dataType);// convolution data type


    /**
     * <pre>
     * cudnnGetConvolutionNdDescriptor_v3 is now mapped to cudnnGetConvolutionNdDescriptor
       and will be removed at the same time thancudnnGetConvolutionNdDescriptor_v2
       Use cudnnGetConvolutionNdDescriptor instead
     * </pre>
     */
    public static int cudnnGetConvolutionNdDescriptor_v3(
        cudnnConvolutionDescriptor convDesc, 
        int arrayLengthRequested, 
        int[] arrayLength, 
        int[] padA, 
        int[] strideA, 
        int[] upscaleA, 
        int[] mode, 
        int[] dataType)// convolution data type
    {
        return checkResult(cudnnGetConvolutionNdDescriptor_v3Native(convDesc, arrayLengthRequested, arrayLength, padA, strideA, upscaleA, mode, dataType));
    }
    private static native int cudnnGetConvolutionNdDescriptor_v3Native(
        cudnnConvolutionDescriptor convDesc, 
        int arrayLengthRequested, 
        int[] arrayLength, 
        int[] padA, 
        int[] strideA, 
        int[] upscaleA, 
        int[] mode, 
        int[] dataType);// convolution data type


    /** Helper function to return the dimensions of the output tensor given a convolution descriptor */
    public static int cudnnGetConvolutionNdForwardOutputDim(
        cudnnConvolutionDescriptor convDesc, 
        cudnnTensorDescriptor inputTensorDesc, 
        cudnnFilterDescriptor filterDesc, 
        int nbDims, 
        int[] tensorOuputDimA)
    {
        return checkResult(cudnnGetConvolutionNdForwardOutputDimNative(convDesc, inputTensorDesc, filterDesc, nbDims, tensorOuputDimA));
    }
    private static native int cudnnGetConvolutionNdForwardOutputDimNative(
        cudnnConvolutionDescriptor convDesc, 
        cudnnTensorDescriptor inputTensorDesc, 
        cudnnFilterDescriptor filterDesc, 
        int nbDims, 
        int[] tensorOuputDimA);


    /** Destroy an instance of convolution descriptor */
    public static int cudnnDestroyConvolutionDescriptor(
        cudnnConvolutionDescriptor convDesc)
    {
        return checkResult(cudnnDestroyConvolutionDescriptorNative(convDesc));
    }
    private static native int cudnnDestroyConvolutionDescriptorNative(
        cudnnConvolutionDescriptor convDesc);


    public static int cudnnFindConvolutionForwardAlgorithm(
        cudnnHandle handle, 
        cudnnTensorDescriptor xDesc, 
        cudnnFilterDescriptor wDesc, 
        cudnnConvolutionDescriptor convDesc, 
        cudnnTensorDescriptor yDesc, 
        int requestedAlgoCount, 
        int[] returnedAlgoCount, 
        cudnnConvolutionFwdAlgoPerf[] perfResults)
    {
        return checkResult(cudnnFindConvolutionForwardAlgorithmNative(handle, xDesc, wDesc, convDesc, yDesc, requestedAlgoCount, returnedAlgoCount, perfResults));
    }
    private static native int cudnnFindConvolutionForwardAlgorithmNative(
        cudnnHandle handle, 
        cudnnTensorDescriptor xDesc, 
        cudnnFilterDescriptor wDesc, 
        cudnnConvolutionDescriptor convDesc, 
        cudnnTensorDescriptor yDesc, 
        int requestedAlgoCount, 
        int[] returnedAlgoCount, 
        cudnnConvolutionFwdAlgoPerf[] perfResults);


    public static int cudnnGetConvolutionForwardAlgorithm(
        cudnnHandle handle, 
        cudnnTensorDescriptor xDesc, 
        cudnnFilterDescriptor wDesc, 
        cudnnConvolutionDescriptor convDesc, 
        cudnnTensorDescriptor yDesc, 
        int preference, 
        long memoryLimitInBytes, 
        int[] algo)
    {
        return checkResult(cudnnGetConvolutionForwardAlgorithmNative(handle, xDesc, wDesc, convDesc, yDesc, preference, memoryLimitInBytes, algo));
    }
    private static native int cudnnGetConvolutionForwardAlgorithmNative(
        cudnnHandle handle, 
        cudnnTensorDescriptor xDesc, 
        cudnnFilterDescriptor wDesc, 
        cudnnConvolutionDescriptor convDesc, 
        cudnnTensorDescriptor yDesc, 
        int preference, 
        long memoryLimitInBytes, 
        int[] algo);


    /**
     *  convolution algorithm (which requires potentially some workspace)
     */
    /** Helper function to return the minimum size of the workspace to be passed to the convolution given an algo*/
    public static int cudnnGetConvolutionForwardWorkspaceSize(
        cudnnHandle handle, 
        cudnnTensorDescriptor xDesc, 
        cudnnFilterDescriptor wDesc, 
        cudnnConvolutionDescriptor convDesc, 
        cudnnTensorDescriptor yDesc, 
        int algo, 
        long[] sizeInBytes)
    {
        return checkResult(cudnnGetConvolutionForwardWorkspaceSizeNative(handle, xDesc, wDesc, convDesc, yDesc, algo, sizeInBytes));
    }
    private static native int cudnnGetConvolutionForwardWorkspaceSizeNative(
        cudnnHandle handle, 
        cudnnTensorDescriptor xDesc, 
        cudnnFilterDescriptor wDesc, 
        cudnnConvolutionDescriptor convDesc, 
        cudnnTensorDescriptor yDesc, 
        int algo, 
        long[] sizeInBytes);


    /** Convolution functions: All of the form "output = alpha * Op(inputs) + beta * output" */
    /** Function to perform the forward pass for batch convolution */
    public static int cudnnConvolutionForward(
        cudnnHandle handle, 
        Pointer alpha, 
        cudnnTensorDescriptor xDesc, 
        Pointer x, 
        cudnnFilterDescriptor wDesc, 
        Pointer w, 
        cudnnConvolutionDescriptor convDesc, 
        int algo, 
        Pointer workSpace, 
        long workSpaceSizeInBytes, 
        Pointer beta, 
        cudnnTensorDescriptor yDesc, 
        Pointer y)
    {
        return checkResult(cudnnConvolutionForwardNative(handle, alpha, xDesc, x, wDesc, w, convDesc, algo, workSpace, workSpaceSizeInBytes, beta, yDesc, y));
    }
    private static native int cudnnConvolutionForwardNative(
        cudnnHandle handle, 
        Pointer alpha, 
        cudnnTensorDescriptor xDesc, 
        Pointer x, 
        cudnnFilterDescriptor wDesc, 
        Pointer w, 
        cudnnConvolutionDescriptor convDesc, 
        int algo, 
        Pointer workSpace, 
        long workSpaceSizeInBytes, 
        Pointer beta, 
        cudnnTensorDescriptor yDesc, 
        Pointer y);


    /** Function to compute the bias gradient for batch convolution */
    public static int cudnnConvolutionBackwardBias(
        cudnnHandle handle, 
        Pointer alpha, 
        cudnnTensorDescriptor dyDesc, 
        Pointer dy, 
        Pointer beta, 
        cudnnTensorDescriptor dbDesc, 
        Pointer db)
    {
        return checkResult(cudnnConvolutionBackwardBiasNative(handle, alpha, dyDesc, dy, beta, dbDesc, db));
    }
    private static native int cudnnConvolutionBackwardBiasNative(
        cudnnHandle handle, 
        Pointer alpha, 
        cudnnTensorDescriptor dyDesc, 
        Pointer dy, 
        Pointer beta, 
        cudnnTensorDescriptor dbDesc, 
        Pointer db);


    public static int cudnnFindConvolutionBackwardFilterAlgorithm(
        cudnnHandle handle, 
        cudnnTensorDescriptor xDesc, 
        cudnnTensorDescriptor dyDesc, 
        cudnnConvolutionDescriptor convDesc, 
        cudnnFilterDescriptor dwDesc, 
        int requestedAlgoCount, 
        int[] returnedAlgoCount, 
        cudnnConvolutionBwdFilterAlgoPerf[] perfResults)
    {
        return checkResult(cudnnFindConvolutionBackwardFilterAlgorithmNative(handle, xDesc, dyDesc, convDesc, dwDesc, requestedAlgoCount, returnedAlgoCount, perfResults));
    }
    private static native int cudnnFindConvolutionBackwardFilterAlgorithmNative(
        cudnnHandle handle, 
        cudnnTensorDescriptor xDesc, 
        cudnnTensorDescriptor dyDesc, 
        cudnnConvolutionDescriptor convDesc, 
        cudnnFilterDescriptor dwDesc, 
        int requestedAlgoCount, 
        int[] returnedAlgoCount, 
        cudnnConvolutionBwdFilterAlgoPerf[] perfResults);


    public static int cudnnGetConvolutionBackwardFilterAlgorithm(
        cudnnHandle handle, 
        cudnnTensorDescriptor xDesc, 
        cudnnTensorDescriptor dyDesc, 
        cudnnConvolutionDescriptor convDesc, 
        cudnnFilterDescriptor dwDesc, 
        int preference, 
        long memoryLimitInBytes, 
        int[] algo)
    {
        return checkResult(cudnnGetConvolutionBackwardFilterAlgorithmNative(handle, xDesc, dyDesc, convDesc, dwDesc, preference, memoryLimitInBytes, algo));
    }
    private static native int cudnnGetConvolutionBackwardFilterAlgorithmNative(
        cudnnHandle handle, 
        cudnnTensorDescriptor xDesc, 
        cudnnTensorDescriptor dyDesc, 
        cudnnConvolutionDescriptor convDesc, 
        cudnnFilterDescriptor dwDesc, 
        int preference, 
        long memoryLimitInBytes, 
        int[] algo);


    /**
     *  convolution algorithm (which requires potentially some workspace)
     */
    /** Helper function to return the minimum size of the workspace to be passed to the convolution given an algo*/
    public static int cudnnGetConvolutionBackwardFilterWorkspaceSize(
        cudnnHandle handle, 
        cudnnTensorDescriptor xDesc, 
        cudnnTensorDescriptor dyDesc, 
        cudnnConvolutionDescriptor convDesc, 
        cudnnFilterDescriptor gradDesc, 
        int algo, 
        long[] sizeInBytes)
    {
        return checkResult(cudnnGetConvolutionBackwardFilterWorkspaceSizeNative(handle, xDesc, dyDesc, convDesc, gradDesc, algo, sizeInBytes));
    }
    private static native int cudnnGetConvolutionBackwardFilterWorkspaceSizeNative(
        cudnnHandle handle, 
        cudnnTensorDescriptor xDesc, 
        cudnnTensorDescriptor dyDesc, 
        cudnnConvolutionDescriptor convDesc, 
        cudnnFilterDescriptor gradDesc, 
        int algo, 
        long[] sizeInBytes);


    public static int cudnnConvolutionBackwardFilter(
        cudnnHandle handle, 
        Pointer alpha, 
        cudnnTensorDescriptor xDesc, 
        Pointer x, 
        cudnnTensorDescriptor dyDesc, 
        Pointer dy, 
        cudnnConvolutionDescriptor convDesc, 
        int algo, 
        Pointer workSpace, 
        long workSpaceSizeInBytes, 
        Pointer beta, 
        cudnnFilterDescriptor dwDesc, 
        Pointer dw)
    {
        return checkResult(cudnnConvolutionBackwardFilterNative(handle, alpha, xDesc, x, dyDesc, dy, convDesc, algo, workSpace, workSpaceSizeInBytes, beta, dwDesc, dw));
    }
    private static native int cudnnConvolutionBackwardFilterNative(
        cudnnHandle handle, 
        Pointer alpha, 
        cudnnTensorDescriptor xDesc, 
        Pointer x, 
        cudnnTensorDescriptor dyDesc, 
        Pointer dy, 
        cudnnConvolutionDescriptor convDesc, 
        int algo, 
        Pointer workSpace, 
        long workSpaceSizeInBytes, 
        Pointer beta, 
        cudnnFilterDescriptor dwDesc, 
        Pointer dw);


    /** cudnnConvolutionBackwardFilter_v3 is now mapped to cudnnConvolutionBackwardFilter
       and will be removed at the same time thancudnnConvolutionBackwardFilter_v2
       Use cudnnConvolutionBackwardFilter instead */
    public static int cudnnConvolutionBackwardFilter_v3(
        cudnnHandle handle, 
        Pointer alpha, 
        cudnnTensorDescriptor xDesc, 
        Pointer x, 
        cudnnTensorDescriptor dyDesc, 
        Pointer dy, 
        cudnnConvolutionDescriptor convDesc, 
        int algo, 
        Pointer workSpace, 
        long workSpaceSizeInBytes, 
        Pointer beta, 
        cudnnFilterDescriptor dwDesc, 
        Pointer dw)
    {
        return checkResult(cudnnConvolutionBackwardFilter_v3Native(handle, alpha, xDesc, x, dyDesc, dy, convDesc, algo, workSpace, workSpaceSizeInBytes, beta, dwDesc, dw));
    }
    private static native int cudnnConvolutionBackwardFilter_v3Native(
        cudnnHandle handle, 
        Pointer alpha, 
        cudnnTensorDescriptor xDesc, 
        Pointer x, 
        cudnnTensorDescriptor dyDesc, 
        Pointer dy, 
        cudnnConvolutionDescriptor convDesc, 
        int algo, 
        Pointer workSpace, 
        long workSpaceSizeInBytes, 
        Pointer beta, 
        cudnnFilterDescriptor dwDesc, 
        Pointer dw);


    public static int cudnnFindConvolutionBackwardDataAlgorithm(
        cudnnHandle handle, 
        cudnnFilterDescriptor wDesc, 
        cudnnTensorDescriptor dyDesc, 
        cudnnConvolutionDescriptor convDesc, 
        cudnnTensorDescriptor dxDesc, 
        int requestedAlgoCount, 
        int[] returnedAlgoCount, 
        cudnnConvolutionBwdDataAlgoPerf[] perfResults)
    {
        return checkResult(cudnnFindConvolutionBackwardDataAlgorithmNative(handle, wDesc, dyDesc, convDesc, dxDesc, requestedAlgoCount, returnedAlgoCount, perfResults));
    }
    private static native int cudnnFindConvolutionBackwardDataAlgorithmNative(
        cudnnHandle handle, 
        cudnnFilterDescriptor wDesc, 
        cudnnTensorDescriptor dyDesc, 
        cudnnConvolutionDescriptor convDesc, 
        cudnnTensorDescriptor dxDesc, 
        int requestedAlgoCount, 
        int[] returnedAlgoCount, 
        cudnnConvolutionBwdDataAlgoPerf[] perfResults);


    public static int cudnnGetConvolutionBackwardDataAlgorithm(
        cudnnHandle handle, 
        cudnnFilterDescriptor wDesc, 
        cudnnTensorDescriptor dyDesc, 
        cudnnConvolutionDescriptor convDesc, 
        cudnnTensorDescriptor dxDesc, 
        int preference, 
        long memoryLimitInBytes, 
        int[] algo)
    {
        return checkResult(cudnnGetConvolutionBackwardDataAlgorithmNative(handle, wDesc, dyDesc, convDesc, dxDesc, preference, memoryLimitInBytes, algo));
    }
    private static native int cudnnGetConvolutionBackwardDataAlgorithmNative(
        cudnnHandle handle, 
        cudnnFilterDescriptor wDesc, 
        cudnnTensorDescriptor dyDesc, 
        cudnnConvolutionDescriptor convDesc, 
        cudnnTensorDescriptor dxDesc, 
        int preference, 
        long memoryLimitInBytes, 
        int[] algo);


    /** Helper function to return the minimum size of the workspace to be passed to the convolution given an algo*/
    public static int cudnnGetConvolutionBackwardDataWorkspaceSize(
        cudnnHandle handle, 
        cudnnFilterDescriptor wDesc, 
        cudnnTensorDescriptor dyDesc, 
        cudnnConvolutionDescriptor convDesc, 
        cudnnTensorDescriptor dxDesc, 
        int algo, 
        long[] sizeInBytes)
    {
        return checkResult(cudnnGetConvolutionBackwardDataWorkspaceSizeNative(handle, wDesc, dyDesc, convDesc, dxDesc, algo, sizeInBytes));
    }
    private static native int cudnnGetConvolutionBackwardDataWorkspaceSizeNative(
        cudnnHandle handle, 
        cudnnFilterDescriptor wDesc, 
        cudnnTensorDescriptor dyDesc, 
        cudnnConvolutionDescriptor convDesc, 
        cudnnTensorDescriptor dxDesc, 
        int algo, 
        long[] sizeInBytes);


    public static int cudnnConvolutionBackwardData(
        cudnnHandle handle, 
        Pointer alpha, 
        cudnnFilterDescriptor wDesc, 
        Pointer w, 
        cudnnTensorDescriptor dyDesc, 
        Pointer dy, 
        cudnnConvolutionDescriptor convDesc, 
        int algo, 
        Pointer workSpace, 
        long workSpaceSizeInBytes, 
        Pointer beta, 
        cudnnTensorDescriptor dxDesc, 
        Pointer dx)
    {
        return checkResult(cudnnConvolutionBackwardDataNative(handle, alpha, wDesc, w, dyDesc, dy, convDesc, algo, workSpace, workSpaceSizeInBytes, beta, dxDesc, dx));
    }
    private static native int cudnnConvolutionBackwardDataNative(
        cudnnHandle handle, 
        Pointer alpha, 
        cudnnFilterDescriptor wDesc, 
        Pointer w, 
        cudnnTensorDescriptor dyDesc, 
        Pointer dy, 
        cudnnConvolutionDescriptor convDesc, 
        int algo, 
        Pointer workSpace, 
        long workSpaceSizeInBytes, 
        Pointer beta, 
        cudnnTensorDescriptor dxDesc, 
        Pointer dx);


    /** cudnnConvolutionBackwardData_v3 is now mapped to cudnnConvolutionBackwardData
       and will be removed at the same time thancudnnConvolutionBackwardData_v2
       Use cudnnConvolutionBackwardData instead */
    public static int cudnnConvolutionBackwardData_v3(
        cudnnHandle handle, 
        Pointer alpha, 
        cudnnFilterDescriptor wDesc, 
        Pointer w, 
        cudnnTensorDescriptor dyDesc, 
        Pointer dy, 
        cudnnConvolutionDescriptor convDesc, 
        int algo, 
        Pointer workSpace, 
        long workSpaceSizeInBytes, 
        Pointer beta, 
        cudnnTensorDescriptor dxDesc, 
        Pointer dx)
    {
        return checkResult(cudnnConvolutionBackwardData_v3Native(handle, alpha, wDesc, w, dyDesc, dy, convDesc, algo, workSpace, workSpaceSizeInBytes, beta, dxDesc, dx));
    }
    private static native int cudnnConvolutionBackwardData_v3Native(
        cudnnHandle handle, 
        Pointer alpha, 
        cudnnFilterDescriptor wDesc, 
        Pointer w, 
        cudnnTensorDescriptor dyDesc, 
        Pointer dy, 
        cudnnConvolutionDescriptor convDesc, 
        int algo, 
        Pointer workSpace, 
        long workSpaceSizeInBytes, 
        Pointer beta, 
        cudnnTensorDescriptor dxDesc, 
        Pointer dx);


    public static int cudnnIm2Col(
        cudnnHandle handle, 
        cudnnTensorDescriptor xDesc, 
        Pointer x, 
        cudnnFilterDescriptor wDesc, 
        cudnnConvolutionDescriptor convDesc, 
        Pointer colBuffer)
    {
        return checkResult(cudnnIm2ColNative(handle, xDesc, x, wDesc, convDesc, colBuffer));
    }
    private static native int cudnnIm2ColNative(
        cudnnHandle handle, 
        cudnnTensorDescriptor xDesc, 
        Pointer x, 
        cudnnFilterDescriptor wDesc, 
        cudnnConvolutionDescriptor convDesc, 
        Pointer colBuffer);


    /** Softmax functions: All of the form "output = alpha * Op(inputs) + beta * output" */
    /** Function to perform forward softmax */
    public static int cudnnSoftmaxForward(
        cudnnHandle handle, 
        int algo, 
        int mode, 
        Pointer alpha, 
        cudnnTensorDescriptor xDesc, 
        Pointer x, 
        Pointer beta, 
        cudnnTensorDescriptor yDesc, 
        Pointer y)
    {
        return checkResult(cudnnSoftmaxForwardNative(handle, algo, mode, alpha, xDesc, x, beta, yDesc, y));
    }
    private static native int cudnnSoftmaxForwardNative(
        cudnnHandle handle, 
        int algo, 
        int mode, 
        Pointer alpha, 
        cudnnTensorDescriptor xDesc, 
        Pointer x, 
        Pointer beta, 
        cudnnTensorDescriptor yDesc, 
        Pointer y);


    /** Function to perform backward softmax */
    public static int cudnnSoftmaxBackward(
        cudnnHandle handle, 
        int algo, 
        int mode, 
        Pointer alpha, 
        cudnnTensorDescriptor yDesc, 
        Pointer y, 
        cudnnTensorDescriptor dyDesc, 
        Pointer dy, 
        Pointer beta, 
        cudnnTensorDescriptor dxDesc, 
        Pointer dx)
    {
        return checkResult(cudnnSoftmaxBackwardNative(handle, algo, mode, alpha, yDesc, y, dyDesc, dy, beta, dxDesc, dx));
    }
    private static native int cudnnSoftmaxBackwardNative(
        cudnnHandle handle, 
        int algo, 
        int mode, 
        Pointer alpha, 
        cudnnTensorDescriptor yDesc, 
        Pointer y, 
        cudnnTensorDescriptor dyDesc, 
        Pointer dy, 
        Pointer beta, 
        cudnnTensorDescriptor dxDesc, 
        Pointer dx);


    /** Create an instance of pooling descriptor */
    public static int cudnnCreatePoolingDescriptor(
        cudnnPoolingDescriptor poolingDesc)
    {
        return checkResult(cudnnCreatePoolingDescriptorNative(poolingDesc));
    }
    private static native int cudnnCreatePoolingDescriptorNative(
        cudnnPoolingDescriptor poolingDesc);


    public static int cudnnSetPooling2dDescriptor(
        cudnnPoolingDescriptor poolingDesc, 
        int mode, 
        int windowHeight, 
        int windowWidth, 
        int verticalPadding, 
        int horizontalPadding, 
        int verticalStride, 
        int horizontalStride)
    {
        return checkResult(cudnnSetPooling2dDescriptorNative(poolingDesc, mode, windowHeight, windowWidth, verticalPadding, horizontalPadding, verticalStride, horizontalStride));
    }
    private static native int cudnnSetPooling2dDescriptorNative(
        cudnnPoolingDescriptor poolingDesc, 
        int mode, 
        int windowHeight, 
        int windowWidth, 
        int verticalPadding, 
        int horizontalPadding, 
        int verticalStride, 
        int horizontalStride);


    public static int cudnnSetPooling2dDescriptor_v4(
        cudnnPoolingDescriptor poolingDesc, 
        int mode, 
        int maxpoolingNanOpt, 
        int windowHeight, 
        int windowWidth, 
        int verticalPadding, 
        int horizontalPadding, 
        int verticalStride, 
        int horizontalStride)
    {
        return checkResult(cudnnSetPooling2dDescriptor_v4Native(poolingDesc, mode, maxpoolingNanOpt, windowHeight, windowWidth, verticalPadding, horizontalPadding, verticalStride, horizontalStride));
    }
    private static native int cudnnSetPooling2dDescriptor_v4Native(
        cudnnPoolingDescriptor poolingDesc, 
        int mode, 
        int maxpoolingNanOpt, 
        int windowHeight, 
        int windowWidth, 
        int verticalPadding, 
        int horizontalPadding, 
        int verticalStride, 
        int horizontalStride);


    public static int cudnnGetPooling2dDescriptor(
        cudnnPoolingDescriptor poolingDesc, 
        int[] mode, 
        Pointer windowHeight, 
        Pointer windowWidth, 
        Pointer verticalPadding, 
        Pointer horizontalPadding, 
        Pointer verticalStride, 
        Pointer horizontalStride)
    {
        return checkResult(cudnnGetPooling2dDescriptorNative(poolingDesc, mode, windowHeight, windowWidth, verticalPadding, horizontalPadding, verticalStride, horizontalStride));
    }
    private static native int cudnnGetPooling2dDescriptorNative(
        cudnnPoolingDescriptor poolingDesc, 
        int[] mode, 
        Pointer windowHeight, 
        Pointer windowWidth, 
        Pointer verticalPadding, 
        Pointer horizontalPadding, 
        Pointer verticalStride, 
        Pointer horizontalStride);


    public static int cudnnGetPooling2dDescriptor_v4(
        cudnnPoolingDescriptor poolingDesc, 
        int[] mode, 
        int[] maxpoolingNanOpt, 
        Pointer windowHeight, 
        Pointer windowWidth, 
        Pointer verticalPadding, 
        Pointer horizontalPadding, 
        Pointer verticalStride, 
        Pointer horizontalStride)
    {
        return checkResult(cudnnGetPooling2dDescriptor_v4Native(poolingDesc, mode, maxpoolingNanOpt, windowHeight, windowWidth, verticalPadding, horizontalPadding, verticalStride, horizontalStride));
    }
    private static native int cudnnGetPooling2dDescriptor_v4Native(
        cudnnPoolingDescriptor poolingDesc, 
        int[] mode, 
        int[] maxpoolingNanOpt, 
        Pointer windowHeight, 
        Pointer windowWidth, 
        Pointer verticalPadding, 
        Pointer horizontalPadding, 
        Pointer verticalStride, 
        Pointer horizontalStride);


    public static int cudnnSetPoolingNdDescriptor(
        cudnnPoolingDescriptor poolingDesc, 
        int mode, 
        int nbDims, 
        int[] windowDimA, 
        int[] paddingA, 
        int[] strideA)
    {
        return checkResult(cudnnSetPoolingNdDescriptorNative(poolingDesc, mode, nbDims, windowDimA, paddingA, strideA));
    }
    private static native int cudnnSetPoolingNdDescriptorNative(
        cudnnPoolingDescriptor poolingDesc, 
        int mode, 
        int nbDims, 
        int[] windowDimA, 
        int[] paddingA, 
        int[] strideA);


    public static int cudnnSetPoolingNdDescriptor_v4(
        cudnnPoolingDescriptor poolingDesc, 
        int mode, 
        int maxpoolingNanOpt, 
        int nbDims, 
        int[] windowDimA, 
        int[] paddingA, 
        int[] strideA)
    {
        return checkResult(cudnnSetPoolingNdDescriptor_v4Native(poolingDesc, mode, maxpoolingNanOpt, nbDims, windowDimA, paddingA, strideA));
    }
    private static native int cudnnSetPoolingNdDescriptor_v4Native(
        cudnnPoolingDescriptor poolingDesc, 
        int mode, 
        int maxpoolingNanOpt, 
        int nbDims, 
        int[] windowDimA, 
        int[] paddingA, 
        int[] strideA);


    public static int cudnnGetPoolingNdDescriptor(
        cudnnPoolingDescriptor poolingDesc, 
        int nbDimsRequested, 
        int[] mode, 
        Pointer nbDims, 
        int[] windowDimA, 
        int[] paddingA, 
        int[] strideA)
    {
        return checkResult(cudnnGetPoolingNdDescriptorNative(poolingDesc, nbDimsRequested, mode, nbDims, windowDimA, paddingA, strideA));
    }
    private static native int cudnnGetPoolingNdDescriptorNative(
        cudnnPoolingDescriptor poolingDesc, 
        int nbDimsRequested, 
        int[] mode, 
        Pointer nbDims, 
        int[] windowDimA, 
        int[] paddingA, 
        int[] strideA);


    public static int cudnnGetPoolingNdDescriptor_v4(
        cudnnPoolingDescriptor poolingDesc, 
        int nbDimsRequested, 
        int[] mode, 
        int[] maxpoolingNanOpt, 
        Pointer nbDims, 
        int[] windowDimA, 
        int[] paddingA, 
        int[] strideA)
    {
        return checkResult(cudnnGetPoolingNdDescriptor_v4Native(poolingDesc, nbDimsRequested, mode, maxpoolingNanOpt, nbDims, windowDimA, paddingA, strideA));
    }
    private static native int cudnnGetPoolingNdDescriptor_v4Native(
        cudnnPoolingDescriptor poolingDesc, 
        int nbDimsRequested, 
        int[] mode, 
        int[] maxpoolingNanOpt, 
        Pointer nbDims, 
        int[] windowDimA, 
        int[] paddingA, 
        int[] strideA);


    public static int cudnnGetPoolingNdForwardOutputDim(
        cudnnPoolingDescriptor poolingDesc, 
        cudnnTensorDescriptor inputTensorDesc, 
        int nbDims, 
        int[] outputTensorDimA)
    {
        return checkResult(cudnnGetPoolingNdForwardOutputDimNative(poolingDesc, inputTensorDesc, nbDims, outputTensorDimA));
    }
    private static native int cudnnGetPoolingNdForwardOutputDimNative(
        cudnnPoolingDescriptor poolingDesc, 
        cudnnTensorDescriptor inputTensorDesc, 
        int nbDims, 
        int[] outputTensorDimA);


    public static int cudnnGetPooling2dForwardOutputDim(
        cudnnPoolingDescriptor poolingDesc, 
        cudnnTensorDescriptor inputTensorDesc, 
        Pointer n, 
        Pointer c, 
        Pointer h, 
        Pointer w)
    {
        return checkResult(cudnnGetPooling2dForwardOutputDimNative(poolingDesc, inputTensorDesc, n, c, h, w));
    }
    private static native int cudnnGetPooling2dForwardOutputDimNative(
        cudnnPoolingDescriptor poolingDesc, 
        cudnnTensorDescriptor inputTensorDesc, 
        Pointer n, 
        Pointer c, 
        Pointer h, 
        Pointer w);


    /** Destroy an instance of pooling descriptor */
    public static int cudnnDestroyPoolingDescriptor(
        cudnnPoolingDescriptor poolingDesc)
    {
        return checkResult(cudnnDestroyPoolingDescriptorNative(poolingDesc));
    }
    private static native int cudnnDestroyPoolingDescriptorNative(
        cudnnPoolingDescriptor poolingDesc);


    /** Pooling functions: All of the form "output = alpha * Op(inputs) + beta * output" */
    /** Function to perform forward pooling */
    public static int cudnnPoolingForward(
        cudnnHandle handle, 
        cudnnPoolingDescriptor poolingDesc, 
        Pointer alpha, 
        cudnnTensorDescriptor xDesc, 
        Pointer x, 
        Pointer beta, 
        cudnnTensorDescriptor yDesc, 
        Pointer y)
    {
        return checkResult(cudnnPoolingForwardNative(handle, poolingDesc, alpha, xDesc, x, beta, yDesc, y));
    }
    private static native int cudnnPoolingForwardNative(
        cudnnHandle handle, 
        cudnnPoolingDescriptor poolingDesc, 
        Pointer alpha, 
        cudnnTensorDescriptor xDesc, 
        Pointer x, 
        Pointer beta, 
        cudnnTensorDescriptor yDesc, 
        Pointer y);


    /** Function to perform backward pooling */
    public static int cudnnPoolingBackward(
        cudnnHandle handle, 
        cudnnPoolingDescriptor poolingDesc, 
        Pointer alpha, 
        cudnnTensorDescriptor yDesc, 
        Pointer y, 
        cudnnTensorDescriptor dyDesc, 
        Pointer dy, 
        cudnnTensorDescriptor xDesc, 
        Pointer x, 
        Pointer beta, 
        cudnnTensorDescriptor dxDesc, 
        Pointer dx)
    {
        return checkResult(cudnnPoolingBackwardNative(handle, poolingDesc, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx));
    }
    private static native int cudnnPoolingBackwardNative(
        cudnnHandle handle, 
        cudnnPoolingDescriptor poolingDesc, 
        Pointer alpha, 
        cudnnTensorDescriptor yDesc, 
        Pointer y, 
        cudnnTensorDescriptor dyDesc, 
        Pointer dy, 
        cudnnTensorDescriptor xDesc, 
        Pointer x, 
        Pointer beta, 
        cudnnTensorDescriptor dxDesc, 
        Pointer dx);


    /** Activation functions: All of the form "output = alpha * Op(inputs) + beta * output" */
    public static int cudnnCreateActivationDescriptor(
        cudnnActivationDescriptor activationDesc)
    {
        return checkResult(cudnnCreateActivationDescriptorNative(activationDesc));
    }
    private static native int cudnnCreateActivationDescriptorNative(
        cudnnActivationDescriptor activationDesc);


    public static int cudnnSetActivationDescriptor(
        cudnnActivationDescriptor activationDesc, 
        int mode, 
        int reluNanOpt, 
        double reluCeiling)
    {
        return checkResult(cudnnSetActivationDescriptorNative(activationDesc, mode, reluNanOpt, reluCeiling));
    }
    private static native int cudnnSetActivationDescriptorNative(
        cudnnActivationDescriptor activationDesc, 
        int mode, 
        int reluNanOpt, 
        double reluCeiling);


    public static int cudnnGetActivationDescriptor(
        cudnnActivationDescriptor activationDesc, 
        int[] mode, 
        int[] reluNanOpt, 
        Pointer reluCeiling)
    {
        return checkResult(cudnnGetActivationDescriptorNative(activationDesc, mode, reluNanOpt, reluCeiling));
    }
    private static native int cudnnGetActivationDescriptorNative(
        cudnnActivationDescriptor activationDesc, 
        int[] mode, 
        int[] reluNanOpt, 
        Pointer reluCeiling);


    public static int cudnnDestroyActivationDescriptor(
        cudnnActivationDescriptor activationDesc)
    {
        return checkResult(cudnnDestroyActivationDescriptorNative(activationDesc));
    }
    private static native int cudnnDestroyActivationDescriptorNative(
        cudnnActivationDescriptor activationDesc);


    /** Function to perform forward activation  */
    public static int cudnnActivationForward(
        cudnnHandle handle, 
        int mode, 
        Pointer alpha, 
        cudnnTensorDescriptor xDesc, 
        Pointer x, 
        Pointer beta, 
        cudnnTensorDescriptor yDesc, 
        Pointer y)
    {
        return checkResult(cudnnActivationForwardNative(handle, mode, alpha, xDesc, x, beta, yDesc, y));
    }
    private static native int cudnnActivationForwardNative(
        cudnnHandle handle, 
        int mode, 
        Pointer alpha, 
        cudnnTensorDescriptor xDesc, 
        Pointer x, 
        Pointer beta, 
        cudnnTensorDescriptor yDesc, 
        Pointer y);


    public static int cudnnActivationForward_v4(
        cudnnHandle handle, 
        cudnnActivationDescriptor activationDesc, 
        Pointer alpha, 
        cudnnTensorDescriptor xDesc, 
        Pointer x, 
        Pointer beta, 
        cudnnTensorDescriptor yDesc, 
        Pointer y)
    {
        return checkResult(cudnnActivationForward_v4Native(handle, activationDesc, alpha, xDesc, x, beta, yDesc, y));
    }
    private static native int cudnnActivationForward_v4Native(
        cudnnHandle handle, 
        cudnnActivationDescriptor activationDesc, 
        Pointer alpha, 
        cudnnTensorDescriptor xDesc, 
        Pointer x, 
        Pointer beta, 
        cudnnTensorDescriptor yDesc, 
        Pointer y);


    /** Function to perform backward activation  */
    public static int cudnnActivationBackward(
        cudnnHandle handle, 
        int mode, 
        Pointer alpha, 
        cudnnTensorDescriptor yDesc, 
        Pointer y, 
        cudnnTensorDescriptor dyDesc, 
        Pointer dy, 
        cudnnTensorDescriptor xDesc, 
        Pointer x, 
        Pointer beta, 
        cudnnTensorDescriptor dxDesc, 
        Pointer dx)
    {
        return checkResult(cudnnActivationBackwardNative(handle, mode, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx));
    }
    private static native int cudnnActivationBackwardNative(
        cudnnHandle handle, 
        int mode, 
        Pointer alpha, 
        cudnnTensorDescriptor yDesc, 
        Pointer y, 
        cudnnTensorDescriptor dyDesc, 
        Pointer dy, 
        cudnnTensorDescriptor xDesc, 
        Pointer x, 
        Pointer beta, 
        cudnnTensorDescriptor dxDesc, 
        Pointer dx);


    public static int cudnnActivationBackward_v4(
        cudnnHandle handle, 
        cudnnActivationDescriptor activationDesc, 
        Pointer alpha, 
        cudnnTensorDescriptor yDesc, 
        Pointer y, 
        cudnnTensorDescriptor dyDesc, 
        Pointer dy, 
        cudnnTensorDescriptor xDesc, 
        Pointer x, 
        Pointer beta, 
        cudnnTensorDescriptor dxDesc, 
        Pointer dx)
    {
        return checkResult(cudnnActivationBackward_v4Native(handle, activationDesc, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx));
    }
    private static native int cudnnActivationBackward_v4Native(
        cudnnHandle handle, 
        cudnnActivationDescriptor activationDesc, 
        Pointer alpha, 
        cudnnTensorDescriptor yDesc, 
        Pointer y, 
        cudnnTensorDescriptor dyDesc, 
        Pointer dy, 
        cudnnTensorDescriptor xDesc, 
        Pointer x, 
        Pointer beta, 
        cudnnTensorDescriptor dxDesc, 
        Pointer dx);


    /**
     * <pre>
    * Create an instance of LRN (Local Response Normalization) descriptor
    * Uses lrnN=5, lrnAlpha=1e-4, lrnBeta=0.75, lrnK=2.0 as defaults from Krizhevsky'12 ImageNet paper
     * </pre>
     */
    public static int cudnnCreateLRNDescriptor(
        cudnnLRNDescriptor normDesc)
    {
        return checkResult(cudnnCreateLRNDescriptorNative(normDesc));
    }
    private static native int cudnnCreateLRNDescriptorNative(
        cudnnLRNDescriptor normDesc);


    /**
     * <pre>
    * Uses a window [center-lookBehind, center+lookAhead], where
    * lookBehind = floor( (lrnN-1)/2 ), lookAhead = lrnN-lookBehind-1.
    * Values of double parameters cast to tensor data type.
     * </pre>
     */
    public static int cudnnSetLRNDescriptor(
        cudnnLRNDescriptor normDesc, 
        int lrnN, 
        double lrnAlpha, 
        double lrnBeta, 
        double lrnK)
    {
        return checkResult(cudnnSetLRNDescriptorNative(normDesc, lrnN, lrnAlpha, lrnBeta, lrnK));
    }
    private static native int cudnnSetLRNDescriptorNative(
        cudnnLRNDescriptor normDesc, 
        int lrnN, 
        double lrnAlpha, 
        double lrnBeta, 
        double lrnK);


    /**
     * <pre>
    * Retrieve the settings currently stored in an LRN layer descriptor
    * Any of the provided pointers can be NULL (no corresponding value will be returned)
     * </pre>
     */
    public static int cudnnGetLRNDescriptor(
        cudnnLRNDescriptor normDesc, 
        int[] lrnN, 
        Pointer lrnAlpha, 
        Pointer lrnBeta, 
        Pointer lrnK)
    {
        return checkResult(cudnnGetLRNDescriptorNative(normDesc, lrnN, lrnAlpha, lrnBeta, lrnK));
    }
    private static native int cudnnGetLRNDescriptorNative(
        cudnnLRNDescriptor normDesc, 
        int[] lrnN, 
        Pointer lrnAlpha, 
        Pointer lrnBeta, 
        Pointer lrnK);


    /** Destroy an instance of LRN descriptor */
    public static int cudnnDestroyLRNDescriptor(
        cudnnLRNDescriptor lrnDesc)
    {
        return checkResult(cudnnDestroyLRNDescriptorNative(lrnDesc));
    }
    private static native int cudnnDestroyLRNDescriptorNative(
        cudnnLRNDescriptor lrnDesc);


    /** LRN functions: output = alpha * normalize(x) + beta * old_y */
    /** LRN cross-channel forward computation. Double parameters cast to tensor data type */
    public static int cudnnLRNCrossChannelForward(
        cudnnHandle handle, 
        cudnnLRNDescriptor normDesc, 
        int lrnMode, 
        Pointer alpha, 
        cudnnTensorDescriptor xDesc, 
        Pointer x, 
        Pointer beta, 
        cudnnTensorDescriptor yDesc, 
        Pointer y)
    {
        return checkResult(cudnnLRNCrossChannelForwardNative(handle, normDesc, lrnMode, alpha, xDesc, x, beta, yDesc, y));
    }
    private static native int cudnnLRNCrossChannelForwardNative(
        cudnnHandle handle, 
        cudnnLRNDescriptor normDesc, 
        int lrnMode, 
        Pointer alpha, 
        cudnnTensorDescriptor xDesc, 
        Pointer x, 
        Pointer beta, 
        cudnnTensorDescriptor yDesc, 
        Pointer y);


    /** LRN cross-channel backward computation. Double parameters cast to tensor data type */
    public static int cudnnLRNCrossChannelBackward(
        cudnnHandle handle, 
        cudnnLRNDescriptor normDesc, 
        int lrnMode, 
        Pointer alpha, 
        cudnnTensorDescriptor yDesc, 
        Pointer y, 
        cudnnTensorDescriptor dyDesc, 
        Pointer dy, 
        cudnnTensorDescriptor xDesc, 
        Pointer x, 
        Pointer beta, 
        cudnnTensorDescriptor dxDesc, 
        Pointer dx)
    {
        return checkResult(cudnnLRNCrossChannelBackwardNative(handle, normDesc, lrnMode, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx));
    }
    private static native int cudnnLRNCrossChannelBackwardNative(
        cudnnHandle handle, 
        cudnnLRNDescriptor normDesc, 
        int lrnMode, 
        Pointer alpha, 
        cudnnTensorDescriptor yDesc, 
        Pointer y, 
        cudnnTensorDescriptor dyDesc, 
        Pointer dy, 
        cudnnTensorDescriptor xDesc, 
        Pointer x, 
        Pointer beta, 
        cudnnTensorDescriptor dxDesc, 
        Pointer dx);


    /** LCN/divisive normalization functions: y = alpha * normalize(x) + beta * y */
    public static int cudnnDivisiveNormalizationForward(
        cudnnHandle handle, 
        cudnnLRNDescriptor normDesc, 
        int mode, 
        Pointer alpha, 
        cudnnTensorDescriptor xDesc, // same desc for means, temp, temp2
        Pointer x, 
        Pointer means, // if NULL, means are assumed to be zero
        Pointer temp, 
        Pointer temp2, 
        Pointer beta, 
        cudnnTensorDescriptor yDesc, 
        Pointer y)
    {
        return checkResult(cudnnDivisiveNormalizationForwardNative(handle, normDesc, mode, alpha, xDesc, x, means, temp, temp2, beta, yDesc, y));
    }
    private static native int cudnnDivisiveNormalizationForwardNative(
        cudnnHandle handle, 
        cudnnLRNDescriptor normDesc, 
        int mode, 
        Pointer alpha, 
        cudnnTensorDescriptor xDesc, // same desc for means, temp, temp2
        Pointer x, 
        Pointer means, // if NULL, means are assumed to be zero
        Pointer temp, 
        Pointer temp2, 
        Pointer beta, 
        cudnnTensorDescriptor yDesc, 
        Pointer y);


    public static int cudnnDivisiveNormalizationBackward(
        cudnnHandle handle, 
        cudnnLRNDescriptor normDesc, 
        int mode, 
        Pointer alpha, 
        cudnnTensorDescriptor xDesc, // same desc for x, means, dy, temp, temp2
        Pointer x, 
        Pointer means, // if NULL, means are assumed to be zero
        Pointer dy, 
        Pointer temp, 
        Pointer temp2, 
        Pointer beta, 
        cudnnTensorDescriptor dXdMeansDesc, // same desc for dx, dMeans
        Pointer dx, // output x differential
        Pointer dMeans)// output means differential, can be NULL
    {
        return checkResult(cudnnDivisiveNormalizationBackwardNative(handle, normDesc, mode, alpha, xDesc, x, means, dy, temp, temp2, beta, dXdMeansDesc, dx, dMeans));
    }
    private static native int cudnnDivisiveNormalizationBackwardNative(
        cudnnHandle handle, 
        cudnnLRNDescriptor normDesc, 
        int mode, 
        Pointer alpha, 
        cudnnTensorDescriptor xDesc, // same desc for x, means, dy, temp, temp2
        Pointer x, 
        Pointer means, // if NULL, means are assumed to be zero
        Pointer dy, 
        Pointer temp, 
        Pointer temp2, 
        Pointer beta, 
        cudnnTensorDescriptor dXdMeansDesc, // same desc for dx, dMeans
        Pointer dx, // output x differential
        Pointer dMeans);// output means differential, can be NULL


    /**
     * <pre>
    * Derives a tensor descriptor from layer data descriptor for BatchNormalization 
    * scale, invVariance, bnBias, bnScale tensors. Use this tensor desc for 
    * bnScaleBiasMeanVarDesc and bnScaleBiasDiffDesc in Batch Normalization forward and backward functions.
     * </pre>
     */
    public static int cudnnDeriveBNTensorDescriptor(
        cudnnTensorDescriptor derivedBnDesc, 
        cudnnTensorDescriptor xDesc, 
        int mode)
    {
        return checkResult(cudnnDeriveBNTensorDescriptorNative(derivedBnDesc, xDesc, mode));
    }
    private static native int cudnnDeriveBNTensorDescriptorNative(
        cudnnTensorDescriptor derivedBnDesc, 
        cudnnTensorDescriptor xDesc, 
        int mode);


    /** Computes y = BN(x). Also accumulates moving averages of mean and inverse variances */
    public static int cudnnBatchNormalizationForwardTraining(
        cudnnHandle handle, 
        int mode, 
        Pointer alpha, // alpha[0] = result blend factor
        Pointer beta, // beta[0] = dest layer blend factor
        cudnnTensorDescriptor xDesc, 
        Pointer x, // NxCxHxW
        cudnnTensorDescriptor yDesc, 
        Pointer y, // NxCxHxW
        /**
         * <pre>
         * Shared desc for the next 6 tensors in the argument list.
                                           Data type to be set as follows:
                                           type = (typeOf(x) == double) ? double : float
                                           Dimensions for this descriptor depend on normalization mode
                                           - Spatial Normalization : tensors are expected to have dims 1xCx1x1
                                            (normalization is performed across NxHxW)
                                           - Per-Activation Normalization : tensors are expected to have dims of 1xCxHxW 
         * (normalization is performed across N)
         * </pre>
         */
        cudnnTensorDescriptor bnScaleBiasMeanVarDesc, 
        // 'Gamma' and 'Beta' respectively in Ioffe and Szegedy's paper's notation
        Pointer bnScale, 
        Pointer bnBias, 
        /**
         * <pre>
         * MUST use factor=1 in the very first call of a complete training cycle.
                                           Use a factor=1/(1+n) at N-th call to the function to get
                                           Cumulative Moving Average (CMA) behavior
                                           CMA[n] = (x[1]+...+x[n])/n
                                           Since CMA[n+1] = (n*CMA[n]+x[n+1])/(n+1) =
                                           ((n+1)*CMA[n]-CMA[n])/(n+1) + x[n+1]/(n+1) =
         * CMA[n]*(1-1/(n+1)) + x[n+1]*1/(n+1)
         * </pre>
         */
        double exponentialAverageFactor, 
        /** Used in Training phase only. 
                                           runningMean = newMean*factor + runningMean*(1-factor) */
        Pointer resultRunningMean, 
        /** Output in training mode, input in inference. Is the moving average
                                           of 1 / sqrt( epsilon + variance[x] ) */
        Pointer resultRunningInvVariance, 
        /** Has to be >= CUDNN_BN_MIN_EPSILON. Should be the same in forward and backward functions. */
        double epsilon, 
        /** Optionally save intermediate results from the forward pass here
                                           - can be reused to speed up backward pass. NULL if unused */
        Pointer resultSaveMean, 
        Pointer resultSaveInvVariance)
    {
        return checkResult(cudnnBatchNormalizationForwardTrainingNative(handle, mode, alpha, beta, xDesc, x, yDesc, y, bnScaleBiasMeanVarDesc, bnScale, bnBias, exponentialAverageFactor, resultRunningMean, resultRunningInvVariance, epsilon, resultSaveMean, resultSaveInvVariance));
    }
    private static native int cudnnBatchNormalizationForwardTrainingNative(
        cudnnHandle handle, 
        int mode, 
        Pointer alpha, // alpha[0] = result blend factor
        Pointer beta, // beta[0] = dest layer blend factor
        cudnnTensorDescriptor xDesc, 
        Pointer x, // NxCxHxW
        cudnnTensorDescriptor yDesc, 
        Pointer y, // NxCxHxW
        /**
         * <pre>
         * Shared desc for the next 6 tensors in the argument list.
                                           Data type to be set as follows:
                                           type = (typeOf(x) == double) ? double : float
                                           Dimensions for this descriptor depend on normalization mode
                                           - Spatial Normalization : tensors are expected to have dims 1xCx1x1
                                            (normalization is performed across NxHxW)
                                           - Per-Activation Normalization : tensors are expected to have dims of 1xCxHxW 
         * (normalization is performed across N)
         * </pre>
         */
        cudnnTensorDescriptor bnScaleBiasMeanVarDesc, 
        // 'Gamma' and 'Beta' respectively in Ioffe and Szegedy's paper's notation
        Pointer bnScale, 
        Pointer bnBias, 
        /**
         * <pre>
         * MUST use factor=1 in the very first call of a complete training cycle.
                                           Use a factor=1/(1+n) at N-th call to the function to get
                                           Cumulative Moving Average (CMA) behavior
                                           CMA[n] = (x[1]+...+x[n])/n
                                           Since CMA[n+1] = (n*CMA[n]+x[n+1])/(n+1) =
                                           ((n+1)*CMA[n]-CMA[n])/(n+1) + x[n+1]/(n+1) =
         * CMA[n]*(1-1/(n+1)) + x[n+1]*1/(n+1)
         * </pre>
         */
        double exponentialAverageFactor, 
        /** Used in Training phase only. 
                                           runningMean = newMean*factor + runningMean*(1-factor) */
        Pointer resultRunningMean, 
        /** Output in training mode, input in inference. Is the moving average
                                           of 1 / sqrt( epsilon + variance[x] ) */
        Pointer resultRunningInvVariance, 
        /** Has to be >= CUDNN_BN_MIN_EPSILON. Should be the same in forward and backward functions. */
        double epsilon, 
        /** Optionally save intermediate results from the forward pass here
                                           - can be reused to speed up backward pass. NULL if unused */
        Pointer resultSaveMean, 
        Pointer resultSaveInvVariance);


    /**
     * <pre>
    * Performs Batch Normalization during Inference: 
    * y[i] = bnScale[k]*(x[i]-estimatedMean[k])*estimatedInvVariance[k] + bnBias[k]
    * with bnScale, bnBias, runningMean, runningInvVariance tensors indexed
    * according to spatial or per-activation mode. Refer to cudnnBatchNormalizationForwardTraining
    * above for notes on function arguments.
     * </pre>
     */
    public static int cudnnBatchNormalizationForwardInference(
        cudnnHandle handle, 
        int mode, 
        Pointer alpha, // alpha[0] = result blend factor
        Pointer beta, // beta[0] = dest layer blend factor
        cudnnTensorDescriptor xDesc, 
        Pointer x, // NxCxHxW
        cudnnTensorDescriptor yDesc, 
        Pointer y, // NxCxHxW
        cudnnTensorDescriptor bnScaleBiasMeanVarDesc, 
        Pointer bnScale, 
        Pointer bnBias, 
        Pointer estimatedMean, 
        Pointer estimatedInvVariance, 
        double epsilon)
    {
        return checkResult(cudnnBatchNormalizationForwardInferenceNative(handle, mode, alpha, beta, xDesc, x, yDesc, y, bnScaleBiasMeanVarDesc, bnScale, bnBias, estimatedMean, estimatedInvVariance, epsilon));
    }
    private static native int cudnnBatchNormalizationForwardInferenceNative(
        cudnnHandle handle, 
        int mode, 
        Pointer alpha, // alpha[0] = result blend factor
        Pointer beta, // beta[0] = dest layer blend factor
        cudnnTensorDescriptor xDesc, 
        Pointer x, // NxCxHxW
        cudnnTensorDescriptor yDesc, 
        Pointer y, // NxCxHxW
        cudnnTensorDescriptor bnScaleBiasMeanVarDesc, 
        Pointer bnScale, 
        Pointer bnBias, 
        Pointer estimatedMean, 
        Pointer estimatedInvVariance, 
        double epsilon);


    /** Performs backward pass of Batch Normalization layer. Returns x gradient,
    * bnScale gradient and bnBias gradient */
    public static int cudnnBatchNormalizationBackward(
        cudnnHandle handle, 
        int mode, 
        Pointer alpha, 
        Pointer beta, 
        cudnnTensorDescriptor xDesc, // same desc for x, dx, dy
        Pointer x, 
        cudnnTensorDescriptor dyDesc, 
        Pointer dy, 
        cudnnTensorDescriptor dxDesc, 
        Pointer dx, 
        /** Shared tensor desc for the 4 tensors below */
        cudnnTensorDescriptor dBnScaleBiasDesc, 
        Pointer bnScale, // bnBias doesn't affect backpropagation
        /** scale and bias diff are not backpropagated below this layer */
        Pointer dBnScaleResult, 
        Pointer dBnBiasResult, 
        /** Same epsilon as forward pass */
        double epsilon, 
        /** Optionally cached intermediate results from
                                           forward pass */
        Pointer savedMean, 
        Pointer savedInvVariance)
    {
        return checkResult(cudnnBatchNormalizationBackwardNative(handle, mode, alpha, beta, xDesc, x, dyDesc, dy, dxDesc, dx, dBnScaleBiasDesc, bnScale, dBnScaleResult, dBnBiasResult, epsilon, savedMean, savedInvVariance));
    }
    private static native int cudnnBatchNormalizationBackwardNative(
        cudnnHandle handle, 
        int mode, 
        Pointer alpha, 
        Pointer beta, 
        cudnnTensorDescriptor xDesc, // same desc for x, dx, dy
        Pointer x, 
        cudnnTensorDescriptor dyDesc, 
        Pointer dy, 
        cudnnTensorDescriptor dxDesc, 
        Pointer dx, 
        /** Shared tensor desc for the 4 tensors below */
        cudnnTensorDescriptor dBnScaleBiasDesc, 
        Pointer bnScale, // bnBias doesn't affect backpropagation
        /** scale and bias diff are not backpropagated below this layer */
        Pointer dBnScaleResult, 
        Pointer dBnBiasResult, 
        /** Same epsilon as forward pass */
        double epsilon, 
        /** Optionally cached intermediate results from
                                           forward pass */
        Pointer savedMean, 
        Pointer savedInvVariance);


    /** DEPRECATED API THAT WILL BE REMOVED FROM UPCOMING RELEASES */
    public static int cudnnSetConvolutionNdDescriptor_v2(
        cudnnConvolutionDescriptor convDesc, 
        int arrayLength, /** nbDims-2 size */
        int[] padA, 
        int[] filterStrideA, 
        int[] upscaleA, 
        int mode)
    {
        return checkResult(cudnnSetConvolutionNdDescriptor_v2Native(convDesc, arrayLength, padA, filterStrideA, upscaleA, mode));
    }
    private static native int cudnnSetConvolutionNdDescriptor_v2Native(
        cudnnConvolutionDescriptor convDesc, 
        int arrayLength, /** nbDims-2 size */
        int[] padA, 
        int[] filterStrideA, 
        int[] upscaleA, 
        int mode);


    public static int cudnnGetConvolutionNdDescriptor_v2(
        cudnnConvolutionDescriptor convDesc, 
        int arrayLengthRequested, 
        int[] arrayLength, 
        int[] padA, 
        int[] strideA, 
        int[] upscaleA, 
        int[] mode)
    {
        return checkResult(cudnnGetConvolutionNdDescriptor_v2Native(convDesc, arrayLengthRequested, arrayLength, padA, strideA, upscaleA, mode));
    }
    private static native int cudnnGetConvolutionNdDescriptor_v2Native(
        cudnnConvolutionDescriptor convDesc, 
        int arrayLengthRequested, 
        int[] arrayLength, 
        int[] padA, 
        int[] strideA, 
        int[] upscaleA, 
        int[] mode);


    public static int cudnnAddTensor_v2(
        cudnnHandle handle, 
        int mode, 
        Pointer alpha, 
        cudnnTensorDescriptor bDesc, 
        Pointer b, 
        Pointer beta, 
        cudnnTensorDescriptor yDesc, 
        Pointer y)
    {
        return checkResult(cudnnAddTensor_v2Native(handle, mode, alpha, bDesc, b, beta, yDesc, y));
    }
    private static native int cudnnAddTensor_v2Native(
        cudnnHandle handle, 
        int mode, 
        Pointer alpha, 
        cudnnTensorDescriptor bDesc, 
        Pointer b, 
        Pointer beta, 
        cudnnTensorDescriptor yDesc, 
        Pointer y);


    public static int cudnnConvolutionBackwardFilter_v2(
        cudnnHandle handle, 
        Pointer alpha, 
        cudnnTensorDescriptor xDesc, 
        Pointer x, 
        cudnnTensorDescriptor dyDesc, 
        Pointer dy, 
        cudnnConvolutionDescriptor convDesc, 
        Pointer beta, 
        cudnnFilterDescriptor dxDesc, 
        Pointer dx)
    {
        return checkResult(cudnnConvolutionBackwardFilter_v2Native(handle, alpha, xDesc, x, dyDesc, dy, convDesc, beta, dxDesc, dx));
    }
    private static native int cudnnConvolutionBackwardFilter_v2Native(
        cudnnHandle handle, 
        Pointer alpha, 
        cudnnTensorDescriptor xDesc, 
        Pointer x, 
        cudnnTensorDescriptor dyDesc, 
        Pointer dy, 
        cudnnConvolutionDescriptor convDesc, 
        Pointer beta, 
        cudnnFilterDescriptor dxDesc, 
        Pointer dx);


    public static int cudnnConvolutionBackwardData_v2(
        cudnnHandle handle, 
        Pointer alpha, 
        cudnnFilterDescriptor xDesc, 
        Pointer x, 
        cudnnTensorDescriptor dyDesc, 
        Pointer dy, 
        cudnnConvolutionDescriptor convDesc, 
        Pointer beta, 
        cudnnTensorDescriptor dxDesc, 
        Pointer dx)
    {
        return checkResult(cudnnConvolutionBackwardData_v2Native(handle, alpha, xDesc, x, dyDesc, dy, convDesc, beta, dxDesc, dx));
    }
    private static native int cudnnConvolutionBackwardData_v2Native(
        cudnnHandle handle, 
        Pointer alpha, 
        cudnnFilterDescriptor xDesc, 
        Pointer x, 
        cudnnTensorDescriptor dyDesc, 
        Pointer dy, 
        cudnnConvolutionDescriptor convDesc, 
        Pointer beta, 
        cudnnTensorDescriptor dxDesc, 
        Pointer dx);


}

