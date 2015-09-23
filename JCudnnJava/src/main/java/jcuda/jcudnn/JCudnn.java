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
    public static final int CUDNN_MAJOR      = 3;
    public static final int CUDNN_MINOR      = 0;
    public static final int CUDNN_PATCHLEVEL = 07;

    public static final int CUDNN_VERSION    =
        (CUDNN_MAJOR * 1000 + CUDNN_MINOR * 100 + CUDNN_PATCHLEVEL);
    
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


    /** Tensor layout conversion helper (dest = alpha * src + beta * dest) */
    public static int cudnnTransformTensor(
        cudnnHandle handle, 
        Pointer alpha, 
        cudnnTensorDescriptor srcDesc, 
        Pointer srcData, 
        Pointer beta, 
        cudnnTensorDescriptor destDesc, 
        Pointer destData)
    {
        return checkResult(cudnnTransformTensorNative(handle, alpha, srcDesc, srcData, beta, destDesc, destData));
    }
    private static native int cudnnTransformTensorNative(
        cudnnHandle handle, 
        Pointer alpha, 
        cudnnTensorDescriptor srcDesc, 
        Pointer srcData, 
        Pointer beta, 
        cudnnTensorDescriptor destDesc, 
        Pointer destData);


    /** Tensor Bias addition : srcDest = alpha * bias + beta * srcDestDesc  */
    /** DEPRECATED AS OF v3 */
    public static int cudnnAddTensor(
        cudnnHandle handle, 
        int mode, 
        Pointer alpha, 
        cudnnTensorDescriptor biasDesc, 
        Pointer biasData, 
        Pointer beta, 
        cudnnTensorDescriptor srcDestDesc, 
        Pointer srcDestData)
    {
        return checkResult(cudnnAddTensorNative(handle, mode, alpha, biasDesc, biasData, beta, srcDestDesc, srcDestData));
    }
    private static native int cudnnAddTensorNative(
        cudnnHandle handle, 
        int mode, 
        Pointer alpha, 
        cudnnTensorDescriptor biasDesc, 
        Pointer biasData, 
        Pointer beta, 
        cudnnTensorDescriptor srcDestDesc, 
        Pointer srcDestData);


    /** Tensor Bias addition : srcDest = alpha * bias + beta * srcDestDesc  */
    public static int cudnnAddTensor_v3(
        cudnnHandle handle, 
        Pointer alpha, 
        cudnnTensorDescriptor biasDesc, 
        Pointer biasData, 
        Pointer beta, 
        cudnnTensorDescriptor srcDestDesc, 
        Pointer srcDestData)
    {
        return checkResult(cudnnAddTensor_v3Native(handle, alpha, biasDesc, biasData, beta, srcDestDesc, srcDestData));
    }
    private static native int cudnnAddTensor_v3Native(
        cudnnHandle handle, 
        Pointer alpha, 
        cudnnTensorDescriptor biasDesc, 
        Pointer biasData, 
        Pointer beta, 
        cudnnTensorDescriptor srcDestDesc, 
        Pointer srcDestData);


    /** Set all data points of a tensor to a given value : srcDest = value */
    public static int cudnnSetTensor(
        cudnnHandle handle, 
        cudnnTensorDescriptor srcDestDesc, 
        Pointer srcDestData, 
        Pointer value)
    {
        return checkResult(cudnnSetTensorNative(handle, srcDestDesc, srcDestData, value));
    }
    private static native int cudnnSetTensorNative(
        cudnnHandle handle, 
        cudnnTensorDescriptor srcDestDesc, 
        Pointer srcDestData, 
        Pointer value);


    /** Set all data points of a tensor to a given value : srcDest = alpha * srcDest */
    public static int cudnnScaleTensor(
        cudnnHandle handle, 
        cudnnTensorDescriptor srcDestDesc, 
        Pointer srcDestData, 
        Pointer alpha)
    {
        return checkResult(cudnnScaleTensorNative(handle, srcDestDesc, srcDestData, alpha));
    }
    private static native int cudnnScaleTensorNative(
        cudnnHandle handle, 
        cudnnTensorDescriptor srcDestDesc, 
        Pointer srcDestData, 
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
        int w)// width of  each input fitler
    {
        return checkResult(cudnnSetFilter4dDescriptorNative(filterDesc, dataType, k, c, h, w));
    }
    private static native int cudnnSetFilter4dDescriptorNative(
        cudnnFilterDescriptor filterDesc, 
        int dataType, // image data type
        int k, // number of output feature maps
        int c, // number of input feature maps
        int h, // height of each input filter
        int w);// width of  each input fitler


    public static int cudnnGetFilter4dDescriptor(
        cudnnFilterDescriptor filterDesc, 
        int[] dataType, // image data type
        Pointer k, // number of output feature maps
        Pointer c, // number of input feature maps
        Pointer h, // height of each input filter
        Pointer w)// width of  each input fitler
    {
        return checkResult(cudnnGetFilter4dDescriptorNative(filterDesc, dataType, k, c, h, w));
    }
    private static native int cudnnGetFilter4dDescriptorNative(
        cudnnFilterDescriptor filterDesc, 
        int[] dataType, // image data type
        Pointer k, // number of output feature maps
        Pointer c, // number of input feature maps
        Pointer h, // height of each input filter
        Pointer w);// width of  each input fitler


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
        int mode)
    {
        return checkResult(cudnnSetConvolutionNdDescriptorNative(convDesc, arrayLength, padA, filterStrideA, upscaleA, mode));
    }
    private static native int cudnnSetConvolutionNdDescriptorNative(
        cudnnConvolutionDescriptor convDesc, 
        int arrayLength, /** nbDims-2 size */
        int[] padA, 
        int[] filterStrideA, 
        int[] upscaleA, 
        int mode);


    public static int cudnnGetConvolutionNdDescriptor(
        cudnnConvolutionDescriptor convDesc, 
        int arrayLengthRequested, 
        int[] arrayLength, 
        int[] padA, 
        int[] strideA, 
        int[] upscaleA, 
        int[] mode)
    {
        return checkResult(cudnnGetConvolutionNdDescriptorNative(convDesc, arrayLengthRequested, arrayLength, padA, strideA, upscaleA, mode));
    }
    private static native int cudnnGetConvolutionNdDescriptorNative(
        cudnnConvolutionDescriptor convDesc, 
        int arrayLengthRequested, 
        int[] arrayLength, 
        int[] padA, 
        int[] strideA, 
        int[] upscaleA, 
        int[] mode);


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
        cudnnTensorDescriptor srcDesc, 
        cudnnFilterDescriptor filterDesc, 
        cudnnConvolutionDescriptor convDesc, 
        cudnnTensorDescriptor destDesc, 
        int requestedAlgoCount, 
        int[] returnedAlgoCount, 
        cudnnConvolutionFwdAlgoPerf[] perfResults)
    {
        return checkResult(cudnnFindConvolutionForwardAlgorithmNative(handle, srcDesc, filterDesc, convDesc, destDesc, requestedAlgoCount, returnedAlgoCount, perfResults));
    }
    private static native int cudnnFindConvolutionForwardAlgorithmNative(
        cudnnHandle handle, 
        cudnnTensorDescriptor srcDesc, 
        cudnnFilterDescriptor filterDesc, 
        cudnnConvolutionDescriptor convDesc, 
        cudnnTensorDescriptor destDesc, 
        int requestedAlgoCount, 
        int[] returnedAlgoCount, 
        cudnnConvolutionFwdAlgoPerf[] perfResults);


    public static int cudnnGetConvolutionForwardAlgorithm(
        cudnnHandle handle, 
        cudnnTensorDescriptor srcDesc, 
        cudnnFilterDescriptor filterDesc, 
        cudnnConvolutionDescriptor convDesc, 
        cudnnTensorDescriptor destDesc, 
        int preference, 
        long memoryLimitInbytes, 
        int[] algo)
    {
        return checkResult(cudnnGetConvolutionForwardAlgorithmNative(handle, srcDesc, filterDesc, convDesc, destDesc, preference, memoryLimitInbytes, algo));
    }
    private static native int cudnnGetConvolutionForwardAlgorithmNative(
        cudnnHandle handle, 
        cudnnTensorDescriptor srcDesc, 
        cudnnFilterDescriptor filterDesc, 
        cudnnConvolutionDescriptor convDesc, 
        cudnnTensorDescriptor destDesc, 
        int preference, 
        long memoryLimitInbytes, 
        int[] algo);


    /**
     *  convolution algorithm (which requires potentially some workspace)
     */
    /** Helper function to return the minimum size of the workspace to be passed to the convolution given an algo*/
    public static int cudnnGetConvolutionForwardWorkspaceSize(
        cudnnHandle handle, 
        cudnnTensorDescriptor srcDesc, 
        cudnnFilterDescriptor filterDesc, 
        cudnnConvolutionDescriptor convDesc, 
        cudnnTensorDescriptor destDesc, 
        int algo, 
        long[] sizeInBytes)
    {
        return checkResult(cudnnGetConvolutionForwardWorkspaceSizeNative(handle, srcDesc, filterDesc, convDesc, destDesc, algo, sizeInBytes));
    }
    private static native int cudnnGetConvolutionForwardWorkspaceSizeNative(
        cudnnHandle handle, 
        cudnnTensorDescriptor srcDesc, 
        cudnnFilterDescriptor filterDesc, 
        cudnnConvolutionDescriptor convDesc, 
        cudnnTensorDescriptor destDesc, 
        int algo, 
        long[] sizeInBytes);


    /** Convolution functions: All of the form "output = alpha * Op(inputs) + beta * output" */
    /** Function to perform the forward multiconvolution */
    public static int cudnnConvolutionForward(
        cudnnHandle handle, 
        Pointer alpha, 
        cudnnTensorDescriptor srcDesc, 
        Pointer srcData, 
        cudnnFilterDescriptor filterDesc, 
        Pointer filterData, 
        cudnnConvolutionDescriptor convDesc, 
        int algo, 
        Pointer workSpace, 
        long workSpaceSizeInBytes, 
        Pointer beta, 
        cudnnTensorDescriptor destDesc, 
        Pointer destData)
    {
        return checkResult(cudnnConvolutionForwardNative(handle, alpha, srcDesc, srcData, filterDesc, filterData, convDesc, algo, workSpace, workSpaceSizeInBytes, beta, destDesc, destData));
    }
    private static native int cudnnConvolutionForwardNative(
        cudnnHandle handle, 
        Pointer alpha, 
        cudnnTensorDescriptor srcDesc, 
        Pointer srcData, 
        cudnnFilterDescriptor filterDesc, 
        Pointer filterData, 
        cudnnConvolutionDescriptor convDesc, 
        int algo, 
        Pointer workSpace, 
        long workSpaceSizeInBytes, 
        Pointer beta, 
        cudnnTensorDescriptor destDesc, 
        Pointer destData);


    /** Functions to perform the backward multiconvolution */
    public static int cudnnConvolutionBackwardBias(
        cudnnHandle handle, 
        Pointer alpha, 
        cudnnTensorDescriptor srcDesc, 
        Pointer srcData, 
        Pointer beta, 
        cudnnTensorDescriptor destDesc, 
        Pointer destData)
    {
        return checkResult(cudnnConvolutionBackwardBiasNative(handle, alpha, srcDesc, srcData, beta, destDesc, destData));
    }
    private static native int cudnnConvolutionBackwardBiasNative(
        cudnnHandle handle, 
        Pointer alpha, 
        cudnnTensorDescriptor srcDesc, 
        Pointer srcData, 
        Pointer beta, 
        cudnnTensorDescriptor destDesc, 
        Pointer destData);


    public static int cudnnFindConvolutionBackwardFilterAlgorithm(
        cudnnHandle handle, 
        cudnnTensorDescriptor srcDesc, 
        cudnnTensorDescriptor diffDesc, 
        cudnnConvolutionDescriptor convDesc, 
        cudnnFilterDescriptor gradDesc, 
        int requestedAlgoCount, 
        int[] returnedAlgoCount, 
        cudnnConvolutionBwdFilterAlgoPerf[] perfResults)
    {
        return checkResult(cudnnFindConvolutionBackwardFilterAlgorithmNative(handle, srcDesc, diffDesc, convDesc, gradDesc, requestedAlgoCount, returnedAlgoCount, perfResults));
    }
    private static native int cudnnFindConvolutionBackwardFilterAlgorithmNative(
        cudnnHandle handle, 
        cudnnTensorDescriptor srcDesc, 
        cudnnTensorDescriptor diffDesc, 
        cudnnConvolutionDescriptor convDesc, 
        cudnnFilterDescriptor gradDesc, 
        int requestedAlgoCount, 
        int[] returnedAlgoCount, 
        cudnnConvolutionBwdFilterAlgoPerf[] perfResults);


    public static int cudnnGetConvolutionBackwardFilterAlgorithm(
        cudnnHandle handle, 
        cudnnTensorDescriptor srcDesc, 
        cudnnTensorDescriptor diffDesc, 
        cudnnConvolutionDescriptor convDesc, 
        cudnnFilterDescriptor gradDesc, 
        int preference, 
        long memoryLimitInbytes, 
        int[] algo)
    {
        return checkResult(cudnnGetConvolutionBackwardFilterAlgorithmNative(handle, srcDesc, diffDesc, convDesc, gradDesc, preference, memoryLimitInbytes, algo));
    }
    private static native int cudnnGetConvolutionBackwardFilterAlgorithmNative(
        cudnnHandle handle, 
        cudnnTensorDescriptor srcDesc, 
        cudnnTensorDescriptor diffDesc, 
        cudnnConvolutionDescriptor convDesc, 
        cudnnFilterDescriptor gradDesc, 
        int preference, 
        long memoryLimitInbytes, 
        int[] algo);


    /**
     *  convolution algorithm (which requires potentially some workspace)
     */
    /** Helper function to return the minimum size of the workspace to be passed to the convolution given an algo*/
    public static int cudnnGetConvolutionBackwardFilterWorkspaceSize(
        cudnnHandle handle, 
        cudnnTensorDescriptor srcDesc, 
        cudnnTensorDescriptor diffDesc, 
        cudnnConvolutionDescriptor convDesc, 
        cudnnFilterDescriptor gradDesc, 
        int algo, 
        long[] sizeInBytes)
    {
        return checkResult(cudnnGetConvolutionBackwardFilterWorkspaceSizeNative(handle, srcDesc, diffDesc, convDesc, gradDesc, algo, sizeInBytes));
    }
    private static native int cudnnGetConvolutionBackwardFilterWorkspaceSizeNative(
        cudnnHandle handle, 
        cudnnTensorDescriptor srcDesc, 
        cudnnTensorDescriptor diffDesc, 
        cudnnConvolutionDescriptor convDesc, 
        cudnnFilterDescriptor gradDesc, 
        int algo, 
        long[] sizeInBytes);


    public static int cudnnConvolutionBackwardFilter_v3(
        cudnnHandle handle, 
        Pointer alpha, 
        cudnnTensorDescriptor srcDesc, 
        Pointer srcData, 
        cudnnTensorDescriptor diffDesc, 
        Pointer diffData, 
        cudnnConvolutionDescriptor convDesc, 
        int algo, 
        Pointer workSpace, 
        long workSpaceSizeInBytes, 
        Pointer beta, 
        cudnnFilterDescriptor gradDesc, 
        Pointer gradData)
    {
        return checkResult(cudnnConvolutionBackwardFilter_v3Native(handle, alpha, srcDesc, srcData, diffDesc, diffData, convDesc, algo, workSpace, workSpaceSizeInBytes, beta, gradDesc, gradData));
    }
    private static native int cudnnConvolutionBackwardFilter_v3Native(
        cudnnHandle handle, 
        Pointer alpha, 
        cudnnTensorDescriptor srcDesc, 
        Pointer srcData, 
        cudnnTensorDescriptor diffDesc, 
        Pointer diffData, 
        cudnnConvolutionDescriptor convDesc, 
        int algo, 
        Pointer workSpace, 
        long workSpaceSizeInBytes, 
        Pointer beta, 
        cudnnFilterDescriptor gradDesc, 
        Pointer gradData);


    public static int cudnnConvolutionBackwardFilter(
        cudnnHandle handle, 
        Pointer alpha, 
        cudnnTensorDescriptor srcDesc, 
        Pointer srcData, 
        cudnnTensorDescriptor diffDesc, 
        Pointer diffData, 
        cudnnConvolutionDescriptor convDesc, 
        Pointer beta, 
        cudnnFilterDescriptor gradDesc, 
        Pointer gradData)
    {
        return checkResult(cudnnConvolutionBackwardFilterNative(handle, alpha, srcDesc, srcData, diffDesc, diffData, convDesc, beta, gradDesc, gradData));
    }
    private static native int cudnnConvolutionBackwardFilterNative(
        cudnnHandle handle, 
        Pointer alpha, 
        cudnnTensorDescriptor srcDesc, 
        Pointer srcData, 
        cudnnTensorDescriptor diffDesc, 
        Pointer diffData, 
        cudnnConvolutionDescriptor convDesc, 
        Pointer beta, 
        cudnnFilterDescriptor gradDesc, 
        Pointer gradData);


    public static int cudnnFindConvolutionBackwardDataAlgorithm(
        cudnnHandle handle, 
        cudnnFilterDescriptor filterDesc, 
        cudnnTensorDescriptor diffDesc, 
        cudnnConvolutionDescriptor convDesc, 
        cudnnTensorDescriptor gradDesc, 
        int requestedAlgoCount, 
        int[] returnedAlgoCount, 
        cudnnConvolutionBwdDataAlgoPerf[] perfResults)
    {
        return checkResult(cudnnFindConvolutionBackwardDataAlgorithmNative(handle, filterDesc, diffDesc, convDesc, gradDesc, requestedAlgoCount, returnedAlgoCount, perfResults));
    }
    private static native int cudnnFindConvolutionBackwardDataAlgorithmNative(
        cudnnHandle handle, 
        cudnnFilterDescriptor filterDesc, 
        cudnnTensorDescriptor diffDesc, 
        cudnnConvolutionDescriptor convDesc, 
        cudnnTensorDescriptor gradDesc, 
        int requestedAlgoCount, 
        int[] returnedAlgoCount, 
        cudnnConvolutionBwdDataAlgoPerf[] perfResults);


    public static int cudnnGetConvolutionBackwardDataAlgorithm(
        cudnnHandle handle, 
        cudnnFilterDescriptor filterDesc, 
        cudnnTensorDescriptor diffDesc, 
        cudnnConvolutionDescriptor convDesc, 
        cudnnTensorDescriptor gradDesc, 
        int preference, 
        long memoryLimitInbytes, 
        int[] algo)
    {
        return checkResult(cudnnGetConvolutionBackwardDataAlgorithmNative(handle, filterDesc, diffDesc, convDesc, gradDesc, preference, memoryLimitInbytes, algo));
    }
    private static native int cudnnGetConvolutionBackwardDataAlgorithmNative(
        cudnnHandle handle, 
        cudnnFilterDescriptor filterDesc, 
        cudnnTensorDescriptor diffDesc, 
        cudnnConvolutionDescriptor convDesc, 
        cudnnTensorDescriptor gradDesc, 
        int preference, 
        long memoryLimitInbytes, 
        int[] algo);


    /** Helper function to return the minimum size of the workspace to be passed to the convolution given an algo*/
    public static int cudnnGetConvolutionBackwardDataWorkspaceSize(
        cudnnHandle handle, 
        cudnnFilterDescriptor filterDesc, 
        cudnnTensorDescriptor diffDesc, 
        cudnnConvolutionDescriptor convDesc, 
        cudnnTensorDescriptor gradDesc, 
        int algo, 
        long[] sizeInBytes)
    {
        return checkResult(cudnnGetConvolutionBackwardDataWorkspaceSizeNative(handle, filterDesc, diffDesc, convDesc, gradDesc, algo, sizeInBytes));
    }
    private static native int cudnnGetConvolutionBackwardDataWorkspaceSizeNative(
        cudnnHandle handle, 
        cudnnFilterDescriptor filterDesc, 
        cudnnTensorDescriptor diffDesc, 
        cudnnConvolutionDescriptor convDesc, 
        cudnnTensorDescriptor gradDesc, 
        int algo, 
        long[] sizeInBytes);


    public static int cudnnConvolutionBackwardData_v3(
        cudnnHandle handle, 
        Pointer alpha, 
        cudnnFilterDescriptor filterDesc, 
        Pointer filterData, 
        cudnnTensorDescriptor diffDesc, 
        Pointer diffData, 
        cudnnConvolutionDescriptor convDesc, 
        int algo, 
        Pointer workSpace, 
        long workSpaceSizeInBytes, 
        Pointer beta, 
        cudnnTensorDescriptor gradDesc, 
        Pointer gradData)
    {
        return checkResult(cudnnConvolutionBackwardData_v3Native(handle, alpha, filterDesc, filterData, diffDesc, diffData, convDesc, algo, workSpace, workSpaceSizeInBytes, beta, gradDesc, gradData));
    }
    private static native int cudnnConvolutionBackwardData_v3Native(
        cudnnHandle handle, 
        Pointer alpha, 
        cudnnFilterDescriptor filterDesc, 
        Pointer filterData, 
        cudnnTensorDescriptor diffDesc, 
        Pointer diffData, 
        cudnnConvolutionDescriptor convDesc, 
        int algo, 
        Pointer workSpace, 
        long workSpaceSizeInBytes, 
        Pointer beta, 
        cudnnTensorDescriptor gradDesc, 
        Pointer gradData);


    public static int cudnnConvolutionBackwardData(
        cudnnHandle handle, 
        Pointer alpha, 
        cudnnFilterDescriptor filterDesc, 
        Pointer filterData, 
        cudnnTensorDescriptor diffDesc, 
        Pointer diffData, 
        cudnnConvolutionDescriptor convDesc, 
        Pointer beta, 
        cudnnTensorDescriptor gradDesc, 
        Pointer gradData)
    {
        return checkResult(cudnnConvolutionBackwardDataNative(handle, alpha, filterDesc, filterData, diffDesc, diffData, convDesc, beta, gradDesc, gradData));
    }
    private static native int cudnnConvolutionBackwardDataNative(
        cudnnHandle handle, 
        Pointer alpha, 
        cudnnFilterDescriptor filterDesc, 
        Pointer filterData, 
        cudnnTensorDescriptor diffDesc, 
        Pointer diffData, 
        cudnnConvolutionDescriptor convDesc, 
        Pointer beta, 
        cudnnTensorDescriptor gradDesc, 
        Pointer gradData);


    public static int cudnnIm2Col(
        cudnnHandle handle, 
        cudnnTensorDescriptor srcDesc, 
        Pointer srcData, 
        cudnnFilterDescriptor filterDesc, 
        cudnnConvolutionDescriptor convDesc, 
        Pointer colBuffer)
    {
        return checkResult(cudnnIm2ColNative(handle, srcDesc, srcData, filterDesc, convDesc, colBuffer));
    }
    private static native int cudnnIm2ColNative(
        cudnnHandle handle, 
        cudnnTensorDescriptor srcDesc, 
        Pointer srcData, 
        cudnnFilterDescriptor filterDesc, 
        cudnnConvolutionDescriptor convDesc, 
        Pointer colBuffer);


    /** Softmax functions: All of the form "output = alpha * Op(inputs) + beta * output" */
    /** Function to perform forward softmax */
    public static int cudnnSoftmaxForward(
        cudnnHandle handle, 
        int algorithm, 
        int mode, 
        Pointer alpha, 
        cudnnTensorDescriptor srcDesc, 
        Pointer srcData, 
        Pointer beta, 
        cudnnTensorDescriptor destDesc, 
        Pointer destData)
    {
        return checkResult(cudnnSoftmaxForwardNative(handle, algorithm, mode, alpha, srcDesc, srcData, beta, destDesc, destData));
    }
    private static native int cudnnSoftmaxForwardNative(
        cudnnHandle handle, 
        int algorithm, 
        int mode, 
        Pointer alpha, 
        cudnnTensorDescriptor srcDesc, 
        Pointer srcData, 
        Pointer beta, 
        cudnnTensorDescriptor destDesc, 
        Pointer destData);


    /** Function to perform backward softmax */
    public static int cudnnSoftmaxBackward(
        cudnnHandle handle, 
        int algorithm, 
        int mode, 
        Pointer alpha, 
        cudnnTensorDescriptor srcDesc, 
        Pointer srcData, 
        cudnnTensorDescriptor srcDiffDesc, 
        Pointer srcDiffData, 
        Pointer beta, 
        cudnnTensorDescriptor destDiffDesc, 
        Pointer destDiffData)
    {
        return checkResult(cudnnSoftmaxBackwardNative(handle, algorithm, mode, alpha, srcDesc, srcData, srcDiffDesc, srcDiffData, beta, destDiffDesc, destDiffData));
    }
    private static native int cudnnSoftmaxBackwardNative(
        cudnnHandle handle, 
        int algorithm, 
        int mode, 
        Pointer alpha, 
        cudnnTensorDescriptor srcDesc, 
        Pointer srcData, 
        cudnnTensorDescriptor srcDiffDesc, 
        Pointer srcDiffData, 
        Pointer beta, 
        cudnnTensorDescriptor destDiffDesc, 
        Pointer destDiffData);


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
        Pointer outN, 
        Pointer outC, 
        Pointer outH, 
        Pointer outW)
    {
        return checkResult(cudnnGetPooling2dForwardOutputDimNative(poolingDesc, inputTensorDesc, outN, outC, outH, outW));
    }
    private static native int cudnnGetPooling2dForwardOutputDimNative(
        cudnnPoolingDescriptor poolingDesc, 
        cudnnTensorDescriptor inputTensorDesc, 
        Pointer outN, 
        Pointer outC, 
        Pointer outH, 
        Pointer outW);


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
        cudnnTensorDescriptor srcDesc, 
        Pointer srcData, 
        Pointer beta, 
        cudnnTensorDescriptor destDesc, 
        Pointer destData)
    {
        return checkResult(cudnnPoolingForwardNative(handle, poolingDesc, alpha, srcDesc, srcData, beta, destDesc, destData));
    }
    private static native int cudnnPoolingForwardNative(
        cudnnHandle handle, 
        cudnnPoolingDescriptor poolingDesc, 
        Pointer alpha, 
        cudnnTensorDescriptor srcDesc, 
        Pointer srcData, 
        Pointer beta, 
        cudnnTensorDescriptor destDesc, 
        Pointer destData);


    /** Function to perform backward pooling */
    public static int cudnnPoolingBackward(
        cudnnHandle handle, 
        cudnnPoolingDescriptor poolingDesc, 
        Pointer alpha, 
        cudnnTensorDescriptor srcDesc, 
        Pointer srcData, 
        cudnnTensorDescriptor srcDiffDesc, 
        Pointer srcDiffData, 
        cudnnTensorDescriptor destDesc, 
        Pointer destData, 
        Pointer beta, 
        cudnnTensorDescriptor destDiffDesc, 
        Pointer destDiffData)
    {
        return checkResult(cudnnPoolingBackwardNative(handle, poolingDesc, alpha, srcDesc, srcData, srcDiffDesc, srcDiffData, destDesc, destData, beta, destDiffDesc, destDiffData));
    }
    private static native int cudnnPoolingBackwardNative(
        cudnnHandle handle, 
        cudnnPoolingDescriptor poolingDesc, 
        Pointer alpha, 
        cudnnTensorDescriptor srcDesc, 
        Pointer srcData, 
        cudnnTensorDescriptor srcDiffDesc, 
        Pointer srcDiffData, 
        cudnnTensorDescriptor destDesc, 
        Pointer destData, 
        Pointer beta, 
        cudnnTensorDescriptor destDiffDesc, 
        Pointer destDiffData);


    /** Activation functions: All of the form "output = alpha * Op(inputs) + beta * output" */
    /** Function to perform forward activation  */
    public static int cudnnActivationForward(
        cudnnHandle handle, 
        int mode, 
        Pointer alpha, 
        cudnnTensorDescriptor srcDesc, 
        Pointer srcData, 
        Pointer beta, 
        cudnnTensorDescriptor destDesc, 
        Pointer destData)
    {
        return checkResult(cudnnActivationForwardNative(handle, mode, alpha, srcDesc, srcData, beta, destDesc, destData));
    }
    private static native int cudnnActivationForwardNative(
        cudnnHandle handle, 
        int mode, 
        Pointer alpha, 
        cudnnTensorDescriptor srcDesc, 
        Pointer srcData, 
        Pointer beta, 
        cudnnTensorDescriptor destDesc, 
        Pointer destData);


    /** Function to perform backward activation  */
    public static int cudnnActivationBackward(
        cudnnHandle handle, 
        int mode, 
        Pointer alpha, 
        cudnnTensorDescriptor srcDesc, 
        Pointer srcData, 
        cudnnTensorDescriptor srcDiffDesc, 
        Pointer srcDiffData, 
        cudnnTensorDescriptor destDesc, 
        Pointer destData, 
        Pointer beta, 
        cudnnTensorDescriptor destDiffDesc, 
        Pointer destDiffData)
    {
        return checkResult(cudnnActivationBackwardNative(handle, mode, alpha, srcDesc, srcData, srcDiffDesc, srcDiffData, destDesc, destData, beta, destDiffDesc, destDiffData));
    }
    private static native int cudnnActivationBackwardNative(
        cudnnHandle handle, 
        int mode, 
        Pointer alpha, 
        cudnnTensorDescriptor srcDesc, 
        Pointer srcData, 
        cudnnTensorDescriptor srcDiffDesc, 
        Pointer srcDiffData, 
        cudnnTensorDescriptor destDesc, 
        Pointer destData, 
        Pointer beta, 
        cudnnTensorDescriptor destDiffDesc, 
        Pointer destDiffData);


    // Create an instance of LRN (Local Response Normalization) descriptor
    // This function will set lrnN=5, lrnAlpha=1e-4, lrnBeta=0.75, lrnK=2.0 as defaults from Krizhevsky'12 ImageNet paper
    public static int cudnnCreateLRNDescriptor(
        cudnnLRNDescriptor normDesc)
    {
        return checkResult(cudnnCreateLRNDescriptorNative(normDesc));
    }
    private static native int cudnnCreateLRNDescriptorNative(
        cudnnLRNDescriptor normDesc);


    // LRN uses a window [center-lookBehind, center+lookAhead], where
    // lookBehind = floor( (lrnN-1)/2 ), lookAhead = lrnN-lookBehind-1.
    // So for n=10, the window is [k-4...k...k+5] with a total of 10 samples.
    // Values of double parameters will be cast down to tensor data type.
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


    // Retrieve the settings currently stored in an LRN layer descriptor
    // Any of the provided pointers can be NULL (no corresponding value will be returned)
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


    // Destroy an instance of LRN descriptor
    public static int cudnnDestroyLRNDescriptor(
        cudnnLRNDescriptor lrnDesc)
    {
        return checkResult(cudnnDestroyLRNDescriptorNative(lrnDesc));
    }
    private static native int cudnnDestroyLRNDescriptorNative(
        cudnnLRNDescriptor lrnDesc);


    // LRN functions: of the form "output = alpha * normalize(srcData) + beta * destData"
    // Function to perform LRN forward cross-channel computation
    // Values of double parameters will be cast down to tensor data type
    public static int cudnnLRNCrossChannelForward(
        cudnnHandle handle, 
        cudnnLRNDescriptor normDesc, 
        int lrnMode, 
        Pointer alpha, 
        cudnnTensorDescriptor srcDesc, 
        Pointer srcData, 
        Pointer beta, 
        cudnnTensorDescriptor destDesc, 
        Pointer destData)
    {
        return checkResult(cudnnLRNCrossChannelForwardNative(handle, normDesc, lrnMode, alpha, srcDesc, srcData, beta, destDesc, destData));
    }
    private static native int cudnnLRNCrossChannelForwardNative(
        cudnnHandle handle, 
        cudnnLRNDescriptor normDesc, 
        int lrnMode, 
        Pointer alpha, 
        cudnnTensorDescriptor srcDesc, 
        Pointer srcData, 
        Pointer beta, 
        cudnnTensorDescriptor destDesc, 
        Pointer destData);


    // Function to perform LRN cross-channel backpropagation
    // values of double parameters will be cast down to tensor data type
    // src is the front layer, dst is the back layer
    public static int cudnnLRNCrossChannelBackward(
        cudnnHandle handle, 
        cudnnLRNDescriptor normDesc, 
        int lrnMode, 
        Pointer alpha, 
        cudnnTensorDescriptor srcDesc, 
        Pointer srcData, 
        cudnnTensorDescriptor srcDiffDesc, 
        Pointer srcDiffData, 
        cudnnTensorDescriptor destDesc, 
        Pointer destData, 
        Pointer beta, 
        cudnnTensorDescriptor destDiffDesc, 
        Pointer destDiffData)
    {
        return checkResult(cudnnLRNCrossChannelBackwardNative(handle, normDesc, lrnMode, alpha, srcDesc, srcData, srcDiffDesc, srcDiffData, destDesc, destData, beta, destDiffDesc, destDiffData));
    }
    private static native int cudnnLRNCrossChannelBackwardNative(
        cudnnHandle handle, 
        cudnnLRNDescriptor normDesc, 
        int lrnMode, 
        Pointer alpha, 
        cudnnTensorDescriptor srcDesc, 
        Pointer srcData, 
        cudnnTensorDescriptor srcDiffDesc, 
        Pointer srcDiffData, 
        cudnnTensorDescriptor destDesc, 
        Pointer destData, 
        Pointer beta, 
        cudnnTensorDescriptor destDiffDesc, 
        Pointer destDiffData);


    // LCN/divisive normalization functions: of the form "output = alpha * normalize(srcData) + beta * destData"
    // srcMeansData can be NULL to reproduce Caffe's LRN within-channel behavior
    public static int cudnnDivisiveNormalizationForward(
        cudnnHandle handle, 
        cudnnLRNDescriptor normDesc, 
        int mode, 
        Pointer alpha, 
        cudnnTensorDescriptor srcDesc, // same desc for means, temp, temp2
        Pointer srcData, 
        Pointer srcMeansData, // if NULL, means are assumed to be zero
        Pointer tempData, 
        Pointer tempData2, 
        Pointer beta, 
        cudnnTensorDescriptor destDesc, 
        Pointer destData)
    {
        return checkResult(cudnnDivisiveNormalizationForwardNative(handle, normDesc, mode, alpha, srcDesc, srcData, srcMeansData, tempData, tempData2, beta, destDesc, destData));
    }
    private static native int cudnnDivisiveNormalizationForwardNative(
        cudnnHandle handle, 
        cudnnLRNDescriptor normDesc, 
        int mode, 
        Pointer alpha, 
        cudnnTensorDescriptor srcDesc, // same desc for means, temp, temp2
        Pointer srcData, 
        Pointer srcMeansData, // if NULL, means are assumed to be zero
        Pointer tempData, 
        Pointer tempData2, 
        Pointer beta, 
        cudnnTensorDescriptor destDesc, 
        Pointer destData);


    public static int cudnnDivisiveNormalizationBackward(
        cudnnHandle handle, 
        cudnnLRNDescriptor normDesc, 
        int mode, 
        Pointer alpha, 
        cudnnTensorDescriptor srcDesc, // same desc for diff, means, temp, temp2
        Pointer srcData, 
        Pointer srcMeansData, // if NULL, means are assumed to be zero
        Pointer srcDiffData, 
        Pointer tempData, 
        Pointer tempData2, 
        Pointer betaData, 
        cudnnTensorDescriptor destDataDesc, // same desc for dest, means, meansDiff
        Pointer destDataDiff, // output data differential
        Pointer destMeansDiff)// output means differential, can be NULL
    {
        return checkResult(cudnnDivisiveNormalizationBackwardNative(handle, normDesc, mode, alpha, srcDesc, srcData, srcMeansData, srcDiffData, tempData, tempData2, betaData, destDataDesc, destDataDiff, destMeansDiff));
    }
    private static native int cudnnDivisiveNormalizationBackwardNative(
        cudnnHandle handle, 
        cudnnLRNDescriptor normDesc, 
        int mode, 
        Pointer alpha, 
        cudnnTensorDescriptor srcDesc, // same desc for diff, means, temp, temp2
        Pointer srcData, 
        Pointer srcMeansData, // if NULL, means are assumed to be zero
        Pointer srcDiffData, 
        Pointer tempData, 
        Pointer tempData2, 
        Pointer betaData, 
        cudnnTensorDescriptor destDataDesc, // same desc for dest, means, meansDiff
        Pointer destDataDiff, // output data differential
        Pointer destMeansDiff);// output means differential, can be NULL


}

