/*
 * JCudnn - Java bindings for cuDNN, the NVIDIA CUDA
 * Deep Neural Network library, to be used with JCuda
 *
 * Copyright (c) 2015-2018 Marco Hutter - http://www.jcuda.org
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

public class cudnnConvolutionBwdDataAlgo
{
    /**
     * non-deterministic 
     */
    public static final int CUDNN_CONVOLUTION_BWD_DATA_ALGO_0 = 0;
    public static final int CUDNN_CONVOLUTION_BWD_DATA_ALGO_1 = 1;
    public static final int CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT = 2;
    public static final int CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING = 3;
    public static final int CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD = 4;
    public static final int CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED = 5;
    public static final int CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT = 6;

    /**
     * Private constructor to prevent instantiation
     */
    private cudnnConvolutionBwdDataAlgo()
    {
        // Private constructor to prevent instantiation
    }

    /**
     * Returns a string representation of the given constant
     *
     * @return A string representation of the given constant
     */
    public static String stringFor(int n)
    {
        switch (n)
        {
            case CUDNN_CONVOLUTION_BWD_DATA_ALGO_0: return "CUDNN_CONVOLUTION_BWD_DATA_ALGO_0";
            case CUDNN_CONVOLUTION_BWD_DATA_ALGO_1: return "CUDNN_CONVOLUTION_BWD_DATA_ALGO_1";
            case CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT: return "CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT";
            case CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING: return "CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING";
            case CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD: return "CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD";
            case CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED: return "CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED";
            case CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT: return "CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT";
        }
        return "INVALID cudnnConvolutionBwdDataAlgo: "+n;
    }
}

