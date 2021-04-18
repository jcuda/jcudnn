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

public class cudnnPointwiseMode
{
    public static final int CUDNN_POINTWISE_ADD = 0;
    public static final int CUDNN_POINTWISE_MUL = 1;
    public static final int CUDNN_POINTWISE_MIN = 2;
    public static final int CUDNN_POINTWISE_MAX = 3;
    public static final int CUDNN_POINTWISE_SQRT = 4;
    public static final int CUDNN_POINTWISE_RELU_FWD = 100;
    public static final int CUDNN_POINTWISE_TANH_FWD = 101;
    public static final int CUDNN_POINTWISE_SIGMOID_FWD = 102;
    public static final int CUDNN_POINTWISE_ELU_FWD = 103;
    public static final int CUDNN_POINTWISE_GELU_FWD = 104;
    public static final int CUDNN_POINTWISE_SOFTPLUS_FWD = 105;
    public static final int CUDNN_POINTWISE_SWISH_FWD = 106;
    public static final int CUDNN_POINTWISE_RELU_BWD = 200;
    public static final int CUDNN_POINTWISE_TANH_BWD = 201;
    public static final int CUDNN_POINTWISE_SIGMOID_BWD = 202;
    public static final int CUDNN_POINTWISE_ELU_BWD = 203;
    public static final int CUDNN_POINTWISE_GELU_BWD = 204;
    public static final int CUDNN_POINTWISE_SOFTPLUS_BWD = 205;
    public static final int CUDNN_POINTWISE_SWISH_BWD = 206;

    /**
     * Private constructor to prevent instantiation
     */
    private cudnnPointwiseMode()
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
            case CUDNN_POINTWISE_ADD: return "CUDNN_POINTWISE_ADD";
            case CUDNN_POINTWISE_MUL: return "CUDNN_POINTWISE_MUL";
            case CUDNN_POINTWISE_MIN: return "CUDNN_POINTWISE_MIN";
            case CUDNN_POINTWISE_MAX: return "CUDNN_POINTWISE_MAX";
            case CUDNN_POINTWISE_SQRT: return "CUDNN_POINTWISE_SQRT";
            case CUDNN_POINTWISE_RELU_FWD: return "CUDNN_POINTWISE_RELU_FWD";
            case CUDNN_POINTWISE_TANH_FWD: return "CUDNN_POINTWISE_TANH_FWD";
            case CUDNN_POINTWISE_SIGMOID_FWD: return "CUDNN_POINTWISE_SIGMOID_FWD";
            case CUDNN_POINTWISE_ELU_FWD: return "CUDNN_POINTWISE_ELU_FWD";
            case CUDNN_POINTWISE_GELU_FWD: return "CUDNN_POINTWISE_GELU_FWD";
            case CUDNN_POINTWISE_SOFTPLUS_FWD: return "CUDNN_POINTWISE_SOFTPLUS_FWD";
            case CUDNN_POINTWISE_SWISH_FWD: return "CUDNN_POINTWISE_SWISH_FWD";
            case CUDNN_POINTWISE_RELU_BWD: return "CUDNN_POINTWISE_RELU_BWD";
            case CUDNN_POINTWISE_TANH_BWD: return "CUDNN_POINTWISE_TANH_BWD";
            case CUDNN_POINTWISE_SIGMOID_BWD: return "CUDNN_POINTWISE_SIGMOID_BWD";
            case CUDNN_POINTWISE_ELU_BWD: return "CUDNN_POINTWISE_ELU_BWD";
            case CUDNN_POINTWISE_GELU_BWD: return "CUDNN_POINTWISE_GELU_BWD";
            case CUDNN_POINTWISE_SOFTPLUS_BWD: return "CUDNN_POINTWISE_SOFTPLUS_BWD";
            case CUDNN_POINTWISE_SWISH_BWD: return "CUDNN_POINTWISE_SWISH_BWD";
        }
        return "INVALID cudnnPointwiseMode: "+n;
    }
}

