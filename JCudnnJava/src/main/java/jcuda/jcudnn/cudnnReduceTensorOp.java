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

/**
 * CUDNN ReduceTensor op type
 */
public class cudnnReduceTensorOp
{
    public static final int CUDNN_REDUCE_TENSOR_ADD = 0;
    public static final int CUDNN_REDUCE_TENSOR_MUL = 1;
    public static final int CUDNN_REDUCE_TENSOR_MIN = 2;
    public static final int CUDNN_REDUCE_TENSOR_MAX = 3;
    public static final int CUDNN_REDUCE_TENSOR_AMAX = 4;
    public static final int CUDNN_REDUCE_TENSOR_AVG = 5;
    public static final int CUDNN_REDUCE_TENSOR_NORM1 = 6;
    public static final int CUDNN_REDUCE_TENSOR_NORM2 = 7;
    public static final int CUDNN_REDUCE_TENSOR_MUL_NO_ZEROS = 8;

    /**
     * Private constructor to prevent instantiation
     */
    private cudnnReduceTensorOp()
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
            case CUDNN_REDUCE_TENSOR_ADD: return "CUDNN_REDUCE_TENSOR_ADD";
            case CUDNN_REDUCE_TENSOR_MUL: return "CUDNN_REDUCE_TENSOR_MUL";
            case CUDNN_REDUCE_TENSOR_MIN: return "CUDNN_REDUCE_TENSOR_MIN";
            case CUDNN_REDUCE_TENSOR_MAX: return "CUDNN_REDUCE_TENSOR_MAX";
            case CUDNN_REDUCE_TENSOR_AMAX: return "CUDNN_REDUCE_TENSOR_AMAX";
            case CUDNN_REDUCE_TENSOR_AVG: return "CUDNN_REDUCE_TENSOR_AVG";
            case CUDNN_REDUCE_TENSOR_NORM1: return "CUDNN_REDUCE_TENSOR_NORM1";
            case CUDNN_REDUCE_TENSOR_NORM2: return "CUDNN_REDUCE_TENSOR_NORM2";
            case CUDNN_REDUCE_TENSOR_MUL_NO_ZEROS: return "CUDNN_REDUCE_TENSOR_MUL_NO_ZEROS";
        }
        return "INVALID cudnnReduceTensorOp: "+n;
    }
}

