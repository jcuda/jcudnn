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

public class cudnnRNNBiasMode
{
    /**
     * rnn cell formulas do not use biases 
     */
    public static final int CUDNN_RNN_NO_BIAS = 0;
    /**
     * rnn cell formulas use one input bias in input GEMM 
     */
    public static final int CUDNN_RNN_SINGLE_INP_BIAS = 1;
    /**
     * default, rnn cell formulas use two bias vectors 
     */
    public static final int CUDNN_RNN_DOUBLE_BIAS = 2;
    /**
     * rnn cell formulas use one recurrent bias in recurrent GEMM 
     */
    public static final int CUDNN_RNN_SINGLE_REC_BIAS = 3;

    /**
     * Private constructor to prevent instantiation
     */
    private cudnnRNNBiasMode()
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
            case CUDNN_RNN_NO_BIAS: return "CUDNN_RNN_NO_BIAS";
            case CUDNN_RNN_SINGLE_INP_BIAS: return "CUDNN_RNN_SINGLE_INP_BIAS";
            case CUDNN_RNN_DOUBLE_BIAS: return "CUDNN_RNN_DOUBLE_BIAS";
            case CUDNN_RNN_SINGLE_REC_BIAS: return "CUDNN_RNN_SINGLE_REC_BIAS";
        }
        return "INVALID cudnnRNNBiasMode: "+n;
    }
}

