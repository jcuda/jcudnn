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

/** RNN API */
public class cudnnRNNMode
{
    /**
     * Stock RNN with ReLu activation 
     */
    public static final int CUDNN_RNN_RELU = 0;
    /**
     * Stock RNN with tanh activation 
     */
    public static final int CUDNN_RNN_TANH = 1;
    /**
     * LSTM with no peephole connections 
     */
    public static final int CUDNN_LSTM = 2;
    /**
     * Using h' = tanh(r * Uh(t-1) + Wx) and h = (1 - z) * h' + z * h(t-1); 
     */
    public static final int CUDNN_GRU = 3;

    /**
     * Private constructor to prevent instantiation
     */
    private cudnnRNNMode()
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
            case CUDNN_RNN_RELU: return "CUDNN_RNN_RELU";
            case CUDNN_RNN_TANH: return "CUDNN_RNN_TANH";
            case CUDNN_LSTM: return "CUDNN_LSTM";
            case CUDNN_GRU: return "CUDNN_GRU";
        }
        return "INVALID cudnnRNNMode: "+n;
    }
}

