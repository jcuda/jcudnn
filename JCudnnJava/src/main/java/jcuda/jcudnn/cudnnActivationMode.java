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
 * activation mode
 */
public class cudnnActivationMode
{
    public static final int CUDNN_ACTIVATION_SIGMOID = 0;
    public static final int CUDNN_ACTIVATION_RELU = 1;
    public static final int CUDNN_ACTIVATION_TANH = 2;
    public static final int CUDNN_ACTIVATION_CLIPPED_RELU = 3;
    public static final int CUDNN_ACTIVATION_ELU = 4;
    public static final int CUDNN_ACTIVATION_IDENTITY = 5;

    /**
     * Private constructor to prevent instantiation
     */
    private cudnnActivationMode()
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
            case CUDNN_ACTIVATION_SIGMOID: return "CUDNN_ACTIVATION_SIGMOID";
            case CUDNN_ACTIVATION_RELU: return "CUDNN_ACTIVATION_RELU";
            case CUDNN_ACTIVATION_TANH: return "CUDNN_ACTIVATION_TANH";
            case CUDNN_ACTIVATION_CLIPPED_RELU: return "CUDNN_ACTIVATION_CLIPPED_RELU";
            case CUDNN_ACTIVATION_ELU: return "CUDNN_ACTIVATION_ELU";
            case CUDNN_ACTIVATION_IDENTITY: return "CUDNN_ACTIVATION_IDENTITY";
        }
        return "INVALID cudnnActivationMode: "+n;
    }
}

