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

public class cudnnBackendBehaviorNote
{
    public static final int CUDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION = 0;
    public static final int CUDNN_BEHAVIOR_NOTE_REQUIRES_FILTER_INT8x32_REORDER = 1;
    public static final int CUDNN_BEHAVIOR_NOTE_REQUIRES_BIAS_INT8x32_REORDER = 2;
    public static final int CUDNN_BEHAVIOR_NOTE_SUPPORTS_CUDA_GRAPH_NATIVE_API = 3;
    public static final int CUDNN_BEHAVIOR_NOTE_TYPE_COUNT = 4;

    /**
     * Private constructor to prevent instantiation
     */
    private cudnnBackendBehaviorNote()
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
            case CUDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION: return "CUDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION";
            case CUDNN_BEHAVIOR_NOTE_REQUIRES_FILTER_INT8x32_REORDER: return "CUDNN_BEHAVIOR_NOTE_REQUIRES_FILTER_INT8x32_REORDER";
            case CUDNN_BEHAVIOR_NOTE_REQUIRES_BIAS_INT8x32_REORDER: return "CUDNN_BEHAVIOR_NOTE_REQUIRES_BIAS_INT8x32_REORDER";
            case CUDNN_BEHAVIOR_NOTE_SUPPORTS_CUDA_GRAPH_NATIVE_API: return "CUDNN_BEHAVIOR_NOTE_SUPPORTS_CUDA_GRAPH_NATIVE_API";
            case CUDNN_BEHAVIOR_NOTE_TYPE_COUNT: return "CUDNN_BEHAVIOR_NOTE_TYPE_COUNT";
        }
        return "INVALID cudnnBackendBehaviorNote: "+n;
    }
}

