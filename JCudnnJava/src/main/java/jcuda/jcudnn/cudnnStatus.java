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
 * CUDNN return codes
 */
public class cudnnStatus
{
    public static final int CUDNN_STATUS_SUCCESS = 0;
    public static final int CUDNN_STATUS_NOT_INITIALIZED = 1;
    public static final int CUDNN_STATUS_ALLOC_FAILED = 2;
    public static final int CUDNN_STATUS_BAD_PARAM = 3;
    public static final int CUDNN_STATUS_INTERNAL_ERROR = 4;
    public static final int CUDNN_STATUS_INVALID_VALUE = 5;
    public static final int CUDNN_STATUS_ARCH_MISMATCH = 6;
    public static final int CUDNN_STATUS_MAPPING_ERROR = 7;
    public static final int CUDNN_STATUS_EXECUTION_FAILED = 8;
    public static final int CUDNN_STATUS_NOT_SUPPORTED = 9;
    public static final int CUDNN_STATUS_LICENSE_ERROR = 10;
    public static final int CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING = 11;
    public static final int CUDNN_STATUS_RUNTIME_IN_PROGRESS = 12;
    public static final int CUDNN_STATUS_RUNTIME_FP_OVERFLOW = 13;
    public static final int CUDNN_STATUS_VERSION_MISMATCH = 14;

    /**
     * Private constructor to prevent instantiation
     */
    private cudnnStatus()
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
            case CUDNN_STATUS_SUCCESS: return "CUDNN_STATUS_SUCCESS";
            case CUDNN_STATUS_NOT_INITIALIZED: return "CUDNN_STATUS_NOT_INITIALIZED";
            case CUDNN_STATUS_ALLOC_FAILED: return "CUDNN_STATUS_ALLOC_FAILED";
            case CUDNN_STATUS_BAD_PARAM: return "CUDNN_STATUS_BAD_PARAM";
            case CUDNN_STATUS_INTERNAL_ERROR: return "CUDNN_STATUS_INTERNAL_ERROR";
            case CUDNN_STATUS_INVALID_VALUE: return "CUDNN_STATUS_INVALID_VALUE";
            case CUDNN_STATUS_ARCH_MISMATCH: return "CUDNN_STATUS_ARCH_MISMATCH";
            case CUDNN_STATUS_MAPPING_ERROR: return "CUDNN_STATUS_MAPPING_ERROR";
            case CUDNN_STATUS_EXECUTION_FAILED: return "CUDNN_STATUS_EXECUTION_FAILED";
            case CUDNN_STATUS_NOT_SUPPORTED: return "CUDNN_STATUS_NOT_SUPPORTED";
            case CUDNN_STATUS_LICENSE_ERROR: return "CUDNN_STATUS_LICENSE_ERROR";
            case CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING: return "CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING";
            case CUDNN_STATUS_RUNTIME_IN_PROGRESS: return "CUDNN_STATUS_RUNTIME_IN_PROGRESS";
            case CUDNN_STATUS_RUNTIME_FP_OVERFLOW: return "CUDNN_STATUS_RUNTIME_FP_OVERFLOW";
            case CUDNN_STATUS_VERSION_MISMATCH: return "CUDNN_STATUS_VERSION_MISMATCH";
        }
        return "INVALID cudnnStatus: "+n;
    }
}

