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

public class cudnnBackendAttributeType
{
    public static final int CUDNN_TYPE_HANDLE = 0;
    public static final int CUDNN_TYPE_DATA_TYPE = 1;
    public static final int CUDNN_TYPE_BOOLEAN = 2;
    public static final int CUDNN_TYPE_INT64 = 3;
    public static final int CUDNN_TYPE_FLOAT = 4;
    public static final int CUDNN_TYPE_DOUBLE = 5;
    public static final int CUDNN_TYPE_VOID_PTR = 6;
    public static final int CUDNN_TYPE_CONVOLUTION_MODE = 7;
    public static final int CUDNN_TYPE_HEUR_MODE = 8;
    public static final int CUDNN_TYPE_KNOB_TYPE = 9;
    public static final int CUDNN_TYPE_NAN_PROPOGATION = 10;
    public static final int CUDNN_TYPE_NUMERICAL_NOTE = 11;
    public static final int CUDNN_TYPE_LAYOUT_TYPE = 12;
    public static final int CUDNN_TYPE_ATTRIB_NAME = 13;
    public static final int CUDNN_TYPE_POINTWISE_MODE = 14;
    public static final int CUDNN_TYPE_BACKEND_DESCRIPTOR = 15;
    public static final int CUDNN_TYPE_GENSTATS_MODE = 16;
    public static final int CUDNN_TYPE_BN_FINALIZE_STATS_MODE = 17;
    public static final int CUDNN_TYPE_REDUCTION_OPERATOR_TYPE = 18;
    public static final int CUDNN_TYPE_BEHAVIOR_NOTE = 19;

    /**
     * Private constructor to prevent instantiation
     */
    private cudnnBackendAttributeType()
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
            case CUDNN_TYPE_HANDLE: return "CUDNN_TYPE_HANDLE";
            case CUDNN_TYPE_DATA_TYPE: return "CUDNN_TYPE_DATA_TYPE";
            case CUDNN_TYPE_BOOLEAN: return "CUDNN_TYPE_BOOLEAN";
            case CUDNN_TYPE_INT64: return "CUDNN_TYPE_INT64";
            case CUDNN_TYPE_FLOAT: return "CUDNN_TYPE_FLOAT";
            case CUDNN_TYPE_DOUBLE: return "CUDNN_TYPE_DOUBLE";
            case CUDNN_TYPE_VOID_PTR: return "CUDNN_TYPE_VOID_PTR";
            case CUDNN_TYPE_CONVOLUTION_MODE: return "CUDNN_TYPE_CONVOLUTION_MODE";
            case CUDNN_TYPE_HEUR_MODE: return "CUDNN_TYPE_HEUR_MODE";
            case CUDNN_TYPE_KNOB_TYPE: return "CUDNN_TYPE_KNOB_TYPE";
            case CUDNN_TYPE_NAN_PROPOGATION: return "CUDNN_TYPE_NAN_PROPOGATION";
            case CUDNN_TYPE_NUMERICAL_NOTE: return "CUDNN_TYPE_NUMERICAL_NOTE";
            case CUDNN_TYPE_LAYOUT_TYPE: return "CUDNN_TYPE_LAYOUT_TYPE";
            case CUDNN_TYPE_ATTRIB_NAME: return "CUDNN_TYPE_ATTRIB_NAME";
            case CUDNN_TYPE_POINTWISE_MODE: return "CUDNN_TYPE_POINTWISE_MODE";
            case CUDNN_TYPE_BACKEND_DESCRIPTOR: return "CUDNN_TYPE_BACKEND_DESCRIPTOR";
            case CUDNN_TYPE_GENSTATS_MODE: return "CUDNN_TYPE_GENSTATS_MODE";
            case CUDNN_TYPE_BN_FINALIZE_STATS_MODE: return "CUDNN_TYPE_BN_FINALIZE_STATS_MODE";
            case CUDNN_TYPE_REDUCTION_OPERATOR_TYPE: return "CUDNN_TYPE_REDUCTION_OPERATOR_TYPE";
            case CUDNN_TYPE_BEHAVIOR_NOTE: return "CUDNN_TYPE_BEHAVIOR_NOTE";
        }
        return "INVALID cudnnBackendAttributeType: "+n;
    }
}

