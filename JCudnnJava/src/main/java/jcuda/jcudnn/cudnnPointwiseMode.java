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
    public static final int CUDNN_POINTWISE_ADD_SQUARE = 5;
    public static final int CUDNN_POINTWISE_DIV = 6;
    public static final int CUDNN_POINTWISE_MAX = 3;
    public static final int CUDNN_POINTWISE_MIN = 2;
    public static final int CUDNN_POINTWISE_MOD = 7;
    public static final int CUDNN_POINTWISE_MUL = 1;
    public static final int CUDNN_POINTWISE_POW = 8;
    public static final int CUDNN_POINTWISE_SUB = 9;
    public static final int CUDNN_POINTWISE_ABS = 10;
    public static final int CUDNN_POINTWISE_CEIL = 11;
    public static final int CUDNN_POINTWISE_COS = 12;
    public static final int CUDNN_POINTWISE_EXP = 13;
    public static final int CUDNN_POINTWISE_FLOOR = 14;
    public static final int CUDNN_POINTWISE_LOG = 15;
    public static final int CUDNN_POINTWISE_NEG = 16;
    public static final int CUDNN_POINTWISE_RSQRT = 17;
    public static final int CUDNN_POINTWISE_SIN = 18;
    public static final int CUDNN_POINTWISE_SQRT = 4;
    public static final int CUDNN_POINTWISE_TAN = 19;
    public static final int CUDNN_POINTWISE_ERF = 20;
    public static final int CUDNN_POINTWISE_IDENTITY = 21;
    public static final int CUDNN_POINTWISE_RECIPROCAL = 22;
    public static final int CUDNN_POINTWISE_RELU_FWD = 100;
    public static final int CUDNN_POINTWISE_TANH_FWD = 101;
    public static final int CUDNN_POINTWISE_SIGMOID_FWD = 102;
    public static final int CUDNN_POINTWISE_ELU_FWD = 103;
    public static final int CUDNN_POINTWISE_GELU_FWD = 104;
    public static final int CUDNN_POINTWISE_SOFTPLUS_FWD = 105;
    public static final int CUDNN_POINTWISE_SWISH_FWD = 106;
    public static final int CUDNN_POINTWISE_GELU_APPROX_TANH_FWD = 107;
    public static final int CUDNN_POINTWISE_RELU_BWD = 200;
    public static final int CUDNN_POINTWISE_TANH_BWD = 201;
    public static final int CUDNN_POINTWISE_SIGMOID_BWD = 202;
    public static final int CUDNN_POINTWISE_ELU_BWD = 203;
    public static final int CUDNN_POINTWISE_GELU_BWD = 204;
    public static final int CUDNN_POINTWISE_SOFTPLUS_BWD = 205;
    public static final int CUDNN_POINTWISE_SWISH_BWD = 206;
    public static final int CUDNN_POINTWISE_GELU_APPROX_TANH_BWD = 207;
    public static final int CUDNN_POINTWISE_CMP_EQ = 300;
    public static final int CUDNN_POINTWISE_CMP_NEQ = 301;
    public static final int CUDNN_POINTWISE_CMP_GT = 302;
    public static final int CUDNN_POINTWISE_CMP_GE = 303;
    public static final int CUDNN_POINTWISE_CMP_LT = 304;
    public static final int CUDNN_POINTWISE_CMP_LE = 305;
    public static final int CUDNN_POINTWISE_LOGICAL_AND = 400;
    public static final int CUDNN_POINTWISE_LOGICAL_OR = 401;
    public static final int CUDNN_POINTWISE_LOGICAL_NOT = 402;
    public static final int CUDNN_POINTWISE_GEN_INDEX = 501;
    public static final int CUDNN_POINTWISE_BINARY_SELECT = 601;

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
            case CUDNN_POINTWISE_ADD_SQUARE: return "CUDNN_POINTWISE_ADD_SQUARE";
            case CUDNN_POINTWISE_DIV: return "CUDNN_POINTWISE_DIV";
            case CUDNN_POINTWISE_MAX: return "CUDNN_POINTWISE_MAX";
            case CUDNN_POINTWISE_MIN: return "CUDNN_POINTWISE_MIN";
            case CUDNN_POINTWISE_MOD: return "CUDNN_POINTWISE_MOD";
            case CUDNN_POINTWISE_MUL: return "CUDNN_POINTWISE_MUL";
            case CUDNN_POINTWISE_POW: return "CUDNN_POINTWISE_POW";
            case CUDNN_POINTWISE_SUB: return "CUDNN_POINTWISE_SUB";
            case CUDNN_POINTWISE_ABS: return "CUDNN_POINTWISE_ABS";
            case CUDNN_POINTWISE_CEIL: return "CUDNN_POINTWISE_CEIL";
            case CUDNN_POINTWISE_COS: return "CUDNN_POINTWISE_COS";
            case CUDNN_POINTWISE_EXP: return "CUDNN_POINTWISE_EXP";
            case CUDNN_POINTWISE_FLOOR: return "CUDNN_POINTWISE_FLOOR";
            case CUDNN_POINTWISE_LOG: return "CUDNN_POINTWISE_LOG";
            case CUDNN_POINTWISE_NEG: return "CUDNN_POINTWISE_NEG";
            case CUDNN_POINTWISE_RSQRT: return "CUDNN_POINTWISE_RSQRT";
            case CUDNN_POINTWISE_SIN: return "CUDNN_POINTWISE_SIN";
            case CUDNN_POINTWISE_SQRT: return "CUDNN_POINTWISE_SQRT";
            case CUDNN_POINTWISE_TAN: return "CUDNN_POINTWISE_TAN";
            case CUDNN_POINTWISE_ERF: return "CUDNN_POINTWISE_ERF";
            case CUDNN_POINTWISE_IDENTITY: return "CUDNN_POINTWISE_IDENTITY";
            case CUDNN_POINTWISE_RECIPROCAL: return "CUDNN_POINTWISE_RECIPROCAL";
            case CUDNN_POINTWISE_RELU_FWD: return "CUDNN_POINTWISE_RELU_FWD";
            case CUDNN_POINTWISE_TANH_FWD: return "CUDNN_POINTWISE_TANH_FWD";
            case CUDNN_POINTWISE_SIGMOID_FWD: return "CUDNN_POINTWISE_SIGMOID_FWD";
            case CUDNN_POINTWISE_ELU_FWD: return "CUDNN_POINTWISE_ELU_FWD";
            case CUDNN_POINTWISE_GELU_FWD: return "CUDNN_POINTWISE_GELU_FWD";
            case CUDNN_POINTWISE_SOFTPLUS_FWD: return "CUDNN_POINTWISE_SOFTPLUS_FWD";
            case CUDNN_POINTWISE_SWISH_FWD: return "CUDNN_POINTWISE_SWISH_FWD";
            case CUDNN_POINTWISE_GELU_APPROX_TANH_FWD: return "CUDNN_POINTWISE_GELU_APPROX_TANH_FWD";
            case CUDNN_POINTWISE_RELU_BWD: return "CUDNN_POINTWISE_RELU_BWD";
            case CUDNN_POINTWISE_TANH_BWD: return "CUDNN_POINTWISE_TANH_BWD";
            case CUDNN_POINTWISE_SIGMOID_BWD: return "CUDNN_POINTWISE_SIGMOID_BWD";
            case CUDNN_POINTWISE_ELU_BWD: return "CUDNN_POINTWISE_ELU_BWD";
            case CUDNN_POINTWISE_GELU_BWD: return "CUDNN_POINTWISE_GELU_BWD";
            case CUDNN_POINTWISE_SOFTPLUS_BWD: return "CUDNN_POINTWISE_SOFTPLUS_BWD";
            case CUDNN_POINTWISE_SWISH_BWD: return "CUDNN_POINTWISE_SWISH_BWD";
            case CUDNN_POINTWISE_GELU_APPROX_TANH_BWD: return "CUDNN_POINTWISE_GELU_APPROX_TANH_BWD";
            case CUDNN_POINTWISE_CMP_EQ: return "CUDNN_POINTWISE_CMP_EQ";
            case CUDNN_POINTWISE_CMP_NEQ: return "CUDNN_POINTWISE_CMP_NEQ";
            case CUDNN_POINTWISE_CMP_GT: return "CUDNN_POINTWISE_CMP_GT";
            case CUDNN_POINTWISE_CMP_GE: return "CUDNN_POINTWISE_CMP_GE";
            case CUDNN_POINTWISE_CMP_LT: return "CUDNN_POINTWISE_CMP_LT";
            case CUDNN_POINTWISE_CMP_LE: return "CUDNN_POINTWISE_CMP_LE";
            case CUDNN_POINTWISE_LOGICAL_AND: return "CUDNN_POINTWISE_LOGICAL_AND";
            case CUDNN_POINTWISE_LOGICAL_OR: return "CUDNN_POINTWISE_LOGICAL_OR";
            case CUDNN_POINTWISE_LOGICAL_NOT: return "CUDNN_POINTWISE_LOGICAL_NOT";
            case CUDNN_POINTWISE_GEN_INDEX: return "CUDNN_POINTWISE_GEN_INDEX";
            case CUDNN_POINTWISE_BINARY_SELECT: return "CUDNN_POINTWISE_BINARY_SELECT";
        }
        return "INVALID cudnnPointwiseMode: "+n;
    }
}

