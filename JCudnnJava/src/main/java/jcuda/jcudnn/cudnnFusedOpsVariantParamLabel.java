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

public class cudnnFusedOpsVariantParamLabel
{
    public static final int CUDNN_PTR_XDATA = 0;
    public static final int CUDNN_PTR_BN_EQSCALE = 1;
    public static final int CUDNN_PTR_BN_EQBIAS = 2;
    public static final int CUDNN_PTR_WDATA = 3;
    public static final int CUDNN_PTR_DWDATA = 4;
    public static final int CUDNN_PTR_YDATA = 5;
    public static final int CUDNN_PTR_DYDATA = 6;
    public static final int CUDNN_PTR_YSUM = 7;
    public static final int CUDNN_PTR_YSQSUM = 8;
    public static final int CUDNN_PTR_WORKSPACE = 9;
    public static final int CUDNN_PTR_BN_SCALE = 10;
    public static final int CUDNN_PTR_BN_BIAS = 11;
    public static final int CUDNN_PTR_BN_SAVED_MEAN = 12;
    public static final int CUDNN_PTR_BN_SAVED_INVSTD = 13;
    public static final int CUDNN_PTR_BN_RUNNING_MEAN = 14;
    public static final int CUDNN_PTR_BN_RUNNING_VAR = 15;
    public static final int CUDNN_PTR_ZDATA = 16;
    public static final int CUDNN_PTR_BN_Z_EQSCALE = 17;
    public static final int CUDNN_PTR_BN_Z_EQBIAS = 18;
    public static final int CUDNN_PTR_ACTIVATION_BITMASK = 19;
    public static final int CUDNN_PTR_DXDATA = 20;
    public static final int CUDNN_PTR_DZDATA = 21;
    public static final int CUDNN_PTR_BN_DSCALE = 22;
    public static final int CUDNN_PTR_BN_DBIAS = 23;
    public static final int CUDNN_SCALAR_SIZE_T_WORKSPACE_SIZE_IN_BYTES = 100;
    public static final int CUDNN_SCALAR_INT64_T_BN_ACCUMULATION_COUNT = 101;
    public static final int CUDNN_SCALAR_DOUBLE_BN_EXP_AVG_FACTOR = 102;
    public static final int CUDNN_SCALAR_DOUBLE_BN_EPSILON = 103;

    /**
     * Private constructor to prevent instantiation
     */
    private cudnnFusedOpsVariantParamLabel()
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
            case CUDNN_PTR_XDATA: return "CUDNN_PTR_XDATA";
            case CUDNN_PTR_BN_EQSCALE: return "CUDNN_PTR_BN_EQSCALE";
            case CUDNN_PTR_BN_EQBIAS: return "CUDNN_PTR_BN_EQBIAS";
            case CUDNN_PTR_WDATA: return "CUDNN_PTR_WDATA";
            case CUDNN_PTR_DWDATA: return "CUDNN_PTR_DWDATA";
            case CUDNN_PTR_YDATA: return "CUDNN_PTR_YDATA";
            case CUDNN_PTR_DYDATA: return "CUDNN_PTR_DYDATA";
            case CUDNN_PTR_YSUM: return "CUDNN_PTR_YSUM";
            case CUDNN_PTR_YSQSUM: return "CUDNN_PTR_YSQSUM";
            case CUDNN_PTR_WORKSPACE: return "CUDNN_PTR_WORKSPACE";
            case CUDNN_PTR_BN_SCALE: return "CUDNN_PTR_BN_SCALE";
            case CUDNN_PTR_BN_BIAS: return "CUDNN_PTR_BN_BIAS";
            case CUDNN_PTR_BN_SAVED_MEAN: return "CUDNN_PTR_BN_SAVED_MEAN";
            case CUDNN_PTR_BN_SAVED_INVSTD: return "CUDNN_PTR_BN_SAVED_INVSTD";
            case CUDNN_PTR_BN_RUNNING_MEAN: return "CUDNN_PTR_BN_RUNNING_MEAN";
            case CUDNN_PTR_BN_RUNNING_VAR: return "CUDNN_PTR_BN_RUNNING_VAR";
            case CUDNN_PTR_ZDATA: return "CUDNN_PTR_ZDATA";
            case CUDNN_PTR_BN_Z_EQSCALE: return "CUDNN_PTR_BN_Z_EQSCALE";
            case CUDNN_PTR_BN_Z_EQBIAS: return "CUDNN_PTR_BN_Z_EQBIAS";
            case CUDNN_PTR_ACTIVATION_BITMASK: return "CUDNN_PTR_ACTIVATION_BITMASK";
            case CUDNN_PTR_DXDATA: return "CUDNN_PTR_DXDATA";
            case CUDNN_PTR_DZDATA: return "CUDNN_PTR_DZDATA";
            case CUDNN_PTR_BN_DSCALE: return "CUDNN_PTR_BN_DSCALE";
            case CUDNN_PTR_BN_DBIAS: return "CUDNN_PTR_BN_DBIAS";
            case CUDNN_SCALAR_SIZE_T_WORKSPACE_SIZE_IN_BYTES: return "CUDNN_SCALAR_SIZE_T_WORKSPACE_SIZE_IN_BYTES";
            case CUDNN_SCALAR_INT64_T_BN_ACCUMULATION_COUNT: return "CUDNN_SCALAR_INT64_T_BN_ACCUMULATION_COUNT";
            case CUDNN_SCALAR_DOUBLE_BN_EXP_AVG_FACTOR: return "CUDNN_SCALAR_DOUBLE_BN_EXP_AVG_FACTOR";
            case CUDNN_SCALAR_DOUBLE_BN_EPSILON: return "CUDNN_SCALAR_DOUBLE_BN_EPSILON";
        }
        return "INVALID cudnnFusedOpsVariantParamLabel: "+n;
    }
}

