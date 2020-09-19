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

public class cudnnFusedOpsConstParamLabel
{
    public static final int CUDNN_PARAM_XDESC = 0;
    public static final int CUDNN_PARAM_XDATA_PLACEHOLDER = 1;
    public static final int CUDNN_PARAM_BN_MODE = 2;
    public static final int CUDNN_PARAM_BN_EQSCALEBIAS_DESC = 3;
    public static final int CUDNN_PARAM_BN_EQSCALE_PLACEHOLDER = 4;
    public static final int CUDNN_PARAM_BN_EQBIAS_PLACEHOLDER = 5;
    public static final int CUDNN_PARAM_ACTIVATION_DESC = 6;
    public static final int CUDNN_PARAM_CONV_DESC = 7;
    public static final int CUDNN_PARAM_WDESC = 8;
    public static final int CUDNN_PARAM_WDATA_PLACEHOLDER = 9;
    public static final int CUDNN_PARAM_DWDESC = 10;
    public static final int CUDNN_PARAM_DWDATA_PLACEHOLDER = 11;
    public static final int CUDNN_PARAM_YDESC = 12;
    public static final int CUDNN_PARAM_YDATA_PLACEHOLDER = 13;
    public static final int CUDNN_PARAM_DYDESC = 14;
    public static final int CUDNN_PARAM_DYDATA_PLACEHOLDER = 15;
    public static final int CUDNN_PARAM_YSTATS_DESC = 16;
    public static final int CUDNN_PARAM_YSUM_PLACEHOLDER = 17;
    public static final int CUDNN_PARAM_YSQSUM_PLACEHOLDER = 18;
    public static final int CUDNN_PARAM_BN_SCALEBIAS_MEANVAR_DESC = 19;
    public static final int CUDNN_PARAM_BN_SCALE_PLACEHOLDER = 20;
    public static final int CUDNN_PARAM_BN_BIAS_PLACEHOLDER = 21;
    public static final int CUDNN_PARAM_BN_SAVED_MEAN_PLACEHOLDER = 22;
    public static final int CUDNN_PARAM_BN_SAVED_INVSTD_PLACEHOLDER = 23;
    public static final int CUDNN_PARAM_BN_RUNNING_MEAN_PLACEHOLDER = 24;
    public static final int CUDNN_PARAM_BN_RUNNING_VAR_PLACEHOLDER = 25;
    public static final int CUDNN_PARAM_ZDESC = 26;
    public static final int CUDNN_PARAM_ZDATA_PLACEHOLDER = 27;
    public static final int CUDNN_PARAM_BN_Z_EQSCALEBIAS_DESC = 28;
    public static final int CUDNN_PARAM_BN_Z_EQSCALE_PLACEHOLDER = 29;
    public static final int CUDNN_PARAM_BN_Z_EQBIAS_PLACEHOLDER = 30;
    public static final int CUDNN_PARAM_ACTIVATION_BITMASK_DESC = 31;
    public static final int CUDNN_PARAM_ACTIVATION_BITMASK_PLACEHOLDER = 32;
    public static final int CUDNN_PARAM_DXDESC = 33;
    public static final int CUDNN_PARAM_DXDATA_PLACEHOLDER = 34;
    public static final int CUDNN_PARAM_DZDESC = 35;
    public static final int CUDNN_PARAM_DZDATA_PLACEHOLDER = 36;
    public static final int CUDNN_PARAM_BN_DSCALE_PLACEHOLDER = 37;
    public static final int CUDNN_PARAM_BN_DBIAS_PLACEHOLDER = 38;

    /**
     * Private constructor to prevent instantiation
     */
    private cudnnFusedOpsConstParamLabel()
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
            case CUDNN_PARAM_XDESC: return "CUDNN_PARAM_XDESC";
            case CUDNN_PARAM_XDATA_PLACEHOLDER: return "CUDNN_PARAM_XDATA_PLACEHOLDER";
            case CUDNN_PARAM_BN_MODE: return "CUDNN_PARAM_BN_MODE";
            case CUDNN_PARAM_BN_EQSCALEBIAS_DESC: return "CUDNN_PARAM_BN_EQSCALEBIAS_DESC";
            case CUDNN_PARAM_BN_EQSCALE_PLACEHOLDER: return "CUDNN_PARAM_BN_EQSCALE_PLACEHOLDER";
            case CUDNN_PARAM_BN_EQBIAS_PLACEHOLDER: return "CUDNN_PARAM_BN_EQBIAS_PLACEHOLDER";
            case CUDNN_PARAM_ACTIVATION_DESC: return "CUDNN_PARAM_ACTIVATION_DESC";
            case CUDNN_PARAM_CONV_DESC: return "CUDNN_PARAM_CONV_DESC";
            case CUDNN_PARAM_WDESC: return "CUDNN_PARAM_WDESC";
            case CUDNN_PARAM_WDATA_PLACEHOLDER: return "CUDNN_PARAM_WDATA_PLACEHOLDER";
            case CUDNN_PARAM_DWDESC: return "CUDNN_PARAM_DWDESC";
            case CUDNN_PARAM_DWDATA_PLACEHOLDER: return "CUDNN_PARAM_DWDATA_PLACEHOLDER";
            case CUDNN_PARAM_YDESC: return "CUDNN_PARAM_YDESC";
            case CUDNN_PARAM_YDATA_PLACEHOLDER: return "CUDNN_PARAM_YDATA_PLACEHOLDER";
            case CUDNN_PARAM_DYDESC: return "CUDNN_PARAM_DYDESC";
            case CUDNN_PARAM_DYDATA_PLACEHOLDER: return "CUDNN_PARAM_DYDATA_PLACEHOLDER";
            case CUDNN_PARAM_YSTATS_DESC: return "CUDNN_PARAM_YSTATS_DESC";
            case CUDNN_PARAM_YSUM_PLACEHOLDER: return "CUDNN_PARAM_YSUM_PLACEHOLDER";
            case CUDNN_PARAM_YSQSUM_PLACEHOLDER: return "CUDNN_PARAM_YSQSUM_PLACEHOLDER";
            case CUDNN_PARAM_BN_SCALEBIAS_MEANVAR_DESC: return "CUDNN_PARAM_BN_SCALEBIAS_MEANVAR_DESC";
            case CUDNN_PARAM_BN_SCALE_PLACEHOLDER: return "CUDNN_PARAM_BN_SCALE_PLACEHOLDER";
            case CUDNN_PARAM_BN_BIAS_PLACEHOLDER: return "CUDNN_PARAM_BN_BIAS_PLACEHOLDER";
            case CUDNN_PARAM_BN_SAVED_MEAN_PLACEHOLDER: return "CUDNN_PARAM_BN_SAVED_MEAN_PLACEHOLDER";
            case CUDNN_PARAM_BN_SAVED_INVSTD_PLACEHOLDER: return "CUDNN_PARAM_BN_SAVED_INVSTD_PLACEHOLDER";
            case CUDNN_PARAM_BN_RUNNING_MEAN_PLACEHOLDER: return "CUDNN_PARAM_BN_RUNNING_MEAN_PLACEHOLDER";
            case CUDNN_PARAM_BN_RUNNING_VAR_PLACEHOLDER: return "CUDNN_PARAM_BN_RUNNING_VAR_PLACEHOLDER";
            case CUDNN_PARAM_ZDESC: return "CUDNN_PARAM_ZDESC";
            case CUDNN_PARAM_ZDATA_PLACEHOLDER: return "CUDNN_PARAM_ZDATA_PLACEHOLDER";
            case CUDNN_PARAM_BN_Z_EQSCALEBIAS_DESC: return "CUDNN_PARAM_BN_Z_EQSCALEBIAS_DESC";
            case CUDNN_PARAM_BN_Z_EQSCALE_PLACEHOLDER: return "CUDNN_PARAM_BN_Z_EQSCALE_PLACEHOLDER";
            case CUDNN_PARAM_BN_Z_EQBIAS_PLACEHOLDER: return "CUDNN_PARAM_BN_Z_EQBIAS_PLACEHOLDER";
            case CUDNN_PARAM_ACTIVATION_BITMASK_DESC: return "CUDNN_PARAM_ACTIVATION_BITMASK_DESC";
            case CUDNN_PARAM_ACTIVATION_BITMASK_PLACEHOLDER: return "CUDNN_PARAM_ACTIVATION_BITMASK_PLACEHOLDER";
            case CUDNN_PARAM_DXDESC: return "CUDNN_PARAM_DXDESC";
            case CUDNN_PARAM_DXDATA_PLACEHOLDER: return "CUDNN_PARAM_DXDATA_PLACEHOLDER";
            case CUDNN_PARAM_DZDESC: return "CUDNN_PARAM_DZDESC";
            case CUDNN_PARAM_DZDATA_PLACEHOLDER: return "CUDNN_PARAM_DZDATA_PLACEHOLDER";
            case CUDNN_PARAM_BN_DSCALE_PLACEHOLDER: return "CUDNN_PARAM_BN_DSCALE_PLACEHOLDER";
            case CUDNN_PARAM_BN_DBIAS_PLACEHOLDER: return "CUDNN_PARAM_BN_DBIAS_PLACEHOLDER";
        }
        return "INVALID cudnnFusedOpsConstParamLabel: "+n;
    }
}

