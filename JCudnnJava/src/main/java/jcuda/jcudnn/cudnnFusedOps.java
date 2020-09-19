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

public class cudnnFusedOps
{
    public static final int CUDNN_FUSED_SCALE_BIAS_ACTIVATION_CONV_BNSTATS = 0;
    public static final int CUDNN_FUSED_SCALE_BIAS_ACTIVATION_WGRAD = 1;
    public static final int CUDNN_FUSED_BN_FINALIZE_STATISTICS_TRAINING = 2;
    public static final int CUDNN_FUSED_BN_FINALIZE_STATISTICS_INFERENCE = 3;
    public static final int CUDNN_FUSED_CONV_SCALE_BIAS_ADD_ACTIVATION = 4;
    public static final int CUDNN_FUSED_SCALE_BIAS_ADD_ACTIVATION_GEN_BITMASK = 5;
    public static final int CUDNN_FUSED_DACTIVATION_FORK_DBATCHNORM = 6;

    /**
     * Private constructor to prevent instantiation
     */
    private cudnnFusedOps()
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
            case CUDNN_FUSED_SCALE_BIAS_ACTIVATION_CONV_BNSTATS: return "CUDNN_FUSED_SCALE_BIAS_ACTIVATION_CONV_BNSTATS";
            case CUDNN_FUSED_SCALE_BIAS_ACTIVATION_WGRAD: return "CUDNN_FUSED_SCALE_BIAS_ACTIVATION_WGRAD";
            case CUDNN_FUSED_BN_FINALIZE_STATISTICS_TRAINING: return "CUDNN_FUSED_BN_FINALIZE_STATISTICS_TRAINING";
            case CUDNN_FUSED_BN_FINALIZE_STATISTICS_INFERENCE: return "CUDNN_FUSED_BN_FINALIZE_STATISTICS_INFERENCE";
            case CUDNN_FUSED_CONV_SCALE_BIAS_ADD_ACTIVATION: return "CUDNN_FUSED_CONV_SCALE_BIAS_ADD_ACTIVATION";
            case CUDNN_FUSED_SCALE_BIAS_ADD_ACTIVATION_GEN_BITMASK: return "CUDNN_FUSED_SCALE_BIAS_ADD_ACTIVATION_GEN_BITMASK";
            case CUDNN_FUSED_DACTIVATION_FORK_DBATCHNORM: return "CUDNN_FUSED_DACTIVATION_FORK_DBATCHNORM";
        }
        return "INVALID cudnnFusedOps: "+n;
    }
}

