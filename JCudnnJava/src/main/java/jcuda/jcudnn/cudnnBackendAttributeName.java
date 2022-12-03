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

public class cudnnBackendAttributeName
{
    public static final int CUDNN_ATTR_POINTWISE_MODE = 0;
    public static final int CUDNN_ATTR_POINTWISE_MATH_PREC = 1;
    public static final int CUDNN_ATTR_POINTWISE_NAN_PROPAGATION = 2;
    public static final int CUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP = 3;
    public static final int CUDNN_ATTR_POINTWISE_RELU_UPPER_CLIP = 4;
    public static final int CUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP_SLOPE = 5;
    public static final int CUDNN_ATTR_POINTWISE_ELU_ALPHA = 6;
    public static final int CUDNN_ATTR_POINTWISE_SOFTPLUS_BETA = 7;
    public static final int CUDNN_ATTR_POINTWISE_SWISH_BETA = 8;
    public static final int CUDNN_ATTR_POINTWISE_AXIS = 9;
    public static final int CUDNN_ATTR_CONVOLUTION_COMP_TYPE = 100;
    public static final int CUDNN_ATTR_CONVOLUTION_CONV_MODE = 101;
    public static final int CUDNN_ATTR_CONVOLUTION_DILATIONS = 102;
    public static final int CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES = 103;
    public static final int CUDNN_ATTR_CONVOLUTION_POST_PADDINGS = 104;
    public static final int CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS = 105;
    public static final int CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS = 106;
    public static final int CUDNN_ATTR_ENGINEHEUR_MODE = 200;
    public static final int CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH = 201;
    public static final int CUDNN_ATTR_ENGINEHEUR_RESULTS = 202;
    public static final int CUDNN_ATTR_ENGINECFG_ENGINE = 300;
    public static final int CUDNN_ATTR_ENGINECFG_INTERMEDIATE_INFO = 301;
    public static final int CUDNN_ATTR_ENGINECFG_KNOB_CHOICES = 302;
    public static final int CUDNN_ATTR_EXECUTION_PLAN_HANDLE = 400;
    public static final int CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG = 401;
    public static final int CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE = 402;
    public static final int CUDNN_ATTR_EXECUTION_PLAN_COMPUTED_INTERMEDIATE_UIDS = 403;
    public static final int CUDNN_ATTR_EXECUTION_PLAN_RUN_ONLY_INTERMEDIATE_UIDS = 404;
    public static final int CUDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION = 405;
    public static final int CUDNN_ATTR_INTERMEDIATE_INFO_UNIQUE_ID = 500;
    public static final int CUDNN_ATTR_INTERMEDIATE_INFO_SIZE = 501;
    public static final int CUDNN_ATTR_INTERMEDIATE_INFO_DEPENDENT_DATA_UIDS = 502;
    public static final int CUDNN_ATTR_INTERMEDIATE_INFO_DEPENDENT_ATTRIBUTES = 503;
    public static final int CUDNN_ATTR_KNOB_CHOICE_KNOB_TYPE = 600;
    public static final int CUDNN_ATTR_KNOB_CHOICE_KNOB_VALUE = 601;
    public static final int CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA = 700;
    public static final int CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA = 701;
    public static final int CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC = 702;
    public static final int CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W = 703;
    public static final int CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X = 704;
    public static final int CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y = 705;
    public static final int CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA = 706;
    public static final int CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA = 707;
    public static final int CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC = 708;
    public static final int CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W = 709;
    public static final int CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX = 710;
    public static final int CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY = 711;
    public static final int CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_ALPHA = 712;
    public static final int CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_BETA = 713;
    public static final int CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_CONV_DESC = 714;
    public static final int CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW = 715;
    public static final int CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X = 716;
    public static final int CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY = 717;
    public static final int CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR = 750;
    public static final int CUDNN_ATTR_OPERATION_POINTWISE_XDESC = 751;
    public static final int CUDNN_ATTR_OPERATION_POINTWISE_BDESC = 752;
    public static final int CUDNN_ATTR_OPERATION_POINTWISE_YDESC = 753;
    public static final int CUDNN_ATTR_OPERATION_POINTWISE_ALPHA1 = 754;
    public static final int CUDNN_ATTR_OPERATION_POINTWISE_ALPHA2 = 755;
    public static final int CUDNN_ATTR_OPERATION_POINTWISE_DXDESC = 756;
    public static final int CUDNN_ATTR_OPERATION_POINTWISE_DYDESC = 757;
    public static final int CUDNN_ATTR_OPERATION_POINTWISE_TDESC = 758;
    public static final int CUDNN_ATTR_OPERATION_GENSTATS_MODE = 770;
    public static final int CUDNN_ATTR_OPERATION_GENSTATS_MATH_PREC = 771;
    public static final int CUDNN_ATTR_OPERATION_GENSTATS_XDESC = 772;
    public static final int CUDNN_ATTR_OPERATION_GENSTATS_SUMDESC = 773;
    public static final int CUDNN_ATTR_OPERATION_GENSTATS_SQSUMDESC = 774;
    public static final int CUDNN_ATTR_OPERATION_BN_FINALIZE_STATS_MODE = 780;
    public static final int CUDNN_ATTR_OPERATION_BN_FINALIZE_MATH_PREC = 781;
    public static final int CUDNN_ATTR_OPERATION_BN_FINALIZE_Y_SUM_DESC = 782;
    public static final int CUDNN_ATTR_OPERATION_BN_FINALIZE_Y_SQ_SUM_DESC = 783;
    public static final int CUDNN_ATTR_OPERATION_BN_FINALIZE_SCALE_DESC = 784;
    public static final int CUDNN_ATTR_OPERATION_BN_FINALIZE_BIAS_DESC = 785;
    public static final int CUDNN_ATTR_OPERATION_BN_FINALIZE_PREV_RUNNING_MEAN_DESC = 786;
    public static final int CUDNN_ATTR_OPERATION_BN_FINALIZE_PREV_RUNNING_VAR_DESC = 787;
    public static final int CUDNN_ATTR_OPERATION_BN_FINALIZE_UPDATED_RUNNING_MEAN_DESC = 788;
    public static final int CUDNN_ATTR_OPERATION_BN_FINALIZE_UPDATED_RUNNING_VAR_DESC = 789;
    public static final int CUDNN_ATTR_OPERATION_BN_FINALIZE_SAVED_MEAN_DESC = 790;
    public static final int CUDNN_ATTR_OPERATION_BN_FINALIZE_SAVED_INV_STD_DESC = 791;
    public static final int CUDNN_ATTR_OPERATION_BN_FINALIZE_EQ_SCALE_DESC = 792;
    public static final int CUDNN_ATTR_OPERATION_BN_FINALIZE_EQ_BIAS_DESC = 793;
    public static final int CUDNN_ATTR_OPERATION_BN_FINALIZE_ACCUM_COUNT_DESC = 794;
    public static final int CUDNN_ATTR_OPERATION_BN_FINALIZE_EPSILON_DESC = 795;
    public static final int CUDNN_ATTR_OPERATION_BN_FINALIZE_EXP_AVERATE_FACTOR_DESC = 796;
    public static final int CUDNN_ATTR_OPERATIONGRAPH_HANDLE = 800;
    public static final int CUDNN_ATTR_OPERATIONGRAPH_OPS = 801;
    public static final int CUDNN_ATTR_OPERATIONGRAPH_ENGINE_GLOBAL_COUNT = 802;
    public static final int CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT = 900;
    public static final int CUDNN_ATTR_TENSOR_DATA_TYPE = 901;
    public static final int CUDNN_ATTR_TENSOR_DIMENSIONS = 902;
    public static final int CUDNN_ATTR_TENSOR_STRIDES = 903;
    public static final int CUDNN_ATTR_TENSOR_VECTOR_COUNT = 904;
    public static final int CUDNN_ATTR_TENSOR_VECTORIZED_DIMENSION = 905;
    public static final int CUDNN_ATTR_TENSOR_UNIQUE_ID = 906;
    public static final int CUDNN_ATTR_TENSOR_IS_VIRTUAL = 907;
    public static final int CUDNN_ATTR_TENSOR_IS_BY_VALUE = 908;
    public static final int CUDNN_ATTR_TENSOR_REORDERING_MODE = 909;
    public static final int CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS = 1000;
    public static final int CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS = 1001;
    public static final int CUDNN_ATTR_VARIANT_PACK_INTERMEDIATES = 1002;
    public static final int CUDNN_ATTR_VARIANT_PACK_WORKSPACE = 1003;
    public static final int CUDNN_ATTR_LAYOUT_INFO_TENSOR_UID = 1100;
    public static final int CUDNN_ATTR_LAYOUT_INFO_TYPES = 1101;
    public static final int CUDNN_ATTR_KNOB_INFO_TYPE = 1200;
    public static final int CUDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE = 1201;
    public static final int CUDNN_ATTR_KNOB_INFO_MINIMUM_VALUE = 1202;
    public static final int CUDNN_ATTR_KNOB_INFO_STRIDE = 1203;
    public static final int CUDNN_ATTR_ENGINE_OPERATION_GRAPH = 1300;
    public static final int CUDNN_ATTR_ENGINE_GLOBAL_INDEX = 1301;
    public static final int CUDNN_ATTR_ENGINE_KNOB_INFO = 1302;
    public static final int CUDNN_ATTR_ENGINE_NUMERICAL_NOTE = 1303;
    public static final int CUDNN_ATTR_ENGINE_LAYOUT_INFO = 1304;
    public static final int CUDNN_ATTR_ENGINE_BEHAVIOR_NOTE = 1305;
    public static final int CUDNN_ATTR_MATMUL_COMP_TYPE = 1500;
    public static final int CUDNN_ATTR_OPERATION_MATMUL_ADESC = 1520;
    public static final int CUDNN_ATTR_OPERATION_MATMUL_BDESC = 1521;
    public static final int CUDNN_ATTR_OPERATION_MATMUL_CDESC = 1522;
    public static final int CUDNN_ATTR_OPERATION_MATMUL_DESC = 1523;
    public static final int CUDNN_ATTR_OPERATION_MATMUL_IRREGULARLY_STRIDED_BATCH_COUNT = 1524;
    public static final int CUDNN_ATTR_OPERATION_MATMUL_GEMM_M_OVERRIDE_DESC = 1525;
    public static final int CUDNN_ATTR_OPERATION_MATMUL_GEMM_N_OVERRIDE_DESC = 1526;
    public static final int CUDNN_ATTR_OPERATION_MATMUL_GEMM_K_OVERRIDE_DESC = 1527;
    public static final int CUDNN_ATTR_REDUCTION_OPERATOR = 1600;
    public static final int CUDNN_ATTR_REDUCTION_COMP_TYPE = 1601;
    public static final int CUDNN_ATTR_OPERATION_REDUCTION_XDESC = 1610;
    public static final int CUDNN_ATTR_OPERATION_REDUCTION_YDESC = 1611;
    public static final int CUDNN_ATTR_OPERATION_REDUCTION_DESC = 1612;
    public static final int CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_MATH_PREC = 1620;
    public static final int CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_MEAN_DESC = 1621;
    public static final int CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_INVSTD_DESC = 1622;
    public static final int CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_BN_SCALE_DESC = 1623;
    public static final int CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_X_DESC = 1624;
    public static final int CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DY_DESC = 1625;
    public static final int CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DBN_SCALE_DESC = 1626;
    public static final int CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DBN_BIAS_DESC = 1627;
    public static final int CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_DY_SCALE_DESC = 1628;
    public static final int CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_X_SCALE_DESC = 1629;
    public static final int CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_BIAS = 1630;
    public static final int CUDNN_ATTR_RESAMPLE_MODE = 1700;
    public static final int CUDNN_ATTR_RESAMPLE_COMP_TYPE = 1701;
    public static final int CUDNN_ATTR_RESAMPLE_SPATIAL_DIMS = 1702;
    public static final int CUDNN_ATTR_RESAMPLE_POST_PADDINGS = 1703;
    public static final int CUDNN_ATTR_RESAMPLE_PRE_PADDINGS = 1704;
    public static final int CUDNN_ATTR_RESAMPLE_STRIDES = 1705;
    public static final int CUDNN_ATTR_RESAMPLE_WINDOW_DIMS = 1706;
    public static final int CUDNN_ATTR_RESAMPLE_NAN_PROPAGATION = 1707;
    public static final int CUDNN_ATTR_RESAMPLE_PADDING_MODE = 1708;
    public static final int CUDNN_ATTR_OPERATION_RESAMPLE_FWD_XDESC = 1710;
    public static final int CUDNN_ATTR_OPERATION_RESAMPLE_FWD_YDESC = 1711;
    public static final int CUDNN_ATTR_OPERATION_RESAMPLE_FWD_IDXDESC = 1712;
    public static final int CUDNN_ATTR_OPERATION_RESAMPLE_FWD_ALPHA = 1713;
    public static final int CUDNN_ATTR_OPERATION_RESAMPLE_FWD_BETA = 1714;
    public static final int CUDNN_ATTR_OPERATION_RESAMPLE_FWD_DESC = 1716;
    public static final int CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DXDESC = 1720;
    public static final int CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DYDESC = 1721;
    public static final int CUDNN_ATTR_OPERATION_RESAMPLE_BWD_IDXDESC = 1722;
    public static final int CUDNN_ATTR_OPERATION_RESAMPLE_BWD_ALPHA = 1723;
    public static final int CUDNN_ATTR_OPERATION_RESAMPLE_BWD_BETA = 1724;
    public static final int CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DESC = 1725;
    public static final int CUDNN_ATTR_OPERATION_RESAMPLE_BWD_XDESC = 1726;
    public static final int CUDNN_ATTR_OPERATION_RESAMPLE_BWD_YDESC = 1727;
    public static final int CUDNN_ATTR_OPERATION_CONCAT_AXIS = 1800;
    public static final int CUDNN_ATTR_OPERATION_CONCAT_INPUT_DESCS = 1801;
    public static final int CUDNN_ATTR_OPERATION_CONCAT_INPLACE_INDEX = 1802;
    public static final int CUDNN_ATTR_OPERATION_CONCAT_OUTPUT_DESC = 1803;
    public static final int CUDNN_ATTR_OPERATION_SIGNAL_MODE = 1900;
    public static final int CUDNN_ATTR_OPERATION_SIGNAL_FLAGDESC = 1901;
    public static final int CUDNN_ATTR_OPERATION_SIGNAL_VALUE = 1902;
    public static final int CUDNN_ATTR_OPERATION_SIGNAL_XDESC = 1903;
    public static final int CUDNN_ATTR_OPERATION_SIGNAL_YDESC = 1904;
    public static final int CUDNN_ATTR_OPERATION_NORM_FWD_MODE = 2000;
    public static final int CUDNN_ATTR_OPERATION_NORM_FWD_PHASE = 2001;
    public static final int CUDNN_ATTR_OPERATION_NORM_FWD_XDESC = 2002;
    public static final int CUDNN_ATTR_OPERATION_NORM_FWD_MEAN_DESC = 2003;
    public static final int CUDNN_ATTR_OPERATION_NORM_FWD_INV_VARIANCE_DESC = 2004;
    public static final int CUDNN_ATTR_OPERATION_NORM_FWD_SCALE_DESC = 2005;
    public static final int CUDNN_ATTR_OPERATION_NORM_FWD_BIAS_DESC = 2006;
    public static final int CUDNN_ATTR_OPERATION_NORM_FWD_EPSILON_DESC = 2007;
    public static final int CUDNN_ATTR_OPERATION_NORM_FWD_EXP_AVG_FACTOR_DESC = 2008;
    public static final int CUDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_MEAN_DESC = 2009;
    public static final int CUDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_VAR_DESC = 2010;
    public static final int CUDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_MEAN_DESC = 2011;
    public static final int CUDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_VAR_DESC = 2012;
    public static final int CUDNN_ATTR_OPERATION_NORM_FWD_YDESC = 2013;
    public static final int CUDNN_ATTR_OPERATION_NORM_FWD_PEER_STAT_DESCS = 2014;
    public static final int CUDNN_ATTR_OPERATION_NORM_BWD_MODE = 2100;
    public static final int CUDNN_ATTR_OPERATION_NORM_BWD_XDESC = 2101;
    public static final int CUDNN_ATTR_OPERATION_NORM_BWD_MEAN_DESC = 2102;
    public static final int CUDNN_ATTR_OPERATION_NORM_BWD_INV_VARIANCE_DESC = 2103;
    public static final int CUDNN_ATTR_OPERATION_NORM_BWD_DYDESC = 2104;
    public static final int CUDNN_ATTR_OPERATION_NORM_BWD_SCALE_DESC = 2105;
    public static final int CUDNN_ATTR_OPERATION_NORM_BWD_EPSILON_DESC = 2106;
    public static final int CUDNN_ATTR_OPERATION_NORM_BWD_DSCALE_DESC = 2107;
    public static final int CUDNN_ATTR_OPERATION_NORM_BWD_DBIAS_DESC = 2108;
    public static final int CUDNN_ATTR_OPERATION_NORM_BWD_DXDESC = 2109;
    public static final int CUDNN_ATTR_OPERATION_NORM_BWD_PEER_STAT_DESCS = 2110;
    public static final int CUDNN_ATTR_OPERATION_RESHAPE_XDESC = 2200;
    public static final int CUDNN_ATTR_OPERATION_RESHAPE_YDESC = 2201;
    public static final int CUDNN_ATTR_RNG_DISTRIBUTION = 2300;
    public static final int CUDNN_ATTR_RNG_NORMAL_DIST_MEAN = 2301;
    public static final int CUDNN_ATTR_RNG_NORMAL_DIST_STANDARD_DEVIATION = 2302;
    public static final int CUDNN_ATTR_RNG_UNIFORM_DIST_MAXIMUM = 2303;
    public static final int CUDNN_ATTR_RNG_UNIFORM_DIST_MINIMUM = 2304;
    public static final int CUDNN_ATTR_RNG_BERNOULLI_DIST_PROBABILITY = 2305;
    public static final int CUDNN_ATTR_OPERATION_RNG_YDESC = 2310;
    public static final int CUDNN_ATTR_OPERATION_RNG_SEED = 2311;
    public static final int CUDNN_ATTR_OPERATION_RNG_DESC = 2312;

    /**
     * Private constructor to prevent instantiation
     */
    private cudnnBackendAttributeName()
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
            case CUDNN_ATTR_POINTWISE_MODE: return "CUDNN_ATTR_POINTWISE_MODE";
            case CUDNN_ATTR_POINTWISE_MATH_PREC: return "CUDNN_ATTR_POINTWISE_MATH_PREC";
            case CUDNN_ATTR_POINTWISE_NAN_PROPAGATION: return "CUDNN_ATTR_POINTWISE_NAN_PROPAGATION";
            case CUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP: return "CUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP";
            case CUDNN_ATTR_POINTWISE_RELU_UPPER_CLIP: return "CUDNN_ATTR_POINTWISE_RELU_UPPER_CLIP";
            case CUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP_SLOPE: return "CUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP_SLOPE";
            case CUDNN_ATTR_POINTWISE_ELU_ALPHA: return "CUDNN_ATTR_POINTWISE_ELU_ALPHA";
            case CUDNN_ATTR_POINTWISE_SOFTPLUS_BETA: return "CUDNN_ATTR_POINTWISE_SOFTPLUS_BETA";
            case CUDNN_ATTR_POINTWISE_SWISH_BETA: return "CUDNN_ATTR_POINTWISE_SWISH_BETA";
            case CUDNN_ATTR_POINTWISE_AXIS: return "CUDNN_ATTR_POINTWISE_AXIS";
            case CUDNN_ATTR_CONVOLUTION_COMP_TYPE: return "CUDNN_ATTR_CONVOLUTION_COMP_TYPE";
            case CUDNN_ATTR_CONVOLUTION_CONV_MODE: return "CUDNN_ATTR_CONVOLUTION_CONV_MODE";
            case CUDNN_ATTR_CONVOLUTION_DILATIONS: return "CUDNN_ATTR_CONVOLUTION_DILATIONS";
            case CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES: return "CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES";
            case CUDNN_ATTR_CONVOLUTION_POST_PADDINGS: return "CUDNN_ATTR_CONVOLUTION_POST_PADDINGS";
            case CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS: return "CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS";
            case CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS: return "CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS";
            case CUDNN_ATTR_ENGINEHEUR_MODE: return "CUDNN_ATTR_ENGINEHEUR_MODE";
            case CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH: return "CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH";
            case CUDNN_ATTR_ENGINEHEUR_RESULTS: return "CUDNN_ATTR_ENGINEHEUR_RESULTS";
            case CUDNN_ATTR_ENGINECFG_ENGINE: return "CUDNN_ATTR_ENGINECFG_ENGINE";
            case CUDNN_ATTR_ENGINECFG_INTERMEDIATE_INFO: return "CUDNN_ATTR_ENGINECFG_INTERMEDIATE_INFO";
            case CUDNN_ATTR_ENGINECFG_KNOB_CHOICES: return "CUDNN_ATTR_ENGINECFG_KNOB_CHOICES";
            case CUDNN_ATTR_EXECUTION_PLAN_HANDLE: return "CUDNN_ATTR_EXECUTION_PLAN_HANDLE";
            case CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG: return "CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG";
            case CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE: return "CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE";
            case CUDNN_ATTR_EXECUTION_PLAN_COMPUTED_INTERMEDIATE_UIDS: return "CUDNN_ATTR_EXECUTION_PLAN_COMPUTED_INTERMEDIATE_UIDS";
            case CUDNN_ATTR_EXECUTION_PLAN_RUN_ONLY_INTERMEDIATE_UIDS: return "CUDNN_ATTR_EXECUTION_PLAN_RUN_ONLY_INTERMEDIATE_UIDS";
            case CUDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION: return "CUDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION";
            case CUDNN_ATTR_INTERMEDIATE_INFO_UNIQUE_ID: return "CUDNN_ATTR_INTERMEDIATE_INFO_UNIQUE_ID";
            case CUDNN_ATTR_INTERMEDIATE_INFO_SIZE: return "CUDNN_ATTR_INTERMEDIATE_INFO_SIZE";
            case CUDNN_ATTR_INTERMEDIATE_INFO_DEPENDENT_DATA_UIDS: return "CUDNN_ATTR_INTERMEDIATE_INFO_DEPENDENT_DATA_UIDS";
            case CUDNN_ATTR_INTERMEDIATE_INFO_DEPENDENT_ATTRIBUTES: return "CUDNN_ATTR_INTERMEDIATE_INFO_DEPENDENT_ATTRIBUTES";
            case CUDNN_ATTR_KNOB_CHOICE_KNOB_TYPE: return "CUDNN_ATTR_KNOB_CHOICE_KNOB_TYPE";
            case CUDNN_ATTR_KNOB_CHOICE_KNOB_VALUE: return "CUDNN_ATTR_KNOB_CHOICE_KNOB_VALUE";
            case CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA: return "CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA";
            case CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA: return "CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA";
            case CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC: return "CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC";
            case CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W: return "CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W";
            case CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X: return "CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X";
            case CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y: return "CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y";
            case CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA: return "CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA";
            case CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA: return "CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA";
            case CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC: return "CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC";
            case CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W: return "CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W";
            case CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX: return "CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX";
            case CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY: return "CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY";
            case CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_ALPHA: return "CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_ALPHA";
            case CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_BETA: return "CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_BETA";
            case CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_CONV_DESC: return "CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_CONV_DESC";
            case CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW: return "CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW";
            case CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X: return "CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X";
            case CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY: return "CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY";
            case CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR: return "CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR";
            case CUDNN_ATTR_OPERATION_POINTWISE_XDESC: return "CUDNN_ATTR_OPERATION_POINTWISE_XDESC";
            case CUDNN_ATTR_OPERATION_POINTWISE_BDESC: return "CUDNN_ATTR_OPERATION_POINTWISE_BDESC";
            case CUDNN_ATTR_OPERATION_POINTWISE_YDESC: return "CUDNN_ATTR_OPERATION_POINTWISE_YDESC";
            case CUDNN_ATTR_OPERATION_POINTWISE_ALPHA1: return "CUDNN_ATTR_OPERATION_POINTWISE_ALPHA1";
            case CUDNN_ATTR_OPERATION_POINTWISE_ALPHA2: return "CUDNN_ATTR_OPERATION_POINTWISE_ALPHA2";
            case CUDNN_ATTR_OPERATION_POINTWISE_DXDESC: return "CUDNN_ATTR_OPERATION_POINTWISE_DXDESC";
            case CUDNN_ATTR_OPERATION_POINTWISE_DYDESC: return "CUDNN_ATTR_OPERATION_POINTWISE_DYDESC";
            case CUDNN_ATTR_OPERATION_POINTWISE_TDESC: return "CUDNN_ATTR_OPERATION_POINTWISE_TDESC";
            case CUDNN_ATTR_OPERATION_GENSTATS_MODE: return "CUDNN_ATTR_OPERATION_GENSTATS_MODE";
            case CUDNN_ATTR_OPERATION_GENSTATS_MATH_PREC: return "CUDNN_ATTR_OPERATION_GENSTATS_MATH_PREC";
            case CUDNN_ATTR_OPERATION_GENSTATS_XDESC: return "CUDNN_ATTR_OPERATION_GENSTATS_XDESC";
            case CUDNN_ATTR_OPERATION_GENSTATS_SUMDESC: return "CUDNN_ATTR_OPERATION_GENSTATS_SUMDESC";
            case CUDNN_ATTR_OPERATION_GENSTATS_SQSUMDESC: return "CUDNN_ATTR_OPERATION_GENSTATS_SQSUMDESC";
            case CUDNN_ATTR_OPERATION_BN_FINALIZE_STATS_MODE: return "CUDNN_ATTR_OPERATION_BN_FINALIZE_STATS_MODE";
            case CUDNN_ATTR_OPERATION_BN_FINALIZE_MATH_PREC: return "CUDNN_ATTR_OPERATION_BN_FINALIZE_MATH_PREC";
            case CUDNN_ATTR_OPERATION_BN_FINALIZE_Y_SUM_DESC: return "CUDNN_ATTR_OPERATION_BN_FINALIZE_Y_SUM_DESC";
            case CUDNN_ATTR_OPERATION_BN_FINALIZE_Y_SQ_SUM_DESC: return "CUDNN_ATTR_OPERATION_BN_FINALIZE_Y_SQ_SUM_DESC";
            case CUDNN_ATTR_OPERATION_BN_FINALIZE_SCALE_DESC: return "CUDNN_ATTR_OPERATION_BN_FINALIZE_SCALE_DESC";
            case CUDNN_ATTR_OPERATION_BN_FINALIZE_BIAS_DESC: return "CUDNN_ATTR_OPERATION_BN_FINALIZE_BIAS_DESC";
            case CUDNN_ATTR_OPERATION_BN_FINALIZE_PREV_RUNNING_MEAN_DESC: return "CUDNN_ATTR_OPERATION_BN_FINALIZE_PREV_RUNNING_MEAN_DESC";
            case CUDNN_ATTR_OPERATION_BN_FINALIZE_PREV_RUNNING_VAR_DESC: return "CUDNN_ATTR_OPERATION_BN_FINALIZE_PREV_RUNNING_VAR_DESC";
            case CUDNN_ATTR_OPERATION_BN_FINALIZE_UPDATED_RUNNING_MEAN_DESC: return "CUDNN_ATTR_OPERATION_BN_FINALIZE_UPDATED_RUNNING_MEAN_DESC";
            case CUDNN_ATTR_OPERATION_BN_FINALIZE_UPDATED_RUNNING_VAR_DESC: return "CUDNN_ATTR_OPERATION_BN_FINALIZE_UPDATED_RUNNING_VAR_DESC";
            case CUDNN_ATTR_OPERATION_BN_FINALIZE_SAVED_MEAN_DESC: return "CUDNN_ATTR_OPERATION_BN_FINALIZE_SAVED_MEAN_DESC";
            case CUDNN_ATTR_OPERATION_BN_FINALIZE_SAVED_INV_STD_DESC: return "CUDNN_ATTR_OPERATION_BN_FINALIZE_SAVED_INV_STD_DESC";
            case CUDNN_ATTR_OPERATION_BN_FINALIZE_EQ_SCALE_DESC: return "CUDNN_ATTR_OPERATION_BN_FINALIZE_EQ_SCALE_DESC";
            case CUDNN_ATTR_OPERATION_BN_FINALIZE_EQ_BIAS_DESC: return "CUDNN_ATTR_OPERATION_BN_FINALIZE_EQ_BIAS_DESC";
            case CUDNN_ATTR_OPERATION_BN_FINALIZE_ACCUM_COUNT_DESC: return "CUDNN_ATTR_OPERATION_BN_FINALIZE_ACCUM_COUNT_DESC";
            case CUDNN_ATTR_OPERATION_BN_FINALIZE_EPSILON_DESC: return "CUDNN_ATTR_OPERATION_BN_FINALIZE_EPSILON_DESC";
            case CUDNN_ATTR_OPERATION_BN_FINALIZE_EXP_AVERATE_FACTOR_DESC: return "CUDNN_ATTR_OPERATION_BN_FINALIZE_EXP_AVERATE_FACTOR_DESC";
            case CUDNN_ATTR_OPERATIONGRAPH_HANDLE: return "CUDNN_ATTR_OPERATIONGRAPH_HANDLE";
            case CUDNN_ATTR_OPERATIONGRAPH_OPS: return "CUDNN_ATTR_OPERATIONGRAPH_OPS";
            case CUDNN_ATTR_OPERATIONGRAPH_ENGINE_GLOBAL_COUNT: return "CUDNN_ATTR_OPERATIONGRAPH_ENGINE_GLOBAL_COUNT";
            case CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT: return "CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT";
            case CUDNN_ATTR_TENSOR_DATA_TYPE: return "CUDNN_ATTR_TENSOR_DATA_TYPE";
            case CUDNN_ATTR_TENSOR_DIMENSIONS: return "CUDNN_ATTR_TENSOR_DIMENSIONS";
            case CUDNN_ATTR_TENSOR_STRIDES: return "CUDNN_ATTR_TENSOR_STRIDES";
            case CUDNN_ATTR_TENSOR_VECTOR_COUNT: return "CUDNN_ATTR_TENSOR_VECTOR_COUNT";
            case CUDNN_ATTR_TENSOR_VECTORIZED_DIMENSION: return "CUDNN_ATTR_TENSOR_VECTORIZED_DIMENSION";
            case CUDNN_ATTR_TENSOR_UNIQUE_ID: return "CUDNN_ATTR_TENSOR_UNIQUE_ID";
            case CUDNN_ATTR_TENSOR_IS_VIRTUAL: return "CUDNN_ATTR_TENSOR_IS_VIRTUAL";
            case CUDNN_ATTR_TENSOR_IS_BY_VALUE: return "CUDNN_ATTR_TENSOR_IS_BY_VALUE";
            case CUDNN_ATTR_TENSOR_REORDERING_MODE: return "CUDNN_ATTR_TENSOR_REORDERING_MODE";
            case CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS: return "CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS";
            case CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS: return "CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS";
            case CUDNN_ATTR_VARIANT_PACK_INTERMEDIATES: return "CUDNN_ATTR_VARIANT_PACK_INTERMEDIATES";
            case CUDNN_ATTR_VARIANT_PACK_WORKSPACE: return "CUDNN_ATTR_VARIANT_PACK_WORKSPACE";
            case CUDNN_ATTR_LAYOUT_INFO_TENSOR_UID: return "CUDNN_ATTR_LAYOUT_INFO_TENSOR_UID";
            case CUDNN_ATTR_LAYOUT_INFO_TYPES: return "CUDNN_ATTR_LAYOUT_INFO_TYPES";
            case CUDNN_ATTR_KNOB_INFO_TYPE: return "CUDNN_ATTR_KNOB_INFO_TYPE";
            case CUDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE: return "CUDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE";
            case CUDNN_ATTR_KNOB_INFO_MINIMUM_VALUE: return "CUDNN_ATTR_KNOB_INFO_MINIMUM_VALUE";
            case CUDNN_ATTR_KNOB_INFO_STRIDE: return "CUDNN_ATTR_KNOB_INFO_STRIDE";
            case CUDNN_ATTR_ENGINE_OPERATION_GRAPH: return "CUDNN_ATTR_ENGINE_OPERATION_GRAPH";
            case CUDNN_ATTR_ENGINE_GLOBAL_INDEX: return "CUDNN_ATTR_ENGINE_GLOBAL_INDEX";
            case CUDNN_ATTR_ENGINE_KNOB_INFO: return "CUDNN_ATTR_ENGINE_KNOB_INFO";
            case CUDNN_ATTR_ENGINE_NUMERICAL_NOTE: return "CUDNN_ATTR_ENGINE_NUMERICAL_NOTE";
            case CUDNN_ATTR_ENGINE_LAYOUT_INFO: return "CUDNN_ATTR_ENGINE_LAYOUT_INFO";
            case CUDNN_ATTR_ENGINE_BEHAVIOR_NOTE: return "CUDNN_ATTR_ENGINE_BEHAVIOR_NOTE";
            case CUDNN_ATTR_MATMUL_COMP_TYPE: return "CUDNN_ATTR_MATMUL_COMP_TYPE";
            case CUDNN_ATTR_OPERATION_MATMUL_ADESC: return "CUDNN_ATTR_OPERATION_MATMUL_ADESC";
            case CUDNN_ATTR_OPERATION_MATMUL_BDESC: return "CUDNN_ATTR_OPERATION_MATMUL_BDESC";
            case CUDNN_ATTR_OPERATION_MATMUL_CDESC: return "CUDNN_ATTR_OPERATION_MATMUL_CDESC";
            case CUDNN_ATTR_OPERATION_MATMUL_DESC: return "CUDNN_ATTR_OPERATION_MATMUL_DESC";
            case CUDNN_ATTR_OPERATION_MATMUL_IRREGULARLY_STRIDED_BATCH_COUNT: return "CUDNN_ATTR_OPERATION_MATMUL_IRREGULARLY_STRIDED_BATCH_COUNT";
            case CUDNN_ATTR_OPERATION_MATMUL_GEMM_M_OVERRIDE_DESC: return "CUDNN_ATTR_OPERATION_MATMUL_GEMM_M_OVERRIDE_DESC";
            case CUDNN_ATTR_OPERATION_MATMUL_GEMM_N_OVERRIDE_DESC: return "CUDNN_ATTR_OPERATION_MATMUL_GEMM_N_OVERRIDE_DESC";
            case CUDNN_ATTR_OPERATION_MATMUL_GEMM_K_OVERRIDE_DESC: return "CUDNN_ATTR_OPERATION_MATMUL_GEMM_K_OVERRIDE_DESC";
            case CUDNN_ATTR_REDUCTION_OPERATOR: return "CUDNN_ATTR_REDUCTION_OPERATOR";
            case CUDNN_ATTR_REDUCTION_COMP_TYPE: return "CUDNN_ATTR_REDUCTION_COMP_TYPE";
            case CUDNN_ATTR_OPERATION_REDUCTION_XDESC: return "CUDNN_ATTR_OPERATION_REDUCTION_XDESC";
            case CUDNN_ATTR_OPERATION_REDUCTION_YDESC: return "CUDNN_ATTR_OPERATION_REDUCTION_YDESC";
            case CUDNN_ATTR_OPERATION_REDUCTION_DESC: return "CUDNN_ATTR_OPERATION_REDUCTION_DESC";
            case CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_MATH_PREC: return "CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_MATH_PREC";
            case CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_MEAN_DESC: return "CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_MEAN_DESC";
            case CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_INVSTD_DESC: return "CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_INVSTD_DESC";
            case CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_BN_SCALE_DESC: return "CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_BN_SCALE_DESC";
            case CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_X_DESC: return "CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_X_DESC";
            case CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DY_DESC: return "CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DY_DESC";
            case CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DBN_SCALE_DESC: return "CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DBN_SCALE_DESC";
            case CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DBN_BIAS_DESC: return "CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DBN_BIAS_DESC";
            case CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_DY_SCALE_DESC: return "CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_DY_SCALE_DESC";
            case CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_X_SCALE_DESC: return "CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_X_SCALE_DESC";
            case CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_BIAS: return "CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_BIAS";
            case CUDNN_ATTR_RESAMPLE_MODE: return "CUDNN_ATTR_RESAMPLE_MODE";
            case CUDNN_ATTR_RESAMPLE_COMP_TYPE: return "CUDNN_ATTR_RESAMPLE_COMP_TYPE";
            case CUDNN_ATTR_RESAMPLE_SPATIAL_DIMS: return "CUDNN_ATTR_RESAMPLE_SPATIAL_DIMS";
            case CUDNN_ATTR_RESAMPLE_POST_PADDINGS: return "CUDNN_ATTR_RESAMPLE_POST_PADDINGS";
            case CUDNN_ATTR_RESAMPLE_PRE_PADDINGS: return "CUDNN_ATTR_RESAMPLE_PRE_PADDINGS";
            case CUDNN_ATTR_RESAMPLE_STRIDES: return "CUDNN_ATTR_RESAMPLE_STRIDES";
            case CUDNN_ATTR_RESAMPLE_WINDOW_DIMS: return "CUDNN_ATTR_RESAMPLE_WINDOW_DIMS";
            case CUDNN_ATTR_RESAMPLE_NAN_PROPAGATION: return "CUDNN_ATTR_RESAMPLE_NAN_PROPAGATION";
            case CUDNN_ATTR_RESAMPLE_PADDING_MODE: return "CUDNN_ATTR_RESAMPLE_PADDING_MODE";
            case CUDNN_ATTR_OPERATION_RESAMPLE_FWD_XDESC: return "CUDNN_ATTR_OPERATION_RESAMPLE_FWD_XDESC";
            case CUDNN_ATTR_OPERATION_RESAMPLE_FWD_YDESC: return "CUDNN_ATTR_OPERATION_RESAMPLE_FWD_YDESC";
            case CUDNN_ATTR_OPERATION_RESAMPLE_FWD_IDXDESC: return "CUDNN_ATTR_OPERATION_RESAMPLE_FWD_IDXDESC";
            case CUDNN_ATTR_OPERATION_RESAMPLE_FWD_ALPHA: return "CUDNN_ATTR_OPERATION_RESAMPLE_FWD_ALPHA";
            case CUDNN_ATTR_OPERATION_RESAMPLE_FWD_BETA: return "CUDNN_ATTR_OPERATION_RESAMPLE_FWD_BETA";
            case CUDNN_ATTR_OPERATION_RESAMPLE_FWD_DESC: return "CUDNN_ATTR_OPERATION_RESAMPLE_FWD_DESC";
            case CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DXDESC: return "CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DXDESC";
            case CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DYDESC: return "CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DYDESC";
            case CUDNN_ATTR_OPERATION_RESAMPLE_BWD_IDXDESC: return "CUDNN_ATTR_OPERATION_RESAMPLE_BWD_IDXDESC";
            case CUDNN_ATTR_OPERATION_RESAMPLE_BWD_ALPHA: return "CUDNN_ATTR_OPERATION_RESAMPLE_BWD_ALPHA";
            case CUDNN_ATTR_OPERATION_RESAMPLE_BWD_BETA: return "CUDNN_ATTR_OPERATION_RESAMPLE_BWD_BETA";
            case CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DESC: return "CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DESC";
            case CUDNN_ATTR_OPERATION_RESAMPLE_BWD_XDESC: return "CUDNN_ATTR_OPERATION_RESAMPLE_BWD_XDESC";
            case CUDNN_ATTR_OPERATION_RESAMPLE_BWD_YDESC: return "CUDNN_ATTR_OPERATION_RESAMPLE_BWD_YDESC";
            case CUDNN_ATTR_OPERATION_CONCAT_AXIS: return "CUDNN_ATTR_OPERATION_CONCAT_AXIS";
            case CUDNN_ATTR_OPERATION_CONCAT_INPUT_DESCS: return "CUDNN_ATTR_OPERATION_CONCAT_INPUT_DESCS";
            case CUDNN_ATTR_OPERATION_CONCAT_INPLACE_INDEX: return "CUDNN_ATTR_OPERATION_CONCAT_INPLACE_INDEX";
            case CUDNN_ATTR_OPERATION_CONCAT_OUTPUT_DESC: return "CUDNN_ATTR_OPERATION_CONCAT_OUTPUT_DESC";
            case CUDNN_ATTR_OPERATION_SIGNAL_MODE: return "CUDNN_ATTR_OPERATION_SIGNAL_MODE";
            case CUDNN_ATTR_OPERATION_SIGNAL_FLAGDESC: return "CUDNN_ATTR_OPERATION_SIGNAL_FLAGDESC";
            case CUDNN_ATTR_OPERATION_SIGNAL_VALUE: return "CUDNN_ATTR_OPERATION_SIGNAL_VALUE";
            case CUDNN_ATTR_OPERATION_SIGNAL_XDESC: return "CUDNN_ATTR_OPERATION_SIGNAL_XDESC";
            case CUDNN_ATTR_OPERATION_SIGNAL_YDESC: return "CUDNN_ATTR_OPERATION_SIGNAL_YDESC";
            case CUDNN_ATTR_OPERATION_NORM_FWD_MODE: return "CUDNN_ATTR_OPERATION_NORM_FWD_MODE";
            case CUDNN_ATTR_OPERATION_NORM_FWD_PHASE: return "CUDNN_ATTR_OPERATION_NORM_FWD_PHASE";
            case CUDNN_ATTR_OPERATION_NORM_FWD_XDESC: return "CUDNN_ATTR_OPERATION_NORM_FWD_XDESC";
            case CUDNN_ATTR_OPERATION_NORM_FWD_MEAN_DESC: return "CUDNN_ATTR_OPERATION_NORM_FWD_MEAN_DESC";
            case CUDNN_ATTR_OPERATION_NORM_FWD_INV_VARIANCE_DESC: return "CUDNN_ATTR_OPERATION_NORM_FWD_INV_VARIANCE_DESC";
            case CUDNN_ATTR_OPERATION_NORM_FWD_SCALE_DESC: return "CUDNN_ATTR_OPERATION_NORM_FWD_SCALE_DESC";
            case CUDNN_ATTR_OPERATION_NORM_FWD_BIAS_DESC: return "CUDNN_ATTR_OPERATION_NORM_FWD_BIAS_DESC";
            case CUDNN_ATTR_OPERATION_NORM_FWD_EPSILON_DESC: return "CUDNN_ATTR_OPERATION_NORM_FWD_EPSILON_DESC";
            case CUDNN_ATTR_OPERATION_NORM_FWD_EXP_AVG_FACTOR_DESC: return "CUDNN_ATTR_OPERATION_NORM_FWD_EXP_AVG_FACTOR_DESC";
            case CUDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_MEAN_DESC: return "CUDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_MEAN_DESC";
            case CUDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_VAR_DESC: return "CUDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_VAR_DESC";
            case CUDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_MEAN_DESC: return "CUDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_MEAN_DESC";
            case CUDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_VAR_DESC: return "CUDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_VAR_DESC";
            case CUDNN_ATTR_OPERATION_NORM_FWD_YDESC: return "CUDNN_ATTR_OPERATION_NORM_FWD_YDESC";
            case CUDNN_ATTR_OPERATION_NORM_FWD_PEER_STAT_DESCS: return "CUDNN_ATTR_OPERATION_NORM_FWD_PEER_STAT_DESCS";
            case CUDNN_ATTR_OPERATION_NORM_BWD_MODE: return "CUDNN_ATTR_OPERATION_NORM_BWD_MODE";
            case CUDNN_ATTR_OPERATION_NORM_BWD_XDESC: return "CUDNN_ATTR_OPERATION_NORM_BWD_XDESC";
            case CUDNN_ATTR_OPERATION_NORM_BWD_MEAN_DESC: return "CUDNN_ATTR_OPERATION_NORM_BWD_MEAN_DESC";
            case CUDNN_ATTR_OPERATION_NORM_BWD_INV_VARIANCE_DESC: return "CUDNN_ATTR_OPERATION_NORM_BWD_INV_VARIANCE_DESC";
            case CUDNN_ATTR_OPERATION_NORM_BWD_DYDESC: return "CUDNN_ATTR_OPERATION_NORM_BWD_DYDESC";
            case CUDNN_ATTR_OPERATION_NORM_BWD_SCALE_DESC: return "CUDNN_ATTR_OPERATION_NORM_BWD_SCALE_DESC";
            case CUDNN_ATTR_OPERATION_NORM_BWD_EPSILON_DESC: return "CUDNN_ATTR_OPERATION_NORM_BWD_EPSILON_DESC";
            case CUDNN_ATTR_OPERATION_NORM_BWD_DSCALE_DESC: return "CUDNN_ATTR_OPERATION_NORM_BWD_DSCALE_DESC";
            case CUDNN_ATTR_OPERATION_NORM_BWD_DBIAS_DESC: return "CUDNN_ATTR_OPERATION_NORM_BWD_DBIAS_DESC";
            case CUDNN_ATTR_OPERATION_NORM_BWD_DXDESC: return "CUDNN_ATTR_OPERATION_NORM_BWD_DXDESC";
            case CUDNN_ATTR_OPERATION_NORM_BWD_PEER_STAT_DESCS: return "CUDNN_ATTR_OPERATION_NORM_BWD_PEER_STAT_DESCS";
            case CUDNN_ATTR_OPERATION_RESHAPE_XDESC: return "CUDNN_ATTR_OPERATION_RESHAPE_XDESC";
            case CUDNN_ATTR_OPERATION_RESHAPE_YDESC: return "CUDNN_ATTR_OPERATION_RESHAPE_YDESC";
            case CUDNN_ATTR_RNG_DISTRIBUTION: return "CUDNN_ATTR_RNG_DISTRIBUTION";
            case CUDNN_ATTR_RNG_NORMAL_DIST_MEAN: return "CUDNN_ATTR_RNG_NORMAL_DIST_MEAN";
            case CUDNN_ATTR_RNG_NORMAL_DIST_STANDARD_DEVIATION: return "CUDNN_ATTR_RNG_NORMAL_DIST_STANDARD_DEVIATION";
            case CUDNN_ATTR_RNG_UNIFORM_DIST_MAXIMUM: return "CUDNN_ATTR_RNG_UNIFORM_DIST_MAXIMUM";
            case CUDNN_ATTR_RNG_UNIFORM_DIST_MINIMUM: return "CUDNN_ATTR_RNG_UNIFORM_DIST_MINIMUM";
            case CUDNN_ATTR_RNG_BERNOULLI_DIST_PROBABILITY: return "CUDNN_ATTR_RNG_BERNOULLI_DIST_PROBABILITY";
            case CUDNN_ATTR_OPERATION_RNG_YDESC: return "CUDNN_ATTR_OPERATION_RNG_YDESC";
            case CUDNN_ATTR_OPERATION_RNG_SEED: return "CUDNN_ATTR_OPERATION_RNG_SEED";
            case CUDNN_ATTR_OPERATION_RNG_DESC: return "CUDNN_ATTR_OPERATION_RNG_DESC";
        }
        return "INVALID cudnnBackendAttributeName: "+n;
    }
}

