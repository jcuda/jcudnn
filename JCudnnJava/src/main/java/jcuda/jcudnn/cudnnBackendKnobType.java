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

public class cudnnBackendKnobType
{
    public static final int CUDNN_KNOB_TYPE_SPLIT_K = 0;
    public static final int CUDNN_KNOB_TYPE_SWIZZLE = 1;
    public static final int CUDNN_KNOB_TYPE_TILE_SIZE = 2;
    public static final int CUDNN_KNOB_TYPE_USE_TEX = 3;
    public static final int CUDNN_KNOB_TYPE_EDGE = 4;
    public static final int CUDNN_KNOB_TYPE_KBLOCK = 5;
    public static final int CUDNN_KNOB_TYPE_LDGA = 6;
    public static final int CUDNN_KNOB_TYPE_LDGB = 7;
    public static final int CUDNN_KNOB_TYPE_CHUNK_K = 8;
    public static final int CUDNN_KNOB_TYPE_SPLIT_H = 9;
    public static final int CUDNN_KNOB_TYPE_WINO_TILE = 10;
    public static final int CUDNN_KNOB_TYPE_MULTIPLY = 11;
    public static final int CUDNN_KNOB_TYPE_SPLIT_K_BUF = 12;
    public static final int CUDNN_KNOB_TYPE_TILEK = 13;
    public static final int CUDNN_KNOB_TYPE_STAGES = 14;
    public static final int CUDNN_KNOB_TYPE_REDUCTION_MODE = 15;
    public static final int CUDNN_KNOB_TYPE_CTA_SPLIT_K_MODE = 16;
    public static final int CUDNN_KNOB_TYPE_SPLIT_K_SLC = 17;
    public static final int CUDNN_KNOB_TYPE_IDX_MODE = 18;
    public static final int CUDNN_KNOB_TYPE_SLICED = 19;
    public static final int CUDNN_KNOB_TYPE_SPLIT_RS = 20;
    public static final int CUDNN_KNOB_TYPE_SINGLEBUFFER = 21;
    public static final int CUDNN_KNOB_TYPE_LDGC = 22;
    public static final int CUDNN_KNOB_TYPE_SPECFILT = 23;
    public static final int CUDNN_KNOB_TYPE_KERNEL_CFG = 24;
    public static final int CUDNN_KNOB_TYPE_WORKSPACE = 25;
    public static final int CUDNN_KNOB_TYPE_TILE_CGA = 26;
    public static final int CUDNN_KNOB_TYPE_TILE_CGA_M = 27;
    public static final int CUDNN_KNOB_TYPE_TILE_CGA_N = 28;
    public static final int CUDNN_KNOB_TYPE_BLOCK_SIZE = 29;
    public static final int CUDNN_KNOB_TYPE_OCCUPANCY = 30;
    public static final int CUDNN_KNOB_TYPE_ARRAY_SIZE_PER_THREAD = 31;
    public static final int CUDNN_KNOB_TYPE_NUM_C_PER_BLOCK = 32;
    public static final int CUDNN_KNOB_TYPE_COUNTS = 33;

    /**
     * Private constructor to prevent instantiation
     */
    private cudnnBackendKnobType()
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
            case CUDNN_KNOB_TYPE_SPLIT_K: return "CUDNN_KNOB_TYPE_SPLIT_K";
            case CUDNN_KNOB_TYPE_SWIZZLE: return "CUDNN_KNOB_TYPE_SWIZZLE";
            case CUDNN_KNOB_TYPE_TILE_SIZE: return "CUDNN_KNOB_TYPE_TILE_SIZE";
            case CUDNN_KNOB_TYPE_USE_TEX: return "CUDNN_KNOB_TYPE_USE_TEX";
            case CUDNN_KNOB_TYPE_EDGE: return "CUDNN_KNOB_TYPE_EDGE";
            case CUDNN_KNOB_TYPE_KBLOCK: return "CUDNN_KNOB_TYPE_KBLOCK";
            case CUDNN_KNOB_TYPE_LDGA: return "CUDNN_KNOB_TYPE_LDGA";
            case CUDNN_KNOB_TYPE_LDGB: return "CUDNN_KNOB_TYPE_LDGB";
            case CUDNN_KNOB_TYPE_CHUNK_K: return "CUDNN_KNOB_TYPE_CHUNK_K";
            case CUDNN_KNOB_TYPE_SPLIT_H: return "CUDNN_KNOB_TYPE_SPLIT_H";
            case CUDNN_KNOB_TYPE_WINO_TILE: return "CUDNN_KNOB_TYPE_WINO_TILE";
            case CUDNN_KNOB_TYPE_MULTIPLY: return "CUDNN_KNOB_TYPE_MULTIPLY";
            case CUDNN_KNOB_TYPE_SPLIT_K_BUF: return "CUDNN_KNOB_TYPE_SPLIT_K_BUF";
            case CUDNN_KNOB_TYPE_TILEK: return "CUDNN_KNOB_TYPE_TILEK";
            case CUDNN_KNOB_TYPE_STAGES: return "CUDNN_KNOB_TYPE_STAGES";
            case CUDNN_KNOB_TYPE_REDUCTION_MODE: return "CUDNN_KNOB_TYPE_REDUCTION_MODE";
            case CUDNN_KNOB_TYPE_CTA_SPLIT_K_MODE: return "CUDNN_KNOB_TYPE_CTA_SPLIT_K_MODE";
            case CUDNN_KNOB_TYPE_SPLIT_K_SLC: return "CUDNN_KNOB_TYPE_SPLIT_K_SLC";
            case CUDNN_KNOB_TYPE_IDX_MODE: return "CUDNN_KNOB_TYPE_IDX_MODE";
            case CUDNN_KNOB_TYPE_SLICED: return "CUDNN_KNOB_TYPE_SLICED";
            case CUDNN_KNOB_TYPE_SPLIT_RS: return "CUDNN_KNOB_TYPE_SPLIT_RS";
            case CUDNN_KNOB_TYPE_SINGLEBUFFER: return "CUDNN_KNOB_TYPE_SINGLEBUFFER";
            case CUDNN_KNOB_TYPE_LDGC: return "CUDNN_KNOB_TYPE_LDGC";
            case CUDNN_KNOB_TYPE_SPECFILT: return "CUDNN_KNOB_TYPE_SPECFILT";
            case CUDNN_KNOB_TYPE_KERNEL_CFG: return "CUDNN_KNOB_TYPE_KERNEL_CFG";
            case CUDNN_KNOB_TYPE_WORKSPACE: return "CUDNN_KNOB_TYPE_WORKSPACE";
            case CUDNN_KNOB_TYPE_TILE_CGA: return "CUDNN_KNOB_TYPE_TILE_CGA";
            case CUDNN_KNOB_TYPE_TILE_CGA_M: return "CUDNN_KNOB_TYPE_TILE_CGA_M";
            case CUDNN_KNOB_TYPE_TILE_CGA_N: return "CUDNN_KNOB_TYPE_TILE_CGA_N";
            case CUDNN_KNOB_TYPE_BLOCK_SIZE: return "CUDNN_KNOB_TYPE_BLOCK_SIZE";
            case CUDNN_KNOB_TYPE_OCCUPANCY: return "CUDNN_KNOB_TYPE_OCCUPANCY";
            case CUDNN_KNOB_TYPE_ARRAY_SIZE_PER_THREAD: return "CUDNN_KNOB_TYPE_ARRAY_SIZE_PER_THREAD";
            case CUDNN_KNOB_TYPE_NUM_C_PER_BLOCK: return "CUDNN_KNOB_TYPE_NUM_C_PER_BLOCK";
            case CUDNN_KNOB_TYPE_COUNTS: return "CUDNN_KNOB_TYPE_COUNTS";
        }
        return "INVALID cudnnBackendKnobType: "+n;
    }
}

