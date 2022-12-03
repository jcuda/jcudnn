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

public class cudnnBackendNumericalNote
{
    public static final int CUDNN_NUMERICAL_NOTE_TENSOR_CORE = 0;
    public static final int CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS = 1;
    public static final int CUDNN_NUMERICAL_NOTE_REDUCED_PRECISION_REDUCTION = 2;
    public static final int CUDNN_NUMERICAL_NOTE_FFT = 3;
    public static final int CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC = 4;
    public static final int CUDNN_NUMERICAL_NOTE_WINOGRAD = 5;
    public static final int CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_4x4 = 6;
    public static final int CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_6x6 = 7;
    public static final int CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_13x13 = 8;
    public static final int CUDNN_NUMERICAL_NOTE_TYPE_COUNT = 9;

    /**
     * Private constructor to prevent instantiation
     */
    private cudnnBackendNumericalNote()
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
            case CUDNN_NUMERICAL_NOTE_TENSOR_CORE: return "CUDNN_NUMERICAL_NOTE_TENSOR_CORE";
            case CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS: return "CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS";
            case CUDNN_NUMERICAL_NOTE_REDUCED_PRECISION_REDUCTION: return "CUDNN_NUMERICAL_NOTE_REDUCED_PRECISION_REDUCTION";
            case CUDNN_NUMERICAL_NOTE_FFT: return "CUDNN_NUMERICAL_NOTE_FFT";
            case CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC: return "CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC";
            case CUDNN_NUMERICAL_NOTE_WINOGRAD: return "CUDNN_NUMERICAL_NOTE_WINOGRAD";
            case CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_4x4: return "CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_4x4";
            case CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_6x6: return "CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_6x6";
            case CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_13x13: return "CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_13x13";
            case CUDNN_NUMERICAL_NOTE_TYPE_COUNT: return "CUDNN_NUMERICAL_NOTE_TYPE_COUNT";
        }
        return "INVALID cudnnBackendNumericalNote: "+n;
    }
}

