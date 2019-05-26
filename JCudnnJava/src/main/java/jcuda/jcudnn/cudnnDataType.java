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
* CUDNN data type
*/
public class cudnnDataType
{
    public static final int CUDNN_DATA_FLOAT = 0;
    public static final int CUDNN_DATA_DOUBLE = 1;
    public static final int CUDNN_DATA_HALF = 2;
    public static final int CUDNN_DATA_INT8 = 3;
    public static final int CUDNN_DATA_INT32 = 4;
    public static final int CUDNN_DATA_INT8x4 = 5;
    public static final int CUDNN_DATA_UINT8 = 6;
    public static final int CUDNN_DATA_UINT8x4 = 7;
    public static final int CUDNN_DATA_INT8x32 = 8;

    /**
     * Private constructor to prevent instantiation
     */
    private cudnnDataType()
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
            case CUDNN_DATA_FLOAT: return "CUDNN_DATA_FLOAT";
            case CUDNN_DATA_DOUBLE: return "CUDNN_DATA_DOUBLE";
            case CUDNN_DATA_HALF: return "CUDNN_DATA_HALF";
            case CUDNN_DATA_INT8: return "CUDNN_DATA_INT8";
            case CUDNN_DATA_INT32: return "CUDNN_DATA_INT32";
            case CUDNN_DATA_INT8x4: return "CUDNN_DATA_INT8x4";
            case CUDNN_DATA_UINT8: return "CUDNN_DATA_UINT8";
            case CUDNN_DATA_UINT8x4: return "CUDNN_DATA_UINT8x4";
            case CUDNN_DATA_INT8x32: return "CUDNN_DATA_INT8x32";
        }
        return "INVALID cudnnDataType: "+n;
    }
}

