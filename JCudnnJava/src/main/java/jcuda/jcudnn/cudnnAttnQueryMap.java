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
 * Multihead Attention
 * 
 * @deprecated Declared as a "legacy type" in CUDNN 7.6.5. Seems to be 
 * replaced by the {@link JCudnn#CUDNN_ATTN_QUERYMAP_ALL_TO_ONE 
 * CUDNN_ATTN_QUERYMAP constants} 
 */
public class cudnnAttnQueryMap
{
    /**
     * multiple Q-s when beam width > 1 map to a single (K,V) set 
     */
    public static final int CUDNN_ATTN_QUERYMAP_ALL_TO_ONE = 0;
    /**
     * multiple Q-s when beam width > 1 map to corresponding (K,V) sets 
     */
    public static final int CUDNN_ATTN_QUERYMAP_ONE_TO_ONE = 1;

    /**
     * Private constructor to prevent instantiation
     */
    private cudnnAttnQueryMap()
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
            case CUDNN_ATTN_QUERYMAP_ALL_TO_ONE: return "CUDNN_ATTN_QUERYMAP_ALL_TO_ONE";
            case CUDNN_ATTN_QUERYMAP_ONE_TO_ONE: return "CUDNN_ATTN_QUERYMAP_ONE_TO_ONE";
        }
        return "INVALID cudnnAttnQueryMap: "+n;
    }
}
