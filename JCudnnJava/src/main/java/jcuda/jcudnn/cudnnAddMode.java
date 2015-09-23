/*
 * JCudnn - Java bindings for cuDNN, the NVIDIA CUDA
 * Deep Neural Network library, to be used with JCuda
 *
 * Copyright (c) 2015-2015 Marco Hutter - http://www.jcuda.org
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

public class cudnnAddMode
{
    /**
     * add one image to every feature maps of each input 
     */
    public static final int CUDNN_ADD_IMAGE = 0;
    public static final int CUDNN_ADD_SAME_HW = 0;
    /**
     * add a set of feature maps to a batch of inputs : tensorBias has n=1 , same nb feature than Src/dest 
     */
    public static final int CUDNN_ADD_FEATURE_MAP = 1;
    public static final int CUDNN_ADD_SAME_CHW = 1;
    /**
     * add a tensor of size 1,c,1,1 to every corresponding point of n,c,h,w input 
     */
    public static final int CUDNN_ADD_SAME_C = 2;
    /**
     * add 2 tensors with same n,c,h,w 
     */
    public static final int CUDNN_ADD_FULL_TENSOR = 3;

    /**
     * Private constructor to prevent instantiation
     */
    private cudnnAddMode(){}

    /**
     * Returns a string representation of the given constant
     *
     * @return A string representation of the given constant
     */
    public static String stringFor(int n)
    {
        switch (n)
        {
            case CUDNN_ADD_IMAGE: 
            //case CUDNN_ADD_SAME_HW: 
                return "(CUDNN_ADD_IMAGE or CUDNN_ADD_SAME_HW)";
            case CUDNN_ADD_FEATURE_MAP: 
            //case CUDNN_ADD_SAME_CHW: 
                return "(CUDNN_ADD_FEATURE_MAP or CUDNN_ADD_SAME_CHW)";
            case CUDNN_ADD_SAME_C: return "CUDNN_ADD_SAME_C";
            case CUDNN_ADD_FULL_TENSOR: return "CUDNN_ADD_FULL_TENSOR";
        }
        return "INVALID cudnnAddMode: "+n;
    }
}

