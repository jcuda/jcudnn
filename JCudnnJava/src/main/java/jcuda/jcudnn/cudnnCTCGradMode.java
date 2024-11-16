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
 * <pre>
 * Behavior for OOB samples. OOB samples are samples where L+R > T is encountered during the gradient calculation. If
 * gradMode is set to CUDNN_CTC_SKIP_OOB_GRADIENTS, then the CTC loss function does not write to the gradient buffer for
 * that sample. Instead, the current values, even not finite, are retained. If gradMode is set to
 * CUDNN_CTC_ZERO_OOB_GRADIENTS, then the gradient for that sample is set to zero. This guarantees a finite gradient.
 * </pre>
 */
public class cudnnCTCGradMode
{
    public static final int CUDNN_CTC_ZERO_OOB_GRADIENTS = 0;
    public static final int CUDNN_CTC_SKIP_OOB_GRADIENTS = 1;

    /**
     * Private constructor to prevent instantiation
     */
    private cudnnCTCGradMode()
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
            case CUDNN_CTC_ZERO_OOB_GRADIENTS: return "CUDNN_CTC_ZERO_OOB_GRADIENTS";
            case CUDNN_CTC_SKIP_OOB_GRADIENTS: return "CUDNN_CTC_SKIP_OOB_GRADIENTS";
        }
        return "INVALID cudnnCTCGradMode: "+n;
    }
}

