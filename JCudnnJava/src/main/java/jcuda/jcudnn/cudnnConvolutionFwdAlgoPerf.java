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
 * Java port of a cudnnConvolutionFwdAlgoPerf
 */
public class cudnnConvolutionFwdAlgoPerf
{
    public int algo;
    public int status;
    public float time;
    public long memory;
    public int determinism;
    public int mathType;
    public int[] reserved;

    /**
     * Creates a new, uninitialized cudnnConvolutionFwdAlgoPerf
     */
    public cudnnConvolutionFwdAlgoPerf()
    {
        // Default constructor
    }

    /**
     * Creates a new cudnnConvolutionFwdAlgoPerf with the given values
     *
     * @param algo The algo value
     * @param status The status value
     * @param time The time value
     * @param memory The memory value
     * @param determinism The determinism value
     * @param mathType The mathType value
     * @param reserved The reserved value
     */
    public cudnnConvolutionFwdAlgoPerf(int algo, int status, float time, long memory, int determinism, int mathType, int[] reserved)
    {
        this.algo = algo;
        this.status = status;
        this.time = time;
        this.memory = memory;
        this.determinism = determinism;
        this.mathType = mathType;
        this.reserved = reserved;
    }

    @Override
    public String toString()
    {
        return "cudnnConvolutionFwdAlgoPerf["+
            "algo="+algo+","+
            "status="+status+","+
            "time="+time+","+
            "memory="+memory+","+
            "determinism="+determinism+","+
            "mathType="+mathType+","+
            "reserved="+reserved+"]";
    }
}


