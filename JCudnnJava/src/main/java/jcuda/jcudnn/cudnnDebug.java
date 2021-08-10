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

import jcuda.runtime.cudaStream_t;

/** struct containing useful informaiton for each API call */
public class cudnnDebug
{
    public int cudnn_version;
    public int cudnnStatus;
    /**
     * epoch time in seconds 
     */
    public int time_sec;
    /**
     * microseconds part of epoch time 
     */
    public int time_usec;
    /**
     * time since start in seconds 
     */
    public int time_delta;
    /**
     * cudnn handle 
     */
    public cudnnHandle handle;
    /**
     * cuda stream ID 
     */
    public cudaStream_t stream;
    /**
     * process ID 
     */
    public long pid;
    /**
     * thread ID 
     */
    public long tid;
    /**
     * CUDA device ID 
     */
    public int cudaDeviceId;
    /**
     * reserved for future use 
     */
    public int[] reserved;

    /**
     * Creates a new, uninitialized cudnnDebug
     */
    public cudnnDebug()
    {
        // Default constructor
    }

    /**
     * Creates a new cudnnDebug with the given values
     *
     * @param cudnn_version The cudnn_version value
     * @param cudnnStatus The cudnnStatus value
     * @param time_sec The time_sec value
     * @param time_usec The time_usec value
     * @param time_delta The time_delta value
     * @param handle The handle value
     * @param stream The stream value
     * @param pid The pid value
     * @param tid The tid value
     * @param cudaDeviceId The cudaDeviceId value
     * @param reserved The reserved value
     */
    public cudnnDebug(int cudnn_version, int cudnnStatus, int time_sec, int time_usec, int time_delta, cudnnHandle handle, cudaStream_t stream, long pid, long tid, int cudaDeviceId, int[] reserved)
    {
        this.cudnn_version = cudnn_version;
        this.cudnnStatus = cudnnStatus;
        this.time_sec = time_sec;
        this.time_usec = time_usec;
        this.time_delta = time_delta;
        this.handle = handle;
        this.stream = stream;
        this.pid = pid;
        this.tid = tid;
        this.cudaDeviceId = cudaDeviceId;
        this.reserved = reserved;
    }

    @Override
    public String toString()
    {
        return "cudnnDebug["+
            "cudnn_version="+cudnn_version+","+
            "cudnnStatus="+cudnnStatus+","+
            "time_sec="+time_sec+","+
            "time_usec="+time_usec+","+
            "time_delta="+time_delta+","+
            "handle="+handle+","+
            "stream="+stream+","+
            "pid="+pid+","+
            "tid="+tid+","+
            "cudaDeviceId="+cudaDeviceId+","+
            "reserved="+reserved+"]";
    }
}


