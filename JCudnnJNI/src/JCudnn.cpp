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

#include "JCudnn.hpp"
#include "JCudnn_common.hpp"
#include <iostream>
#include <string>
#include <map>

// Class and method ID for cudnnConvolutionFwdAlgoPerf and its constructor
jclass cudnnConvolutionFwdAlgoPerf_Class;
jmethodID cudnnConvolutionFwdAlgoPerf_Constructor;

// Field IDs for cudnnConvolutionFwdAlgoPerf
jfieldID cudnnConvolutionFwdAlgoPerf_algo; // cudnnConvolutionFwdAlgo_t
jfieldID cudnnConvolutionFwdAlgoPerf_status; // cudnnStatus_t
jfieldID cudnnConvolutionFwdAlgoPerf_time; // float
jfieldID cudnnConvolutionFwdAlgoPerf_memory; // size_t

// Class and method ID for cudnnConvolutionBwdFilterAlgoPerf and its constructor
jclass cudnnConvolutionBwdFilterAlgoPerf_Class;
jmethodID cudnnConvolutionBwdFilterAlgoPerf_Constructor;

// Field IDs for cudnnConvolutionBwdFilterAlgoPerf
jfieldID cudnnConvolutionBwdFilterAlgoPerf_algo; // cudnnConvolutionBwdFilterAlgo_t
jfieldID cudnnConvolutionBwdFilterAlgoPerf_status; // cudnnStatus_t
jfieldID cudnnConvolutionBwdFilterAlgoPerf_time; // float
jfieldID cudnnConvolutionBwdFilterAlgoPerf_memory; // size_t

// Class and method ID for cudnnConvolutionBwdDataAlgoPerf and its constructor
jclass cudnnConvolutionBwdDataAlgoPerf_Class;
jmethodID cudnnConvolutionBwdDataAlgoPerf_Constructor;

// Field IDs for cudnnConvolutionBwdDataAlgoPerf
jfieldID cudnnConvolutionBwdDataAlgoPerf_algo; // cudnnConvolutionBwdDataAlgo_t
jfieldID cudnnConvolutionBwdDataAlgoPerf_status; // cudnnStatus_t
jfieldID cudnnConvolutionBwdDataAlgoPerf_time; // float
jfieldID cudnnConvolutionBwdDataAlgoPerf_memory; // size_t


/**
 * Called when the library is loaded. Will initialize all
 * required field and method IDs
 */
JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *jvm, void *reserved)
{
    JNIEnv *env = NULL;
    if (jvm->GetEnv((void **)&env, JNI_VERSION_1_4))
    {
        return JNI_ERR;
    }

    Logger::log(LOG_TRACE, "Initializing JCudnn\n");


    // Initialize the JNIUtils and PointerUtils
    if (initJNIUtils(env) == JNI_ERR) return JNI_ERR;
    if (initPointerUtils(env) == JNI_ERR) return JNI_ERR;

    // Obtain classes and constructors of performance info structures
    if (!init(env, cudnnConvolutionFwdAlgoPerf_Class,       cudnnConvolutionFwdAlgoPerf_Constructor,       "jcuda/jcudnn/cudnnConvolutionFwdAlgoPerf"      )) return JNI_ERR;
    if (!init(env, cudnnConvolutionBwdFilterAlgoPerf_Class, cudnnConvolutionBwdFilterAlgoPerf_Constructor, "jcuda/jcudnn/cudnnConvolutionBwdFilterAlgoPerf")) return JNI_ERR;
    if (!init(env, cudnnConvolutionBwdDataAlgoPerf_Class,   cudnnConvolutionBwdDataAlgoPerf_Constructor,   "jcuda/jcudnn/cudnnConvolutionBwdDataAlgoPerf"  )) return JNI_ERR;

    jclass cls = NULL;

    // Obtain the fieldIDs for cudnnConvolutionFwdAlgoPerf
    if (!init(env, cls, "jcuda/jcudnn/cudnnConvolutionFwdAlgoPerf")) return JNI_ERR;
    if (!init(env, cls, cudnnConvolutionFwdAlgoPerf_algo, "algo", "I")) return JNI_ERR;
    if (!init(env, cls, cudnnConvolutionFwdAlgoPerf_status, "status", "I")) return JNI_ERR;
    if (!init(env, cls, cudnnConvolutionFwdAlgoPerf_time, "time", "F")) return JNI_ERR;
    if (!init(env, cls, cudnnConvolutionFwdAlgoPerf_memory, "memory", "J")) return JNI_ERR;

    // Obtain the fieldIDs for cudnnConvolutionBwdFilterAlgoPerf
    if (!init(env, cls, "jcuda/jcudnn/cudnnConvolutionBwdFilterAlgoPerf")) return JNI_ERR;
    if (!init(env, cls, cudnnConvolutionBwdFilterAlgoPerf_algo, "algo", "I")) return JNI_ERR;
    if (!init(env, cls, cudnnConvolutionBwdFilterAlgoPerf_status, "status", "I")) return JNI_ERR;
    if (!init(env, cls, cudnnConvolutionBwdFilterAlgoPerf_time, "time", "F")) return JNI_ERR;
    if (!init(env, cls, cudnnConvolutionBwdFilterAlgoPerf_memory, "memory", "J")) return JNI_ERR;

    // Obtain the fieldIDs for cudnnConvolutionBwdDataAlgoPerf
    if (!init(env, cls, "jcuda/jcudnn/cudnnConvolutionBwdDataAlgoPerf")) return JNI_ERR;
    if (!init(env, cls, cudnnConvolutionBwdDataAlgoPerf_algo, "algo", "I")) return JNI_ERR;
    if (!init(env, cls, cudnnConvolutionBwdDataAlgoPerf_status, "status", "I")) return JNI_ERR;
    if (!init(env, cls, cudnnConvolutionBwdDataAlgoPerf_time, "time", "F")) return JNI_ERR;
    if (!init(env, cls, cudnnConvolutionBwdDataAlgoPerf_memory, "memory", "J")) return JNI_ERR;


    return JNI_VERSION_1_4;
}


/**
 * Create an int array with the same size as the given java array,
 * and store it in the given pointer. The caller must delete[] the
 * created array. The fill-flag indicates whether the array should
 * be initialized with the data from the given array
 */
bool initNative(JNIEnv* env, jintArray array, int* &a, bool fill)
{
    a = NULL;
    if (array == NULL)
    {
        return true;
    }
    jsize arrayLength = env->GetArrayLength(array);
    a = new int[arrayLength];
    if (a == NULL)
    {
        ThrowByName(env, "java/lang/OutOfMemoryError", "Not enough memory for array");
        return false;
    }
    if (fill)
    {
        jint *ja = (jint*)env->GetPrimitiveArrayCritical(array, NULL);
        if (ja == NULL)
        {
            ThrowByName(env, "java/lang/OutOfMemoryError", "Not enough memory for array");
            return false;
        }
        for (jsize i = 0; i < arrayLength; ++i)
        {
            a[i] = ja[i];
        }
        env->ReleasePrimitiveArrayCritical(array, ja, JNI_ABORT);
    }
    return true;
}

/**
 * Release the given array by deleting it and setting the pointer to NULL.
 * The writeBack flag indicates whether the data from the given array 
 * should be written into the given java array
 */
bool releaseNative(JNIEnv* env, int* &a, jintArray array, bool writeBack)
{
    if (array == NULL)
    {
        delete[] a;
        a = NULL;
        return true;
    }
    jsize arrayLength = env->GetArrayLength(array);
    if (writeBack)
    {
        jint *ja = (jint*)env->GetPrimitiveArrayCritical(array, NULL);
        if (ja == NULL)
        {
            ThrowByName(env, "java/lang/OutOfMemoryError", "Not enough memory for array");
            return false;
        }
        for (jsize i = 0; i < arrayLength; ++i)
        {
            ja[i] = a[i];
        }
        env->ReleasePrimitiveArrayCritical(array, ja, JNI_COMMIT);
    }
    delete[] a;
    a = NULL;
    return true;
}


/**
 * Initialize the given native output array with the given size.
 * The input array will only be checked to have a size that is
 * at least as large as the given size, but not be used otherwise
 */
 bool initNative(JNIEnv *env, jobjectArray input, cudnnConvolutionFwdAlgoPerf_t* &output, jint size)
{
    jsize arraySize = env->GetArrayLength(input);
    if (arraySize < size)
    {
        ThrowByName(env, "java/lang/ArrayIndexOutOfBoundsException",
            "Array parameter has insufficient size");
        return false;
    }
    output = new cudnnConvolutionFwdAlgoPerf_t[arraySize];
    return true;
}

/**
 * Write the data from the given object to the given java object
 */
bool releaseNative(JNIEnv *env, cudnnConvolutionFwdAlgoPerf_t input, jobject output)
{
    env->SetIntField(output, cudnnConvolutionFwdAlgoPerf_algo, (jint)input.algo);
    env->SetIntField(output, cudnnConvolutionFwdAlgoPerf_status, (jint)input.status);
    env->SetFloatField(output, cudnnConvolutionFwdAlgoPerf_time, input.time);
    env->SetLongField(output, cudnnConvolutionFwdAlgoPerf_memory, (jlong)input.memory);
    return true;
}

/**
 * Release and delete the given input array, writing the values back
 * to the given java array, creating (up to 'size') objects if 
 * necessary
 */
bool releaseNative(JNIEnv *env, cudnnConvolutionFwdAlgoPerf_t* &input, jobjectArray output, int size)
{
    jsize arraySize = env->GetArrayLength(output);
    if (arraySize < size)
    {
        ThrowByName(env, "java/lang/ArrayIndexOutOfBoundsException",
            "Array parameter has insufficient size");
        return false;
    }
    for (jsize i = 0; i < size; i++)
    {
        jobject outputElement = env->GetObjectArrayElement(output, i);
        if (outputElement == NULL)
        {
            outputElement = env->NewObject(cudnnConvolutionFwdAlgoPerf_Class, cudnnConvolutionFwdAlgoPerf_Constructor);
            env->SetObjectArrayElement(output, i, outputElement);
        }
        if (!releaseNative(env, input[i], outputElement))
        {
            return false;
        }
    }
    return true;
}


/**
* Initialize the given native output array with the given size.
* The input array will only be checked to have a size that is
* at least as large as the given size, but not be used otherwise
*/
bool initNative(JNIEnv *env, jobjectArray input, cudnnConvolutionBwdFilterAlgoPerf_t* &output, jint size)
{
    jsize arraySize = env->GetArrayLength(input);
    if (arraySize < size)
    {
        ThrowByName(env, "java/lang/ArrayIndexOutOfBoundsException",
            "Array parameter has insufficient size");
        return false;
    }
    output = new cudnnConvolutionBwdFilterAlgoPerf_t[arraySize];
    return true;
}

/**
* Write the data from the given object to the given java object
*/
bool releaseNative(JNIEnv *env, cudnnConvolutionBwdFilterAlgoPerf_t input, jobject output)
{
    env->SetIntField(output, cudnnConvolutionBwdFilterAlgoPerf_algo, (jint)input.algo);
    env->SetIntField(output, cudnnConvolutionBwdFilterAlgoPerf_status, (jint)input.status);
    env->SetFloatField(output, cudnnConvolutionBwdFilterAlgoPerf_time, input.time);
    env->SetLongField(output, cudnnConvolutionBwdFilterAlgoPerf_memory, (jlong)input.algo);
    return true;
}

/**
* Release and delete the given input array, writing the values back
* to the given java array, creating (up to 'size') objects if
* necessary
*/
bool releaseNative(JNIEnv *env, cudnnConvolutionBwdFilterAlgoPerf_t* &input, jobjectArray output, int size)
{
    jsize arraySize = env->GetArrayLength(output);
    if (arraySize < size)
    {
        ThrowByName(env, "java/lang/ArrayIndexOutOfBoundsException",
            "Array parameter has insufficient size");
        return false;
    }
    for (jsize i = 0; i<size; i++)
    {
        jobject outputElement = env->GetObjectArrayElement(output, i);
        if (outputElement == NULL)
        {
            outputElement = env->NewObject(cudnnConvolutionBwdFilterAlgoPerf_Class, cudnnConvolutionBwdFilterAlgoPerf_Constructor);
            env->SetObjectArrayElement(output, i, outputElement);
        }
        if (!releaseNative(env, input[i], outputElement))
        {
            return false;
        }
    }
    return true;
}

/**
* Initialize the given native output array with the given size.
* The input array will only be checked to have a size that is
* at least as large as the given size, but not be used otherwise
*/
bool initNative(JNIEnv *env, jobjectArray input, cudnnConvolutionBwdDataAlgoPerf_t* &output, jint size)
{
    jsize arraySize = env->GetArrayLength(input);
    if (arraySize < size)
    {
        ThrowByName(env, "java/lang/ArrayIndexOutOfBoundsException",
            "Array parameter has insufficient size");
        return false;
    }
    output = new cudnnConvolutionBwdDataAlgoPerf_t[arraySize];
    return true;
}

/**
* Write the data from the given object to the given java object
*/
bool releaseNative(JNIEnv *env, cudnnConvolutionBwdDataAlgoPerf_t input, jobject output)
{
    env->SetIntField(output, cudnnConvolutionBwdDataAlgoPerf_algo, (jint)input.algo);
    env->SetIntField(output, cudnnConvolutionBwdDataAlgoPerf_status, (jint)input.status);
    env->SetFloatField(output, cudnnConvolutionBwdDataAlgoPerf_time, input.time);
    env->SetLongField(output, cudnnConvolutionBwdDataAlgoPerf_memory, (jlong)input.algo);
    return true;
}

/**
* Release and delete the given input array, writing the values back
* to the given java array, creating (up to 'size') objects if
* necessary
*/
bool releaseNative(JNIEnv *env, cudnnConvolutionBwdDataAlgoPerf_t* &input, jobjectArray output, int size)
{
    jsize arraySize = env->GetArrayLength(output);
    if (arraySize < size)
    {
        ThrowByName(env, "java/lang/ArrayIndexOutOfBoundsException",
            "Array parameter has insufficient size");
        return false;
    }
    for (jsize i = 0; i<size; i++)
    {
        jobject outputElement = env->GetObjectArrayElement(output, i);
        if (outputElement == NULL)
        {
            outputElement = env->NewObject(cudnnConvolutionBwdDataAlgoPerf_Class, cudnnConvolutionBwdDataAlgoPerf_Constructor);
            env->SetObjectArrayElement(output, i, outputElement);
        }
        if (!releaseNative(env, input[i], outputElement))
        {
            return false;
        }
    }
    return true;
}



JNIEXPORT jlong JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetVersionNative(JNIEnv *env, jclass cls)
{
    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetVersion()\n");

    // Native function call
    size_t jniResult_native = cudnnGetVersion();

    // Return the result
    jlong jniResult;
    jniResult = (jlong)jniResult_native;
    return jniResult;
}

// human-readable error messages
JNIEXPORT jstring JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetErrorStringNative(JNIEnv *env, jclass cls, jint status)
{
    // Null-checks for non-primitive arguments
    // status is primitive

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetErrorString(status=%d)\n",
        status);

    // Native variable declarations
    cudnnStatus_t status_native = (cudnnStatus_t)JCUDNN_STATUS_INTERNAL_ERROR;

    // Obtain native variable values
    status_native = (cudnnStatus_t)status;

    // Native function call
    const char* jniResult_native = cudnnGetErrorString(status_native);

    // Write back native variable values
    // status is primitive

    // Return the result
    return env->NewStringUTF(jniResult_native);
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnCreateNative(JNIEnv *env, jclass cls, jobject handle)
{
    // Null-checks for non-primitive arguments
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cudnnCreate");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnCreate(handle=%p)\n",
        handle);

    // Native variable declarations
    cudnnHandle_t handle_native;

    // Obtain native variable values
    // handle is write-only

    // Native function call
    cudnnStatus_t jniResult_native = cudnnCreate(&handle_native);

    // Write back native variable values
    setNativePointerValue(env, handle, (jlong)handle_native);

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnDestroyNative(JNIEnv *env, jclass cls, jobject handle)
{
    // Null-checks for non-primitive arguments
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cudnnDestroy");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnDestroy(handle=%p)\n",
        handle);

    // Native variable declarations
    cudnnHandle_t handle_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnDestroy(handle_native);

    // Write back native variable values
    // handle is a read-only native pointer

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnSetStreamNative(JNIEnv *env, jclass cls, jobject handle, jobject streamId)
{
    // Null-checks for non-primitive arguments
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cudnnSetStream");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    // streamId may be NULL

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnSetStream(handle=%p, streamId=%p)\n",
        handle, streamId);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudaStream_t streamId_native = NULL;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    if (streamId != NULL)
    {
        streamId_native = (cudaStream_t)getNativePointerValue(env, streamId);
    }

    // Native function call
    cudnnStatus_t jniResult_native = cudnnSetStream(handle_native, streamId_native);

    // Write back native variable values
    // handle is a read-only native pointer
    // streamId is a read-only native pointer

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetStreamNative(JNIEnv *env, jclass cls, jobject handle, jobject streamId)
{
    // Null-checks for non-primitive arguments
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cudnnGetStream");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (streamId == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'streamId' is null for cudnnGetStream");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetStream(handle=%p, streamId=%p)\n",
        handle, streamId);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudaStream_t* streamId_native = NULL;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    streamId_native = (cudaStream_t*)getPointer(env, streamId);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetStream(handle_native, streamId_native);

    // Write back native variable values
    // handle is a read-only native pointer
    // streamId is a native pointer

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

/** Create an instance of a generic Tensor descriptor */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnCreateTensorDescriptorNative(JNIEnv *env, jclass cls, jobject tensorDesc)
{
    // Null-checks for non-primitive arguments
    if (tensorDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'tensorDesc' is null for cudnnCreateTensorDescriptor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnCreateTensorDescriptor(tensorDesc=%p)\n",
        tensorDesc);

    // Native variable declarations
    cudnnTensorDescriptor_t tensorDesc_native;

    // Obtain native variable values
    // tensorDesc is write-only

    // Native function call
    cudnnStatus_t jniResult_native = cudnnCreateTensorDescriptor(&tensorDesc_native);

    // Write back native variable values
    setNativePointerValue(env, tensorDesc, (jlong)tensorDesc_native);

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnSetTensor4dDescriptorNative(JNIEnv *env, jclass cls, jobject tensorDesc, jint format, jint dataType, jint n, jint c, jint h, jint w)
{
    // Null-checks for non-primitive arguments
    if (tensorDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'tensorDesc' is null for cudnnSetTensor4dDescriptor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    // format is primitive
    // dataType is primitive
    // n is primitive
    // c is primitive
    // h is primitive
    // w is primitive

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnSetTensor4dDescriptor(tensorDesc=%p, format=%d, dataType=%d, n=%d, c=%d, h=%d, w=%d)\n",
        tensorDesc, format, dataType, n, c, h, w);

    // Native variable declarations
    cudnnTensorDescriptor_t tensorDesc_native;
    cudnnTensorFormat_t format_native;
    cudnnDataType_t dataType_native;
    int n_native = 0;
    int c_native = 0;
    int h_native = 0;
    int w_native = 0;

    // Obtain native variable values
    tensorDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, tensorDesc);
    format_native = (cudnnTensorFormat_t)format;
    dataType_native = (cudnnDataType_t)dataType;
    n_native = (int)n;
    c_native = (int)c;
    h_native = (int)h;
    w_native = (int)w;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnSetTensor4dDescriptor(tensorDesc_native, format_native, dataType_native, n_native, c_native, h_native, w_native);

    // Write back native variable values
    // tensorDesc is a read-only native pointer
    // format is primitive
    // dataType is primitive
    // n is primitive
    // c is primitive
    // h is primitive
    // w is primitive

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnSetTensor4dDescriptorExNative(JNIEnv *env, jclass cls, jobject tensorDesc, jint dataType, jint n, jint c, jint h, jint w, jint nStride, jint cStride, jint hStride, jint wStride)
{
    // Null-checks for non-primitive arguments
    if (tensorDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'tensorDesc' is null for cudnnSetTensor4dDescriptorEx");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    // dataType is primitive
    // n is primitive
    // c is primitive
    // h is primitive
    // w is primitive
    // nStride is primitive
    // cStride is primitive
    // hStride is primitive
    // wStride is primitive

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnSetTensor4dDescriptorEx(tensorDesc=%p, dataType=%d, n=%d, c=%d, h=%d, w=%d, nStride=%d, cStride=%d, hStride=%d, wStride=%d)\n",
        tensorDesc, dataType, n, c, h, w, nStride, cStride, hStride, wStride);

    // Native variable declarations
    cudnnTensorDescriptor_t tensorDesc_native;
    cudnnDataType_t dataType_native;
    int n_native = 0;
    int c_native = 0;
    int h_native = 0;
    int w_native = 0;
    int nStride_native = 0;
    int cStride_native = 0;
    int hStride_native = 0;
    int wStride_native = 0;

    // Obtain native variable values
    tensorDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, tensorDesc);
    dataType_native = (cudnnDataType_t)dataType;
    n_native = (int)n;
    c_native = (int)c;
    h_native = (int)h;
    w_native = (int)w;
    nStride_native = (int)nStride;
    cStride_native = (int)cStride;
    hStride_native = (int)hStride;
    wStride_native = (int)wStride;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnSetTensor4dDescriptorEx(tensorDesc_native, dataType_native, n_native, c_native, h_native, w_native, nStride_native, cStride_native, hStride_native, wStride_native);

    // Write back native variable values
    // tensorDesc is a read-only native pointer
    // dataType is primitive
    // n is primitive
    // c is primitive
    // h is primitive
    // w is primitive
    // nStride is primitive
    // cStride is primitive
    // hStride is primitive
    // wStride is primitive

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetTensor4dDescriptorNative(JNIEnv *env, jclass cls, jobject tensorDesc, jintArray dataType, jobject n, jobject c, jobject h, jobject w, jobject nStride, jobject cStride, jobject hStride, jobject wStride)
{
    // Null-checks for non-primitive arguments
    if (tensorDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'tensorDesc' is null for cudnnGetTensor4dDescriptor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (dataType == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dataType' is null for cudnnGetTensor4dDescriptor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (n == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'n' is null for cudnnGetTensor4dDescriptor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (c == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'c' is null for cudnnGetTensor4dDescriptor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (h == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'h' is null for cudnnGetTensor4dDescriptor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (w == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'w' is null for cudnnGetTensor4dDescriptor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (nStride == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'nStride' is null for cudnnGetTensor4dDescriptor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (cStride == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'cStride' is null for cudnnGetTensor4dDescriptor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (hStride == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hStride' is null for cudnnGetTensor4dDescriptor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (wStride == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'wStride' is null for cudnnGetTensor4dDescriptor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetTensor4dDescriptor(tensorDesc=%p, dataType=%p, n=%p, c=%p, h=%p, w=%p, nStride=%p, cStride=%p, hStride=%p, wStride=%p)\n",
        tensorDesc, dataType, n, c, h, w, nStride, cStride, hStride, wStride);

    // Native variable declarations
    cudnnTensorDescriptor_t tensorDesc_native;
    cudnnDataType_t* dataType_native;
    int* n_native;
    int* c_native;
    int* h_native;
    int* w_native;
    int* nStride_native;
    int* cStride_native;
    int* hStride_native;
    int* wStride_native;

    // Obtain native variable values
    tensorDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, tensorDesc);
    dataType_native = (cudnnDataType_t*)getPointer(env, dataType);
    n_native = (int*)getPointer(env, n);
    c_native = (int*)getPointer(env, c);
    h_native = (int*)getPointer(env, h);
    w_native = (int*)getPointer(env, w);
    nStride_native = (int*)getPointer(env, nStride);
    cStride_native = (int*)getPointer(env, cStride);
    hStride_native = (int*)getPointer(env, hStride);
    wStride_native = (int*)getPointer(env, wStride);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetTensor4dDescriptor(tensorDesc_native, dataType_native, n_native, c_native, h_native, w_native, nStride_native, cStride_native, hStride_native, wStride_native);

    // Write back native variable values
    // tensorDesc is a read-only native pointer
    // dataType is a native pointer
    // n is a native pointer
    // c is a native pointer
    // h is a native pointer
    // w is a native pointer
    // nStride is a native pointer
    // cStride is a native pointer
    // hStride is a native pointer
    // wStride is a native pointer

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnSetTensorNdDescriptorNative(JNIEnv *env, jclass cls, jobject tensorDesc, jint dataType, jint nbDims, jintArray dimA, jintArray strideA)
{
    // Null-checks for non-primitive arguments
    if (tensorDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'tensorDesc' is null for cudnnSetTensorNdDescriptor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    // dataType is primitive
    // nbDims is primitive
    if (dimA == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dimA' is null for cudnnSetTensorNdDescriptor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (strideA == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'strideA' is null for cudnnSetTensorNdDescriptor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnSetTensorNdDescriptor(tensorDesc=%p, dataType=%d, nbDims=%d, dimA=%p, strideA=%p)\n",
        tensorDesc, dataType, nbDims, dimA, strideA);

    // Native variable declarations
    cudnnTensorDescriptor_t tensorDesc_native;
    cudnnDataType_t dataType_native;
    int nbDims_native = 0;
    int* dimA_native = NULL;
    int* strideA_native = NULL;

    // Obtain native variable values
    tensorDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, tensorDesc);
    dataType_native = (cudnnDataType_t)dataType;
    nbDims_native = (int)nbDims;
    if (!initNative(env, dimA, dimA_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!initNative(env, strideA, strideA_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnSetTensorNdDescriptor(tensorDesc_native, dataType_native, nbDims_native, dimA_native, strideA_native);

    // Write back native variable values
    // tensorDesc is a read-only native pointer
    // dataType is primitive
    // nbDims is primitive
    if (!releaseNative(env, dimA_native, dimA, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!releaseNative(env, strideA_native, strideA, true)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetTensorNdDescriptorNative(JNIEnv *env, jclass cls, jobject tensorDesc, jint nbDimsRequested, jintArray dataType, jintArray nbDims, jintArray dimA, jintArray strideA)
{
    // Null-checks for non-primitive arguments
    if (tensorDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'tensorDesc' is null for cudnnGetTensorNdDescriptor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    // nbDimsRequested is primitive
    if (dataType == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dataType' is null for cudnnGetTensorNdDescriptor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (nbDims == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'nbDims' is null for cudnnGetTensorNdDescriptor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (dimA == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dimA' is null for cudnnGetTensorNdDescriptor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (strideA == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'strideA' is null for cudnnGetTensorNdDescriptor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetTensorNdDescriptor(tensorDesc=%p, nbDimsRequested=%d, dataType=%p, nbDims=%p, dimA=%p, strideA=%p)\n",
        tensorDesc, nbDimsRequested, dataType, nbDims, dimA, strideA);

    // Native variable declarations
    cudnnTensorDescriptor_t tensorDesc_native;
    int nbDimsRequested_native = 0;
    cudnnDataType_t* dataType_native;
    int nbDims_native = 0;
    int* dimA_native = NULL;
    int* strideA_native = NULL;

    // Obtain native variable values
    tensorDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, tensorDesc);
    nbDimsRequested_native = (int)nbDimsRequested;
    dataType_native = (cudnnDataType_t*)getPointer(env, dataType);
    // nbDims is set here
    if (!initNative(env, dimA, dimA_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!initNative(env, strideA, strideA_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetTensorNdDescriptor(tensorDesc_native, nbDimsRequested_native, dataType_native, &nbDims_native, dimA_native, strideA_native);

    // Write back native variable values
    // tensorDesc is a read-only native pointer
    // nbDimsRequested is primitive
    // dataType is a native pointer
    if (!set(env, nbDims, 0, (jint)nbDims_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!releaseNative(env, dimA_native, dimA, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!releaseNative(env, strideA_native, strideA, true)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

/**
* <pre>
* PixelOffset( n, c, h, w ) = n *input_stride + c * feature_stride + h * h_stride + w * w_stride

1)Example of all images in row major order one batch of features after the other (with an optional padding on row)
input_stride :  c x h x h_stride
feature_stride : h x h_stride
h_stride  :  >= w  ( h_stride = w if no padding)
w_stride  : 1


2)Example of all images in row major with features maps interleaved
input_stride :  c x h x h_stride
feature_stride : 1
h_stride  :  w x c
w_stride  : c

3)Example of all images in column major order one batch of features after the other (with optional padding on column)
input_stride :  c x w x w_stride
feature_stride : w x w_stride
h_stride  :  1
w_stride  :  >= h

* </pre>
*/
/** Destroy an instance of Tensor4d descriptor */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnDestroyTensorDescriptorNative(JNIEnv *env, jclass cls, jobject tensorDesc)
{
    // Null-checks for non-primitive arguments
    if (tensorDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'tensorDesc' is null for cudnnDestroyTensorDescriptor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnDestroyTensorDescriptor(tensorDesc=%p)\n",
        tensorDesc);

    // Native variable declarations
    cudnnTensorDescriptor_t tensorDesc_native;

    // Obtain native variable values
    tensorDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, tensorDesc);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnDestroyTensorDescriptor(tensorDesc_native);

    // Write back native variable values
    // tensorDesc is a read-only native pointer

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

/** Tensor layout conversion helper (y = alpha * x + beta * y) */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnTransformTensorNative(JNIEnv *env, jclass cls, jobject handle, jobject alpha, jobject xDesc, jobject x, jobject beta, jobject yDesc, jobject y)
{
    // Null-checks for non-primitive arguments
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cudnnTransformTensor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (alpha == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'alpha' is null for cudnnTransformTensor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (xDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'xDesc' is null for cudnnTransformTensor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cudnnTransformTensor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (beta == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'beta' is null for cudnnTransformTensor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (yDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'yDesc' is null for cudnnTransformTensor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (y == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'y' is null for cudnnTransformTensor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnTransformTensor(handle=%p, alpha=%p, xDesc=%p, x=%p, beta=%p, yDesc=%p, y=%p)\n",
        handle, alpha, xDesc, x, beta, yDesc, y);

    // Native variable declarations
    cudnnHandle_t handle_native;
    void* alpha_native;
    cudnnTensorDescriptor_t xDesc_native;
    void* x_native;
    void* beta_native;
    cudnnTensorDescriptor_t yDesc_native;
    void* y_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    PointerData *alpha_pointerData = initPointerData(env, alpha);
    if (alpha_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    alpha_native = (void*)alpha_pointerData->getPointer(env);
    xDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, xDesc);
    x_native = (void*)getPointer(env, x);
    PointerData *beta_pointerData = initPointerData(env, beta);
    if (beta_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    beta_native = (void*)beta_pointerData->getPointer(env);
    yDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, yDesc);
    y_native = (void*)getPointer(env, y);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnTransformTensor(handle_native, alpha_native, xDesc_native, x_native, beta_native, yDesc_native, y_native);

    // Write back native variable values
    // handle is a read-only native pointer
    if (!releasePointerData(env, alpha_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // xDesc is a read-only native pointer
    // x is a native pointer
    if (!releasePointerData(env, beta_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // yDesc is a read-only native pointer
    // y is a native pointer

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

/** Tensor Bias addition : y = alpha * b + beta * y  */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnAddTensorNative(JNIEnv *env, jclass cls, jobject handle, jobject alpha, jobject bDesc, jobject b, jobject beta, jobject yDesc, jobject y)
{
    // Null-checks for non-primitive arguments
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cudnnAddTensor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (alpha == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'alpha' is null for cudnnAddTensor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (bDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'bDesc' is null for cudnnAddTensor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (b == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'b' is null for cudnnAddTensor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (beta == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'beta' is null for cudnnAddTensor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (yDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'yDesc' is null for cudnnAddTensor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (y == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'y' is null for cudnnAddTensor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnAddTensor(handle=%p, alpha=%p, bDesc=%p, b=%p, beta=%p, yDesc=%p, y=%p)\n",
        handle, alpha, bDesc, b, beta, yDesc, y);

    // Native variable declarations
    cudnnHandle_t handle_native;
    void* alpha_native;
    cudnnTensorDescriptor_t bDesc_native;
    void* b_native;
    void* beta_native;
    cudnnTensorDescriptor_t yDesc_native;
    void* y_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    PointerData *alpha_pointerData = initPointerData(env, alpha);
    if (alpha_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    alpha_native = (void*)alpha_pointerData->getPointer(env);
    bDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, bDesc);
    b_native = (void*)getPointer(env, b);
    PointerData *beta_pointerData = initPointerData(env, beta);
    if (beta_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    beta_native = (void*)beta_pointerData->getPointer(env);
    yDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, yDesc);
    y_native = (void*)getPointer(env, y);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnAddTensor(handle_native, alpha_native, bDesc_native, b_native, beta_native, yDesc_native, y_native);

    // Write back native variable values
    // handle is a read-only native pointer
    if (!releasePointerData(env, alpha_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // bDesc is a read-only native pointer
    // b is a native pointer
    if (!releasePointerData(env, beta_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // yDesc is a read-only native pointer
    // y is a native pointer

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

/**
* <pre>
* cudnnAddTensor_v3 is now mapped to cudnnAddTensor
and will be removed at the same time as cudnnAddTensor_v2
Use cudnnAddTensor instead
* </pre>
*/
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnAddTensor_1v3Native(JNIEnv *env, jclass cls, jobject handle, jobject alpha, jobject bDesc, jobject b, jobject beta, jobject yDesc, jobject y)
{
    // Null-checks for non-primitive arguments
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cudnnAddTensor_v3");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (alpha == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'alpha' is null for cudnnAddTensor_v3");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (bDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'bDesc' is null for cudnnAddTensor_v3");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (b == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'b' is null for cudnnAddTensor_v3");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (beta == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'beta' is null for cudnnAddTensor_v3");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (yDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'yDesc' is null for cudnnAddTensor_v3");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (y == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'y' is null for cudnnAddTensor_v3");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnAddTensor_v3(handle=%p, alpha=%p, bDesc=%p, b=%p, beta=%p, yDesc=%p, y=%p)\n",
        handle, alpha, bDesc, b, beta, yDesc, y);

    // Native variable declarations
    cudnnHandle_t handle_native;
    void* alpha_native;
    cudnnTensorDescriptor_t bDesc_native;
    void* b_native;
    void* beta_native;
    cudnnTensorDescriptor_t yDesc_native;
    void* y_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    PointerData *alpha_pointerData = initPointerData(env, alpha);
    if (alpha_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    alpha_native = (void*)alpha_pointerData->getPointer(env);
    bDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, bDesc);
    b_native = (void*)getPointer(env, b);
    PointerData *beta_pointerData = initPointerData(env, beta);
    if (beta_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    beta_native = (void*)beta_pointerData->getPointer(env);
    yDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, yDesc);
    y_native = (void*)getPointer(env, y);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnAddTensor_v3(handle_native, alpha_native, bDesc_native, b_native, beta_native, yDesc_native, y_native);

    // Write back native variable values
    // handle is a read-only native pointer
    if (!releasePointerData(env, alpha_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // bDesc is a read-only native pointer
    // b is a native pointer
    if (!releasePointerData(env, beta_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // yDesc is a read-only native pointer
    // y is a native pointer

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

/** Set all values of a tensor to a given value : y[i] = value[0] */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnSetTensorNative(JNIEnv *env, jclass cls, jobject handle, jobject yDesc, jobject y, jobject valuePtr)
{
    // Null-checks for non-primitive arguments
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cudnnSetTensor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (yDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'yDesc' is null for cudnnSetTensor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (y == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'y' is null for cudnnSetTensor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (valuePtr == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'valuePtr' is null for cudnnSetTensor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnSetTensor(handle=%p, yDesc=%p, y=%p, valuePtr=%p)\n",
        handle, yDesc, y, valuePtr);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnTensorDescriptor_t yDesc_native;
    void* y_native;
    void* valuePtr_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    yDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, yDesc);
    y_native = (void*)getPointer(env, y);
    valuePtr_native = (void*)getPointer(env, valuePtr);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnSetTensor(handle_native, yDesc_native, y_native, valuePtr_native);

    // Write back native variable values
    // handle is a read-only native pointer
    // yDesc is a read-only native pointer
    // y is a native pointer
    // valuePtr is a native pointer

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

/** Scale all values of a tensor by a given factor : y[i] = alpha * y[i] */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnScaleTensorNative(JNIEnv *env, jclass cls, jobject handle, jobject yDesc, jobject y, jobject alpha)
{
    // Null-checks for non-primitive arguments
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cudnnScaleTensor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (yDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'yDesc' is null for cudnnScaleTensor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (y == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'y' is null for cudnnScaleTensor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (alpha == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'alpha' is null for cudnnScaleTensor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnScaleTensor(handle=%p, yDesc=%p, y=%p, alpha=%p)\n",
        handle, yDesc, y, alpha);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnTensorDescriptor_t yDesc_native;
    void* y_native;
    void* alpha_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    yDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, yDesc);
    y_native = (void*)getPointer(env, y);
    PointerData *alpha_pointerData = initPointerData(env, alpha);
    if (alpha_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    alpha_native = (void*)alpha_pointerData->getPointer(env);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnScaleTensor(handle_native, yDesc_native, y_native, alpha_native);

    // Write back native variable values
    // handle is a read-only native pointer
    // yDesc is a read-only native pointer
    // y is a native pointer
    if (!releasePointerData(env, alpha_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

/** Create an instance of FilterStruct */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnCreateFilterDescriptorNative(JNIEnv *env, jclass cls, jobject filterDesc)
{
    // Null-checks for non-primitive arguments
    if (filterDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'filterDesc' is null for cudnnCreateFilterDescriptor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnCreateFilterDescriptor(filterDesc=%p)\n",
        filterDesc);

    // Native variable declarations
    cudnnFilterDescriptor_t filterDesc_native;

    // Obtain native variable values
    // filterDesc is write-only

    // Native function call
    cudnnStatus_t jniResult_native = cudnnCreateFilterDescriptor(&filterDesc_native);

    // Write back native variable values
    setNativePointerValue(env, filterDesc, (jlong)filterDesc_native);

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnSetFilter4dDescriptorNative(JNIEnv *env, jclass cls, jobject filterDesc, jint dataType, jint k, jint c, jint h, jint w)
{
    // Null-checks for non-primitive arguments
    if (filterDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'filterDesc' is null for cudnnSetFilter4dDescriptor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    // dataType is primitive
    // k is primitive
    // c is primitive
    // h is primitive
    // w is primitive

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnSetFilter4dDescriptor(filterDesc=%p, dataType=%d, k=%d, c=%d, h=%d, w=%d)\n",
        filterDesc, dataType, k, c, h, w);

    // Native variable declarations
    cudnnFilterDescriptor_t filterDesc_native;
    cudnnDataType_t dataType_native;
    int k_native = 0;
    int c_native = 0;
    int h_native = 0;
    int w_native = 0;

    // Obtain native variable values
    filterDesc_native = (cudnnFilterDescriptor_t)getNativePointerValue(env, filterDesc);
    dataType_native = (cudnnDataType_t)dataType;
    k_native = (int)k;
    c_native = (int)c;
    h_native = (int)h;
    w_native = (int)w;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnSetFilter4dDescriptor(filterDesc_native, dataType_native, k_native, c_native, h_native, w_native);

    // Write back native variable values
    // filterDesc is a read-only native pointer
    // dataType is primitive
    // k is primitive
    // c is primitive
    // h is primitive
    // w is primitive

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnSetFilter4dDescriptor_1v4Native(JNIEnv *env, jclass cls, jobject filterDesc, jint dataType, jint format, jint k, jint c, jint h, jint w)
{
    // Null-checks for non-primitive arguments
    if (filterDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'filterDesc' is null for cudnnSetFilter4dDescriptor_v4");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    // dataType is primitive
    // format is primitive
    // k is primitive
    // c is primitive
    // h is primitive
    // w is primitive

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnSetFilter4dDescriptor_v4(filterDesc=%p, dataType=%d, format=%d, k=%d, c=%d, h=%d, w=%d)\n",
        filterDesc, dataType, format, k, c, h, w);

    // Native variable declarations
    cudnnFilterDescriptor_t filterDesc_native;
    cudnnDataType_t dataType_native;
    cudnnTensorFormat_t format_native;
    int k_native = 0;
    int c_native = 0;
    int h_native = 0;
    int w_native = 0;

    // Obtain native variable values
    filterDesc_native = (cudnnFilterDescriptor_t)getNativePointerValue(env, filterDesc);
    dataType_native = (cudnnDataType_t)dataType;
    format_native = (cudnnTensorFormat_t)format;
    k_native = (int)k;
    c_native = (int)c;
    h_native = (int)h;
    w_native = (int)w;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnSetFilter4dDescriptor_v4(filterDesc_native, dataType_native, format_native, k_native, c_native, h_native, w_native);

    // Write back native variable values
    // filterDesc is a read-only native pointer
    // dataType is primitive
    // format is primitive
    // k is primitive
    // c is primitive
    // h is primitive
    // w is primitive

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetFilter4dDescriptorNative(JNIEnv *env, jclass cls, jobject filterDesc, jintArray dataType, jobject k, jobject c, jobject h, jobject w)
{
    // Null-checks for non-primitive arguments
    if (filterDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'filterDesc' is null for cudnnGetFilter4dDescriptor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (dataType == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dataType' is null for cudnnGetFilter4dDescriptor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (k == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'k' is null for cudnnGetFilter4dDescriptor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (c == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'c' is null for cudnnGetFilter4dDescriptor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (h == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'h' is null for cudnnGetFilter4dDescriptor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (w == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'w' is null for cudnnGetFilter4dDescriptor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetFilter4dDescriptor(filterDesc=%p, dataType=%p, k=%p, c=%p, h=%p, w=%p)\n",
        filterDesc, dataType, k, c, h, w);

    // Native variable declarations
    cudnnFilterDescriptor_t filterDesc_native;
    cudnnDataType_t* dataType_native;
    int* k_native;
    int* c_native;
    int* h_native;
    int* w_native;

    // Obtain native variable values
    filterDesc_native = (cudnnFilterDescriptor_t)getNativePointerValue(env, filterDesc);
    dataType_native = (cudnnDataType_t*)getPointer(env, dataType);
    k_native = (int*)getPointer(env, k);
    c_native = (int*)getPointer(env, c);
    h_native = (int*)getPointer(env, h);
    w_native = (int*)getPointer(env, w);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetFilter4dDescriptor(filterDesc_native, dataType_native, k_native, c_native, h_native, w_native);

    // Write back native variable values
    // filterDesc is a read-only native pointer
    // dataType is a native pointer
    // k is a native pointer
    // c is a native pointer
    // h is a native pointer
    // w is a native pointer

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetFilter4dDescriptor_1v4Native(JNIEnv *env, jclass cls, jobject filterDesc, jintArray dataType, jintArray format, jobject k, jobject c, jobject h, jobject w)
{
    // Null-checks for non-primitive arguments
    if (filterDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'filterDesc' is null for cudnnGetFilter4dDescriptor_v4");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (dataType == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dataType' is null for cudnnGetFilter4dDescriptor_v4");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (format == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'format' is null for cudnnGetFilter4dDescriptor_v4");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (k == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'k' is null for cudnnGetFilter4dDescriptor_v4");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (c == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'c' is null for cudnnGetFilter4dDescriptor_v4");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (h == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'h' is null for cudnnGetFilter4dDescriptor_v4");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (w == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'w' is null for cudnnGetFilter4dDescriptor_v4");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetFilter4dDescriptor_v4(filterDesc=%p, dataType=%p, format=%p, k=%p, c=%p, h=%p, w=%p)\n",
        filterDesc, dataType, format, k, c, h, w);

    // Native variable declarations
    cudnnFilterDescriptor_t filterDesc_native;
    cudnnDataType_t* dataType_native;
    cudnnTensorFormat_t* format_native;
    int* k_native;
    int* c_native;
    int* h_native;
    int* w_native;

    // Obtain native variable values
    filterDesc_native = (cudnnFilterDescriptor_t)getNativePointerValue(env, filterDesc);
    dataType_native = (cudnnDataType_t*)getPointer(env, dataType);
    format_native = (cudnnTensorFormat_t*)getPointer(env, format);
    k_native = (int*)getPointer(env, k);
    c_native = (int*)getPointer(env, c);
    h_native = (int*)getPointer(env, h);
    w_native = (int*)getPointer(env, w);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetFilter4dDescriptor_v4(filterDesc_native, dataType_native, format_native, k_native, c_native, h_native, w_native);

    // Write back native variable values
    // filterDesc is a read-only native pointer
    // dataType is a native pointer
    // format is a native pointer
    // k is a native pointer
    // c is a native pointer
    // h is a native pointer
    // w is a native pointer

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnSetFilterNdDescriptorNative(JNIEnv *env, jclass cls, jobject filterDesc, jint dataType, jint nbDims, jintArray filterDimA)
{
    // Null-checks for non-primitive arguments
    if (filterDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'filterDesc' is null for cudnnSetFilterNdDescriptor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    // dataType is primitive
    // nbDims is primitive
    if (filterDimA == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'filterDimA' is null for cudnnSetFilterNdDescriptor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnSetFilterNdDescriptor(filterDesc=%p, dataType=%d, nbDims=%d, filterDimA=%p)\n",
        filterDesc, dataType, nbDims, filterDimA);

    // Native variable declarations
    cudnnFilterDescriptor_t filterDesc_native;
    cudnnDataType_t dataType_native;
    int nbDims_native = 0;
    int* filterDimA_native = NULL;

    // Obtain native variable values
    filterDesc_native = (cudnnFilterDescriptor_t)getNativePointerValue(env, filterDesc);
    dataType_native = (cudnnDataType_t)dataType;
    nbDims_native = (int)nbDims;
    if (!initNative(env, filterDimA, filterDimA_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnSetFilterNdDescriptor(filterDesc_native, dataType_native, nbDims_native, filterDimA_native);

    // Write back native variable values
    // filterDesc is a read-only native pointer
    // dataType is primitive
    // nbDims is primitive
    if (!releaseNative(env, filterDimA_native, filterDimA, true)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnSetFilterNdDescriptor_1v4Native(JNIEnv *env, jclass cls, jobject filterDesc, jint dataType, jint format, jint nbDims, jintArray filterDimA)
{
    // Null-checks for non-primitive arguments
    if (filterDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'filterDesc' is null for cudnnSetFilterNdDescriptor_v4");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    // dataType is primitive
    // format is primitive
    // nbDims is primitive
    if (filterDimA == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'filterDimA' is null for cudnnSetFilterNdDescriptor_v4");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnSetFilterNdDescriptor_v4(filterDesc=%p, dataType=%d, format=%d, nbDims=%d, filterDimA=%p)\n",
        filterDesc, dataType, format, nbDims, filterDimA);

    // Native variable declarations
    cudnnFilterDescriptor_t filterDesc_native;
    cudnnDataType_t dataType_native;
    cudnnTensorFormat_t format_native;
    int nbDims_native = 0;
    int* filterDimA_native = NULL;

    // Obtain native variable values
    filterDesc_native = (cudnnFilterDescriptor_t)getNativePointerValue(env, filterDesc);
    dataType_native = (cudnnDataType_t)dataType;
    format_native = (cudnnTensorFormat_t)format;
    nbDims_native = (int)nbDims;
    if (!initNative(env, filterDimA, filterDimA_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnSetFilterNdDescriptor_v4(filterDesc_native, dataType_native, format_native, nbDims_native, filterDimA_native);

    // Write back native variable values
    // filterDesc is a read-only native pointer
    // dataType is primitive
    // format is primitive
    // nbDims is primitive
    if (!releaseNative(env, filterDimA_native, filterDimA, true)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetFilterNdDescriptorNative(JNIEnv *env, jclass cls, jobject filterDesc, jint nbDimsRequested, jintArray dataType, jobject nbDims, jintArray filterDimA)
{
    // Null-checks for non-primitive arguments
    if (filterDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'filterDesc' is null for cudnnGetFilterNdDescriptor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    // nbDimsRequested is primitive
    if (dataType == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dataType' is null for cudnnGetFilterNdDescriptor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (nbDims == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'nbDims' is null for cudnnGetFilterNdDescriptor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (filterDimA == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'filterDimA' is null for cudnnGetFilterNdDescriptor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetFilterNdDescriptor(filterDesc=%p, nbDimsRequested=%d, dataType=%p, nbDims=%p, filterDimA=%p)\n",
        filterDesc, nbDimsRequested, dataType, nbDims, filterDimA);

    // Native variable declarations
    cudnnFilterDescriptor_t filterDesc_native;
    int nbDimsRequested_native = 0;
    cudnnDataType_t* dataType_native;
    int* nbDims_native;
    int* filterDimA_native = NULL;

    // Obtain native variable values
    filterDesc_native = (cudnnFilterDescriptor_t)getNativePointerValue(env, filterDesc);
    nbDimsRequested_native = (int)nbDimsRequested;
    dataType_native = (cudnnDataType_t*)getPointer(env, dataType);
    nbDims_native = (int*)getPointer(env, nbDims);
    if (!initNative(env, filterDimA, filterDimA_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetFilterNdDescriptor(filterDesc_native, nbDimsRequested_native, dataType_native, nbDims_native, filterDimA_native);

    // Write back native variable values
    // filterDesc is a read-only native pointer
    // nbDimsRequested is primitive
    // dataType is a native pointer
    // nbDims is a native pointer
    if (!releaseNative(env, filterDimA_native, filterDimA, true)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetFilterNdDescriptor_1v4Native(JNIEnv *env, jclass cls, jobject filterDesc, jint nbDimsRequested, jintArray dataType, jintArray format, jobject nbDims, jintArray filterDimA)
{
    // Null-checks for non-primitive arguments
    if (filterDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'filterDesc' is null for cudnnGetFilterNdDescriptor_v4");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    // nbDimsRequested is primitive
    if (dataType == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dataType' is null for cudnnGetFilterNdDescriptor_v4");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (format == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'format' is null for cudnnGetFilterNdDescriptor_v4");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (nbDims == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'nbDims' is null for cudnnGetFilterNdDescriptor_v4");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (filterDimA == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'filterDimA' is null for cudnnGetFilterNdDescriptor_v4");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetFilterNdDescriptor_v4(filterDesc=%p, nbDimsRequested=%d, dataType=%p, format=%p, nbDims=%p, filterDimA=%p)\n",
        filterDesc, nbDimsRequested, dataType, format, nbDims, filterDimA);

    // Native variable declarations
    cudnnFilterDescriptor_t filterDesc_native;
    int nbDimsRequested_native = 0;
    cudnnDataType_t* dataType_native;
    cudnnTensorFormat_t* format_native;
    int* nbDims_native;
    int* filterDimA_native = NULL;

    // Obtain native variable values
    filterDesc_native = (cudnnFilterDescriptor_t)getNativePointerValue(env, filterDesc);
    nbDimsRequested_native = (int)nbDimsRequested;
    dataType_native = (cudnnDataType_t*)getPointer(env, dataType);
    format_native = (cudnnTensorFormat_t*)getPointer(env, format);
    nbDims_native = (int*)getPointer(env, nbDims);
    if (!initNative(env, filterDimA, filterDimA_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetFilterNdDescriptor_v4(filterDesc_native, nbDimsRequested_native, dataType_native, format_native, nbDims_native, filterDimA_native);

    // Write back native variable values
    // filterDesc is a read-only native pointer
    // nbDimsRequested is primitive
    // dataType is a native pointer
    // format is a native pointer
    // nbDims is a native pointer
    if (!releaseNative(env, filterDimA_native, filterDimA, true)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnDestroyFilterDescriptorNative(JNIEnv *env, jclass cls, jobject filterDesc)
{
    // Null-checks for non-primitive arguments
    if (filterDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'filterDesc' is null for cudnnDestroyFilterDescriptor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnDestroyFilterDescriptor(filterDesc=%p)\n",
        filterDesc);

    // Native variable declarations
    cudnnFilterDescriptor_t filterDesc_native;

    // Obtain native variable values
    filterDesc_native = (cudnnFilterDescriptor_t)getNativePointerValue(env, filterDesc);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnDestroyFilterDescriptor(filterDesc_native);

    // Write back native variable values
    // filterDesc is a read-only native pointer

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

/** Create an instance of convolution descriptor */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnCreateConvolutionDescriptorNative(JNIEnv *env, jclass cls, jobject convDesc)
{
    // Null-checks for non-primitive arguments
    if (convDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'convDesc' is null for cudnnCreateConvolutionDescriptor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnCreateConvolutionDescriptor(convDesc=%p)\n",
        convDesc);

    // Native variable declarations
    cudnnConvolutionDescriptor_t convDesc_native;

    // Obtain native variable values
    // convDesc is write-only

    // Native function call
    cudnnStatus_t jniResult_native = cudnnCreateConvolutionDescriptor(&convDesc_native);

    // Write back native variable values
    setNativePointerValue(env, convDesc, (jlong)convDesc_native);

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnSetConvolution2dDescriptorNative(JNIEnv *env, jclass cls, jobject convDesc, jint pad_h, jint pad_w, jint u, jint v, jint upscalex, jint upscaley, jint mode)
{
    // Null-checks for non-primitive arguments
    if (convDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'convDesc' is null for cudnnSetConvolution2dDescriptor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    // pad_h is primitive
    // pad_w is primitive
    // u is primitive
    // v is primitive
    // upscalex is primitive
    // upscaley is primitive
    // mode is primitive

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnSetConvolution2dDescriptor(convDesc=%p, pad_h=%d, pad_w=%d, u=%d, v=%d, upscalex=%d, upscaley=%d, mode=%d)\n",
        convDesc, pad_h, pad_w, u, v, upscalex, upscaley, mode);

    // Native variable declarations
    cudnnConvolutionDescriptor_t convDesc_native;
    int pad_h_native = 0;
    int pad_w_native = 0;
    int u_native = 0;
    int v_native = 0;
    int upscalex_native = 0;
    int upscaley_native = 0;
    cudnnConvolutionMode_t mode_native;

    // Obtain native variable values
    convDesc_native = (cudnnConvolutionDescriptor_t)getNativePointerValue(env, convDesc);
    pad_h_native = (int)pad_h;
    pad_w_native = (int)pad_w;
    u_native = (int)u;
    v_native = (int)v;
    upscalex_native = (int)upscalex;
    upscaley_native = (int)upscaley;
    mode_native = (cudnnConvolutionMode_t)mode;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnSetConvolution2dDescriptor(convDesc_native, pad_h_native, pad_w_native, u_native, v_native, upscalex_native, upscaley_native, mode_native);

    // Write back native variable values
    // convDesc is a read-only native pointer
    // pad_h is primitive
    // pad_w is primitive
    // u is primitive
    // v is primitive
    // upscalex is primitive
    // upscaley is primitive
    // mode is primitive

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetConvolution2dDescriptorNative(JNIEnv *env, jclass cls, jobject convDesc, jobject pad_h, jobject pad_w, jobject u, jobject v, jobject upscalex, jobject upscaley, jintArray mode)
{
    // Null-checks for non-primitive arguments
    if (convDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'convDesc' is null for cudnnGetConvolution2dDescriptor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (pad_h == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pad_h' is null for cudnnGetConvolution2dDescriptor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (pad_w == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pad_w' is null for cudnnGetConvolution2dDescriptor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (u == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'u' is null for cudnnGetConvolution2dDescriptor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (v == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'v' is null for cudnnGetConvolution2dDescriptor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (upscalex == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'upscalex' is null for cudnnGetConvolution2dDescriptor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (upscaley == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'upscaley' is null for cudnnGetConvolution2dDescriptor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (mode == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'mode' is null for cudnnGetConvolution2dDescriptor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetConvolution2dDescriptor(convDesc=%p, pad_h=%p, pad_w=%p, u=%p, v=%p, upscalex=%p, upscaley=%p, mode=%p)\n",
        convDesc, pad_h, pad_w, u, v, upscalex, upscaley, mode);

    // Native variable declarations
    cudnnConvolutionDescriptor_t convDesc_native;
    int* pad_h_native;
    int* pad_w_native;
    int* u_native;
    int* v_native;
    int* upscalex_native;
    int* upscaley_native;
    cudnnConvolutionMode_t* mode_native;

    // Obtain native variable values
    convDesc_native = (cudnnConvolutionDescriptor_t)getNativePointerValue(env, convDesc);
    pad_h_native = (int*)getPointer(env, pad_h);
    pad_w_native = (int*)getPointer(env, pad_w);
    u_native = (int*)getPointer(env, u);
    v_native = (int*)getPointer(env, v);
    upscalex_native = (int*)getPointer(env, upscalex);
    upscaley_native = (int*)getPointer(env, upscaley);
    mode_native = (cudnnConvolutionMode_t*)getPointer(env, mode);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetConvolution2dDescriptor(convDesc_native, pad_h_native, pad_w_native, u_native, v_native, upscalex_native, upscaley_native, mode_native);

    // Write back native variable values
    // convDesc is a read-only native pointer
    // pad_h is a native pointer
    // pad_w is a native pointer
    // u is a native pointer
    // v is a native pointer
    // upscalex is a native pointer
    // upscaley is a native pointer
    // mode is a native pointer

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

/** Helper function to return the dimensions of the output tensor given a convolution descriptor */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetConvolution2dForwardOutputDimNative(JNIEnv *env, jclass cls, jobject convDesc, jobject inputTensorDesc, jobject filterDesc, jobject n, jobject c, jobject h, jobject w)
{
    // Null-checks for non-primitive arguments
    if (convDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'convDesc' is null for cudnnGetConvolution2dForwardOutputDim");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (inputTensorDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'inputTensorDesc' is null for cudnnGetConvolution2dForwardOutputDim");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (filterDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'filterDesc' is null for cudnnGetConvolution2dForwardOutputDim");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (n == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'n' is null for cudnnGetConvolution2dForwardOutputDim");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (c == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'c' is null for cudnnGetConvolution2dForwardOutputDim");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (h == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'h' is null for cudnnGetConvolution2dForwardOutputDim");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (w == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'w' is null for cudnnGetConvolution2dForwardOutputDim");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetConvolution2dForwardOutputDim(convDesc=%p, inputTensorDesc=%p, filterDesc=%p, n=%p, c=%p, h=%p, w=%p)\n",
        convDesc, inputTensorDesc, filterDesc, n, c, h, w);

    // Native variable declarations
    cudnnConvolutionDescriptor_t convDesc_native;
    cudnnTensorDescriptor_t inputTensorDesc_native;
    cudnnFilterDescriptor_t filterDesc_native;
    int* n_native;
    int* c_native;
    int* h_native;
    int* w_native;

    // Obtain native variable values
    convDesc_native = (cudnnConvolutionDescriptor_t)getNativePointerValue(env, convDesc);
    inputTensorDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, inputTensorDesc);
    filterDesc_native = (cudnnFilterDescriptor_t)getNativePointerValue(env, filterDesc);
    n_native = (int*)getPointer(env, n);
    c_native = (int*)getPointer(env, c);
    h_native = (int*)getPointer(env, h);
    w_native = (int*)getPointer(env, w);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetConvolution2dForwardOutputDim(convDesc_native, inputTensorDesc_native, filterDesc_native, n_native, c_native, h_native, w_native);

    // Write back native variable values
    // convDesc is a read-only native pointer
    // inputTensorDesc is a read-only native pointer
    // filterDesc is a read-only native pointer
    // n is a native pointer
    // c is a native pointer
    // h is a native pointer
    // w is a native pointer

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnSetConvolutionNdDescriptorNative(JNIEnv *env, jclass cls, jobject convDesc, jint arrayLength, jintArray padA, jintArray filterStrideA, jintArray upscaleA, jint mode, jint dataType)
{
    // Null-checks for non-primitive arguments
    if (convDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'convDesc' is null for cudnnSetConvolutionNdDescriptor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    // arrayLength is primitive
    if (padA == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'padA' is null for cudnnSetConvolutionNdDescriptor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (filterStrideA == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'filterStrideA' is null for cudnnSetConvolutionNdDescriptor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (upscaleA == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'upscaleA' is null for cudnnSetConvolutionNdDescriptor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    // mode is primitive
    // dataType is primitive

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnSetConvolutionNdDescriptor(convDesc=%p, arrayLength=%d, padA=%p, filterStrideA=%p, upscaleA=%p, mode=%d, dataType=%d)\n",
        convDesc, arrayLength, padA, filterStrideA, upscaleA, mode, dataType);

    // Native variable declarations
    cudnnConvolutionDescriptor_t convDesc_native;
    int arrayLength_native = 0;
    int* padA_native = NULL;
    int* filterStrideA_native = NULL;
    int* upscaleA_native = NULL;
    cudnnConvolutionMode_t mode_native;
    cudnnDataType_t dataType_native;

    // Obtain native variable values
    convDesc_native = (cudnnConvolutionDescriptor_t)getNativePointerValue(env, convDesc);
    arrayLength_native = (int)arrayLength;
    if (!initNative(env, padA, padA_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!initNative(env, filterStrideA, filterStrideA_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!initNative(env, upscaleA, upscaleA_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    mode_native = (cudnnConvolutionMode_t)mode;
    dataType_native = (cudnnDataType_t)dataType;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnSetConvolutionNdDescriptor(convDesc_native, arrayLength_native, padA_native, filterStrideA_native, upscaleA_native, mode_native, dataType_native);

    // Write back native variable values
    // convDesc is a read-only native pointer
    // arrayLength is primitive
    if (!releaseNative(env, padA_native, padA, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!releaseNative(env, filterStrideA_native, filterStrideA, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!releaseNative(env, upscaleA_native, upscaleA, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // mode is primitive
    // dataType is primitive

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetConvolutionNdDescriptorNative(JNIEnv *env, jclass cls, jobject convDesc, jint arrayLengthRequested, jintArray arrayLength, jintArray padA, jintArray strideA, jintArray upscaleA, jintArray mode, jintArray dataType)
{
    // Null-checks for non-primitive arguments
    if (convDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'convDesc' is null for cudnnGetConvolutionNdDescriptor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    // arrayLengthRequested is primitive
    if (arrayLength == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'arrayLength' is null for cudnnGetConvolutionNdDescriptor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (padA == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'padA' is null for cudnnGetConvolutionNdDescriptor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (strideA == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'strideA' is null for cudnnGetConvolutionNdDescriptor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (upscaleA == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'upscaleA' is null for cudnnGetConvolutionNdDescriptor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (mode == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'mode' is null for cudnnGetConvolutionNdDescriptor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (dataType == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dataType' is null for cudnnGetConvolutionNdDescriptor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetConvolutionNdDescriptor(convDesc=%p, arrayLengthRequested=%d, arrayLength=%p, padA=%p, strideA=%p, upscaleA=%p, mode=%p, dataType=%p)\n",
        convDesc, arrayLengthRequested, arrayLength, padA, strideA, upscaleA, mode, dataType);

    // Native variable declarations
    cudnnConvolutionDescriptor_t convDesc_native;
    int arrayLengthRequested_native = 0;
    int arrayLength_native = 0;
    int* padA_native = NULL;
    int* strideA_native = NULL;
    int* upscaleA_native = NULL;
    cudnnConvolutionMode_t mode_native;
    cudnnDataType_t dataType_native;

    // Obtain native variable values
    convDesc_native = (cudnnConvolutionDescriptor_t)getNativePointerValue(env, convDesc);
    arrayLengthRequested_native = (int)arrayLengthRequested;
    // arrayLength is set here
    if (!initNative(env, padA, padA_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!initNative(env, strideA, strideA_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!initNative(env, upscaleA, upscaleA_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // mode is set here
    // dataType is set here

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetConvolutionNdDescriptor(convDesc_native, arrayLengthRequested_native, &arrayLength_native, padA_native, strideA_native, upscaleA_native, &mode_native, &dataType_native);

    // Write back native variable values
    // convDesc is a read-only native pointer
    // arrayLengthRequested is primitive
    if (!set(env, arrayLength, 0, (jint)arrayLength_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!releaseNative(env, padA_native, padA, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!releaseNative(env, strideA_native, strideA, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!releaseNative(env, upscaleA_native, upscaleA, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, mode, 0, (jint)mode_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, dataType, 0, (jint)dataType_native)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

/** cudnnSetConvolutionNdDescriptor_v3 is now mapped to cudnnSetConvolutionNdDescriptor
and will be removed at the same time than cudnnSetConvolutionNdDescriptor_v2
Use cudnnSetConvolutionNdDescriptor instead */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnSetConvolutionNdDescriptor_1v3Native(JNIEnv *env, jclass cls, jobject convDesc, jint arrayLength, jintArray padA, jintArray filterStrideA, jintArray upscaleA, jint mode, jint dataType)
{
    // Null-checks for non-primitive arguments
    if (convDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'convDesc' is null for cudnnSetConvolutionNdDescriptor_v3");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    // arrayLength is primitive
    if (padA == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'padA' is null for cudnnSetConvolutionNdDescriptor_v3");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (filterStrideA == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'filterStrideA' is null for cudnnSetConvolutionNdDescriptor_v3");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (upscaleA == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'upscaleA' is null for cudnnSetConvolutionNdDescriptor_v3");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    // mode is primitive
    // dataType is primitive

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnSetConvolutionNdDescriptor_v3(convDesc=%p, arrayLength=%d, padA=%p, filterStrideA=%p, upscaleA=%p, mode=%d, dataType=%d)\n",
        convDesc, arrayLength, padA, filterStrideA, upscaleA, mode, dataType);

    // Native variable declarations
    cudnnConvolutionDescriptor_t convDesc_native;
    int arrayLength_native = 0;
    int* padA_native = NULL;
    int* filterStrideA_native = NULL;
    int* upscaleA_native = NULL;
    cudnnConvolutionMode_t mode_native;
    cudnnDataType_t dataType_native;

    // Obtain native variable values
    convDesc_native = (cudnnConvolutionDescriptor_t)getNativePointerValue(env, convDesc);
    arrayLength_native = (int)arrayLength;
    if (!initNative(env, padA, padA_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!initNative(env, filterStrideA, filterStrideA_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!initNative(env, upscaleA, upscaleA_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    mode_native = (cudnnConvolutionMode_t)mode;
    dataType_native = (cudnnDataType_t)dataType;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnSetConvolutionNdDescriptor_v3(convDesc_native, arrayLength_native, padA_native, filterStrideA_native, upscaleA_native, mode_native, dataType_native);

    // Write back native variable values
    // convDesc is a read-only native pointer
    // arrayLength is primitive
    if (!releaseNative(env, padA_native, padA, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!releaseNative(env, filterStrideA_native, filterStrideA, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!releaseNative(env, upscaleA_native, upscaleA, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // mode is primitive
    // dataType is primitive

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

/**
* <pre>
* cudnnGetConvolutionNdDescriptor_v3 is now mapped to cudnnGetConvolutionNdDescriptor
and will be removed at the same time thancudnnGetConvolutionNdDescriptor_v2
Use cudnnGetConvolutionNdDescriptor instead
* </pre>
*/
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetConvolutionNdDescriptor_1v3Native(JNIEnv *env, jclass cls, jobject convDesc, jint arrayLengthRequested, jintArray arrayLength, jintArray padA, jintArray strideA, jintArray upscaleA, jintArray mode, jintArray dataType)
{
    // Null-checks for non-primitive arguments
    if (convDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'convDesc' is null for cudnnGetConvolutionNdDescriptor_v3");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    // arrayLengthRequested is primitive
    if (arrayLength == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'arrayLength' is null for cudnnGetConvolutionNdDescriptor_v3");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (padA == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'padA' is null for cudnnGetConvolutionNdDescriptor_v3");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (strideA == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'strideA' is null for cudnnGetConvolutionNdDescriptor_v3");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (upscaleA == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'upscaleA' is null for cudnnGetConvolutionNdDescriptor_v3");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (mode == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'mode' is null for cudnnGetConvolutionNdDescriptor_v3");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (dataType == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dataType' is null for cudnnGetConvolutionNdDescriptor_v3");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetConvolutionNdDescriptor_v3(convDesc=%p, arrayLengthRequested=%d, arrayLength=%p, padA=%p, strideA=%p, upscaleA=%p, mode=%p, dataType=%p)\n",
        convDesc, arrayLengthRequested, arrayLength, padA, strideA, upscaleA, mode, dataType);

    // Native variable declarations
    cudnnConvolutionDescriptor_t convDesc_native;
    int arrayLengthRequested_native = 0;
    int arrayLength_native = 0;
    int* padA_native = NULL;
    int* strideA_native = NULL;
    int* upscaleA_native = NULL;
    cudnnConvolutionMode_t mode_native;
    cudnnDataType_t dataType_native;

    // Obtain native variable values
    convDesc_native = (cudnnConvolutionDescriptor_t)getNativePointerValue(env, convDesc);
    arrayLengthRequested_native = (int)arrayLengthRequested;
    // arrayLength is set here
    if (!initNative(env, padA, padA_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!initNative(env, strideA, strideA_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!initNative(env, upscaleA, upscaleA_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // mode is set here
    // dataType is set here

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetConvolutionNdDescriptor_v3(convDesc_native, arrayLengthRequested_native, &arrayLength_native, padA_native, strideA_native, upscaleA_native, &mode_native, &dataType_native);

    // Write back native variable values
    // convDesc is a read-only native pointer
    // arrayLengthRequested is primitive
    if (!set(env, arrayLength, 0, (jint)arrayLength_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!releaseNative(env, padA_native, padA, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!releaseNative(env, strideA_native, strideA, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!releaseNative(env, upscaleA_native, upscaleA, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, mode, 0, (jint)mode_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, dataType, 0, (jint)dataType_native)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

/** Helper function to return the dimensions of the output tensor given a convolution descriptor */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetConvolutionNdForwardOutputDimNative(JNIEnv *env, jclass cls, jobject convDesc, jobject inputTensorDesc, jobject filterDesc, jint nbDims, jintArray tensorOuputDimA)
{
    // Null-checks for non-primitive arguments
    if (convDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'convDesc' is null for cudnnGetConvolutionNdForwardOutputDim");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (inputTensorDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'inputTensorDesc' is null for cudnnGetConvolutionNdForwardOutputDim");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (filterDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'filterDesc' is null for cudnnGetConvolutionNdForwardOutputDim");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    // nbDims is primitive
    if (tensorOuputDimA == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'tensorOuputDimA' is null for cudnnGetConvolutionNdForwardOutputDim");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetConvolutionNdForwardOutputDim(convDesc=%p, inputTensorDesc=%p, filterDesc=%p, nbDims=%d, tensorOuputDimA=%p)\n",
        convDesc, inputTensorDesc, filterDesc, nbDims, tensorOuputDimA);

    // Native variable declarations
    cudnnConvolutionDescriptor_t convDesc_native;
    cudnnTensorDescriptor_t inputTensorDesc_native;
    cudnnFilterDescriptor_t filterDesc_native;
    int nbDims_native = 0;
    int* tensorOuputDimA_native = NULL;

    // Obtain native variable values
    convDesc_native = (cudnnConvolutionDescriptor_t)getNativePointerValue(env, convDesc);
    inputTensorDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, inputTensorDesc);
    filterDesc_native = (cudnnFilterDescriptor_t)getNativePointerValue(env, filterDesc);
    nbDims_native = (int)nbDims;
    if (!initNative(env, tensorOuputDimA, tensorOuputDimA_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetConvolutionNdForwardOutputDim(convDesc_native, inputTensorDesc_native, filterDesc_native, nbDims_native, tensorOuputDimA_native);

    // Write back native variable values
    // convDesc is a read-only native pointer
    // inputTensorDesc is a read-only native pointer
    // filterDesc is a read-only native pointer
    // nbDims is primitive
    if (!releaseNative(env, tensorOuputDimA_native, tensorOuputDimA, true)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

/** Destroy an instance of convolution descriptor */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnDestroyConvolutionDescriptorNative(JNIEnv *env, jclass cls, jobject convDesc)
{
    // Null-checks for non-primitive arguments
    if (convDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'convDesc' is null for cudnnDestroyConvolutionDescriptor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnDestroyConvolutionDescriptor(convDesc=%p)\n",
        convDesc);

    // Native variable declarations
    cudnnConvolutionDescriptor_t convDesc_native;

    // Obtain native variable values
    convDesc_native = (cudnnConvolutionDescriptor_t)getNativePointerValue(env, convDesc);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnDestroyConvolutionDescriptor(convDesc_native);

    // Write back native variable values
    // convDesc is a read-only native pointer

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnFindConvolutionForwardAlgorithmNative(JNIEnv *env, jclass cls, jobject handle, jobject xDesc, jobject wDesc, jobject convDesc, jobject yDesc, jint requestedAlgoCount, jintArray returnedAlgoCount, jobjectArray perfResults)
{
    // Null-checks for non-primitive arguments
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cudnnFindConvolutionForwardAlgorithm");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (xDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'xDesc' is null for cudnnFindConvolutionForwardAlgorithm");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (wDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'wDesc' is null for cudnnFindConvolutionForwardAlgorithm");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (convDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'convDesc' is null for cudnnFindConvolutionForwardAlgorithm");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (yDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'yDesc' is null for cudnnFindConvolutionForwardAlgorithm");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    // requestedAlgoCount is primitive
    if (returnedAlgoCount == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'returnedAlgoCount' is null for cudnnFindConvolutionForwardAlgorithm");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (perfResults == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'perfResults' is null for cudnnFindConvolutionForwardAlgorithm");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnFindConvolutionForwardAlgorithm(handle=%p, xDesc=%p, wDesc=%p, convDesc=%p, yDesc=%p, requestedAlgoCount=%d, returnedAlgoCount=%p, perfResults=%p)\n",
        handle, xDesc, wDesc, convDesc, yDesc, requestedAlgoCount, returnedAlgoCount, perfResults);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnTensorDescriptor_t xDesc_native;
    cudnnFilterDescriptor_t wDesc_native;
    cudnnConvolutionDescriptor_t convDesc_native;
    cudnnTensorDescriptor_t yDesc_native;
    int requestedAlgoCount_native = 0;
    int returnedAlgoCount_native = 0;
    cudnnConvolutionFwdAlgoPerf_t* perfResults_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    xDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, xDesc);
    wDesc_native = (cudnnFilterDescriptor_t)getNativePointerValue(env, wDesc);
    convDesc_native = (cudnnConvolutionDescriptor_t)getNativePointerValue(env, convDesc);
    yDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, yDesc);
    requestedAlgoCount_native = (int)requestedAlgoCount;
    // returnedAlgoCount is set here
    if (!initNative(env, perfResults, perfResults_native, requestedAlgoCount)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnFindConvolutionForwardAlgorithm(handle_native, xDesc_native, wDesc_native, convDesc_native, yDesc_native, requestedAlgoCount_native, &returnedAlgoCount_native, perfResults_native);

    // Write back native variable values
    // handle is a read-only native pointer
    // xDesc is a read-only native pointer
    // wDesc is a read-only native pointer
    // convDesc is a read-only native pointer
    // yDesc is a read-only native pointer
    // requestedAlgoCount is primitive
    if (!set(env, returnedAlgoCount, 0, (jint)returnedAlgoCount_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!releaseNative(env, perfResults_native, perfResults, returnedAlgoCount_native)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetConvolutionForwardAlgorithmNative(JNIEnv *env, jclass cls, jobject handle, jobject xDesc, jobject wDesc, jobject convDesc, jobject yDesc, jint preference, jlong memoryLimitInBytes, jintArray algo)
{
    // Null-checks for non-primitive arguments
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cudnnGetConvolutionForwardAlgorithm");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (xDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'xDesc' is null for cudnnGetConvolutionForwardAlgorithm");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (wDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'wDesc' is null for cudnnGetConvolutionForwardAlgorithm");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (convDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'convDesc' is null for cudnnGetConvolutionForwardAlgorithm");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (yDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'yDesc' is null for cudnnGetConvolutionForwardAlgorithm");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    // preference is primitive
    // memoryLimitInBytes is primitive
    if (algo == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'algo' is null for cudnnGetConvolutionForwardAlgorithm");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetConvolutionForwardAlgorithm(handle=%p, xDesc=%p, wDesc=%p, convDesc=%p, yDesc=%p, preference=%d, memoryLimitInBytes=%ld, algo=%p)\n",
        handle, xDesc, wDesc, convDesc, yDesc, preference, memoryLimitInBytes, algo);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnTensorDescriptor_t xDesc_native;
    cudnnFilterDescriptor_t wDesc_native;
    cudnnConvolutionDescriptor_t convDesc_native;
    cudnnTensorDescriptor_t yDesc_native;
    cudnnConvolutionFwdPreference_t preference_native;
    size_t memoryLimitInBytes_native = 0;
    cudnnConvolutionFwdAlgo_t algo_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    xDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, xDesc);
    wDesc_native = (cudnnFilterDescriptor_t)getNativePointerValue(env, wDesc);
    convDesc_native = (cudnnConvolutionDescriptor_t)getNativePointerValue(env, convDesc);
    yDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, yDesc);
    preference_native = (cudnnConvolutionFwdPreference_t)preference;
    memoryLimitInBytes_native = (size_t)memoryLimitInBytes;
    // algo is set here

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetConvolutionForwardAlgorithm(handle_native, xDesc_native, wDesc_native, convDesc_native, yDesc_native, preference_native, memoryLimitInBytes_native, &algo_native);

    // Write back native variable values
    // handle is a read-only native pointer
    // xDesc is a read-only native pointer
    // wDesc is a read-only native pointer
    // convDesc is a read-only native pointer
    // yDesc is a read-only native pointer
    // preference is primitive
    // memoryLimitInBytes is primitive
    if (!set(env, algo, 0, (jint)algo_native)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

/**
*  convolution algorithm (which requires potentially some workspace)
*/
/** Helper function to return the minimum size of the workspace to be passed to the convolution given an algo*/
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetConvolutionForwardWorkspaceSizeNative(JNIEnv *env, jclass cls, jobject handle, jobject xDesc, jobject wDesc, jobject convDesc, jobject yDesc, jint algo, jlongArray sizeInBytes)
{
    // Null-checks for non-primitive arguments
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cudnnGetConvolutionForwardWorkspaceSize");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (xDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'xDesc' is null for cudnnGetConvolutionForwardWorkspaceSize");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (wDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'wDesc' is null for cudnnGetConvolutionForwardWorkspaceSize");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (convDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'convDesc' is null for cudnnGetConvolutionForwardWorkspaceSize");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (yDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'yDesc' is null for cudnnGetConvolutionForwardWorkspaceSize");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    // algo is primitive
    if (sizeInBytes == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'sizeInBytes' is null for cudnnGetConvolutionForwardWorkspaceSize");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetConvolutionForwardWorkspaceSize(handle=%p, xDesc=%p, wDesc=%p, convDesc=%p, yDesc=%p, algo=%d, sizeInBytes=%p)\n",
        handle, xDesc, wDesc, convDesc, yDesc, algo, sizeInBytes);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnTensorDescriptor_t xDesc_native;
    cudnnFilterDescriptor_t wDesc_native;
    cudnnConvolutionDescriptor_t convDesc_native;
    cudnnTensorDescriptor_t yDesc_native;
    cudnnConvolutionFwdAlgo_t algo_native;
    size_t sizeInBytes_native = 0;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    xDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, xDesc);
    wDesc_native = (cudnnFilterDescriptor_t)getNativePointerValue(env, wDesc);
    convDesc_native = (cudnnConvolutionDescriptor_t)getNativePointerValue(env, convDesc);
    yDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, yDesc);
    algo_native = (cudnnConvolutionFwdAlgo_t)algo;
    // sizeInBytes is set here

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetConvolutionForwardWorkspaceSize(handle_native, xDesc_native, wDesc_native, convDesc_native, yDesc_native, algo_native, &sizeInBytes_native);

    // Write back native variable values
    // handle is a read-only native pointer
    // xDesc is a read-only native pointer
    // wDesc is a read-only native pointer
    // convDesc is a read-only native pointer
    // yDesc is a read-only native pointer
    // algo is primitive
    if (!set(env, sizeInBytes, 0, (jlong)sizeInBytes_native)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

/** Convolution functions: All of the form "output = alpha * Op(inputs) + beta * output" */
/** Function to perform the forward pass for batch convolution */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnConvolutionForwardNative(JNIEnv *env, jclass cls, jobject handle, jobject alpha, jobject xDesc, jobject x, jobject wDesc, jobject w, jobject convDesc, jint algo, jobject workSpace, jlong workSpaceSizeInBytes, jobject beta, jobject yDesc, jobject y)
{
    // Null-checks for non-primitive arguments
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cudnnConvolutionForward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (alpha == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'alpha' is null for cudnnConvolutionForward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (xDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'xDesc' is null for cudnnConvolutionForward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cudnnConvolutionForward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (wDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'wDesc' is null for cudnnConvolutionForward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (w == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'w' is null for cudnnConvolutionForward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (convDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'convDesc' is null for cudnnConvolutionForward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    // algo is primitive
    if (workSpace == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'workSpace' is null for cudnnConvolutionForward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    // workSpaceSizeInBytes is primitive
    if (beta == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'beta' is null for cudnnConvolutionForward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (yDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'yDesc' is null for cudnnConvolutionForward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (y == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'y' is null for cudnnConvolutionForward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnConvolutionForward(handle=%p, alpha=%p, xDesc=%p, x=%p, wDesc=%p, w=%p, convDesc=%p, algo=%d, workSpace=%p, workSpaceSizeInBytes=%ld, beta=%p, yDesc=%p, y=%p)\n",
        handle, alpha, xDesc, x, wDesc, w, convDesc, algo, workSpace, workSpaceSizeInBytes, beta, yDesc, y);

    // Native variable declarations
    cudnnHandle_t handle_native;
    void* alpha_native;
    cudnnTensorDescriptor_t xDesc_native;
    void* x_native;
    cudnnFilterDescriptor_t wDesc_native;
    void* w_native;
    cudnnConvolutionDescriptor_t convDesc_native;
    cudnnConvolutionFwdAlgo_t algo_native;
    void* workSpace_native;
    size_t workSpaceSizeInBytes_native = 0;
    void* beta_native;
    cudnnTensorDescriptor_t yDesc_native;
    void* y_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    PointerData *alpha_pointerData = initPointerData(env, alpha);
    if (alpha_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    alpha_native = (void*)alpha_pointerData->getPointer(env);
    xDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, xDesc);
    x_native = (void*)getPointer(env, x);
    wDesc_native = (cudnnFilterDescriptor_t)getNativePointerValue(env, wDesc);
    w_native = (void*)getPointer(env, w);
    convDesc_native = (cudnnConvolutionDescriptor_t)getNativePointerValue(env, convDesc);
    algo_native = (cudnnConvolutionFwdAlgo_t)algo;
    workSpace_native = (void*)getPointer(env, workSpace);
    workSpaceSizeInBytes_native = (size_t)workSpaceSizeInBytes;
    PointerData *beta_pointerData = initPointerData(env, beta);
    if (beta_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    beta_native = (void*)beta_pointerData->getPointer(env);
    yDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, yDesc);
    y_native = (void*)getPointer(env, y);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnConvolutionForward(handle_native, alpha_native, xDesc_native, x_native, wDesc_native, w_native, convDesc_native, algo_native, workSpace_native, workSpaceSizeInBytes_native, beta_native, yDesc_native, y_native);

    // Write back native variable values
    // handle is a read-only native pointer
    if (!releasePointerData(env, alpha_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // xDesc is a read-only native pointer
    // x is a native pointer
    // wDesc is a read-only native pointer
    // w is a native pointer
    // convDesc is a read-only native pointer
    // algo is primitive
    // workSpace is a native pointer
    // workSpaceSizeInBytes is primitive
    if (!releasePointerData(env, beta_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // yDesc is a read-only native pointer
    // y is a native pointer

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

/** Function to compute the bias gradient for batch convolution */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnConvolutionBackwardBiasNative(JNIEnv *env, jclass cls, jobject handle, jobject alpha, jobject dyDesc, jobject dy, jobject beta, jobject dbDesc, jobject db)
{
    // Null-checks for non-primitive arguments
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cudnnConvolutionBackwardBias");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (alpha == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'alpha' is null for cudnnConvolutionBackwardBias");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (dyDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dyDesc' is null for cudnnConvolutionBackwardBias");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (dy == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dy' is null for cudnnConvolutionBackwardBias");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (beta == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'beta' is null for cudnnConvolutionBackwardBias");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (dbDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dbDesc' is null for cudnnConvolutionBackwardBias");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (db == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'db' is null for cudnnConvolutionBackwardBias");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnConvolutionBackwardBias(handle=%p, alpha=%p, dyDesc=%p, dy=%p, beta=%p, dbDesc=%p, db=%p)\n",
        handle, alpha, dyDesc, dy, beta, dbDesc, db);

    // Native variable declarations
    cudnnHandle_t handle_native;
    void* alpha_native;
    cudnnTensorDescriptor_t dyDesc_native;
    void* dy_native;
    void* beta_native;
    cudnnTensorDescriptor_t dbDesc_native;
    void* db_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    PointerData *alpha_pointerData = initPointerData(env, alpha);
    if (alpha_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    alpha_native = (void*)alpha_pointerData->getPointer(env);
    dyDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, dyDesc);
    dy_native = (void*)getPointer(env, dy);
    PointerData *beta_pointerData = initPointerData(env, beta);
    if (beta_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    beta_native = (void*)beta_pointerData->getPointer(env);
    dbDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, dbDesc);
    db_native = (void*)getPointer(env, db);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnConvolutionBackwardBias(handle_native, alpha_native, dyDesc_native, dy_native, beta_native, dbDesc_native, db_native);

    // Write back native variable values
    // handle is a read-only native pointer
    if (!releasePointerData(env, alpha_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // dyDesc is a read-only native pointer
    // dy is a native pointer
    if (!releasePointerData(env, beta_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // dbDesc is a read-only native pointer
    // db is a native pointer

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnFindConvolutionBackwardFilterAlgorithmNative(JNIEnv *env, jclass cls, jobject handle, jobject xDesc, jobject dyDesc, jobject convDesc, jobject dwDesc, jint requestedAlgoCount, jintArray returnedAlgoCount, jobjectArray perfResults)
{
    // Null-checks for non-primitive arguments
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cudnnFindConvolutionBackwardFilterAlgorithm");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (xDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'xDesc' is null for cudnnFindConvolutionBackwardFilterAlgorithm");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (dyDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dyDesc' is null for cudnnFindConvolutionBackwardFilterAlgorithm");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (convDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'convDesc' is null for cudnnFindConvolutionBackwardFilterAlgorithm");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (dwDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dwDesc' is null for cudnnFindConvolutionBackwardFilterAlgorithm");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    // requestedAlgoCount is primitive
    if (returnedAlgoCount == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'returnedAlgoCount' is null for cudnnFindConvolutionBackwardFilterAlgorithm");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (perfResults == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'perfResults' is null for cudnnFindConvolutionBackwardFilterAlgorithm");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnFindConvolutionBackwardFilterAlgorithm(handle=%p, xDesc=%p, dyDesc=%p, convDesc=%p, dwDesc=%p, requestedAlgoCount=%d, returnedAlgoCount=%p, perfResults=%p)\n",
        handle, xDesc, dyDesc, convDesc, dwDesc, requestedAlgoCount, returnedAlgoCount, perfResults);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnTensorDescriptor_t xDesc_native;
    cudnnTensorDescriptor_t dyDesc_native;
    cudnnConvolutionDescriptor_t convDesc_native;
    cudnnFilterDescriptor_t dwDesc_native;
    int requestedAlgoCount_native = 0;
    int returnedAlgoCount_native = 0;
    cudnnConvolutionBwdFilterAlgoPerf_t* perfResults_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    xDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, xDesc);
    dyDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, dyDesc);
    convDesc_native = (cudnnConvolutionDescriptor_t)getNativePointerValue(env, convDesc);
    dwDesc_native = (cudnnFilterDescriptor_t)getNativePointerValue(env, dwDesc);
    requestedAlgoCount_native = (int)requestedAlgoCount;
    // returnedAlgoCount is set here
    if (!initNative(env, perfResults, perfResults_native, requestedAlgoCount)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnFindConvolutionBackwardFilterAlgorithm(handle_native, xDesc_native, dyDesc_native, convDesc_native, dwDesc_native, requestedAlgoCount_native, &returnedAlgoCount_native, perfResults_native);

    // Write back native variable values
    // handle is a read-only native pointer
    // xDesc is a read-only native pointer
    // dyDesc is a read-only native pointer
    // convDesc is a read-only native pointer
    // dwDesc is a read-only native pointer
    // requestedAlgoCount is primitive
    if (!set(env, returnedAlgoCount, 0, (jint)returnedAlgoCount_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!releaseNative(env, perfResults_native, perfResults, returnedAlgoCount_native)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetConvolutionBackwardFilterAlgorithmNative(JNIEnv *env, jclass cls, jobject handle, jobject xDesc, jobject dyDesc, jobject convDesc, jobject dwDesc, jint preference, jlong memoryLimitInBytes, jintArray algo)
{
    // Null-checks for non-primitive arguments
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cudnnGetConvolutionBackwardFilterAlgorithm");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (xDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'xDesc' is null for cudnnGetConvolutionBackwardFilterAlgorithm");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (dyDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dyDesc' is null for cudnnGetConvolutionBackwardFilterAlgorithm");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (convDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'convDesc' is null for cudnnGetConvolutionBackwardFilterAlgorithm");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (dwDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dwDesc' is null for cudnnGetConvolutionBackwardFilterAlgorithm");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    // preference is primitive
    // memoryLimitInBytes is primitive
    if (algo == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'algo' is null for cudnnGetConvolutionBackwardFilterAlgorithm");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetConvolutionBackwardFilterAlgorithm(handle=%p, xDesc=%p, dyDesc=%p, convDesc=%p, dwDesc=%p, preference=%d, memoryLimitInBytes=%ld, algo=%p)\n",
        handle, xDesc, dyDesc, convDesc, dwDesc, preference, memoryLimitInBytes, algo);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnTensorDescriptor_t xDesc_native;
    cudnnTensorDescriptor_t dyDesc_native;
    cudnnConvolutionDescriptor_t convDesc_native;
    cudnnFilterDescriptor_t dwDesc_native;
    cudnnConvolutionBwdFilterPreference_t preference_native;
    size_t memoryLimitInBytes_native = 0;
    cudnnConvolutionBwdFilterAlgo_t algo_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    xDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, xDesc);
    dyDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, dyDesc);
    convDesc_native = (cudnnConvolutionDescriptor_t)getNativePointerValue(env, convDesc);
    dwDesc_native = (cudnnFilterDescriptor_t)getNativePointerValue(env, dwDesc);
    preference_native = (cudnnConvolutionBwdFilterPreference_t)preference;
    memoryLimitInBytes_native = (size_t)memoryLimitInBytes;
    // algo is set here

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetConvolutionBackwardFilterAlgorithm(handle_native, xDesc_native, dyDesc_native, convDesc_native, dwDesc_native, preference_native, memoryLimitInBytes_native, &algo_native);

    // Write back native variable values
    // handle is a read-only native pointer
    // xDesc is a read-only native pointer
    // dyDesc is a read-only native pointer
    // convDesc is a read-only native pointer
    // dwDesc is a read-only native pointer
    // preference is primitive
    // memoryLimitInBytes is primitive
    if (!set(env, algo, 0, (jint)algo_native)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

/**
*  convolution algorithm (which requires potentially some workspace)
*/
/** Helper function to return the minimum size of the workspace to be passed to the convolution given an algo*/
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetConvolutionBackwardFilterWorkspaceSizeNative(JNIEnv *env, jclass cls, jobject handle, jobject xDesc, jobject dyDesc, jobject convDesc, jobject gradDesc, jint algo, jlongArray sizeInBytes)
{
    // Null-checks for non-primitive arguments
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cudnnGetConvolutionBackwardFilterWorkspaceSize");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (xDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'xDesc' is null for cudnnGetConvolutionBackwardFilterWorkspaceSize");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (dyDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dyDesc' is null for cudnnGetConvolutionBackwardFilterWorkspaceSize");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (convDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'convDesc' is null for cudnnGetConvolutionBackwardFilterWorkspaceSize");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (gradDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'gradDesc' is null for cudnnGetConvolutionBackwardFilterWorkspaceSize");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    // algo is primitive
    if (sizeInBytes == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'sizeInBytes' is null for cudnnGetConvolutionBackwardFilterWorkspaceSize");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetConvolutionBackwardFilterWorkspaceSize(handle=%p, xDesc=%p, dyDesc=%p, convDesc=%p, gradDesc=%p, algo=%d, sizeInBytes=%p)\n",
        handle, xDesc, dyDesc, convDesc, gradDesc, algo, sizeInBytes);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnTensorDescriptor_t xDesc_native;
    cudnnTensorDescriptor_t dyDesc_native;
    cudnnConvolutionDescriptor_t convDesc_native;
    cudnnFilterDescriptor_t gradDesc_native;
    cudnnConvolutionBwdFilterAlgo_t algo_native;
    size_t sizeInBytes_native = 0;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    xDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, xDesc);
    dyDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, dyDesc);
    convDesc_native = (cudnnConvolutionDescriptor_t)getNativePointerValue(env, convDesc);
    gradDesc_native = (cudnnFilterDescriptor_t)getNativePointerValue(env, gradDesc);
    algo_native = (cudnnConvolutionBwdFilterAlgo_t)algo;
    // sizeInBytes is set here

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetConvolutionBackwardFilterWorkspaceSize(handle_native, xDesc_native, dyDesc_native, convDesc_native, gradDesc_native, algo_native, &sizeInBytes_native);

    // Write back native variable values
    // handle is a read-only native pointer
    // xDesc is a read-only native pointer
    // dyDesc is a read-only native pointer
    // convDesc is a read-only native pointer
    // gradDesc is a read-only native pointer
    // algo is primitive
    if (!set(env, sizeInBytes, 0, (jlong)sizeInBytes_native)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnConvolutionBackwardFilterNative(JNIEnv *env, jclass cls, jobject handle, jobject alpha, jobject xDesc, jobject x, jobject dyDesc, jobject dy, jobject convDesc, jint algo, jobject workSpace, jlong workSpaceSizeInBytes, jobject beta, jobject dwDesc, jobject dw)
{
    // Null-checks for non-primitive arguments
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cudnnConvolutionBackwardFilter");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (alpha == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'alpha' is null for cudnnConvolutionBackwardFilter");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (xDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'xDesc' is null for cudnnConvolutionBackwardFilter");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cudnnConvolutionBackwardFilter");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (dyDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dyDesc' is null for cudnnConvolutionBackwardFilter");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (dy == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dy' is null for cudnnConvolutionBackwardFilter");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (convDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'convDesc' is null for cudnnConvolutionBackwardFilter");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    // algo is primitive
    if (workSpace == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'workSpace' is null for cudnnConvolutionBackwardFilter");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    // workSpaceSizeInBytes is primitive
    if (beta == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'beta' is null for cudnnConvolutionBackwardFilter");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (dwDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dwDesc' is null for cudnnConvolutionBackwardFilter");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (dw == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dw' is null for cudnnConvolutionBackwardFilter");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnConvolutionBackwardFilter(handle=%p, alpha=%p, xDesc=%p, x=%p, dyDesc=%p, dy=%p, convDesc=%p, algo=%d, workSpace=%p, workSpaceSizeInBytes=%ld, beta=%p, dwDesc=%p, dw=%p)\n",
        handle, alpha, xDesc, x, dyDesc, dy, convDesc, algo, workSpace, workSpaceSizeInBytes, beta, dwDesc, dw);

    // Native variable declarations
    cudnnHandle_t handle_native;
    void* alpha_native;
    cudnnTensorDescriptor_t xDesc_native;
    void* x_native;
    cudnnTensorDescriptor_t dyDesc_native;
    void* dy_native;
    cudnnConvolutionDescriptor_t convDesc_native;
    cudnnConvolutionBwdFilterAlgo_t algo_native;
    void* workSpace_native;
    size_t workSpaceSizeInBytes_native = 0;
    void* beta_native;
    cudnnFilterDescriptor_t dwDesc_native;
    void* dw_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    PointerData *alpha_pointerData = initPointerData(env, alpha);
    if (alpha_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    alpha_native = (void*)alpha_pointerData->getPointer(env);
    xDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, xDesc);
    x_native = (void*)getPointer(env, x);
    dyDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, dyDesc);
    dy_native = (void*)getPointer(env, dy);
    convDesc_native = (cudnnConvolutionDescriptor_t)getNativePointerValue(env, convDesc);
    algo_native = (cudnnConvolutionBwdFilterAlgo_t)algo;
    workSpace_native = (void*)getPointer(env, workSpace);
    workSpaceSizeInBytes_native = (size_t)workSpaceSizeInBytes;
    PointerData *beta_pointerData = initPointerData(env, beta);
    if (beta_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    beta_native = (void*)beta_pointerData->getPointer(env);
    dwDesc_native = (cudnnFilterDescriptor_t)getNativePointerValue(env, dwDesc);
    dw_native = (void*)getPointer(env, dw);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnConvolutionBackwardFilter(handle_native, alpha_native, xDesc_native, x_native, dyDesc_native, dy_native, convDesc_native, algo_native, workSpace_native, workSpaceSizeInBytes_native, beta_native, dwDesc_native, dw_native);

    // Write back native variable values
    // handle is a read-only native pointer
    if (!releasePointerData(env, alpha_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // xDesc is a read-only native pointer
    // x is a native pointer
    // dyDesc is a read-only native pointer
    // dy is a native pointer
    // convDesc is a read-only native pointer
    // algo is primitive
    // workSpace is a native pointer
    // workSpaceSizeInBytes is primitive
    if (!releasePointerData(env, beta_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // dwDesc is a read-only native pointer
    // dw is a native pointer

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

/** cudnnConvolutionBackwardFilter_v3 is now mapped to cudnnConvolutionBackwardFilter
and will be removed at the same time thancudnnConvolutionBackwardFilter_v2
Use cudnnConvolutionBackwardFilter instead */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnConvolutionBackwardFilter_1v3Native(JNIEnv *env, jclass cls, jobject handle, jobject alpha, jobject xDesc, jobject x, jobject dyDesc, jobject dy, jobject convDesc, jint algo, jobject workSpace, jlong workSpaceSizeInBytes, jobject beta, jobject dwDesc, jobject dw)
{
    // Null-checks for non-primitive arguments
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cudnnConvolutionBackwardFilter_v3");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (alpha == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'alpha' is null for cudnnConvolutionBackwardFilter_v3");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (xDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'xDesc' is null for cudnnConvolutionBackwardFilter_v3");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cudnnConvolutionBackwardFilter_v3");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (dyDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dyDesc' is null for cudnnConvolutionBackwardFilter_v3");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (dy == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dy' is null for cudnnConvolutionBackwardFilter_v3");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (convDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'convDesc' is null for cudnnConvolutionBackwardFilter_v3");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    // algo is primitive
    if (workSpace == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'workSpace' is null for cudnnConvolutionBackwardFilter_v3");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    // workSpaceSizeInBytes is primitive
    if (beta == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'beta' is null for cudnnConvolutionBackwardFilter_v3");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (dwDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dwDesc' is null for cudnnConvolutionBackwardFilter_v3");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (dw == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dw' is null for cudnnConvolutionBackwardFilter_v3");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnConvolutionBackwardFilter_v3(handle=%p, alpha=%p, xDesc=%p, x=%p, dyDesc=%p, dy=%p, convDesc=%p, algo=%d, workSpace=%p, workSpaceSizeInBytes=%ld, beta=%p, dwDesc=%p, dw=%p)\n",
        handle, alpha, xDesc, x, dyDesc, dy, convDesc, algo, workSpace, workSpaceSizeInBytes, beta, dwDesc, dw);

    // Native variable declarations
    cudnnHandle_t handle_native;
    void* alpha_native;
    cudnnTensorDescriptor_t xDesc_native;
    void* x_native;
    cudnnTensorDescriptor_t dyDesc_native;
    void* dy_native;
    cudnnConvolutionDescriptor_t convDesc_native;
    cudnnConvolutionBwdFilterAlgo_t algo_native;
    void* workSpace_native;
    size_t workSpaceSizeInBytes_native = 0;
    void* beta_native;
    cudnnFilterDescriptor_t dwDesc_native;
    void* dw_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    PointerData *alpha_pointerData = initPointerData(env, alpha);
    if (alpha_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    alpha_native = (void*)alpha_pointerData->getPointer(env);
    xDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, xDesc);
    x_native = (void*)getPointer(env, x);
    dyDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, dyDesc);
    dy_native = (void*)getPointer(env, dy);
    convDesc_native = (cudnnConvolutionDescriptor_t)getNativePointerValue(env, convDesc);
    algo_native = (cudnnConvolutionBwdFilterAlgo_t)algo;
    workSpace_native = (void*)getPointer(env, workSpace);
    workSpaceSizeInBytes_native = (size_t)workSpaceSizeInBytes;
    PointerData *beta_pointerData = initPointerData(env, beta);
    if (beta_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    beta_native = (void*)beta_pointerData->getPointer(env);
    dwDesc_native = (cudnnFilterDescriptor_t)getNativePointerValue(env, dwDesc);
    dw_native = (void*)getPointer(env, dw);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnConvolutionBackwardFilter_v3(handle_native, alpha_native, xDesc_native, x_native, dyDesc_native, dy_native, convDesc_native, algo_native, workSpace_native, workSpaceSizeInBytes_native, beta_native, dwDesc_native, dw_native);

    // Write back native variable values
    // handle is a read-only native pointer
    if (!releasePointerData(env, alpha_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // xDesc is a read-only native pointer
    // x is a native pointer
    // dyDesc is a read-only native pointer
    // dy is a native pointer
    // convDesc is a read-only native pointer
    // algo is primitive
    // workSpace is a native pointer
    // workSpaceSizeInBytes is primitive
    if (!releasePointerData(env, beta_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // dwDesc is a read-only native pointer
    // dw is a native pointer

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnFindConvolutionBackwardDataAlgorithmNative(JNIEnv *env, jclass cls, jobject handle, jobject wDesc, jobject dyDesc, jobject convDesc, jobject dxDesc, jint requestedAlgoCount, jintArray returnedAlgoCount, jobjectArray perfResults)
{
    // Null-checks for non-primitive arguments
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cudnnFindConvolutionBackwardDataAlgorithm");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (wDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'wDesc' is null for cudnnFindConvolutionBackwardDataAlgorithm");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (dyDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dyDesc' is null for cudnnFindConvolutionBackwardDataAlgorithm");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (convDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'convDesc' is null for cudnnFindConvolutionBackwardDataAlgorithm");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (dxDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dxDesc' is null for cudnnFindConvolutionBackwardDataAlgorithm");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    // requestedAlgoCount is primitive
    if (returnedAlgoCount == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'returnedAlgoCount' is null for cudnnFindConvolutionBackwardDataAlgorithm");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (perfResults == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'perfResults' is null for cudnnFindConvolutionBackwardDataAlgorithm");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnFindConvolutionBackwardDataAlgorithm(handle=%p, wDesc=%p, dyDesc=%p, convDesc=%p, dxDesc=%p, requestedAlgoCount=%d, returnedAlgoCount=%p, perfResults=%p)\n",
        handle, wDesc, dyDesc, convDesc, dxDesc, requestedAlgoCount, returnedAlgoCount, perfResults);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnFilterDescriptor_t wDesc_native;
    cudnnTensorDescriptor_t dyDesc_native;
    cudnnConvolutionDescriptor_t convDesc_native;
    cudnnTensorDescriptor_t dxDesc_native;
    int requestedAlgoCount_native = 0;
    int returnedAlgoCount_native = 0;
    cudnnConvolutionBwdDataAlgoPerf_t* perfResults_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    wDesc_native = (cudnnFilterDescriptor_t)getNativePointerValue(env, wDesc);
    dyDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, dyDesc);
    convDesc_native = (cudnnConvolutionDescriptor_t)getNativePointerValue(env, convDesc);
    dxDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, dxDesc);
    requestedAlgoCount_native = (int)requestedAlgoCount;
    // returnedAlgoCount is set here
    if (!initNative(env, perfResults, perfResults_native, requestedAlgoCount)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnFindConvolutionBackwardDataAlgorithm(handle_native, wDesc_native, dyDesc_native, convDesc_native, dxDesc_native, requestedAlgoCount_native, &returnedAlgoCount_native, perfResults_native);

    // Write back native variable values
    // handle is a read-only native pointer
    // wDesc is a read-only native pointer
    // dyDesc is a read-only native pointer
    // convDesc is a read-only native pointer
    // dxDesc is a read-only native pointer
    // requestedAlgoCount is primitive
    if (!set(env, returnedAlgoCount, 0, (jint)returnedAlgoCount_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!releaseNative(env, perfResults_native, perfResults, returnedAlgoCount_native)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetConvolutionBackwardDataAlgorithmNative(JNIEnv *env, jclass cls, jobject handle, jobject wDesc, jobject dyDesc, jobject convDesc, jobject dxDesc, jint preference, jlong memoryLimitInBytes, jintArray algo)
{
    // Null-checks for non-primitive arguments
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cudnnGetConvolutionBackwardDataAlgorithm");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (wDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'wDesc' is null for cudnnGetConvolutionBackwardDataAlgorithm");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (dyDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dyDesc' is null for cudnnGetConvolutionBackwardDataAlgorithm");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (convDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'convDesc' is null for cudnnGetConvolutionBackwardDataAlgorithm");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (dxDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dxDesc' is null for cudnnGetConvolutionBackwardDataAlgorithm");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    // preference is primitive
    // memoryLimitInBytes is primitive
    if (algo == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'algo' is null for cudnnGetConvolutionBackwardDataAlgorithm");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetConvolutionBackwardDataAlgorithm(handle=%p, wDesc=%p, dyDesc=%p, convDesc=%p, dxDesc=%p, preference=%d, memoryLimitInBytes=%ld, algo=%p)\n",
        handle, wDesc, dyDesc, convDesc, dxDesc, preference, memoryLimitInBytes, algo);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnFilterDescriptor_t wDesc_native;
    cudnnTensorDescriptor_t dyDesc_native;
    cudnnConvolutionDescriptor_t convDesc_native;
    cudnnTensorDescriptor_t dxDesc_native;
    cudnnConvolutionBwdDataPreference_t preference_native;
    size_t memoryLimitInBytes_native = 0;
    cudnnConvolutionBwdDataAlgo_t algo_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    wDesc_native = (cudnnFilterDescriptor_t)getNativePointerValue(env, wDesc);
    dyDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, dyDesc);
    convDesc_native = (cudnnConvolutionDescriptor_t)getNativePointerValue(env, convDesc);
    dxDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, dxDesc);
    preference_native = (cudnnConvolutionBwdDataPreference_t)preference;
    memoryLimitInBytes_native = (size_t)memoryLimitInBytes;
    // algo is set here

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetConvolutionBackwardDataAlgorithm(handle_native, wDesc_native, dyDesc_native, convDesc_native, dxDesc_native, preference_native, memoryLimitInBytes_native, &algo_native);

    // Write back native variable values
    // handle is a read-only native pointer
    // wDesc is a read-only native pointer
    // dyDesc is a read-only native pointer
    // convDesc is a read-only native pointer
    // dxDesc is a read-only native pointer
    // preference is primitive
    // memoryLimitInBytes is primitive
    if (!set(env, algo, 0, (jint)algo_native)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

/** Helper function to return the minimum size of the workspace to be passed to the convolution given an algo*/
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetConvolutionBackwardDataWorkspaceSizeNative(JNIEnv *env, jclass cls, jobject handle, jobject wDesc, jobject dyDesc, jobject convDesc, jobject dxDesc, jint algo, jlongArray sizeInBytes)
{
    // Null-checks for non-primitive arguments
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cudnnGetConvolutionBackwardDataWorkspaceSize");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (wDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'wDesc' is null for cudnnGetConvolutionBackwardDataWorkspaceSize");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (dyDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dyDesc' is null for cudnnGetConvolutionBackwardDataWorkspaceSize");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (convDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'convDesc' is null for cudnnGetConvolutionBackwardDataWorkspaceSize");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (dxDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dxDesc' is null for cudnnGetConvolutionBackwardDataWorkspaceSize");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    // algo is primitive
    if (sizeInBytes == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'sizeInBytes' is null for cudnnGetConvolutionBackwardDataWorkspaceSize");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetConvolutionBackwardDataWorkspaceSize(handle=%p, wDesc=%p, dyDesc=%p, convDesc=%p, dxDesc=%p, algo=%d, sizeInBytes=%p)\n",
        handle, wDesc, dyDesc, convDesc, dxDesc, algo, sizeInBytes);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnFilterDescriptor_t wDesc_native;
    cudnnTensorDescriptor_t dyDesc_native;
    cudnnConvolutionDescriptor_t convDesc_native;
    cudnnTensorDescriptor_t dxDesc_native;
    cudnnConvolutionBwdDataAlgo_t algo_native;
    size_t sizeInBytes_native = 0;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    wDesc_native = (cudnnFilterDescriptor_t)getNativePointerValue(env, wDesc);
    dyDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, dyDesc);
    convDesc_native = (cudnnConvolutionDescriptor_t)getNativePointerValue(env, convDesc);
    dxDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, dxDesc);
    algo_native = (cudnnConvolutionBwdDataAlgo_t)algo;
    // sizeInBytes is set here

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetConvolutionBackwardDataWorkspaceSize(handle_native, wDesc_native, dyDesc_native, convDesc_native, dxDesc_native, algo_native, &sizeInBytes_native);

    // Write back native variable values
    // handle is a read-only native pointer
    // wDesc is a read-only native pointer
    // dyDesc is a read-only native pointer
    // convDesc is a read-only native pointer
    // dxDesc is a read-only native pointer
    // algo is primitive
    if (!set(env, sizeInBytes, 0, (jlong)sizeInBytes_native)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnConvolutionBackwardDataNative(JNIEnv *env, jclass cls, jobject handle, jobject alpha, jobject wDesc, jobject w, jobject dyDesc, jobject dy, jobject convDesc, jint algo, jobject workSpace, jlong workSpaceSizeInBytes, jobject beta, jobject dxDesc, jobject dx)
{
    // Null-checks for non-primitive arguments
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cudnnConvolutionBackwardData");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (alpha == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'alpha' is null for cudnnConvolutionBackwardData");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (wDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'wDesc' is null for cudnnConvolutionBackwardData");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (w == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'w' is null for cudnnConvolutionBackwardData");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (dyDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dyDesc' is null for cudnnConvolutionBackwardData");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (dy == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dy' is null for cudnnConvolutionBackwardData");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (convDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'convDesc' is null for cudnnConvolutionBackwardData");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    // algo is primitive
    if (workSpace == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'workSpace' is null for cudnnConvolutionBackwardData");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    // workSpaceSizeInBytes is primitive
    if (beta == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'beta' is null for cudnnConvolutionBackwardData");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (dxDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dxDesc' is null for cudnnConvolutionBackwardData");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (dx == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dx' is null for cudnnConvolutionBackwardData");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnConvolutionBackwardData(handle=%p, alpha=%p, wDesc=%p, w=%p, dyDesc=%p, dy=%p, convDesc=%p, algo=%d, workSpace=%p, workSpaceSizeInBytes=%ld, beta=%p, dxDesc=%p, dx=%p)\n",
        handle, alpha, wDesc, w, dyDesc, dy, convDesc, algo, workSpace, workSpaceSizeInBytes, beta, dxDesc, dx);

    // Native variable declarations
    cudnnHandle_t handle_native;
    void* alpha_native;
    cudnnFilterDescriptor_t wDesc_native;
    void* w_native;
    cudnnTensorDescriptor_t dyDesc_native;
    void* dy_native;
    cudnnConvolutionDescriptor_t convDesc_native;
    cudnnConvolutionBwdDataAlgo_t algo_native;
    void* workSpace_native;
    size_t workSpaceSizeInBytes_native = 0;
    void* beta_native;
    cudnnTensorDescriptor_t dxDesc_native;
    void* dx_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    PointerData *alpha_pointerData = initPointerData(env, alpha);
    if (alpha_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    alpha_native = (void*)alpha_pointerData->getPointer(env);
    wDesc_native = (cudnnFilterDescriptor_t)getNativePointerValue(env, wDesc);
    w_native = (void*)getPointer(env, w);
    dyDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, dyDesc);
    dy_native = (void*)getPointer(env, dy);
    convDesc_native = (cudnnConvolutionDescriptor_t)getNativePointerValue(env, convDesc);
    algo_native = (cudnnConvolutionBwdDataAlgo_t)algo;
    workSpace_native = (void*)getPointer(env, workSpace);
    workSpaceSizeInBytes_native = (size_t)workSpaceSizeInBytes;
    PointerData *beta_pointerData = initPointerData(env, beta);
    if (beta_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    beta_native = (void*)beta_pointerData->getPointer(env);
    dxDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, dxDesc);
    dx_native = (void*)getPointer(env, dx);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnConvolutionBackwardData(handle_native, alpha_native, wDesc_native, w_native, dyDesc_native, dy_native, convDesc_native, algo_native, workSpace_native, workSpaceSizeInBytes_native, beta_native, dxDesc_native, dx_native);

    // Write back native variable values
    // handle is a read-only native pointer
    if (!releasePointerData(env, alpha_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // wDesc is a read-only native pointer
    // w is a native pointer
    // dyDesc is a read-only native pointer
    // dy is a native pointer
    // convDesc is a read-only native pointer
    // algo is primitive
    // workSpace is a native pointer
    // workSpaceSizeInBytes is primitive
    if (!releasePointerData(env, beta_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // dxDesc is a read-only native pointer
    // dx is a native pointer

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

/** cudnnConvolutionBackwardData_v3 is now mapped to cudnnConvolutionBackwardData
and will be removed at the same time thancudnnConvolutionBackwardData_v2
Use cudnnConvolutionBackwardData instead */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnConvolutionBackwardData_1v3Native(JNIEnv *env, jclass cls, jobject handle, jobject alpha, jobject wDesc, jobject w, jobject dyDesc, jobject dy, jobject convDesc, jint algo, jobject workSpace, jlong workSpaceSizeInBytes, jobject beta, jobject dxDesc, jobject dx)
{
    // Null-checks for non-primitive arguments
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cudnnConvolutionBackwardData_v3");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (alpha == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'alpha' is null for cudnnConvolutionBackwardData_v3");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (wDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'wDesc' is null for cudnnConvolutionBackwardData_v3");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (w == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'w' is null for cudnnConvolutionBackwardData_v3");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (dyDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dyDesc' is null for cudnnConvolutionBackwardData_v3");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (dy == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dy' is null for cudnnConvolutionBackwardData_v3");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (convDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'convDesc' is null for cudnnConvolutionBackwardData_v3");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    // algo is primitive
    if (workSpace == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'workSpace' is null for cudnnConvolutionBackwardData_v3");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    // workSpaceSizeInBytes is primitive
    if (beta == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'beta' is null for cudnnConvolutionBackwardData_v3");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (dxDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dxDesc' is null for cudnnConvolutionBackwardData_v3");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (dx == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dx' is null for cudnnConvolutionBackwardData_v3");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnConvolutionBackwardData_v3(handle=%p, alpha=%p, wDesc=%p, w=%p, dyDesc=%p, dy=%p, convDesc=%p, algo=%d, workSpace=%p, workSpaceSizeInBytes=%ld, beta=%p, dxDesc=%p, dx=%p)\n",
        handle, alpha, wDesc, w, dyDesc, dy, convDesc, algo, workSpace, workSpaceSizeInBytes, beta, dxDesc, dx);

    // Native variable declarations
    cudnnHandle_t handle_native;
    void* alpha_native;
    cudnnFilterDescriptor_t wDesc_native;
    void* w_native;
    cudnnTensorDescriptor_t dyDesc_native;
    void* dy_native;
    cudnnConvolutionDescriptor_t convDesc_native;
    cudnnConvolutionBwdDataAlgo_t algo_native;
    void* workSpace_native;
    size_t workSpaceSizeInBytes_native = 0;
    void* beta_native;
    cudnnTensorDescriptor_t dxDesc_native;
    void* dx_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    PointerData *alpha_pointerData = initPointerData(env, alpha);
    if (alpha_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    alpha_native = (void*)alpha_pointerData->getPointer(env);
    wDesc_native = (cudnnFilterDescriptor_t)getNativePointerValue(env, wDesc);
    w_native = (void*)getPointer(env, w);
    dyDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, dyDesc);
    dy_native = (void*)getPointer(env, dy);
    convDesc_native = (cudnnConvolutionDescriptor_t)getNativePointerValue(env, convDesc);
    algo_native = (cudnnConvolutionBwdDataAlgo_t)algo;
    workSpace_native = (void*)getPointer(env, workSpace);
    workSpaceSizeInBytes_native = (size_t)workSpaceSizeInBytes;
    PointerData *beta_pointerData = initPointerData(env, beta);
    if (beta_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    beta_native = (void*)beta_pointerData->getPointer(env);
    dxDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, dxDesc);
    dx_native = (void*)getPointer(env, dx);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnConvolutionBackwardData_v3(handle_native, alpha_native, wDesc_native, w_native, dyDesc_native, dy_native, convDesc_native, algo_native, workSpace_native, workSpaceSizeInBytes_native, beta_native, dxDesc_native, dx_native);

    // Write back native variable values
    // handle is a read-only native pointer
    if (!releasePointerData(env, alpha_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // wDesc is a read-only native pointer
    // w is a native pointer
    // dyDesc is a read-only native pointer
    // dy is a native pointer
    // convDesc is a read-only native pointer
    // algo is primitive
    // workSpace is a native pointer
    // workSpaceSizeInBytes is primitive
    if (!releasePointerData(env, beta_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // dxDesc is a read-only native pointer
    // dx is a native pointer

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnIm2ColNative(JNIEnv *env, jclass cls, jobject handle, jobject xDesc, jobject x, jobject wDesc, jobject convDesc, jobject colBuffer)
{
    // Null-checks for non-primitive arguments
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cudnnIm2Col");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (xDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'xDesc' is null for cudnnIm2Col");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cudnnIm2Col");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (wDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'wDesc' is null for cudnnIm2Col");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (convDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'convDesc' is null for cudnnIm2Col");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (colBuffer == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'colBuffer' is null for cudnnIm2Col");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnIm2Col(handle=%p, xDesc=%p, x=%p, wDesc=%p, convDesc=%p, colBuffer=%p)\n",
        handle, xDesc, x, wDesc, convDesc, colBuffer);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnTensorDescriptor_t xDesc_native;
    void* x_native;
    cudnnFilterDescriptor_t wDesc_native;
    cudnnConvolutionDescriptor_t convDesc_native;
    void* colBuffer_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    xDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, xDesc);
    x_native = (void*)getPointer(env, x);
    wDesc_native = (cudnnFilterDescriptor_t)getNativePointerValue(env, wDesc);
    convDesc_native = (cudnnConvolutionDescriptor_t)getNativePointerValue(env, convDesc);
    colBuffer_native = (void*)getPointer(env, colBuffer);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnIm2Col(handle_native, xDesc_native, x_native, wDesc_native, convDesc_native, colBuffer_native);

    // Write back native variable values
    // handle is a read-only native pointer
    // xDesc is a read-only native pointer
    // x is a native pointer
    // wDesc is a read-only native pointer
    // convDesc is a read-only native pointer
    // colBuffer is a native pointer

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

/** Softmax functions: All of the form "output = alpha * Op(inputs) + beta * output" */
/** Function to perform forward softmax */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnSoftmaxForwardNative(JNIEnv *env, jclass cls, jobject handle, jint algo, jint mode, jobject alpha, jobject xDesc, jobject x, jobject beta, jobject yDesc, jobject y)
{
    // Null-checks for non-primitive arguments
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cudnnSoftmaxForward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    // algo is primitive
    // mode is primitive
    if (alpha == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'alpha' is null for cudnnSoftmaxForward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (xDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'xDesc' is null for cudnnSoftmaxForward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cudnnSoftmaxForward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (beta == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'beta' is null for cudnnSoftmaxForward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (yDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'yDesc' is null for cudnnSoftmaxForward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (y == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'y' is null for cudnnSoftmaxForward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnSoftmaxForward(handle=%p, algo=%d, mode=%d, alpha=%p, xDesc=%p, x=%p, beta=%p, yDesc=%p, y=%p)\n",
        handle, algo, mode, alpha, xDesc, x, beta, yDesc, y);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnSoftmaxAlgorithm_t algo_native;
    cudnnSoftmaxMode_t mode_native;
    void* alpha_native;
    cudnnTensorDescriptor_t xDesc_native;
    void* x_native;
    void* beta_native;
    cudnnTensorDescriptor_t yDesc_native;
    void* y_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    algo_native = (cudnnSoftmaxAlgorithm_t)algo;
    mode_native = (cudnnSoftmaxMode_t)mode;
    PointerData *alpha_pointerData = initPointerData(env, alpha);
    if (alpha_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    alpha_native = (void*)alpha_pointerData->getPointer(env);
    xDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, xDesc);
    x_native = (void*)getPointer(env, x);
    PointerData *beta_pointerData = initPointerData(env, beta);
    if (beta_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    beta_native = (void*)beta_pointerData->getPointer(env);
    yDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, yDesc);
    y_native = (void*)getPointer(env, y);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnSoftmaxForward(handle_native, algo_native, mode_native, alpha_native, xDesc_native, x_native, beta_native, yDesc_native, y_native);

    // Write back native variable values
    // handle is a read-only native pointer
    // algo is primitive
    // mode is primitive
    if (!releasePointerData(env, alpha_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // xDesc is a read-only native pointer
    // x is a native pointer
    if (!releasePointerData(env, beta_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // yDesc is a read-only native pointer
    // y is a native pointer

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

/** Function to perform backward softmax */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnSoftmaxBackwardNative(JNIEnv *env, jclass cls, jobject handle, jint algo, jint mode, jobject alpha, jobject yDesc, jobject y, jobject dyDesc, jobject dy, jobject beta, jobject dxDesc, jobject dx)
{
    // Null-checks for non-primitive arguments
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cudnnSoftmaxBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    // algo is primitive
    // mode is primitive
    if (alpha == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'alpha' is null for cudnnSoftmaxBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (yDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'yDesc' is null for cudnnSoftmaxBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (y == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'y' is null for cudnnSoftmaxBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (dyDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dyDesc' is null for cudnnSoftmaxBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (dy == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dy' is null for cudnnSoftmaxBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (beta == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'beta' is null for cudnnSoftmaxBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (dxDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dxDesc' is null for cudnnSoftmaxBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (dx == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dx' is null for cudnnSoftmaxBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnSoftmaxBackward(handle=%p, algo=%d, mode=%d, alpha=%p, yDesc=%p, y=%p, dyDesc=%p, dy=%p, beta=%p, dxDesc=%p, dx=%p)\n",
        handle, algo, mode, alpha, yDesc, y, dyDesc, dy, beta, dxDesc, dx);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnSoftmaxAlgorithm_t algo_native;
    cudnnSoftmaxMode_t mode_native;
    void* alpha_native;
    cudnnTensorDescriptor_t yDesc_native;
    void* y_native;
    cudnnTensorDescriptor_t dyDesc_native;
    void* dy_native;
    void* beta_native;
    cudnnTensorDescriptor_t dxDesc_native;
    void* dx_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    algo_native = (cudnnSoftmaxAlgorithm_t)algo;
    mode_native = (cudnnSoftmaxMode_t)mode;
    PointerData *alpha_pointerData = initPointerData(env, alpha);
    if (alpha_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    alpha_native = (void*)alpha_pointerData->getPointer(env);
    yDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, yDesc);
    y_native = (void*)getPointer(env, y);
    dyDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, dyDesc);
    dy_native = (void*)getPointer(env, dy);
    PointerData *beta_pointerData = initPointerData(env, beta);
    if (beta_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    beta_native = (void*)beta_pointerData->getPointer(env);
    dxDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, dxDesc);
    dx_native = (void*)getPointer(env, dx);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnSoftmaxBackward(handle_native, algo_native, mode_native, alpha_native, yDesc_native, y_native, dyDesc_native, dy_native, beta_native, dxDesc_native, dx_native);

    // Write back native variable values
    // handle is a read-only native pointer
    // algo is primitive
    // mode is primitive
    if (!releasePointerData(env, alpha_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // yDesc is a read-only native pointer
    // y is a native pointer
    // dyDesc is a read-only native pointer
    // dy is a native pointer
    if (!releasePointerData(env, beta_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // dxDesc is a read-only native pointer
    // dx is a native pointer

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

/** Create an instance of pooling descriptor */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnCreatePoolingDescriptorNative(JNIEnv *env, jclass cls, jobject poolingDesc)
{
    // Null-checks for non-primitive arguments
    if (poolingDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'poolingDesc' is null for cudnnCreatePoolingDescriptor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnCreatePoolingDescriptor(poolingDesc=%p)\n",
        poolingDesc);

    // Native variable declarations
    cudnnPoolingDescriptor_t poolingDesc_native;

    // Obtain native variable values
    // poolingDesc is write-only

    // Native function call
    cudnnStatus_t jniResult_native = cudnnCreatePoolingDescriptor(&poolingDesc_native);

    // Write back native variable values
    setNativePointerValue(env, poolingDesc, (jlong)poolingDesc_native);

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnSetPooling2dDescriptorNative(JNIEnv *env, jclass cls, jobject poolingDesc, jint mode, jint windowHeight, jint windowWidth, jint verticalPadding, jint horizontalPadding, jint verticalStride, jint horizontalStride)
{
    // Null-checks for non-primitive arguments
    if (poolingDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'poolingDesc' is null for cudnnSetPooling2dDescriptor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    // mode is primitive
    // windowHeight is primitive
    // windowWidth is primitive
    // verticalPadding is primitive
    // horizontalPadding is primitive
    // verticalStride is primitive
    // horizontalStride is primitive

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnSetPooling2dDescriptor(poolingDesc=%p, mode=%d, windowHeight=%d, windowWidth=%d, verticalPadding=%d, horizontalPadding=%d, verticalStride=%d, horizontalStride=%d)\n",
        poolingDesc, mode, windowHeight, windowWidth, verticalPadding, horizontalPadding, verticalStride, horizontalStride);

    // Native variable declarations
    cudnnPoolingDescriptor_t poolingDesc_native;
    cudnnPoolingMode_t mode_native;
    int windowHeight_native = 0;
    int windowWidth_native = 0;
    int verticalPadding_native = 0;
    int horizontalPadding_native = 0;
    int verticalStride_native = 0;
    int horizontalStride_native = 0;

    // Obtain native variable values
    poolingDesc_native = (cudnnPoolingDescriptor_t)getNativePointerValue(env, poolingDesc);
    mode_native = (cudnnPoolingMode_t)mode;
    windowHeight_native = (int)windowHeight;
    windowWidth_native = (int)windowWidth;
    verticalPadding_native = (int)verticalPadding;
    horizontalPadding_native = (int)horizontalPadding;
    verticalStride_native = (int)verticalStride;
    horizontalStride_native = (int)horizontalStride;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnSetPooling2dDescriptor(poolingDesc_native, mode_native, windowHeight_native, windowWidth_native, verticalPadding_native, horizontalPadding_native, verticalStride_native, horizontalStride_native);

    // Write back native variable values
    // poolingDesc is a read-only native pointer
    // mode is primitive
    // windowHeight is primitive
    // windowWidth is primitive
    // verticalPadding is primitive
    // horizontalPadding is primitive
    // verticalStride is primitive
    // horizontalStride is primitive

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnSetPooling2dDescriptor_1v4Native(JNIEnv *env, jclass cls, jobject poolingDesc, jint mode, jint maxpoolingNanOpt, jint windowHeight, jint windowWidth, jint verticalPadding, jint horizontalPadding, jint verticalStride, jint horizontalStride)
{
    // Null-checks for non-primitive arguments
    if (poolingDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'poolingDesc' is null for cudnnSetPooling2dDescriptor_v4");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    // mode is primitive
    // maxpoolingNanOpt is primitive
    // windowHeight is primitive
    // windowWidth is primitive
    // verticalPadding is primitive
    // horizontalPadding is primitive
    // verticalStride is primitive
    // horizontalStride is primitive

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnSetPooling2dDescriptor_v4(poolingDesc=%p, mode=%d, maxpoolingNanOpt=%d, windowHeight=%d, windowWidth=%d, verticalPadding=%d, horizontalPadding=%d, verticalStride=%d, horizontalStride=%d)\n",
        poolingDesc, mode, maxpoolingNanOpt, windowHeight, windowWidth, verticalPadding, horizontalPadding, verticalStride, horizontalStride);

    // Native variable declarations
    cudnnPoolingDescriptor_t poolingDesc_native;
    cudnnPoolingMode_t mode_native;
    cudnnNanPropagation_t maxpoolingNanOpt_native;
    int windowHeight_native = 0;
    int windowWidth_native = 0;
    int verticalPadding_native = 0;
    int horizontalPadding_native = 0;
    int verticalStride_native = 0;
    int horizontalStride_native = 0;

    // Obtain native variable values
    poolingDesc_native = (cudnnPoolingDescriptor_t)getNativePointerValue(env, poolingDesc);
    mode_native = (cudnnPoolingMode_t)mode;
    maxpoolingNanOpt_native = (cudnnNanPropagation_t)maxpoolingNanOpt;
    windowHeight_native = (int)windowHeight;
    windowWidth_native = (int)windowWidth;
    verticalPadding_native = (int)verticalPadding;
    horizontalPadding_native = (int)horizontalPadding;
    verticalStride_native = (int)verticalStride;
    horizontalStride_native = (int)horizontalStride;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnSetPooling2dDescriptor_v4(poolingDesc_native, mode_native, maxpoolingNanOpt_native, windowHeight_native, windowWidth_native, verticalPadding_native, horizontalPadding_native, verticalStride_native, horizontalStride_native);

    // Write back native variable values
    // poolingDesc is a read-only native pointer
    // mode is primitive
    // maxpoolingNanOpt is primitive
    // windowHeight is primitive
    // windowWidth is primitive
    // verticalPadding is primitive
    // horizontalPadding is primitive
    // verticalStride is primitive
    // horizontalStride is primitive

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetPooling2dDescriptorNative(JNIEnv *env, jclass cls, jobject poolingDesc, jintArray mode, jobject windowHeight, jobject windowWidth, jobject verticalPadding, jobject horizontalPadding, jobject verticalStride, jobject horizontalStride)
{
    // Null-checks for non-primitive arguments
    if (poolingDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'poolingDesc' is null for cudnnGetPooling2dDescriptor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (mode == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'mode' is null for cudnnGetPooling2dDescriptor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (windowHeight == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'windowHeight' is null for cudnnGetPooling2dDescriptor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (windowWidth == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'windowWidth' is null for cudnnGetPooling2dDescriptor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (verticalPadding == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'verticalPadding' is null for cudnnGetPooling2dDescriptor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (horizontalPadding == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'horizontalPadding' is null for cudnnGetPooling2dDescriptor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (verticalStride == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'verticalStride' is null for cudnnGetPooling2dDescriptor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (horizontalStride == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'horizontalStride' is null for cudnnGetPooling2dDescriptor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetPooling2dDescriptor(poolingDesc=%p, mode=%p, windowHeight=%p, windowWidth=%p, verticalPadding=%p, horizontalPadding=%p, verticalStride=%p, horizontalStride=%p)\n",
        poolingDesc, mode, windowHeight, windowWidth, verticalPadding, horizontalPadding, verticalStride, horizontalStride);

    // Native variable declarations
    cudnnPoolingDescriptor_t poolingDesc_native;
    cudnnPoolingMode_t* mode_native;
    int* windowHeight_native;
    int* windowWidth_native;
    int* verticalPadding_native;
    int* horizontalPadding_native;
    int* verticalStride_native;
    int* horizontalStride_native;

    // Obtain native variable values
    poolingDesc_native = (cudnnPoolingDescriptor_t)getNativePointerValue(env, poolingDesc);
    mode_native = (cudnnPoolingMode_t*)getPointer(env, mode);
    windowHeight_native = (int*)getPointer(env, windowHeight);
    windowWidth_native = (int*)getPointer(env, windowWidth);
    verticalPadding_native = (int*)getPointer(env, verticalPadding);
    horizontalPadding_native = (int*)getPointer(env, horizontalPadding);
    verticalStride_native = (int*)getPointer(env, verticalStride);
    horizontalStride_native = (int*)getPointer(env, horizontalStride);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetPooling2dDescriptor(poolingDesc_native, mode_native, windowHeight_native, windowWidth_native, verticalPadding_native, horizontalPadding_native, verticalStride_native, horizontalStride_native);

    // Write back native variable values
    // poolingDesc is a read-only native pointer
    // mode is a native pointer
    // windowHeight is a native pointer
    // windowWidth is a native pointer
    // verticalPadding is a native pointer
    // horizontalPadding is a native pointer
    // verticalStride is a native pointer
    // horizontalStride is a native pointer

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetPooling2dDescriptor_1v4Native(JNIEnv *env, jclass cls, jobject poolingDesc, jintArray mode, jintArray maxpoolingNanOpt, jobject windowHeight, jobject windowWidth, jobject verticalPadding, jobject horizontalPadding, jobject verticalStride, jobject horizontalStride)
{
    // Null-checks for non-primitive arguments
    if (poolingDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'poolingDesc' is null for cudnnGetPooling2dDescriptor_v4");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (mode == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'mode' is null for cudnnGetPooling2dDescriptor_v4");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (maxpoolingNanOpt == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'maxpoolingNanOpt' is null for cudnnGetPooling2dDescriptor_v4");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (windowHeight == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'windowHeight' is null for cudnnGetPooling2dDescriptor_v4");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (windowWidth == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'windowWidth' is null for cudnnGetPooling2dDescriptor_v4");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (verticalPadding == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'verticalPadding' is null for cudnnGetPooling2dDescriptor_v4");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (horizontalPadding == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'horizontalPadding' is null for cudnnGetPooling2dDescriptor_v4");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (verticalStride == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'verticalStride' is null for cudnnGetPooling2dDescriptor_v4");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (horizontalStride == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'horizontalStride' is null for cudnnGetPooling2dDescriptor_v4");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetPooling2dDescriptor_v4(poolingDesc=%p, mode=%p, maxpoolingNanOpt=%p, windowHeight=%p, windowWidth=%p, verticalPadding=%p, horizontalPadding=%p, verticalStride=%p, horizontalStride=%p)\n",
        poolingDesc, mode, maxpoolingNanOpt, windowHeight, windowWidth, verticalPadding, horizontalPadding, verticalStride, horizontalStride);

    // Native variable declarations
    cudnnPoolingDescriptor_t poolingDesc_native;
    cudnnPoolingMode_t* mode_native;
    cudnnNanPropagation_t* maxpoolingNanOpt_native;
    int* windowHeight_native;
    int* windowWidth_native;
    int* verticalPadding_native;
    int* horizontalPadding_native;
    int* verticalStride_native;
    int* horizontalStride_native;

    // Obtain native variable values
    poolingDesc_native = (cudnnPoolingDescriptor_t)getNativePointerValue(env, poolingDesc);
    mode_native = (cudnnPoolingMode_t*)getPointer(env, mode);
    maxpoolingNanOpt_native = (cudnnNanPropagation_t*)getPointer(env, maxpoolingNanOpt);
    windowHeight_native = (int*)getPointer(env, windowHeight);
    windowWidth_native = (int*)getPointer(env, windowWidth);
    verticalPadding_native = (int*)getPointer(env, verticalPadding);
    horizontalPadding_native = (int*)getPointer(env, horizontalPadding);
    verticalStride_native = (int*)getPointer(env, verticalStride);
    horizontalStride_native = (int*)getPointer(env, horizontalStride);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetPooling2dDescriptor_v4(poolingDesc_native, mode_native, maxpoolingNanOpt_native, windowHeight_native, windowWidth_native, verticalPadding_native, horizontalPadding_native, verticalStride_native, horizontalStride_native);

    // Write back native variable values
    // poolingDesc is a read-only native pointer
    // mode is a native pointer
    // maxpoolingNanOpt is a native pointer
    // windowHeight is a native pointer
    // windowWidth is a native pointer
    // verticalPadding is a native pointer
    // horizontalPadding is a native pointer
    // verticalStride is a native pointer
    // horizontalStride is a native pointer

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnSetPoolingNdDescriptorNative(JNIEnv *env, jclass cls, jobject poolingDesc, jint mode, jint nbDims, jintArray windowDimA, jintArray paddingA, jintArray strideA)
{
    // Null-checks for non-primitive arguments
    if (poolingDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'poolingDesc' is null for cudnnSetPoolingNdDescriptor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    // mode is primitive
    // nbDims is primitive
    if (windowDimA == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'windowDimA' is null for cudnnSetPoolingNdDescriptor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (paddingA == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'paddingA' is null for cudnnSetPoolingNdDescriptor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (strideA == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'strideA' is null for cudnnSetPoolingNdDescriptor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnSetPoolingNdDescriptor(poolingDesc=%p, mode=%d, nbDims=%d, windowDimA=%p, paddingA=%p, strideA=%p)\n",
        poolingDesc, mode, nbDims, windowDimA, paddingA, strideA);

    // Native variable declarations
    cudnnPoolingDescriptor_t poolingDesc_native;
    cudnnPoolingMode_t mode_native;
    int nbDims_native = 0;
    int* windowDimA_native = NULL;
    int* paddingA_native = NULL;
    int* strideA_native = NULL;

    // Obtain native variable values
    poolingDesc_native = (cudnnPoolingDescriptor_t)getNativePointerValue(env, poolingDesc);
    mode_native = (cudnnPoolingMode_t)mode;
    nbDims_native = (int)nbDims;
    if (!initNative(env, windowDimA, windowDimA_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!initNative(env, paddingA, paddingA_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!initNative(env, strideA, strideA_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnSetPoolingNdDescriptor(poolingDesc_native, mode_native, nbDims_native, windowDimA_native, paddingA_native, strideA_native);

    // Write back native variable values
    // poolingDesc is a read-only native pointer
    // mode is primitive
    // nbDims is primitive
    if (!releaseNative(env, windowDimA_native, windowDimA, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!releaseNative(env, paddingA_native, paddingA, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!releaseNative(env, strideA_native, strideA, true)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnSetPoolingNdDescriptor_1v4Native(JNIEnv *env, jclass cls, jobject poolingDesc, jint mode, jint maxpoolingNanOpt, jint nbDims, jintArray windowDimA, jintArray paddingA, jintArray strideA)
{
    // Null-checks for non-primitive arguments
    if (poolingDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'poolingDesc' is null for cudnnSetPoolingNdDescriptor_v4");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    // mode is primitive
    // maxpoolingNanOpt is primitive
    // nbDims is primitive
    if (windowDimA == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'windowDimA' is null for cudnnSetPoolingNdDescriptor_v4");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (paddingA == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'paddingA' is null for cudnnSetPoolingNdDescriptor_v4");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (strideA == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'strideA' is null for cudnnSetPoolingNdDescriptor_v4");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnSetPoolingNdDescriptor_v4(poolingDesc=%p, mode=%d, maxpoolingNanOpt=%d, nbDims=%d, windowDimA=%p, paddingA=%p, strideA=%p)\n",
        poolingDesc, mode, maxpoolingNanOpt, nbDims, windowDimA, paddingA, strideA);

    // Native variable declarations
    cudnnPoolingDescriptor_t poolingDesc_native;
    cudnnPoolingMode_t mode_native;
    cudnnNanPropagation_t maxpoolingNanOpt_native;
    int nbDims_native = 0;
    int* windowDimA_native = NULL;
    int* paddingA_native = NULL;
    int* strideA_native = NULL;

    // Obtain native variable values
    poolingDesc_native = (cudnnPoolingDescriptor_t)getNativePointerValue(env, poolingDesc);
    mode_native = (cudnnPoolingMode_t)mode;
    maxpoolingNanOpt_native = (cudnnNanPropagation_t)maxpoolingNanOpt;
    nbDims_native = (int)nbDims;
    if (!initNative(env, windowDimA, windowDimA_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!initNative(env, paddingA, paddingA_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!initNative(env, strideA, strideA_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnSetPoolingNdDescriptor_v4(poolingDesc_native, mode_native, maxpoolingNanOpt_native, nbDims_native, windowDimA_native, paddingA_native, strideA_native);

    // Write back native variable values
    // poolingDesc is a read-only native pointer
    // mode is primitive
    // maxpoolingNanOpt is primitive
    // nbDims is primitive
    if (!releaseNative(env, windowDimA_native, windowDimA, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!releaseNative(env, paddingA_native, paddingA, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!releaseNative(env, strideA_native, strideA, true)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetPoolingNdDescriptorNative(JNIEnv *env, jclass cls, jobject poolingDesc, jint nbDimsRequested, jintArray mode, jobject nbDims, jintArray windowDimA, jintArray paddingA, jintArray strideA)
{
    // Null-checks for non-primitive arguments
    if (poolingDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'poolingDesc' is null for cudnnGetPoolingNdDescriptor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    // nbDimsRequested is primitive
    if (mode == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'mode' is null for cudnnGetPoolingNdDescriptor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (nbDims == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'nbDims' is null for cudnnGetPoolingNdDescriptor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (windowDimA == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'windowDimA' is null for cudnnGetPoolingNdDescriptor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (paddingA == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'paddingA' is null for cudnnGetPoolingNdDescriptor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (strideA == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'strideA' is null for cudnnGetPoolingNdDescriptor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetPoolingNdDescriptor(poolingDesc=%p, nbDimsRequested=%d, mode=%p, nbDims=%p, windowDimA=%p, paddingA=%p, strideA=%p)\n",
        poolingDesc, nbDimsRequested, mode, nbDims, windowDimA, paddingA, strideA);

    // Native variable declarations
    cudnnPoolingDescriptor_t poolingDesc_native;
    int nbDimsRequested_native = 0;
    cudnnPoolingMode_t* mode_native;
    int* nbDims_native;
    int* windowDimA_native = NULL;
    int* paddingA_native = NULL;
    int* strideA_native = NULL;

    // Obtain native variable values
    poolingDesc_native = (cudnnPoolingDescriptor_t)getNativePointerValue(env, poolingDesc);
    nbDimsRequested_native = (int)nbDimsRequested;
    mode_native = (cudnnPoolingMode_t*)getPointer(env, mode);
    nbDims_native = (int*)getPointer(env, nbDims);
    if (!initNative(env, windowDimA, windowDimA_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!initNative(env, paddingA, paddingA_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!initNative(env, strideA, strideA_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetPoolingNdDescriptor(poolingDesc_native, nbDimsRequested_native, mode_native, nbDims_native, windowDimA_native, paddingA_native, strideA_native);

    // Write back native variable values
    // poolingDesc is a read-only native pointer
    // nbDimsRequested is primitive
    // mode is a native pointer
    // nbDims is a native pointer
    if (!releaseNative(env, windowDimA_native, windowDimA, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!releaseNative(env, paddingA_native, paddingA, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!releaseNative(env, strideA_native, strideA, true)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetPoolingNdDescriptor_1v4Native(JNIEnv *env, jclass cls, jobject poolingDesc, jint nbDimsRequested, jintArray mode, jintArray maxpoolingNanOpt, jobject nbDims, jintArray windowDimA, jintArray paddingA, jintArray strideA)
{
    // Null-checks for non-primitive arguments
    if (poolingDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'poolingDesc' is null for cudnnGetPoolingNdDescriptor_v4");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    // nbDimsRequested is primitive
    if (mode == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'mode' is null for cudnnGetPoolingNdDescriptor_v4");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (maxpoolingNanOpt == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'maxpoolingNanOpt' is null for cudnnGetPoolingNdDescriptor_v4");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (nbDims == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'nbDims' is null for cudnnGetPoolingNdDescriptor_v4");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (windowDimA == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'windowDimA' is null for cudnnGetPoolingNdDescriptor_v4");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (paddingA == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'paddingA' is null for cudnnGetPoolingNdDescriptor_v4");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (strideA == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'strideA' is null for cudnnGetPoolingNdDescriptor_v4");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetPoolingNdDescriptor_v4(poolingDesc=%p, nbDimsRequested=%d, mode=%p, maxpoolingNanOpt=%p, nbDims=%p, windowDimA=%p, paddingA=%p, strideA=%p)\n",
        poolingDesc, nbDimsRequested, mode, maxpoolingNanOpt, nbDims, windowDimA, paddingA, strideA);

    // Native variable declarations
    cudnnPoolingDescriptor_t poolingDesc_native;
    int nbDimsRequested_native = 0;
    cudnnPoolingMode_t* mode_native;
    cudnnNanPropagation_t* maxpoolingNanOpt_native;
    int* nbDims_native;
    int* windowDimA_native = NULL;
    int* paddingA_native = NULL;
    int* strideA_native = NULL;

    // Obtain native variable values
    poolingDesc_native = (cudnnPoolingDescriptor_t)getNativePointerValue(env, poolingDesc);
    nbDimsRequested_native = (int)nbDimsRequested;
    mode_native = (cudnnPoolingMode_t*)getPointer(env, mode);
    maxpoolingNanOpt_native = (cudnnNanPropagation_t*)getPointer(env, maxpoolingNanOpt);
    nbDims_native = (int*)getPointer(env, nbDims);
    if (!initNative(env, windowDimA, windowDimA_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!initNative(env, paddingA, paddingA_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!initNative(env, strideA, strideA_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetPoolingNdDescriptor_v4(poolingDesc_native, nbDimsRequested_native, mode_native, maxpoolingNanOpt_native, nbDims_native, windowDimA_native, paddingA_native, strideA_native);

    // Write back native variable values
    // poolingDesc is a read-only native pointer
    // nbDimsRequested is primitive
    // mode is a native pointer
    // maxpoolingNanOpt is a native pointer
    // nbDims is a native pointer
    if (!releaseNative(env, windowDimA_native, windowDimA, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!releaseNative(env, paddingA_native, paddingA, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!releaseNative(env, strideA_native, strideA, true)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetPoolingNdForwardOutputDimNative(JNIEnv *env, jclass cls, jobject poolingDesc, jobject inputTensorDesc, jint nbDims, jintArray outputTensorDimA)
{
    // Null-checks for non-primitive arguments
    if (poolingDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'poolingDesc' is null for cudnnGetPoolingNdForwardOutputDim");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (inputTensorDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'inputTensorDesc' is null for cudnnGetPoolingNdForwardOutputDim");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    // nbDims is primitive
    if (outputTensorDimA == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'outputTensorDimA' is null for cudnnGetPoolingNdForwardOutputDim");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetPoolingNdForwardOutputDim(poolingDesc=%p, inputTensorDesc=%p, nbDims=%d, outputTensorDimA=%p)\n",
        poolingDesc, inputTensorDesc, nbDims, outputTensorDimA);

    // Native variable declarations
    cudnnPoolingDescriptor_t poolingDesc_native;
    cudnnTensorDescriptor_t inputTensorDesc_native;
    int nbDims_native = 0;
    int* outputTensorDimA_native = NULL;

    // Obtain native variable values
    poolingDesc_native = (cudnnPoolingDescriptor_t)getNativePointerValue(env, poolingDesc);
    inputTensorDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, inputTensorDesc);
    nbDims_native = (int)nbDims;
    if (!initNative(env, outputTensorDimA, outputTensorDimA_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetPoolingNdForwardOutputDim(poolingDesc_native, inputTensorDesc_native, nbDims_native, outputTensorDimA_native);

    // Write back native variable values
    // poolingDesc is a read-only native pointer
    // inputTensorDesc is a read-only native pointer
    // nbDims is primitive
    if (!releaseNative(env, outputTensorDimA_native, outputTensorDimA, true)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetPooling2dForwardOutputDimNative(JNIEnv *env, jclass cls, jobject poolingDesc, jobject inputTensorDesc, jobject n, jobject c, jobject h, jobject w)
{
    // Null-checks for non-primitive arguments
    if (poolingDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'poolingDesc' is null for cudnnGetPooling2dForwardOutputDim");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (inputTensorDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'inputTensorDesc' is null for cudnnGetPooling2dForwardOutputDim");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (n == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'n' is null for cudnnGetPooling2dForwardOutputDim");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (c == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'c' is null for cudnnGetPooling2dForwardOutputDim");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (h == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'h' is null for cudnnGetPooling2dForwardOutputDim");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (w == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'w' is null for cudnnGetPooling2dForwardOutputDim");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetPooling2dForwardOutputDim(poolingDesc=%p, inputTensorDesc=%p, n=%p, c=%p, h=%p, w=%p)\n",
        poolingDesc, inputTensorDesc, n, c, h, w);

    // Native variable declarations
    cudnnPoolingDescriptor_t poolingDesc_native;
    cudnnTensorDescriptor_t inputTensorDesc_native;
    int* n_native;
    int* c_native;
    int* h_native;
    int* w_native;

    // Obtain native variable values
    poolingDesc_native = (cudnnPoolingDescriptor_t)getNativePointerValue(env, poolingDesc);
    inputTensorDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, inputTensorDesc);
    n_native = (int*)getPointer(env, n);
    c_native = (int*)getPointer(env, c);
    h_native = (int*)getPointer(env, h);
    w_native = (int*)getPointer(env, w);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetPooling2dForwardOutputDim(poolingDesc_native, inputTensorDesc_native, n_native, c_native, h_native, w_native);

    // Write back native variable values
    // poolingDesc is a read-only native pointer
    // inputTensorDesc is a read-only native pointer
    // n is a native pointer
    // c is a native pointer
    // h is a native pointer
    // w is a native pointer

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

/** Destroy an instance of pooling descriptor */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnDestroyPoolingDescriptorNative(JNIEnv *env, jclass cls, jobject poolingDesc)
{
    // Null-checks for non-primitive arguments
    if (poolingDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'poolingDesc' is null for cudnnDestroyPoolingDescriptor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnDestroyPoolingDescriptor(poolingDesc=%p)\n",
        poolingDesc);

    // Native variable declarations
    cudnnPoolingDescriptor_t poolingDesc_native;

    // Obtain native variable values
    poolingDesc_native = (cudnnPoolingDescriptor_t)getNativePointerValue(env, poolingDesc);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnDestroyPoolingDescriptor(poolingDesc_native);

    // Write back native variable values
    // poolingDesc is a read-only native pointer

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

/** Pooling functions: All of the form "output = alpha * Op(inputs) + beta * output" */
/** Function to perform forward pooling */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnPoolingForwardNative(JNIEnv *env, jclass cls, jobject handle, jobject poolingDesc, jobject alpha, jobject xDesc, jobject x, jobject beta, jobject yDesc, jobject y)
{
    // Null-checks for non-primitive arguments
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cudnnPoolingForward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (poolingDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'poolingDesc' is null for cudnnPoolingForward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (alpha == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'alpha' is null for cudnnPoolingForward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (xDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'xDesc' is null for cudnnPoolingForward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cudnnPoolingForward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (beta == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'beta' is null for cudnnPoolingForward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (yDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'yDesc' is null for cudnnPoolingForward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (y == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'y' is null for cudnnPoolingForward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnPoolingForward(handle=%p, poolingDesc=%p, alpha=%p, xDesc=%p, x=%p, beta=%p, yDesc=%p, y=%p)\n",
        handle, poolingDesc, alpha, xDesc, x, beta, yDesc, y);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnPoolingDescriptor_t poolingDesc_native;
    void* alpha_native;
    cudnnTensorDescriptor_t xDesc_native;
    void* x_native;
    void* beta_native;
    cudnnTensorDescriptor_t yDesc_native;
    void* y_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    poolingDesc_native = (cudnnPoolingDescriptor_t)getNativePointerValue(env, poolingDesc);
    PointerData *alpha_pointerData = initPointerData(env, alpha);
    if (alpha_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    alpha_native = (void*)alpha_pointerData->getPointer(env);
    xDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, xDesc);
    x_native = (void*)getPointer(env, x);
    PointerData *beta_pointerData = initPointerData(env, beta);
    if (beta_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    beta_native = (void*)beta_pointerData->getPointer(env);
    yDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, yDesc);
    y_native = (void*)getPointer(env, y);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnPoolingForward(handle_native, poolingDesc_native, alpha_native, xDesc_native, x_native, beta_native, yDesc_native, y_native);

    // Write back native variable values
    // handle is a read-only native pointer
    // poolingDesc is a read-only native pointer
    if (!releasePointerData(env, alpha_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // xDesc is a read-only native pointer
    // x is a native pointer
    if (!releasePointerData(env, beta_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // yDesc is a read-only native pointer
    // y is a native pointer

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

/** Function to perform backward pooling */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnPoolingBackwardNative(JNIEnv *env, jclass cls, jobject handle, jobject poolingDesc, jobject alpha, jobject yDesc, jobject y, jobject dyDesc, jobject dy, jobject xDesc, jobject x, jobject beta, jobject dxDesc, jobject dx)
{
    // Null-checks for non-primitive arguments
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cudnnPoolingBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (poolingDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'poolingDesc' is null for cudnnPoolingBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (alpha == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'alpha' is null for cudnnPoolingBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (yDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'yDesc' is null for cudnnPoolingBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (y == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'y' is null for cudnnPoolingBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (dyDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dyDesc' is null for cudnnPoolingBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (dy == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dy' is null for cudnnPoolingBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (xDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'xDesc' is null for cudnnPoolingBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cudnnPoolingBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (beta == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'beta' is null for cudnnPoolingBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (dxDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dxDesc' is null for cudnnPoolingBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (dx == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dx' is null for cudnnPoolingBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnPoolingBackward(handle=%p, poolingDesc=%p, alpha=%p, yDesc=%p, y=%p, dyDesc=%p, dy=%p, xDesc=%p, x=%p, beta=%p, dxDesc=%p, dx=%p)\n",
        handle, poolingDesc, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnPoolingDescriptor_t poolingDesc_native;
    void* alpha_native;
    cudnnTensorDescriptor_t yDesc_native;
    void* y_native;
    cudnnTensorDescriptor_t dyDesc_native;
    void* dy_native;
    cudnnTensorDescriptor_t xDesc_native;
    void* x_native;
    void* beta_native;
    cudnnTensorDescriptor_t dxDesc_native;
    void* dx_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    poolingDesc_native = (cudnnPoolingDescriptor_t)getNativePointerValue(env, poolingDesc);
    PointerData *alpha_pointerData = initPointerData(env, alpha);
    if (alpha_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    alpha_native = (void*)alpha_pointerData->getPointer(env);
    yDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, yDesc);
    y_native = (void*)getPointer(env, y);
    dyDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, dyDesc);
    dy_native = (void*)getPointer(env, dy);
    xDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, xDesc);
    x_native = (void*)getPointer(env, x);
    PointerData *beta_pointerData = initPointerData(env, beta);
    if (beta_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    beta_native = (void*)beta_pointerData->getPointer(env);
    dxDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, dxDesc);
    dx_native = (void*)getPointer(env, dx);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnPoolingBackward(handle_native, poolingDesc_native, alpha_native, yDesc_native, y_native, dyDesc_native, dy_native, xDesc_native, x_native, beta_native, dxDesc_native, dx_native);

    // Write back native variable values
    // handle is a read-only native pointer
    // poolingDesc is a read-only native pointer
    if (!releasePointerData(env, alpha_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // yDesc is a read-only native pointer
    // y is a native pointer
    // dyDesc is a read-only native pointer
    // dy is a native pointer
    // xDesc is a read-only native pointer
    // x is a native pointer
    if (!releasePointerData(env, beta_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // dxDesc is a read-only native pointer
    // dx is a native pointer

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

/** Activation functions: All of the form "output = alpha * Op(inputs) + beta * output" */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnCreateActivationDescriptorNative(JNIEnv *env, jclass cls, jobject activationDesc)
{
    // Null-checks for non-primitive arguments
    if (activationDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'activationDesc' is null for cudnnCreateActivationDescriptor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnCreateActivationDescriptor(activationDesc=%p)\n",
        activationDesc);

    // Native variable declarations
    cudnnActivationDescriptor_t activationDesc_native;

    // Obtain native variable values
    // activationDesc is write-only

    // Native function call
    cudnnStatus_t jniResult_native = cudnnCreateActivationDescriptor(&activationDesc_native);

    // Write back native variable values
    setNativePointerValue(env, activationDesc, (jlong)activationDesc_native);

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnSetActivationDescriptorNative(JNIEnv *env, jclass cls, jobject activationDesc, jint mode, jint reluNanOpt, jdouble reluCeiling)
{
    // Null-checks for non-primitive arguments
    if (activationDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'activationDesc' is null for cudnnSetActivationDescriptor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    // mode is primitive
    // reluNanOpt is primitive
    // reluCeiling is primitive

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnSetActivationDescriptor(activationDesc=%p, mode=%d, reluNanOpt=%d, reluCeiling=%lf)\n",
        activationDesc, mode, reluNanOpt, reluCeiling);

    // Native variable declarations
    cudnnActivationDescriptor_t activationDesc_native;
    cudnnActivationMode_t mode_native;
    cudnnNanPropagation_t reluNanOpt_native;
    double reluCeiling_native = 0.0;

    // Obtain native variable values
    activationDesc_native = (cudnnActivationDescriptor_t)getNativePointerValue(env, activationDesc);
    mode_native = (cudnnActivationMode_t)mode;
    reluNanOpt_native = (cudnnNanPropagation_t)reluNanOpt;
    reluCeiling_native = (double)reluCeiling;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnSetActivationDescriptor(activationDesc_native, mode_native, reluNanOpt_native, reluCeiling_native);

    // Write back native variable values
    // activationDesc is a read-only native pointer
    // mode is primitive
    // reluNanOpt is primitive
    // reluCeiling is primitive

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetActivationDescriptorNative(JNIEnv *env, jclass cls, jobject activationDesc, jintArray mode, jintArray reluNanOpt, jobject reluCeiling)
{
    // Null-checks for non-primitive arguments
    if (activationDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'activationDesc' is null for cudnnGetActivationDescriptor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (mode == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'mode' is null for cudnnGetActivationDescriptor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (reluNanOpt == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'reluNanOpt' is null for cudnnGetActivationDescriptor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (reluCeiling == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'reluCeiling' is null for cudnnGetActivationDescriptor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetActivationDescriptor(activationDesc=%p, mode=%p, reluNanOpt=%p, reluCeiling=%p)\n",
        activationDesc, mode, reluNanOpt, reluCeiling);

    // Native variable declarations
    cudnnActivationDescriptor_t activationDesc_native;
    cudnnActivationMode_t* mode_native;
    cudnnNanPropagation_t* reluNanOpt_native;
    double* reluCeiling_native;

    // Obtain native variable values
    activationDesc_native = (cudnnActivationDescriptor_t)getNativePointerValue(env, activationDesc);
    mode_native = (cudnnActivationMode_t*)getPointer(env, mode);
    reluNanOpt_native = (cudnnNanPropagation_t*)getPointer(env, reluNanOpt);
    reluCeiling_native = (double*)getPointer(env, reluCeiling);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetActivationDescriptor(activationDesc_native, mode_native, reluNanOpt_native, reluCeiling_native);

    // Write back native variable values
    // activationDesc is a read-only native pointer
    // mode is a native pointer
    // reluNanOpt is a native pointer
    // reluCeiling is a native pointer

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnDestroyActivationDescriptorNative(JNIEnv *env, jclass cls, jobject activationDesc)
{
    // Null-checks for non-primitive arguments
    if (activationDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'activationDesc' is null for cudnnDestroyActivationDescriptor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnDestroyActivationDescriptor(activationDesc=%p)\n",
        activationDesc);

    // Native variable declarations
    cudnnActivationDescriptor_t activationDesc_native;

    // Obtain native variable values
    activationDesc_native = (cudnnActivationDescriptor_t)getNativePointerValue(env, activationDesc);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnDestroyActivationDescriptor(activationDesc_native);

    // Write back native variable values
    // activationDesc is a read-only native pointer

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

/** Function to perform forward activation  */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnActivationForwardNative(JNIEnv *env, jclass cls, jobject handle, jint mode, jobject alpha, jobject xDesc, jobject x, jobject beta, jobject yDesc, jobject y)
{
    // Null-checks for non-primitive arguments
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cudnnActivationForward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    // mode is primitive
    if (alpha == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'alpha' is null for cudnnActivationForward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (xDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'xDesc' is null for cudnnActivationForward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cudnnActivationForward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (beta == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'beta' is null for cudnnActivationForward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (yDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'yDesc' is null for cudnnActivationForward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (y == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'y' is null for cudnnActivationForward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnActivationForward(handle=%p, mode=%d, alpha=%p, xDesc=%p, x=%p, beta=%p, yDesc=%p, y=%p)\n",
        handle, mode, alpha, xDesc, x, beta, yDesc, y);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnActivationMode_t mode_native;
    void* alpha_native;
    cudnnTensorDescriptor_t xDesc_native;
    void* x_native;
    void* beta_native;
    cudnnTensorDescriptor_t yDesc_native;
    void* y_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    mode_native = (cudnnActivationMode_t)mode;
    PointerData *alpha_pointerData = initPointerData(env, alpha);
    if (alpha_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    alpha_native = (void*)alpha_pointerData->getPointer(env);
    xDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, xDesc);
    x_native = (void*)getPointer(env, x);
    PointerData *beta_pointerData = initPointerData(env, beta);
    if (beta_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    beta_native = (void*)beta_pointerData->getPointer(env);
    yDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, yDesc);
    y_native = (void*)getPointer(env, y);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnActivationForward(handle_native, mode_native, alpha_native, xDesc_native, x_native, beta_native, yDesc_native, y_native);

    // Write back native variable values
    // handle is a read-only native pointer
    // mode is primitive
    if (!releasePointerData(env, alpha_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // xDesc is a read-only native pointer
    // x is a native pointer
    if (!releasePointerData(env, beta_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // yDesc is a read-only native pointer
    // y is a native pointer

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnActivationForward_1v4Native(JNIEnv *env, jclass cls, jobject handle, jobject activationDesc, jobject alpha, jobject xDesc, jobject x, jobject beta, jobject yDesc, jobject y)
{
    // Null-checks for non-primitive arguments
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cudnnActivationForward_v4");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (activationDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'activationDesc' is null for cudnnActivationForward_v4");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (alpha == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'alpha' is null for cudnnActivationForward_v4");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (xDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'xDesc' is null for cudnnActivationForward_v4");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cudnnActivationForward_v4");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (beta == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'beta' is null for cudnnActivationForward_v4");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (yDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'yDesc' is null for cudnnActivationForward_v4");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (y == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'y' is null for cudnnActivationForward_v4");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnActivationForward_v4(handle=%p, activationDesc=%p, alpha=%p, xDesc=%p, x=%p, beta=%p, yDesc=%p, y=%p)\n",
        handle, activationDesc, alpha, xDesc, x, beta, yDesc, y);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnActivationDescriptor_t activationDesc_native;
    void* alpha_native;
    cudnnTensorDescriptor_t xDesc_native;
    void* x_native;
    void* beta_native;
    cudnnTensorDescriptor_t yDesc_native;
    void* y_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    activationDesc_native = (cudnnActivationDescriptor_t)getNativePointerValue(env, activationDesc);
    PointerData *alpha_pointerData = initPointerData(env, alpha);
    if (alpha_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    alpha_native = (void*)alpha_pointerData->getPointer(env);
    xDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, xDesc);
    x_native = (void*)getPointer(env, x);
    PointerData *beta_pointerData = initPointerData(env, beta);
    if (beta_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    beta_native = (void*)beta_pointerData->getPointer(env);
    yDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, yDesc);
    y_native = (void*)getPointer(env, y);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnActivationForward_v4(handle_native, activationDesc_native, alpha_native, xDesc_native, x_native, beta_native, yDesc_native, y_native);

    // Write back native variable values
    // handle is a read-only native pointer
    // activationDesc is a read-only native pointer
    if (!releasePointerData(env, alpha_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // xDesc is a read-only native pointer
    // x is a native pointer
    if (!releasePointerData(env, beta_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // yDesc is a read-only native pointer
    // y is a native pointer

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

/** Function to perform backward activation  */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnActivationBackwardNative(JNIEnv *env, jclass cls, jobject handle, jint mode, jobject alpha, jobject yDesc, jobject y, jobject dyDesc, jobject dy, jobject xDesc, jobject x, jobject beta, jobject dxDesc, jobject dx)
{
    // Null-checks for non-primitive arguments
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cudnnActivationBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    // mode is primitive
    if (alpha == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'alpha' is null for cudnnActivationBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (yDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'yDesc' is null for cudnnActivationBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (y == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'y' is null for cudnnActivationBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (dyDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dyDesc' is null for cudnnActivationBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (dy == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dy' is null for cudnnActivationBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (xDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'xDesc' is null for cudnnActivationBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cudnnActivationBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (beta == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'beta' is null for cudnnActivationBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (dxDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dxDesc' is null for cudnnActivationBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (dx == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dx' is null for cudnnActivationBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnActivationBackward(handle=%p, mode=%d, alpha=%p, yDesc=%p, y=%p, dyDesc=%p, dy=%p, xDesc=%p, x=%p, beta=%p, dxDesc=%p, dx=%p)\n",
        handle, mode, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnActivationMode_t mode_native;
    void* alpha_native;
    cudnnTensorDescriptor_t yDesc_native;
    void* y_native;
    cudnnTensorDescriptor_t dyDesc_native;
    void* dy_native;
    cudnnTensorDescriptor_t xDesc_native;
    void* x_native;
    void* beta_native;
    cudnnTensorDescriptor_t dxDesc_native;
    void* dx_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    mode_native = (cudnnActivationMode_t)mode;
    PointerData *alpha_pointerData = initPointerData(env, alpha);
    if (alpha_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    alpha_native = (void*)alpha_pointerData->getPointer(env);
    yDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, yDesc);
    y_native = (void*)getPointer(env, y);
    dyDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, dyDesc);
    dy_native = (void*)getPointer(env, dy);
    xDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, xDesc);
    x_native = (void*)getPointer(env, x);
    PointerData *beta_pointerData = initPointerData(env, beta);
    if (beta_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    beta_native = (void*)beta_pointerData->getPointer(env);
    dxDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, dxDesc);
    dx_native = (void*)getPointer(env, dx);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnActivationBackward(handle_native, mode_native, alpha_native, yDesc_native, y_native, dyDesc_native, dy_native, xDesc_native, x_native, beta_native, dxDesc_native, dx_native);

    // Write back native variable values
    // handle is a read-only native pointer
    // mode is primitive
    if (!releasePointerData(env, alpha_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // yDesc is a read-only native pointer
    // y is a native pointer
    // dyDesc is a read-only native pointer
    // dy is a native pointer
    // xDesc is a read-only native pointer
    // x is a native pointer
    if (!releasePointerData(env, beta_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // dxDesc is a read-only native pointer
    // dx is a native pointer

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnActivationBackward_1v4Native(JNIEnv *env, jclass cls, jobject handle, jobject activationDesc, jobject alpha, jobject yDesc, jobject y, jobject dyDesc, jobject dy, jobject xDesc, jobject x, jobject beta, jobject dxDesc, jobject dx)
{
    // Null-checks for non-primitive arguments
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cudnnActivationBackward_v4");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (activationDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'activationDesc' is null for cudnnActivationBackward_v4");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (alpha == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'alpha' is null for cudnnActivationBackward_v4");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (yDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'yDesc' is null for cudnnActivationBackward_v4");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (y == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'y' is null for cudnnActivationBackward_v4");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (dyDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dyDesc' is null for cudnnActivationBackward_v4");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (dy == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dy' is null for cudnnActivationBackward_v4");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (xDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'xDesc' is null for cudnnActivationBackward_v4");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cudnnActivationBackward_v4");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (beta == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'beta' is null for cudnnActivationBackward_v4");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (dxDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dxDesc' is null for cudnnActivationBackward_v4");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (dx == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dx' is null for cudnnActivationBackward_v4");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnActivationBackward_v4(handle=%p, activationDesc=%p, alpha=%p, yDesc=%p, y=%p, dyDesc=%p, dy=%p, xDesc=%p, x=%p, beta=%p, dxDesc=%p, dx=%p)\n",
        handle, activationDesc, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnActivationDescriptor_t activationDesc_native;
    void* alpha_native;
    cudnnTensorDescriptor_t yDesc_native;
    void* y_native;
    cudnnTensorDescriptor_t dyDesc_native;
    void* dy_native;
    cudnnTensorDescriptor_t xDesc_native;
    void* x_native;
    void* beta_native;
    cudnnTensorDescriptor_t dxDesc_native;
    void* dx_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    activationDesc_native = (cudnnActivationDescriptor_t)getNativePointerValue(env, activationDesc);
    PointerData *alpha_pointerData = initPointerData(env, alpha);
    if (alpha_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    alpha_native = (void*)alpha_pointerData->getPointer(env);
    yDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, yDesc);
    y_native = (void*)getPointer(env, y);
    dyDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, dyDesc);
    dy_native = (void*)getPointer(env, dy);
    xDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, xDesc);
    x_native = (void*)getPointer(env, x);
    PointerData *beta_pointerData = initPointerData(env, beta);
    if (beta_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    beta_native = (void*)beta_pointerData->getPointer(env);
    dxDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, dxDesc);
    dx_native = (void*)getPointer(env, dx);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnActivationBackward_v4(handle_native, activationDesc_native, alpha_native, yDesc_native, y_native, dyDesc_native, dy_native, xDesc_native, x_native, beta_native, dxDesc_native, dx_native);

    // Write back native variable values
    // handle is a read-only native pointer
    // activationDesc is a read-only native pointer
    if (!releasePointerData(env, alpha_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // yDesc is a read-only native pointer
    // y is a native pointer
    // dyDesc is a read-only native pointer
    // dy is a native pointer
    // xDesc is a read-only native pointer
    // x is a native pointer
    if (!releasePointerData(env, beta_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // dxDesc is a read-only native pointer
    // dx is a native pointer

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

/**
* <pre>
* Create an instance of LRN (Local Response Normalization) descriptor
* Uses lrnN=5, lrnAlpha=1e-4, lrnBeta=0.75, lrnK=2.0 as defaults from Krizhevsky'12 ImageNet paper
* </pre>
*/
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnCreateLRNDescriptorNative(JNIEnv *env, jclass cls, jobject normDesc)
{
    // Null-checks for non-primitive arguments
    if (normDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'normDesc' is null for cudnnCreateLRNDescriptor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnCreateLRNDescriptor(normDesc=%p)\n",
        normDesc);

    // Native variable declarations
    cudnnLRNDescriptor_t normDesc_native;

    // Obtain native variable values
    // normDesc is write-only

    // Native function call
    cudnnStatus_t jniResult_native = cudnnCreateLRNDescriptor(&normDesc_native);

    // Write back native variable values
    setNativePointerValue(env, normDesc, (jlong)normDesc_native);

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

/**
* <pre>
* Uses a window [center-lookBehind, center+lookAhead], where
* lookBehind = floor( (lrnN-1)/2 ), lookAhead = lrnN-lookBehind-1.
* Values of double parameters cast to tensor data type.
* </pre>
*/
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnSetLRNDescriptorNative(JNIEnv *env, jclass cls, jobject normDesc, jint lrnN, jdouble lrnAlpha, jdouble lrnBeta, jdouble lrnK)
{
    // Null-checks for non-primitive arguments
    if (normDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'normDesc' is null for cudnnSetLRNDescriptor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    // lrnN is primitive
    // lrnAlpha is primitive
    // lrnBeta is primitive
    // lrnK is primitive

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnSetLRNDescriptor(normDesc=%p, lrnN=%d, lrnAlpha=%lf, lrnBeta=%lf, lrnK=%lf)\n",
        normDesc, lrnN, lrnAlpha, lrnBeta, lrnK);

    // Native variable declarations
    cudnnLRNDescriptor_t normDesc_native;
    unsigned int lrnN_native = 0;
    double lrnAlpha_native = 0.0;
    double lrnBeta_native = 0.0;
    double lrnK_native = 0.0;

    // Obtain native variable values
    normDesc_native = (cudnnLRNDescriptor_t)getNativePointerValue(env, normDesc);
    lrnN_native = (unsigned int)lrnN;
    lrnAlpha_native = (double)lrnAlpha;
    lrnBeta_native = (double)lrnBeta;
    lrnK_native = (double)lrnK;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnSetLRNDescriptor(normDesc_native, lrnN_native, lrnAlpha_native, lrnBeta_native, lrnK_native);

    // Write back native variable values
    // normDesc is a read-only native pointer
    // lrnN is primitive
    // lrnAlpha is primitive
    // lrnBeta is primitive
    // lrnK is primitive

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

/**
* <pre>
* Retrieve the settings currently stored in an LRN layer descriptor
* Any of the provided pointers can be NULL (no corresponding value will be returned)
* </pre>
*/
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetLRNDescriptorNative(JNIEnv *env, jclass cls, jobject normDesc, jintArray lrnN, jobject lrnAlpha, jobject lrnBeta, jobject lrnK)
{
    // Null-checks for non-primitive arguments
    if (normDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'normDesc' is null for cudnnGetLRNDescriptor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (lrnN == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'lrnN' is null for cudnnGetLRNDescriptor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    // lrnAlpha may be NULL
    // lrnBeta may be NULL
    // lrnK may be NULL

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetLRNDescriptor(normDesc=%p, lrnN=%p, lrnAlpha=%p, lrnBeta=%p, lrnK=%p)\n",
        normDesc, lrnN, lrnAlpha, lrnBeta, lrnK);

    // Native variable declarations
    cudnnLRNDescriptor_t normDesc_native;
    unsigned int* lrnN_native;
    double* lrnAlpha_native;
    double* lrnBeta_native;
    double* lrnK_native;

    // Obtain native variable values
    normDesc_native = (cudnnLRNDescriptor_t)getNativePointerValue(env, normDesc);
    lrnN_native = (unsigned int*)getPointer(env, lrnN);
    if (lrnAlpha != NULL)
    {
        lrnAlpha_native = (double*)getNativePointerValue(env, lrnAlpha);
    }
    if (lrnBeta != NULL)
    {
        lrnBeta_native = (double*)getNativePointerValue(env, lrnBeta);
    }
    if (lrnK != NULL)
    {
        lrnK_native = (double*)getNativePointerValue(env, lrnK);
    }

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetLRNDescriptor(normDesc_native, lrnN_native, lrnAlpha_native, lrnBeta_native, lrnK_native);

    // Write back native variable values
    // normDesc is a read-only native pointer
    // lrnN is a native pointer
    // lrnAlpha is a read-only native pointer
    // lrnBeta is a read-only native pointer
    // lrnK is a read-only native pointer

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

/** Destroy an instance of LRN descriptor */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnDestroyLRNDescriptorNative(JNIEnv *env, jclass cls, jobject lrnDesc)
{
    // Null-checks for non-primitive arguments
    if (lrnDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'lrnDesc' is null for cudnnDestroyLRNDescriptor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnDestroyLRNDescriptor(lrnDesc=%p)\n",
        lrnDesc);

    // Native variable declarations
    cudnnLRNDescriptor_t lrnDesc_native;

    // Obtain native variable values
    lrnDesc_native = (cudnnLRNDescriptor_t)getNativePointerValue(env, lrnDesc);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnDestroyLRNDescriptor(lrnDesc_native);

    // Write back native variable values
    // lrnDesc is a read-only native pointer

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

/** LRN functions: output = alpha * normalize(x) + beta * old_y */
/** LRN cross-channel forward computation. Double parameters cast to tensor data type */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnLRNCrossChannelForwardNative(JNIEnv *env, jclass cls, jobject handle, jobject normDesc, jint lrnMode, jobject alpha, jobject xDesc, jobject x, jobject beta, jobject yDesc, jobject y)
{
    // Null-checks for non-primitive arguments
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cudnnLRNCrossChannelForward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (normDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'normDesc' is null for cudnnLRNCrossChannelForward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    // lrnMode is primitive
    if (alpha == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'alpha' is null for cudnnLRNCrossChannelForward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (xDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'xDesc' is null for cudnnLRNCrossChannelForward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cudnnLRNCrossChannelForward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (beta == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'beta' is null for cudnnLRNCrossChannelForward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (yDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'yDesc' is null for cudnnLRNCrossChannelForward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (y == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'y' is null for cudnnLRNCrossChannelForward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnLRNCrossChannelForward(handle=%p, normDesc=%p, lrnMode=%d, alpha=%p, xDesc=%p, x=%p, beta=%p, yDesc=%p, y=%p)\n",
        handle, normDesc, lrnMode, alpha, xDesc, x, beta, yDesc, y);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnLRNDescriptor_t normDesc_native;
    cudnnLRNMode_t lrnMode_native;
    void* alpha_native;
    cudnnTensorDescriptor_t xDesc_native;
    void* x_native;
    void* beta_native;
    cudnnTensorDescriptor_t yDesc_native;
    void* y_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    normDesc_native = (cudnnLRNDescriptor_t)getNativePointerValue(env, normDesc);
    lrnMode_native = (cudnnLRNMode_t)lrnMode;
    PointerData *alpha_pointerData = initPointerData(env, alpha);
    if (alpha_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    alpha_native = (void*)alpha_pointerData->getPointer(env);
    xDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, xDesc);
    x_native = (void*)getPointer(env, x);
    PointerData *beta_pointerData = initPointerData(env, beta);
    if (beta_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    beta_native = (void*)beta_pointerData->getPointer(env);
    yDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, yDesc);
    y_native = (void*)getPointer(env, y);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnLRNCrossChannelForward(handle_native, normDesc_native, lrnMode_native, alpha_native, xDesc_native, x_native, beta_native, yDesc_native, y_native);

    // Write back native variable values
    // handle is a read-only native pointer
    // normDesc is a read-only native pointer
    // lrnMode is primitive
    if (!releasePointerData(env, alpha_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // xDesc is a read-only native pointer
    // x is a native pointer
    if (!releasePointerData(env, beta_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // yDesc is a read-only native pointer
    // y is a native pointer

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

/** LRN cross-channel backward computation. Double parameters cast to tensor data type */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnLRNCrossChannelBackwardNative(JNIEnv *env, jclass cls, jobject handle, jobject normDesc, jint lrnMode, jobject alpha, jobject yDesc, jobject y, jobject dyDesc, jobject dy, jobject xDesc, jobject x, jobject beta, jobject dxDesc, jobject dx)
{
    // Null-checks for non-primitive arguments
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cudnnLRNCrossChannelBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (normDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'normDesc' is null for cudnnLRNCrossChannelBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    // lrnMode is primitive
    if (alpha == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'alpha' is null for cudnnLRNCrossChannelBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (yDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'yDesc' is null for cudnnLRNCrossChannelBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (y == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'y' is null for cudnnLRNCrossChannelBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (dyDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dyDesc' is null for cudnnLRNCrossChannelBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (dy == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dy' is null for cudnnLRNCrossChannelBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (xDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'xDesc' is null for cudnnLRNCrossChannelBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cudnnLRNCrossChannelBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (beta == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'beta' is null for cudnnLRNCrossChannelBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (dxDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dxDesc' is null for cudnnLRNCrossChannelBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (dx == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dx' is null for cudnnLRNCrossChannelBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnLRNCrossChannelBackward(handle=%p, normDesc=%p, lrnMode=%d, alpha=%p, yDesc=%p, y=%p, dyDesc=%p, dy=%p, xDesc=%p, x=%p, beta=%p, dxDesc=%p, dx=%p)\n",
        handle, normDesc, lrnMode, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnLRNDescriptor_t normDesc_native;
    cudnnLRNMode_t lrnMode_native;
    void* alpha_native;
    cudnnTensorDescriptor_t yDesc_native;
    void* y_native;
    cudnnTensorDescriptor_t dyDesc_native;
    void* dy_native;
    cudnnTensorDescriptor_t xDesc_native;
    void* x_native;
    void* beta_native;
    cudnnTensorDescriptor_t dxDesc_native;
    void* dx_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    normDesc_native = (cudnnLRNDescriptor_t)getNativePointerValue(env, normDesc);
    lrnMode_native = (cudnnLRNMode_t)lrnMode;
    PointerData *alpha_pointerData = initPointerData(env, alpha);
    if (alpha_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    alpha_native = (void*)alpha_pointerData->getPointer(env);
    yDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, yDesc);
    y_native = (void*)getPointer(env, y);
    dyDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, dyDesc);
    dy_native = (void*)getPointer(env, dy);
    xDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, xDesc);
    x_native = (void*)getPointer(env, x);
    PointerData *beta_pointerData = initPointerData(env, beta);
    if (beta_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    beta_native = (void*)beta_pointerData->getPointer(env);
    dxDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, dxDesc);
    dx_native = (void*)getPointer(env, dx);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnLRNCrossChannelBackward(handle_native, normDesc_native, lrnMode_native, alpha_native, yDesc_native, y_native, dyDesc_native, dy_native, xDesc_native, x_native, beta_native, dxDesc_native, dx_native);

    // Write back native variable values
    // handle is a read-only native pointer
    // normDesc is a read-only native pointer
    // lrnMode is primitive
    if (!releasePointerData(env, alpha_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // yDesc is a read-only native pointer
    // y is a native pointer
    // dyDesc is a read-only native pointer
    // dy is a native pointer
    // xDesc is a read-only native pointer
    // x is a native pointer
    if (!releasePointerData(env, beta_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // dxDesc is a read-only native pointer
    // dx is a native pointer

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

/** LCN/divisive normalization functions: y = alpha * normalize(x) + beta * y */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnDivisiveNormalizationForwardNative(JNIEnv *env, jclass cls, jobject handle, jobject normDesc, jint mode, jobject alpha, jobject xDesc, jobject x, jobject means, jobject temp, jobject temp2, jobject beta, jobject yDesc, jobject y)
{
    // Null-checks for non-primitive arguments
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cudnnDivisiveNormalizationForward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (normDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'normDesc' is null for cudnnDivisiveNormalizationForward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    // mode is primitive
    if (alpha == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'alpha' is null for cudnnDivisiveNormalizationForward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (xDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'xDesc' is null for cudnnDivisiveNormalizationForward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cudnnDivisiveNormalizationForward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (means == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'means' is null for cudnnDivisiveNormalizationForward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (temp == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'temp' is null for cudnnDivisiveNormalizationForward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (temp2 == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'temp2' is null for cudnnDivisiveNormalizationForward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (beta == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'beta' is null for cudnnDivisiveNormalizationForward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (yDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'yDesc' is null for cudnnDivisiveNormalizationForward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (y == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'y' is null for cudnnDivisiveNormalizationForward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnDivisiveNormalizationForward(handle=%p, normDesc=%p, mode=%d, alpha=%p, xDesc=%p, x=%p, means=%p, temp=%p, temp2=%p, beta=%p, yDesc=%p, y=%p)\n",
        handle, normDesc, mode, alpha, xDesc, x, means, temp, temp2, beta, yDesc, y);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnLRNDescriptor_t normDesc_native;
    cudnnDivNormMode_t mode_native;
    void* alpha_native;
    cudnnTensorDescriptor_t xDesc_native;
    void* x_native;
    void* means_native;
    void* temp_native;
    void* temp2_native;
    void* beta_native;
    cudnnTensorDescriptor_t yDesc_native;
    void* y_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    normDesc_native = (cudnnLRNDescriptor_t)getNativePointerValue(env, normDesc);
    mode_native = (cudnnDivNormMode_t)mode;
    PointerData *alpha_pointerData = initPointerData(env, alpha);
    if (alpha_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    alpha_native = (void*)alpha_pointerData->getPointer(env);
    xDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, xDesc);
    x_native = (void*)getPointer(env, x);
    means_native = (void*)getPointer(env, means);
    temp_native = (void*)getPointer(env, temp);
    temp2_native = (void*)getPointer(env, temp2);
    PointerData *beta_pointerData = initPointerData(env, beta);
    if (beta_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    beta_native = (void*)beta_pointerData->getPointer(env);
    yDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, yDesc);
    y_native = (void*)getPointer(env, y);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnDivisiveNormalizationForward(handle_native, normDesc_native, mode_native, alpha_native, xDesc_native, x_native, means_native, temp_native, temp2_native, beta_native, yDesc_native, y_native);

    // Write back native variable values
    // handle is a read-only native pointer
    // normDesc is a read-only native pointer
    // mode is primitive
    if (!releasePointerData(env, alpha_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // xDesc is a read-only native pointer
    // x is a native pointer
    // means is a native pointer
    // temp is a native pointer
    // temp2 is a native pointer
    if (!releasePointerData(env, beta_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // yDesc is a read-only native pointer
    // y is a native pointer

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnDivisiveNormalizationBackwardNative(JNIEnv *env, jclass cls, jobject handle, jobject normDesc, jint mode, jobject alpha, jobject xDesc, jobject x, jobject means, jobject dy, jobject temp, jobject temp2, jobject beta, jobject dXdMeansDesc, jobject dx, jobject dMeans)
{
    // Null-checks for non-primitive arguments
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cudnnDivisiveNormalizationBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (normDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'normDesc' is null for cudnnDivisiveNormalizationBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    // mode is primitive
    if (alpha == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'alpha' is null for cudnnDivisiveNormalizationBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (xDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'xDesc' is null for cudnnDivisiveNormalizationBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cudnnDivisiveNormalizationBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (means == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'means' is null for cudnnDivisiveNormalizationBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (dy == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dy' is null for cudnnDivisiveNormalizationBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (temp == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'temp' is null for cudnnDivisiveNormalizationBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (temp2 == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'temp2' is null for cudnnDivisiveNormalizationBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (beta == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'beta' is null for cudnnDivisiveNormalizationBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (dXdMeansDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dXdMeansDesc' is null for cudnnDivisiveNormalizationBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (dx == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dx' is null for cudnnDivisiveNormalizationBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (dMeans == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dMeans' is null for cudnnDivisiveNormalizationBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnDivisiveNormalizationBackward(handle=%p, normDesc=%p, mode=%d, alpha=%p, xDesc=%p, x=%p, means=%p, dy=%p, temp=%p, temp2=%p, beta=%p, dXdMeansDesc=%p, dx=%p, dMeans=%p)\n",
        handle, normDesc, mode, alpha, xDesc, x, means, dy, temp, temp2, beta, dXdMeansDesc, dx, dMeans);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnLRNDescriptor_t normDesc_native;
    cudnnDivNormMode_t mode_native;
    void* alpha_native;
    cudnnTensorDescriptor_t xDesc_native;
    void* x_native;
    void* means_native;
    void* dy_native;
    void* temp_native;
    void* temp2_native;
    void* beta_native;
    cudnnTensorDescriptor_t dXdMeansDesc_native;
    void* dx_native;
    void* dMeans_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    normDesc_native = (cudnnLRNDescriptor_t)getNativePointerValue(env, normDesc);
    mode_native = (cudnnDivNormMode_t)mode;
    PointerData *alpha_pointerData = initPointerData(env, alpha);
    if (alpha_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    alpha_native = (void*)alpha_pointerData->getPointer(env);
    xDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, xDesc);
    x_native = (void*)getPointer(env, x);
    means_native = (void*)getPointer(env, means);
    dy_native = (void*)getPointer(env, dy);
    temp_native = (void*)getPointer(env, temp);
    temp2_native = (void*)getPointer(env, temp2);
    PointerData *beta_pointerData = initPointerData(env, beta);
    if (beta_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    beta_native = (void*)beta_pointerData->getPointer(env);
    dXdMeansDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, dXdMeansDesc);
    dx_native = (void*)getPointer(env, dx);
    dMeans_native = (void*)getPointer(env, dMeans);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnDivisiveNormalizationBackward(handle_native, normDesc_native, mode_native, alpha_native, xDesc_native, x_native, means_native, dy_native, temp_native, temp2_native, beta_native, dXdMeansDesc_native, dx_native, dMeans_native);

    // Write back native variable values
    // handle is a read-only native pointer
    // normDesc is a read-only native pointer
    // mode is primitive
    if (!releasePointerData(env, alpha_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // xDesc is a read-only native pointer
    // x is a native pointer
    // means is a native pointer
    // dy is a native pointer
    // temp is a native pointer
    // temp2 is a native pointer
    if (!releasePointerData(env, beta_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // dXdMeansDesc is a read-only native pointer
    // dx is a native pointer
    // dMeans is a native pointer

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

/**
* <pre>
* Derives a tensor descriptor from layer data descriptor for BatchNormalization
* scale, invVariance, bnBias, bnScale tensors. Use this tensor desc for
* bnScaleBiasMeanVarDesc and bnScaleBiasDiffDesc in Batch Normalization forward and backward functions.
* </pre>
*/
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnDeriveBNTensorDescriptorNative(JNIEnv *env, jclass cls, jobject derivedBnDesc, jobject xDesc, jint mode)
{
    // Null-checks for non-primitive arguments
    if (derivedBnDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'derivedBnDesc' is null for cudnnDeriveBNTensorDescriptor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (xDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'xDesc' is null for cudnnDeriveBNTensorDescriptor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    // mode is primitive

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnDeriveBNTensorDescriptor(derivedBnDesc=%p, xDesc=%p, mode=%d)\n",
        derivedBnDesc, xDesc, mode);

    // Native variable declarations
    cudnnTensorDescriptor_t derivedBnDesc_native;
    cudnnTensorDescriptor_t xDesc_native;
    cudnnBatchNormMode_t mode_native;

    // Obtain native variable values
    derivedBnDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, derivedBnDesc);
    xDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, xDesc);
    mode_native = (cudnnBatchNormMode_t)mode;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnDeriveBNTensorDescriptor(derivedBnDesc_native, xDesc_native, mode_native);

    // Write back native variable values
    // derivedBnDesc is a read-only native pointer
    // xDesc is a read-only native pointer
    // mode is primitive

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

/** Computes y = BN(x). Also accumulates moving averages of mean and inverse variances */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnBatchNormalizationForwardTrainingNative(JNIEnv *env, jclass cls, jobject handle, jint mode, jobject alpha, jobject beta, jobject xDesc, jobject x, jobject yDesc, jobject y, jobject bnScaleBiasMeanVarDesc, jobject bnScale, jobject bnBias, jdouble exponentialAverageFactor, jobject resultRunningMean, jobject resultRunningInvVariance, jdouble epsilon, jobject resultSaveMean, jobject resultSaveInvVariance)
{
    // Null-checks for non-primitive arguments
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cudnnBatchNormalizationForwardTraining");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    // mode is primitive
    if (alpha == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'alpha' is null for cudnnBatchNormalizationForwardTraining");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (beta == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'beta' is null for cudnnBatchNormalizationForwardTraining");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (xDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'xDesc' is null for cudnnBatchNormalizationForwardTraining");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cudnnBatchNormalizationForwardTraining");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (yDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'yDesc' is null for cudnnBatchNormalizationForwardTraining");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (y == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'y' is null for cudnnBatchNormalizationForwardTraining");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (bnScaleBiasMeanVarDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'bnScaleBiasMeanVarDesc' is null for cudnnBatchNormalizationForwardTraining");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (bnScale == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'bnScale' is null for cudnnBatchNormalizationForwardTraining");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (bnBias == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'bnBias' is null for cudnnBatchNormalizationForwardTraining");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    // exponentialAverageFactor is primitive
    if (resultRunningMean == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'resultRunningMean' is null for cudnnBatchNormalizationForwardTraining");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (resultRunningInvVariance == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'resultRunningInvVariance' is null for cudnnBatchNormalizationForwardTraining");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    // epsilon is primitive
    if (resultSaveMean == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'resultSaveMean' is null for cudnnBatchNormalizationForwardTraining");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (resultSaveInvVariance == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'resultSaveInvVariance' is null for cudnnBatchNormalizationForwardTraining");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnBatchNormalizationForwardTraining(handle=%p, mode=%d, alpha=%p, beta=%p, xDesc=%p, x=%p, yDesc=%p, y=%p, bnScaleBiasMeanVarDesc=%p, bnScale=%p, bnBias=%p, exponentialAverageFactor=%lf, resultRunningMean=%p, resultRunningInvVariance=%p, epsilon=%lf, resultSaveMean=%p, resultSaveInvVariance=%p)\n",
        handle, mode, alpha, beta, xDesc, x, yDesc, y, bnScaleBiasMeanVarDesc, bnScale, bnBias, exponentialAverageFactor, resultRunningMean, resultRunningInvVariance, epsilon, resultSaveMean, resultSaveInvVariance);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnBatchNormMode_t mode_native;
    void* alpha_native;
    void* beta_native;
    cudnnTensorDescriptor_t xDesc_native;
    void* x_native;
    cudnnTensorDescriptor_t yDesc_native;
    void* y_native;
    cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc_native;
    void* bnScale_native;
    void* bnBias_native;
    double exponentialAverageFactor_native = 0.0;
    void* resultRunningMean_native;
    void* resultRunningInvVariance_native;
    double epsilon_native = 0.0;
    void* resultSaveMean_native;
    void* resultSaveInvVariance_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    mode_native = (cudnnBatchNormMode_t)mode;
    PointerData *alpha_pointerData = initPointerData(env, alpha);
    if (alpha_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    alpha_native = (void*)alpha_pointerData->getPointer(env);
    PointerData *beta_pointerData = initPointerData(env, beta);
    if (beta_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    beta_native = (void*)beta_pointerData->getPointer(env);
    xDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, xDesc);
    x_native = (void*)getPointer(env, x);
    yDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, yDesc);
    y_native = (void*)getPointer(env, y);
    bnScaleBiasMeanVarDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, bnScaleBiasMeanVarDesc);
    bnScale_native = (void*)getPointer(env, bnScale);
    bnBias_native = (void*)getPointer(env, bnBias);
    exponentialAverageFactor_native = (double)exponentialAverageFactor;
    resultRunningMean_native = (void*)getPointer(env, resultRunningMean);
    resultRunningInvVariance_native = (void*)getPointer(env, resultRunningInvVariance);
    epsilon_native = (double)epsilon;
    resultSaveMean_native = (void*)getPointer(env, resultSaveMean);
    resultSaveInvVariance_native = (void*)getPointer(env, resultSaveInvVariance);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnBatchNormalizationForwardTraining(handle_native, mode_native, alpha_native, beta_native, xDesc_native, x_native, yDesc_native, y_native, bnScaleBiasMeanVarDesc_native, bnScale_native, bnBias_native, exponentialAverageFactor_native, resultRunningMean_native, resultRunningInvVariance_native, epsilon_native, resultSaveMean_native, resultSaveInvVariance_native);

    // Write back native variable values
    // handle is a read-only native pointer
    // mode is primitive
    if (!releasePointerData(env, alpha_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!releasePointerData(env, beta_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // xDesc is a read-only native pointer
    // x is a native pointer
    // yDesc is a read-only native pointer
    // y is a native pointer
    // bnScaleBiasMeanVarDesc is a read-only native pointer
    // bnScale is a native pointer
    // bnBias is a native pointer
    // exponentialAverageFactor is primitive
    // resultRunningMean is a native pointer
    // resultRunningInvVariance is a native pointer
    // epsilon is primitive
    // resultSaveMean is a native pointer
    // resultSaveInvVariance is a native pointer

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

/**
* <pre>
* Performs Batch Normalization during Inference:
* y[i] = bnScale[k]*(x[i]-estimatedMean[k])*estimatedInvVariance[k] + bnBias[k]
* with bnScale, bnBias, runningMean, runningInvVariance tensors indexed
* according to spatial or per-activation mode. Refer to cudnnBatchNormalizationForwardTraining
* above for notes on function arguments.
* </pre>
*/
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnBatchNormalizationForwardInferenceNative(JNIEnv *env, jclass cls, jobject handle, jint mode, jobject alpha, jobject beta, jobject xDesc, jobject x, jobject yDesc, jobject y, jobject bnScaleBiasMeanVarDesc, jobject bnScale, jobject bnBias, jobject estimatedMean, jobject estimatedInvVariance, jdouble epsilon)
{
    // Null-checks for non-primitive arguments
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cudnnBatchNormalizationForwardInference");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    // mode is primitive
    if (alpha == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'alpha' is null for cudnnBatchNormalizationForwardInference");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (beta == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'beta' is null for cudnnBatchNormalizationForwardInference");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (xDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'xDesc' is null for cudnnBatchNormalizationForwardInference");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cudnnBatchNormalizationForwardInference");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (yDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'yDesc' is null for cudnnBatchNormalizationForwardInference");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (y == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'y' is null for cudnnBatchNormalizationForwardInference");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (bnScaleBiasMeanVarDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'bnScaleBiasMeanVarDesc' is null for cudnnBatchNormalizationForwardInference");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (bnScale == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'bnScale' is null for cudnnBatchNormalizationForwardInference");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (bnBias == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'bnBias' is null for cudnnBatchNormalizationForwardInference");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (estimatedMean == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'estimatedMean' is null for cudnnBatchNormalizationForwardInference");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (estimatedInvVariance == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'estimatedInvVariance' is null for cudnnBatchNormalizationForwardInference");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    // epsilon is primitive

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnBatchNormalizationForwardInference(handle=%p, mode=%d, alpha=%p, beta=%p, xDesc=%p, x=%p, yDesc=%p, y=%p, bnScaleBiasMeanVarDesc=%p, bnScale=%p, bnBias=%p, estimatedMean=%p, estimatedInvVariance=%p, epsilon=%lf)\n",
        handle, mode, alpha, beta, xDesc, x, yDesc, y, bnScaleBiasMeanVarDesc, bnScale, bnBias, estimatedMean, estimatedInvVariance, epsilon);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnBatchNormMode_t mode_native;
    void* alpha_native;
    void* beta_native;
    cudnnTensorDescriptor_t xDesc_native;
    void* x_native;
    cudnnTensorDescriptor_t yDesc_native;
    void* y_native;
    cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc_native;
    void* bnScale_native;
    void* bnBias_native;
    void* estimatedMean_native;
    void* estimatedInvVariance_native;
    double epsilon_native = 0.0;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    mode_native = (cudnnBatchNormMode_t)mode;
    PointerData *alpha_pointerData = initPointerData(env, alpha);
    if (alpha_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    alpha_native = (void*)alpha_pointerData->getPointer(env);
    PointerData *beta_pointerData = initPointerData(env, beta);
    if (beta_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    beta_native = (void*)beta_pointerData->getPointer(env);
    xDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, xDesc);
    x_native = (void*)getPointer(env, x);
    yDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, yDesc);
    y_native = (void*)getPointer(env, y);
    bnScaleBiasMeanVarDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, bnScaleBiasMeanVarDesc);
    bnScale_native = (void*)getPointer(env, bnScale);
    bnBias_native = (void*)getPointer(env, bnBias);
    estimatedMean_native = (void*)getPointer(env, estimatedMean);
    estimatedInvVariance_native = (void*)getPointer(env, estimatedInvVariance);
    epsilon_native = (double)epsilon;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnBatchNormalizationForwardInference(handle_native, mode_native, alpha_native, beta_native, xDesc_native, x_native, yDesc_native, y_native, bnScaleBiasMeanVarDesc_native, bnScale_native, bnBias_native, estimatedMean_native, estimatedInvVariance_native, epsilon_native);

    // Write back native variable values
    // handle is a read-only native pointer
    // mode is primitive
    if (!releasePointerData(env, alpha_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!releasePointerData(env, beta_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // xDesc is a read-only native pointer
    // x is a native pointer
    // yDesc is a read-only native pointer
    // y is a native pointer
    // bnScaleBiasMeanVarDesc is a read-only native pointer
    // bnScale is a native pointer
    // bnBias is a native pointer
    // estimatedMean is a native pointer
    // estimatedInvVariance is a native pointer
    // epsilon is primitive

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

/** Performs backward pass of Batch Normalization layer. Returns x gradient,
* bnScale gradient and bnBias gradient */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnBatchNormalizationBackwardNative(JNIEnv *env, jclass cls, jobject handle, jint mode, jobject alpha, jobject beta, jobject xDesc, jobject x, jobject dyDesc, jobject dy, jobject dxDesc, jobject dx, jobject dBnScaleBiasDesc, jobject bnScale, jobject dBnScaleResult, jobject dBnBiasResult, jdouble epsilon, jobject savedMean, jobject savedInvVariance)
{
    // Null-checks for non-primitive arguments
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cudnnBatchNormalizationBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    // mode is primitive
    if (alpha == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'alpha' is null for cudnnBatchNormalizationBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (beta == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'beta' is null for cudnnBatchNormalizationBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (xDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'xDesc' is null for cudnnBatchNormalizationBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cudnnBatchNormalizationBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (dyDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dyDesc' is null for cudnnBatchNormalizationBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (dy == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dy' is null for cudnnBatchNormalizationBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (dxDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dxDesc' is null for cudnnBatchNormalizationBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (dx == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dx' is null for cudnnBatchNormalizationBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (dBnScaleBiasDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dBnScaleBiasDesc' is null for cudnnBatchNormalizationBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (bnScale == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'bnScale' is null for cudnnBatchNormalizationBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (dBnScaleResult == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dBnScaleResult' is null for cudnnBatchNormalizationBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (dBnBiasResult == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dBnBiasResult' is null for cudnnBatchNormalizationBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    // epsilon is primitive
    if (savedMean == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'savedMean' is null for cudnnBatchNormalizationBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (savedInvVariance == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'savedInvVariance' is null for cudnnBatchNormalizationBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnBatchNormalizationBackward(handle=%p, mode=%d, alpha=%p, beta=%p, xDesc=%p, x=%p, dyDesc=%p, dy=%p, dxDesc=%p, dx=%p, dBnScaleBiasDesc=%p, bnScale=%p, dBnScaleResult=%p, dBnBiasResult=%p, epsilon=%lf, savedMean=%p, savedInvVariance=%p)\n",
        handle, mode, alpha, beta, xDesc, x, dyDesc, dy, dxDesc, dx, dBnScaleBiasDesc, bnScale, dBnScaleResult, dBnBiasResult, epsilon, savedMean, savedInvVariance);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnBatchNormMode_t mode_native;
    void* alpha_native;
    void* beta_native;
    cudnnTensorDescriptor_t xDesc_native;
    void* x_native;
    cudnnTensorDescriptor_t dyDesc_native;
    void* dy_native;
    cudnnTensorDescriptor_t dxDesc_native;
    void* dx_native;
    cudnnTensorDescriptor_t dBnScaleBiasDesc_native;
    void* bnScale_native;
    void* dBnScaleResult_native;
    void* dBnBiasResult_native;
    double epsilon_native = 0.0;
    void* savedMean_native;
    void* savedInvVariance_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    mode_native = (cudnnBatchNormMode_t)mode;
    PointerData *alpha_pointerData = initPointerData(env, alpha);
    if (alpha_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    alpha_native = (void*)alpha_pointerData->getPointer(env);
    PointerData *beta_pointerData = initPointerData(env, beta);
    if (beta_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    beta_native = (void*)beta_pointerData->getPointer(env);
    xDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, xDesc);
    x_native = (void*)getPointer(env, x);
    dyDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, dyDesc);
    dy_native = (void*)getPointer(env, dy);
    dxDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, dxDesc);
    dx_native = (void*)getPointer(env, dx);
    dBnScaleBiasDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, dBnScaleBiasDesc);
    bnScale_native = (void*)getPointer(env, bnScale);
    dBnScaleResult_native = (void*)getPointer(env, dBnScaleResult);
    dBnBiasResult_native = (void*)getPointer(env, dBnBiasResult);
    epsilon_native = (double)epsilon;
    savedMean_native = (void*)getPointer(env, savedMean);
    savedInvVariance_native = (void*)getPointer(env, savedInvVariance);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnBatchNormalizationBackward(handle_native, mode_native, alpha_native, beta_native, xDesc_native, x_native, dyDesc_native, dy_native, dxDesc_native, dx_native, dBnScaleBiasDesc_native, bnScale_native, dBnScaleResult_native, dBnBiasResult_native, epsilon_native, savedMean_native, savedInvVariance_native);

    // Write back native variable values
    // handle is a read-only native pointer
    // mode is primitive
    if (!releasePointerData(env, alpha_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!releasePointerData(env, beta_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // xDesc is a read-only native pointer
    // x is a native pointer
    // dyDesc is a read-only native pointer
    // dy is a native pointer
    // dxDesc is a read-only native pointer
    // dx is a native pointer
    // dBnScaleBiasDesc is a read-only native pointer
    // bnScale is a native pointer
    // dBnScaleResult is a native pointer
    // dBnBiasResult is a native pointer
    // epsilon is primitive
    // savedMean is a native pointer
    // savedInvVariance is a native pointer

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

/** DEPRECATED API THAT WILL BE REMOVED FROM UPCOMING RELEASES */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnSetConvolutionNdDescriptor_1v2Native(JNIEnv *env, jclass cls, jobject convDesc, jint arrayLength, jintArray padA, jintArray filterStrideA, jintArray upscaleA, jint mode)
{
    // Null-checks for non-primitive arguments
    if (convDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'convDesc' is null for cudnnSetConvolutionNdDescriptor_v2");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    // arrayLength is primitive
    if (padA == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'padA' is null for cudnnSetConvolutionNdDescriptor_v2");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (filterStrideA == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'filterStrideA' is null for cudnnSetConvolutionNdDescriptor_v2");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (upscaleA == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'upscaleA' is null for cudnnSetConvolutionNdDescriptor_v2");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    // mode is primitive

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnSetConvolutionNdDescriptor_v2(convDesc=%p, arrayLength=%d, padA=%p, filterStrideA=%p, upscaleA=%p, mode=%d)\n",
        convDesc, arrayLength, padA, filterStrideA, upscaleA, mode);

    // Native variable declarations
    cudnnConvolutionDescriptor_t convDesc_native;
    int arrayLength_native = 0;
    int* padA_native = NULL;
    int* filterStrideA_native = NULL;
    int* upscaleA_native = NULL;
    cudnnConvolutionMode_t mode_native;

    // Obtain native variable values
    convDesc_native = (cudnnConvolutionDescriptor_t)getNativePointerValue(env, convDesc);
    arrayLength_native = (int)arrayLength;
    if (!initNative(env, padA, padA_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!initNative(env, filterStrideA, filterStrideA_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!initNative(env, upscaleA, upscaleA_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    mode_native = (cudnnConvolutionMode_t)mode;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnSetConvolutionNdDescriptor_v2(convDesc_native, arrayLength_native, padA_native, filterStrideA_native, upscaleA_native, mode_native);

    // Write back native variable values
    // convDesc is a read-only native pointer
    // arrayLength is primitive
    if (!releaseNative(env, padA_native, padA, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!releaseNative(env, filterStrideA_native, filterStrideA, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!releaseNative(env, upscaleA_native, upscaleA, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // mode is primitive

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetConvolutionNdDescriptor_1v2Native(JNIEnv *env, jclass cls, jobject convDesc, jint arrayLengthRequested, jintArray arrayLength, jintArray padA, jintArray strideA, jintArray upscaleA, jintArray mode)
{
    // Null-checks for non-primitive arguments
    if (convDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'convDesc' is null for cudnnGetConvolutionNdDescriptor_v2");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    // arrayLengthRequested is primitive
    if (arrayLength == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'arrayLength' is null for cudnnGetConvolutionNdDescriptor_v2");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (padA == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'padA' is null for cudnnGetConvolutionNdDescriptor_v2");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (strideA == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'strideA' is null for cudnnGetConvolutionNdDescriptor_v2");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (upscaleA == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'upscaleA' is null for cudnnGetConvolutionNdDescriptor_v2");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (mode == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'mode' is null for cudnnGetConvolutionNdDescriptor_v2");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetConvolutionNdDescriptor_v2(convDesc=%p, arrayLengthRequested=%d, arrayLength=%p, padA=%p, strideA=%p, upscaleA=%p, mode=%p)\n",
        convDesc, arrayLengthRequested, arrayLength, padA, strideA, upscaleA, mode);

    // Native variable declarations
    cudnnConvolutionDescriptor_t convDesc_native;
    int arrayLengthRequested_native = 0;
    int arrayLength_native = 0;
    int* padA_native = NULL;
    int* strideA_native = NULL;
    int* upscaleA_native = NULL;
    cudnnConvolutionMode_t mode_native;

    // Obtain native variable values
    convDesc_native = (cudnnConvolutionDescriptor_t)getNativePointerValue(env, convDesc);
    arrayLengthRequested_native = (int)arrayLengthRequested;
    // arrayLength is set here
    if (!initNative(env, padA, padA_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!initNative(env, strideA, strideA_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!initNative(env, upscaleA, upscaleA_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // mode is set here

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetConvolutionNdDescriptor_v2(convDesc_native, arrayLengthRequested_native, &arrayLength_native, padA_native, strideA_native, upscaleA_native, &mode_native);

    // Write back native variable values
    // convDesc is a read-only native pointer
    // arrayLengthRequested is primitive
    if (!set(env, arrayLength, 0, (jint)arrayLength_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!releaseNative(env, padA_native, padA, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!releaseNative(env, strideA_native, strideA, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!releaseNative(env, upscaleA_native, upscaleA, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, mode, 0, (jint)mode_native)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnAddTensor_1v2Native(JNIEnv *env, jclass cls, jobject handle, jint mode, jobject alpha, jobject bDesc, jobject b, jobject beta, jobject yDesc, jobject y)
{
    // Null-checks for non-primitive arguments
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cudnnAddTensor_v2");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    // mode is primitive
    if (alpha == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'alpha' is null for cudnnAddTensor_v2");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (bDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'bDesc' is null for cudnnAddTensor_v2");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (b == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'b' is null for cudnnAddTensor_v2");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (beta == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'beta' is null for cudnnAddTensor_v2");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (yDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'yDesc' is null for cudnnAddTensor_v2");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (y == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'y' is null for cudnnAddTensor_v2");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnAddTensor_v2(handle=%p, mode=%d, alpha=%p, bDesc=%p, b=%p, beta=%p, yDesc=%p, y=%p)\n",
        handle, mode, alpha, bDesc, b, beta, yDesc, y);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnAddMode_t mode_native;
    void* alpha_native;
    cudnnTensorDescriptor_t bDesc_native;
    void* b_native;
    void* beta_native;
    cudnnTensorDescriptor_t yDesc_native;
    void* y_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    mode_native = (cudnnAddMode_t)mode;
    PointerData *alpha_pointerData = initPointerData(env, alpha);
    if (alpha_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    alpha_native = (void*)alpha_pointerData->getPointer(env);
    bDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, bDesc);
    b_native = (void*)getPointer(env, b);
    PointerData *beta_pointerData = initPointerData(env, beta);
    if (beta_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    beta_native = (void*)beta_pointerData->getPointer(env);
    yDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, yDesc);
    y_native = (void*)getPointer(env, y);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnAddTensor_v2(handle_native, mode_native, alpha_native, bDesc_native, b_native, beta_native, yDesc_native, y_native);

    // Write back native variable values
    // handle is a read-only native pointer
    // mode is primitive
    if (!releasePointerData(env, alpha_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // bDesc is a read-only native pointer
    // b is a native pointer
    if (!releasePointerData(env, beta_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // yDesc is a read-only native pointer
    // y is a native pointer

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnConvolutionBackwardFilter_1v2Native(JNIEnv *env, jclass cls, jobject handle, jobject alpha, jobject xDesc, jobject x, jobject dyDesc, jobject dy, jobject convDesc, jobject beta, jobject dxDesc, jobject dx)
{
    // Null-checks for non-primitive arguments
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cudnnConvolutionBackwardFilter_v2");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (alpha == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'alpha' is null for cudnnConvolutionBackwardFilter_v2");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (xDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'xDesc' is null for cudnnConvolutionBackwardFilter_v2");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cudnnConvolutionBackwardFilter_v2");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (dyDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dyDesc' is null for cudnnConvolutionBackwardFilter_v2");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (dy == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dy' is null for cudnnConvolutionBackwardFilter_v2");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (convDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'convDesc' is null for cudnnConvolutionBackwardFilter_v2");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (beta == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'beta' is null for cudnnConvolutionBackwardFilter_v2");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (dxDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dxDesc' is null for cudnnConvolutionBackwardFilter_v2");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (dx == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dx' is null for cudnnConvolutionBackwardFilter_v2");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnConvolutionBackwardFilter_v2(handle=%p, alpha=%p, xDesc=%p, x=%p, dyDesc=%p, dy=%p, convDesc=%p, beta=%p, dxDesc=%p, dx=%p)\n",
        handle, alpha, xDesc, x, dyDesc, dy, convDesc, beta, dxDesc, dx);

    // Native variable declarations
    cudnnHandle_t handle_native;
    void* alpha_native;
    cudnnTensorDescriptor_t xDesc_native;
    void* x_native;
    cudnnTensorDescriptor_t dyDesc_native;
    void* dy_native;
    cudnnConvolutionDescriptor_t convDesc_native;
    void* beta_native;
    cudnnFilterDescriptor_t dxDesc_native;
    void* dx_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    PointerData *alpha_pointerData = initPointerData(env, alpha);
    if (alpha_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    alpha_native = (void*)alpha_pointerData->getPointer(env);
    xDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, xDesc);
    x_native = (void*)getPointer(env, x);
    dyDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, dyDesc);
    dy_native = (void*)getPointer(env, dy);
    convDesc_native = (cudnnConvolutionDescriptor_t)getNativePointerValue(env, convDesc);
    PointerData *beta_pointerData = initPointerData(env, beta);
    if (beta_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    beta_native = (void*)beta_pointerData->getPointer(env);
    dxDesc_native = (cudnnFilterDescriptor_t)getNativePointerValue(env, dxDesc);
    dx_native = (void*)getPointer(env, dx);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnConvolutionBackwardFilter_v2(handle_native, alpha_native, xDesc_native, x_native, dyDesc_native, dy_native, convDesc_native, beta_native, dxDesc_native, dx_native);

    // Write back native variable values
    // handle is a read-only native pointer
    if (!releasePointerData(env, alpha_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // xDesc is a read-only native pointer
    // x is a native pointer
    // dyDesc is a read-only native pointer
    // dy is a native pointer
    // convDesc is a read-only native pointer
    if (!releasePointerData(env, beta_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // dxDesc is a read-only native pointer
    // dx is a native pointer

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnConvolutionBackwardData_1v2Native(JNIEnv *env, jclass cls, jobject handle, jobject alpha, jobject xDesc, jobject x, jobject dyDesc, jobject dy, jobject convDesc, jobject beta, jobject dxDesc, jobject dx)
{
    // Null-checks for non-primitive arguments
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cudnnConvolutionBackwardData_v2");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (alpha == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'alpha' is null for cudnnConvolutionBackwardData_v2");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (xDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'xDesc' is null for cudnnConvolutionBackwardData_v2");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cudnnConvolutionBackwardData_v2");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (dyDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dyDesc' is null for cudnnConvolutionBackwardData_v2");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (dy == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dy' is null for cudnnConvolutionBackwardData_v2");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (convDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'convDesc' is null for cudnnConvolutionBackwardData_v2");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (beta == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'beta' is null for cudnnConvolutionBackwardData_v2");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (dxDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dxDesc' is null for cudnnConvolutionBackwardData_v2");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (dx == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dx' is null for cudnnConvolutionBackwardData_v2");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnConvolutionBackwardData_v2(handle=%p, alpha=%p, xDesc=%p, x=%p, dyDesc=%p, dy=%p, convDesc=%p, beta=%p, dxDesc=%p, dx=%p)\n",
        handle, alpha, xDesc, x, dyDesc, dy, convDesc, beta, dxDesc, dx);

    // Native variable declarations
    cudnnHandle_t handle_native;
    void* alpha_native;
    cudnnFilterDescriptor_t xDesc_native;
    void* x_native;
    cudnnTensorDescriptor_t dyDesc_native;
    void* dy_native;
    cudnnConvolutionDescriptor_t convDesc_native;
    void* beta_native;
    cudnnTensorDescriptor_t dxDesc_native;
    void* dx_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    PointerData *alpha_pointerData = initPointerData(env, alpha);
    if (alpha_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    alpha_native = (void*)alpha_pointerData->getPointer(env);
    xDesc_native = (cudnnFilterDescriptor_t)getNativePointerValue(env, xDesc);
    x_native = (void*)getPointer(env, x);
    dyDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, dyDesc);
    dy_native = (void*)getPointer(env, dy);
    convDesc_native = (cudnnConvolutionDescriptor_t)getNativePointerValue(env, convDesc);
    PointerData *beta_pointerData = initPointerData(env, beta);
    if (beta_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    beta_native = (void*)beta_pointerData->getPointer(env);
    dxDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, dxDesc);
    dx_native = (void*)getPointer(env, dx);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnConvolutionBackwardData_v2(handle_native, alpha_native, xDesc_native, x_native, dyDesc_native, dy_native, convDesc_native, beta_native, dxDesc_native, dx_native);

    // Write back native variable values
    // handle is a read-only native pointer
    if (!releasePointerData(env, alpha_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // xDesc is a read-only native pointer
    // x is a native pointer
    // dyDesc is a read-only native pointer
    // dy is a native pointer
    // convDesc is a read-only native pointer
    if (!releasePointerData(env, beta_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // dxDesc is a read-only native pointer
    // dx is a native pointer

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}


