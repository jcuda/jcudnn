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

/** Tensor layout conversion helper (dest = alpha * src + beta * dest) */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnTransformTensorNative(JNIEnv *env, jclass cls, jobject handle, jobject alpha, jobject srcDesc, jobject srcData, jobject beta, jobject destDesc, jobject destData)
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
    if (srcDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'srcDesc' is null for cudnnTransformTensor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (srcData == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'srcData' is null for cudnnTransformTensor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (beta == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'beta' is null for cudnnTransformTensor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (destDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'destDesc' is null for cudnnTransformTensor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (destData == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'destData' is null for cudnnTransformTensor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnTransformTensor(handle=%p, alpha=%p, srcDesc=%p, srcData=%p, beta=%p, destDesc=%p, destData=%p)\n",
        handle, alpha, srcDesc, srcData, beta, destDesc, destData);

    // Native variable declarations
    cudnnHandle_t handle_native;
    void* alpha_native;
    cudnnTensorDescriptor_t srcDesc_native;
    void* srcData_native;
    void* beta_native;
    cudnnTensorDescriptor_t destDesc_native;
    void* destData_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    PointerData *alpha_pointerData = initPointerData(env, alpha);
    if (alpha_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    alpha_native = (void*)alpha_pointerData->getPointer(env);
    srcDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, srcDesc);
    srcData_native = (void*)getPointer(env, srcData);
    PointerData *beta_pointerData = initPointerData(env, beta);
    if (beta_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    beta_native = (void*)beta_pointerData->getPointer(env);
    destDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, destDesc);
    destData_native = (void*)getPointer(env, destData);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnTransformTensor(handle_native, alpha_native, srcDesc_native, srcData_native, beta_native, destDesc_native, destData_native);

    // Write back native variable values
    // handle is a read-only native pointer
    if (!releasePointerData(env, alpha_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // srcDesc is a read-only native pointer
    // srcData is a native pointer
    if (!releasePointerData(env, beta_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // destDesc is a read-only native pointer
    // destData is a native pointer

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

/** Tensor Bias addition : srcDest = alpha * bias + beta * srcDestDesc  */
/** DEPRECATED AS OF v3 */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnAddTensorNative(JNIEnv *env, jclass cls, jobject handle, jint mode, jobject alpha, jobject biasDesc, jobject biasData, jobject beta, jobject srcDestDesc, jobject srcDestData)
{
    // Null-checks for non-primitive arguments
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cudnnAddTensor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    // mode is primitive
    if (alpha == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'alpha' is null for cudnnAddTensor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (biasDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'biasDesc' is null for cudnnAddTensor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (biasData == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'biasData' is null for cudnnAddTensor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (beta == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'beta' is null for cudnnAddTensor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (srcDestDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'srcDestDesc' is null for cudnnAddTensor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (srcDestData == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'srcDestData' is null for cudnnAddTensor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnAddTensor(handle=%p, mode=%d, alpha=%p, biasDesc=%p, biasData=%p, beta=%p, srcDestDesc=%p, srcDestData=%p)\n",
        handle, mode, alpha, biasDesc, biasData, beta, srcDestDesc, srcDestData);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnAddMode_t mode_native;
    void* alpha_native;
    cudnnTensorDescriptor_t biasDesc_native;
    void* biasData_native;
    void* beta_native;
    cudnnTensorDescriptor_t srcDestDesc_native;
    void* srcDestData_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    mode_native = (cudnnAddMode_t)mode;
    PointerData *alpha_pointerData = initPointerData(env, alpha);
    if (alpha_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    alpha_native = (void*)alpha_pointerData->getPointer(env);
    biasDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, biasDesc);
    biasData_native = (void*)getPointer(env, biasData);
    PointerData *beta_pointerData = initPointerData(env, beta);
    if (beta_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    beta_native = (void*)beta_pointerData->getPointer(env);
    srcDestDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, srcDestDesc);
    srcDestData_native = (void*)getPointer(env, srcDestData);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnAddTensor(handle_native, mode_native, alpha_native, biasDesc_native, biasData_native, beta_native, srcDestDesc_native, srcDestData_native);

    // Write back native variable values
    // handle is a read-only native pointer
    // mode is primitive
    if (!releasePointerData(env, alpha_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // biasDesc is a read-only native pointer
    // biasData is a native pointer
    if (!releasePointerData(env, beta_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // srcDestDesc is a read-only native pointer
    // srcDestData is a native pointer

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

/** Tensor Bias addition : srcDest = alpha * bias + beta * srcDestDesc  */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnAddTensor_1v3Native(JNIEnv *env, jclass cls, jobject handle, jobject alpha, jobject biasDesc, jobject biasData, jobject beta, jobject srcDestDesc, jobject srcDestData)
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
    if (biasDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'biasDesc' is null for cudnnAddTensor_v3");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (biasData == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'biasData' is null for cudnnAddTensor_v3");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (beta == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'beta' is null for cudnnAddTensor_v3");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (srcDestDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'srcDestDesc' is null for cudnnAddTensor_v3");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (srcDestData == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'srcDestData' is null for cudnnAddTensor_v3");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnAddTensor_v3(handle=%p, alpha=%p, biasDesc=%p, biasData=%p, beta=%p, srcDestDesc=%p, srcDestData=%p)\n",
        handle, alpha, biasDesc, biasData, beta, srcDestDesc, srcDestData);

    // Native variable declarations
    cudnnHandle_t handle_native;
    void* alpha_native;
    cudnnTensorDescriptor_t biasDesc_native;
    void* biasData_native;
    void* beta_native;
    cudnnTensorDescriptor_t srcDestDesc_native;
    void* srcDestData_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    PointerData *alpha_pointerData = initPointerData(env, alpha);
    if (alpha_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    alpha_native = (void*)alpha_pointerData->getPointer(env);
    biasDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, biasDesc);
    biasData_native = (void*)getPointer(env, biasData);
    PointerData *beta_pointerData = initPointerData(env, beta);
    if (beta_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    beta_native = (void*)beta_pointerData->getPointer(env);
    srcDestDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, srcDestDesc);
    srcDestData_native = (void*)getPointer(env, srcDestData);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnAddTensor_v3(handle_native, alpha_native, biasDesc_native, biasData_native, beta_native, srcDestDesc_native, srcDestData_native);

    // Write back native variable values
    // handle is a read-only native pointer
    if (!releasePointerData(env, alpha_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // biasDesc is a read-only native pointer
    // biasData is a native pointer
    if (!releasePointerData(env, beta_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // srcDestDesc is a read-only native pointer
    // srcDestData is a native pointer

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

/** Set all data points of a tensor to a given value : srcDest = value */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnSetTensorNative(JNIEnv *env, jclass cls, jobject handle, jobject srcDestDesc, jobject srcDestData, jobject value)
{
    // Null-checks for non-primitive arguments
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cudnnSetTensor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (srcDestDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'srcDestDesc' is null for cudnnSetTensor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (srcDestData == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'srcDestData' is null for cudnnSetTensor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (value == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'value' is null for cudnnSetTensor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnSetTensor(handle=%p, srcDestDesc=%p, srcDestData=%p, value=%p)\n",
        handle, srcDestDesc, srcDestData, value);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnTensorDescriptor_t srcDestDesc_native;
    void* srcDestData_native;
    void* value_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    srcDestDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, srcDestDesc);
    srcDestData_native = (void*)getPointer(env, srcDestData);
    PointerData *value_pointerData = initPointerData(env, value);
    if (value_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    value_native = (void*)value_pointerData->getPointer(env);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnSetTensor(handle_native, srcDestDesc_native, srcDestData_native, value_native);

    // Write back native variable values
    // handle is a read-only native pointer
    // srcDestDesc is a read-only native pointer
    // srcDestData is a native pointer
    if (!releasePointerData(env, value_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

/** Set all data points of a tensor to a given value : srcDest = alpha * srcDest */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnScaleTensorNative(JNIEnv *env, jclass cls, jobject handle, jobject srcDestDesc, jobject srcDestData, jobject alpha)
{
    // Null-checks for non-primitive arguments
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cudnnScaleTensor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (srcDestDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'srcDestDesc' is null for cudnnScaleTensor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (srcDestData == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'srcDestData' is null for cudnnScaleTensor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (alpha == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'alpha' is null for cudnnScaleTensor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnScaleTensor(handle=%p, srcDestDesc=%p, srcDestData=%p, alpha=%p)\n",
        handle, srcDestDesc, srcDestData, alpha);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnTensorDescriptor_t srcDestDesc_native;
    void* srcDestData_native;
    void* alpha_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    srcDestDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, srcDestDesc);
    srcDestData_native = (void*)getPointer(env, srcDestData);
    PointerData *alpha_pointerData = initPointerData(env, alpha);
    if (alpha_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    alpha_native = (void*)alpha_pointerData->getPointer(env);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnScaleTensor(handle_native, srcDestDesc_native, srcDestData_native, alpha_native);

    // Write back native variable values
    // handle is a read-only native pointer
    // srcDestDesc is a read-only native pointer
    // srcDestData is a native pointer
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

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnSetConvolutionNdDescriptorNative(JNIEnv *env, jclass cls, jobject convDesc, jint arrayLength, jintArray padA, jintArray filterStrideA, jintArray upscaleA, jint mode)
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

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnSetConvolutionNdDescriptor(convDesc=%p, arrayLength=%d, padA=%p, filterStrideA=%p, upscaleA=%p, mode=%d)\n",
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
    cudnnStatus_t jniResult_native = cudnnSetConvolutionNdDescriptor(convDesc_native, arrayLength_native, padA_native, filterStrideA_native, upscaleA_native, mode_native);

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

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetConvolutionNdDescriptorNative(JNIEnv *env, jclass cls, jobject convDesc, jint arrayLengthRequested, jintArray arrayLength, jintArray padA, jintArray strideA, jintArray upscaleA, jintArray mode)
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

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetConvolutionNdDescriptor(convDesc=%p, arrayLengthRequested=%d, arrayLength=%p, padA=%p, strideA=%p, upscaleA=%p, mode=%p)\n",
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
    cudnnStatus_t jniResult_native = cudnnGetConvolutionNdDescriptor(convDesc_native, arrayLengthRequested_native, &arrayLength_native, padA_native, strideA_native, upscaleA_native, &mode_native);

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

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnFindConvolutionForwardAlgorithmNative(JNIEnv *env, jclass cls, jobject handle, jobject srcDesc, jobject filterDesc, jobject convDesc, jobject destDesc, jint requestedAlgoCount, jintArray returnedAlgoCount, jobjectArray perfResults)
{
    // Null-checks for non-primitive arguments
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cudnnFindConvolutionForwardAlgorithm");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (srcDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'srcDesc' is null for cudnnFindConvolutionForwardAlgorithm");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (filterDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'filterDesc' is null for cudnnFindConvolutionForwardAlgorithm");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (convDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'convDesc' is null for cudnnFindConvolutionForwardAlgorithm");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (destDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'destDesc' is null for cudnnFindConvolutionForwardAlgorithm");
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
    Logger::log(LOG_TRACE, "Executing cudnnFindConvolutionForwardAlgorithm(handle=%p, srcDesc=%p, filterDesc=%p, convDesc=%p, destDesc=%p, requestedAlgoCount=%d, returnedAlgoCount=%p, perfResults=%p)\n",
        handle, srcDesc, filterDesc, convDesc, destDesc, requestedAlgoCount, returnedAlgoCount, perfResults);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnTensorDescriptor_t srcDesc_native;
    cudnnFilterDescriptor_t filterDesc_native;
    cudnnConvolutionDescriptor_t convDesc_native;
    cudnnTensorDescriptor_t destDesc_native;
    int requestedAlgoCount_native = 0;
    int returnedAlgoCount_native = 0;
    cudnnConvolutionFwdAlgoPerf_t* perfResults_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    srcDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, srcDesc);
    filterDesc_native = (cudnnFilterDescriptor_t)getNativePointerValue(env, filterDesc);
    convDesc_native = (cudnnConvolutionDescriptor_t)getNativePointerValue(env, convDesc);
    destDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, destDesc);
    requestedAlgoCount_native = (int)requestedAlgoCount;
    // returnedAlgoCount is set here
    if (!initNative(env, perfResults, perfResults_native, requestedAlgoCount)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnFindConvolutionForwardAlgorithm(handle_native, srcDesc_native, filterDesc_native, convDesc_native, destDesc_native, requestedAlgoCount_native, &returnedAlgoCount_native, perfResults_native);

    // Write back native variable values
    // handle is a read-only native pointer
    // srcDesc is a read-only native pointer
    // filterDesc is a read-only native pointer
    // convDesc is a read-only native pointer
    // destDesc is a read-only native pointer
    // requestedAlgoCount is primitive
    if (!set(env, returnedAlgoCount, 0, (jint)returnedAlgoCount_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!releaseNative(env, perfResults_native, perfResults, returnedAlgoCount_native)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetConvolutionForwardAlgorithmNative(JNIEnv *env, jclass cls, jobject handle, jobject srcDesc, jobject filterDesc, jobject convDesc, jobject destDesc, jint preference, jlong memoryLimitInbytes, jintArray algo)
{
    // Null-checks for non-primitive arguments
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cudnnGetConvolutionForwardAlgorithm");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (srcDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'srcDesc' is null for cudnnGetConvolutionForwardAlgorithm");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (filterDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'filterDesc' is null for cudnnGetConvolutionForwardAlgorithm");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (convDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'convDesc' is null for cudnnGetConvolutionForwardAlgorithm");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (destDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'destDesc' is null for cudnnGetConvolutionForwardAlgorithm");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    // preference is primitive
    // memoryLimitInbytes is primitive
    if (algo == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'algo' is null for cudnnGetConvolutionForwardAlgorithm");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetConvolutionForwardAlgorithm(handle=%p, srcDesc=%p, filterDesc=%p, convDesc=%p, destDesc=%p, preference=%d, memoryLimitInbytes=%ld, algo=%p)\n",
        handle, srcDesc, filterDesc, convDesc, destDesc, preference, memoryLimitInbytes, algo);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnTensorDescriptor_t srcDesc_native;
    cudnnFilterDescriptor_t filterDesc_native;
    cudnnConvolutionDescriptor_t convDesc_native;
    cudnnTensorDescriptor_t destDesc_native;
    cudnnConvolutionFwdPreference_t preference_native;
    size_t memoryLimitInbytes_native = 0;
    cudnnConvolutionFwdAlgo_t algo_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    srcDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, srcDesc);
    filterDesc_native = (cudnnFilterDescriptor_t)getNativePointerValue(env, filterDesc);
    convDesc_native = (cudnnConvolutionDescriptor_t)getNativePointerValue(env, convDesc);
    destDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, destDesc);
    preference_native = (cudnnConvolutionFwdPreference_t)preference;
    memoryLimitInbytes_native = (size_t)memoryLimitInbytes;
    // algo is set here

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetConvolutionForwardAlgorithm(handle_native, srcDesc_native, filterDesc_native, convDesc_native, destDesc_native, preference_native, memoryLimitInbytes_native, &algo_native);

    // Write back native variable values
    // handle is a read-only native pointer
    // srcDesc is a read-only native pointer
    // filterDesc is a read-only native pointer
    // convDesc is a read-only native pointer
    // destDesc is a read-only native pointer
    // preference is primitive
    // memoryLimitInbytes is primitive
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
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetConvolutionForwardWorkspaceSizeNative(JNIEnv *env, jclass cls, jobject handle, jobject srcDesc, jobject filterDesc, jobject convDesc, jobject destDesc, jint algo, jlongArray sizeInBytes)
{
    // Null-checks for non-primitive arguments
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cudnnGetConvolutionForwardWorkspaceSize");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (srcDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'srcDesc' is null for cudnnGetConvolutionForwardWorkspaceSize");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (filterDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'filterDesc' is null for cudnnGetConvolutionForwardWorkspaceSize");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (convDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'convDesc' is null for cudnnGetConvolutionForwardWorkspaceSize");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (destDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'destDesc' is null for cudnnGetConvolutionForwardWorkspaceSize");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    // algo is primitive
    if (sizeInBytes == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'sizeInBytes' is null for cudnnGetConvolutionForwardWorkspaceSize");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetConvolutionForwardWorkspaceSize(handle=%p, srcDesc=%p, filterDesc=%p, convDesc=%p, destDesc=%p, algo=%d, sizeInBytes=%p)\n",
        handle, srcDesc, filterDesc, convDesc, destDesc, algo, sizeInBytes);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnTensorDescriptor_t srcDesc_native;
    cudnnFilterDescriptor_t filterDesc_native;
    cudnnConvolutionDescriptor_t convDesc_native;
    cudnnTensorDescriptor_t destDesc_native;
    cudnnConvolutionFwdAlgo_t algo_native;
    size_t sizeInBytes_native = 0;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    srcDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, srcDesc);
    filterDesc_native = (cudnnFilterDescriptor_t)getNativePointerValue(env, filterDesc);
    convDesc_native = (cudnnConvolutionDescriptor_t)getNativePointerValue(env, convDesc);
    destDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, destDesc);
    algo_native = (cudnnConvolutionFwdAlgo_t)algo;
    // sizeInBytes is set here

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetConvolutionForwardWorkspaceSize(handle_native, srcDesc_native, filterDesc_native, convDesc_native, destDesc_native, algo_native, &sizeInBytes_native);

    // Write back native variable values
    // handle is a read-only native pointer
    // srcDesc is a read-only native pointer
    // filterDesc is a read-only native pointer
    // convDesc is a read-only native pointer
    // destDesc is a read-only native pointer
    // algo is primitive
    if (!set(env, sizeInBytes, 0, (jlong)sizeInBytes_native)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

/** Convolution functions: All of the form "output = alpha * Op(inputs) + beta * output" */
/** Function to perform the forward multiconvolution */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnConvolutionForwardNative(JNIEnv *env, jclass cls, jobject handle, jobject alpha, jobject srcDesc, jobject srcData, jobject filterDesc, jobject filterData, jobject convDesc, jint algo, jobject workSpace, jlong workSpaceSizeInBytes, jobject beta, jobject destDesc, jobject destData)
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
    if (srcDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'srcDesc' is null for cudnnConvolutionForward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (srcData == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'srcData' is null for cudnnConvolutionForward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (filterDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'filterDesc' is null for cudnnConvolutionForward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (filterData == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'filterData' is null for cudnnConvolutionForward");
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
    if (destDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'destDesc' is null for cudnnConvolutionForward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (destData == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'destData' is null for cudnnConvolutionForward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnConvolutionForward(handle=%p, alpha=%p, srcDesc=%p, srcData=%p, filterDesc=%p, filterData=%p, convDesc=%p, algo=%d, workSpace=%p, workSpaceSizeInBytes=%ld, beta=%p, destDesc=%p, destData=%p)\n",
        handle, alpha, srcDesc, srcData, filterDesc, filterData, convDesc, algo, workSpace, workSpaceSizeInBytes, beta, destDesc, destData);

    // Native variable declarations
    cudnnHandle_t handle_native;
    void* alpha_native;
    cudnnTensorDescriptor_t srcDesc_native;
    void* srcData_native;
    cudnnFilterDescriptor_t filterDesc_native;
    void* filterData_native;
    cudnnConvolutionDescriptor_t convDesc_native;
    cudnnConvolutionFwdAlgo_t algo_native;
    void* workSpace_native;
    size_t workSpaceSizeInBytes_native = 0;
    void* beta_native;
    cudnnTensorDescriptor_t destDesc_native;
    void* destData_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    PointerData *alpha_pointerData = initPointerData(env, alpha);
    if (alpha_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    alpha_native = (void*)alpha_pointerData->getPointer(env);
    srcDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, srcDesc);
    srcData_native = (void*)getPointer(env, srcData);
    filterDesc_native = (cudnnFilterDescriptor_t)getNativePointerValue(env, filterDesc);
    filterData_native = (void*)getPointer(env, filterData);
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
    destDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, destDesc);
    destData_native = (void*)getPointer(env, destData);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnConvolutionForward(handle_native, alpha_native, srcDesc_native, srcData_native, filterDesc_native, filterData_native, convDesc_native, algo_native, workSpace_native, workSpaceSizeInBytes_native, beta_native, destDesc_native, destData_native);

    // Write back native variable values
    // handle is a read-only native pointer
    if (!releasePointerData(env, alpha_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // srcDesc is a read-only native pointer
    // srcData is a native pointer
    // filterDesc is a read-only native pointer
    // filterData is a native pointer
    // convDesc is a read-only native pointer
    // algo is primitive
    // workSpace is a native pointer
    // workSpaceSizeInBytes is primitive
    if (!releasePointerData(env, beta_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // destDesc is a read-only native pointer
    // destData is a native pointer

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

/** Functions to perform the backward multiconvolution */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnConvolutionBackwardBiasNative(JNIEnv *env, jclass cls, jobject handle, jobject alpha, jobject srcDesc, jobject srcData, jobject beta, jobject destDesc, jobject destData)
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
    if (srcDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'srcDesc' is null for cudnnConvolutionBackwardBias");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (srcData == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'srcData' is null for cudnnConvolutionBackwardBias");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (beta == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'beta' is null for cudnnConvolutionBackwardBias");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (destDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'destDesc' is null for cudnnConvolutionBackwardBias");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (destData == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'destData' is null for cudnnConvolutionBackwardBias");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnConvolutionBackwardBias(handle=%p, alpha=%p, srcDesc=%p, srcData=%p, beta=%p, destDesc=%p, destData=%p)\n",
        handle, alpha, srcDesc, srcData, beta, destDesc, destData);

    // Native variable declarations
    cudnnHandle_t handle_native;
    void* alpha_native;
    cudnnTensorDescriptor_t srcDesc_native;
    void* srcData_native;
    void* beta_native;
    cudnnTensorDescriptor_t destDesc_native;
    void* destData_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    PointerData *alpha_pointerData = initPointerData(env, alpha);
    if (alpha_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    alpha_native = (void*)alpha_pointerData->getPointer(env);
    srcDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, srcDesc);
    srcData_native = (void*)getPointer(env, srcData);
    PointerData *beta_pointerData = initPointerData(env, beta);
    if (beta_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    beta_native = (void*)beta_pointerData->getPointer(env);
    destDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, destDesc);
    destData_native = (void*)getPointer(env, destData);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnConvolutionBackwardBias(handle_native, alpha_native, srcDesc_native, srcData_native, beta_native, destDesc_native, destData_native);

    // Write back native variable values
    // handle is a read-only native pointer
    if (!releasePointerData(env, alpha_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // srcDesc is a read-only native pointer
    // srcData is a native pointer
    if (!releasePointerData(env, beta_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // destDesc is a read-only native pointer
    // destData is a native pointer

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnFindConvolutionBackwardFilterAlgorithmNative(JNIEnv *env, jclass cls, jobject handle, jobject srcDesc, jobject diffDesc, jobject convDesc, jobject gradDesc, jint requestedAlgoCount, jintArray returnedAlgoCount, jobjectArray perfResults)
{
    // Null-checks for non-primitive arguments
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cudnnFindConvolutionBackwardFilterAlgorithm");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (srcDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'srcDesc' is null for cudnnFindConvolutionBackwardFilterAlgorithm");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (diffDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'diffDesc' is null for cudnnFindConvolutionBackwardFilterAlgorithm");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (convDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'convDesc' is null for cudnnFindConvolutionBackwardFilterAlgorithm");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (gradDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'gradDesc' is null for cudnnFindConvolutionBackwardFilterAlgorithm");
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
    Logger::log(LOG_TRACE, "Executing cudnnFindConvolutionBackwardFilterAlgorithm(handle=%p, srcDesc=%p, diffDesc=%p, convDesc=%p, gradDesc=%p, requestedAlgoCount=%d, returnedAlgoCount=%p, perfResults=%p)\n",
        handle, srcDesc, diffDesc, convDesc, gradDesc, requestedAlgoCount, returnedAlgoCount, perfResults);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnTensorDescriptor_t srcDesc_native;
    cudnnTensorDescriptor_t diffDesc_native;
    cudnnConvolutionDescriptor_t convDesc_native;
    cudnnFilterDescriptor_t gradDesc_native;
    int requestedAlgoCount_native = 0;
    int returnedAlgoCount_native = 0;
    cudnnConvolutionBwdFilterAlgoPerf_t* perfResults_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    srcDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, srcDesc);
    diffDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, diffDesc);
    convDesc_native = (cudnnConvolutionDescriptor_t)getNativePointerValue(env, convDesc);
    gradDesc_native = (cudnnFilterDescriptor_t)getNativePointerValue(env, gradDesc);
    requestedAlgoCount_native = (int)requestedAlgoCount;
    // returnedAlgoCount is set here
    if (!initNative(env, perfResults, perfResults_native, requestedAlgoCount)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnFindConvolutionBackwardFilterAlgorithm(handle_native, srcDesc_native, diffDesc_native, convDesc_native, gradDesc_native, requestedAlgoCount_native, &returnedAlgoCount_native, perfResults_native);

    // Write back native variable values
    // handle is a read-only native pointer
    // srcDesc is a read-only native pointer
    // diffDesc is a read-only native pointer
    // convDesc is a read-only native pointer
    // gradDesc is a read-only native pointer
    // requestedAlgoCount is primitive
    if (!set(env, returnedAlgoCount, 0, (jint)returnedAlgoCount_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!releaseNative(env, perfResults_native, perfResults, returnedAlgoCount_native)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetConvolutionBackwardFilterAlgorithmNative(JNIEnv *env, jclass cls, jobject handle, jobject srcDesc, jobject diffDesc, jobject convDesc, jobject gradDesc, jint preference, jlong memoryLimitInbytes, jintArray algo)
{
    // Null-checks for non-primitive arguments
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cudnnGetConvolutionBackwardFilterAlgorithm");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (srcDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'srcDesc' is null for cudnnGetConvolutionBackwardFilterAlgorithm");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (diffDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'diffDesc' is null for cudnnGetConvolutionBackwardFilterAlgorithm");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (convDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'convDesc' is null for cudnnGetConvolutionBackwardFilterAlgorithm");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (gradDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'gradDesc' is null for cudnnGetConvolutionBackwardFilterAlgorithm");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    // preference is primitive
    // memoryLimitInbytes is primitive
    if (algo == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'algo' is null for cudnnGetConvolutionBackwardFilterAlgorithm");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetConvolutionBackwardFilterAlgorithm(handle=%p, srcDesc=%p, diffDesc=%p, convDesc=%p, gradDesc=%p, preference=%d, memoryLimitInbytes=%ld, algo=%p)\n",
        handle, srcDesc, diffDesc, convDesc, gradDesc, preference, memoryLimitInbytes, algo);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnTensorDescriptor_t srcDesc_native;
    cudnnTensorDescriptor_t diffDesc_native;
    cudnnConvolutionDescriptor_t convDesc_native;
    cudnnFilterDescriptor_t gradDesc_native;
    cudnnConvolutionBwdFilterPreference_t preference_native;
    size_t memoryLimitInbytes_native = 0;
    cudnnConvolutionBwdFilterAlgo_t algo_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    srcDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, srcDesc);
    diffDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, diffDesc);
    convDesc_native = (cudnnConvolutionDescriptor_t)getNativePointerValue(env, convDesc);
    gradDesc_native = (cudnnFilterDescriptor_t)getNativePointerValue(env, gradDesc);
    preference_native = (cudnnConvolutionBwdFilterPreference_t)preference;
    memoryLimitInbytes_native = (size_t)memoryLimitInbytes;
    // algo is set here

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetConvolutionBackwardFilterAlgorithm(handle_native, srcDesc_native, diffDesc_native, convDesc_native, gradDesc_native, preference_native, memoryLimitInbytes_native, &algo_native);

    // Write back native variable values
    // handle is a read-only native pointer
    // srcDesc is a read-only native pointer
    // diffDesc is a read-only native pointer
    // convDesc is a read-only native pointer
    // gradDesc is a read-only native pointer
    // preference is primitive
    // memoryLimitInbytes is primitive
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
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetConvolutionBackwardFilterWorkspaceSizeNative(JNIEnv *env, jclass cls, jobject handle, jobject srcDesc, jobject diffDesc, jobject convDesc, jobject gradDesc, jint algo, jlongArray sizeInBytes)
{
    // Null-checks for non-primitive arguments
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cudnnGetConvolutionBackwardFilterWorkspaceSize");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (srcDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'srcDesc' is null for cudnnGetConvolutionBackwardFilterWorkspaceSize");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (diffDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'diffDesc' is null for cudnnGetConvolutionBackwardFilterWorkspaceSize");
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
    Logger::log(LOG_TRACE, "Executing cudnnGetConvolutionBackwardFilterWorkspaceSize(handle=%p, srcDesc=%p, diffDesc=%p, convDesc=%p, gradDesc=%p, algo=%d, sizeInBytes=%p)\n",
        handle, srcDesc, diffDesc, convDesc, gradDesc, algo, sizeInBytes);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnTensorDescriptor_t srcDesc_native;
    cudnnTensorDescriptor_t diffDesc_native;
    cudnnConvolutionDescriptor_t convDesc_native;
    cudnnFilterDescriptor_t gradDesc_native;
    cudnnConvolutionBwdFilterAlgo_t algo_native;
    size_t sizeInBytes_native = 0;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    srcDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, srcDesc);
    diffDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, diffDesc);
    convDesc_native = (cudnnConvolutionDescriptor_t)getNativePointerValue(env, convDesc);
    gradDesc_native = (cudnnFilterDescriptor_t)getNativePointerValue(env, gradDesc);
    algo_native = (cudnnConvolutionBwdFilterAlgo_t)algo;
    // sizeInBytes is set here

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetConvolutionBackwardFilterWorkspaceSize(handle_native, srcDesc_native, diffDesc_native, convDesc_native, gradDesc_native, algo_native, &sizeInBytes_native);

    // Write back native variable values
    // handle is a read-only native pointer
    // srcDesc is a read-only native pointer
    // diffDesc is a read-only native pointer
    // convDesc is a read-only native pointer
    // gradDesc is a read-only native pointer
    // algo is primitive
    if (!set(env, sizeInBytes, 0, (jlong)sizeInBytes_native)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnConvolutionBackwardFilter_1v3Native(JNIEnv *env, jclass cls, jobject handle, jobject alpha, jobject srcDesc, jobject srcData, jobject diffDesc, jobject diffData, jobject convDesc, jint algo, jobject workSpace, jlong workSpaceSizeInBytes, jobject beta, jobject gradDesc, jobject gradData)
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
    if (srcDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'srcDesc' is null for cudnnConvolutionBackwardFilter_v3");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (srcData == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'srcData' is null for cudnnConvolutionBackwardFilter_v3");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (diffDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'diffDesc' is null for cudnnConvolutionBackwardFilter_v3");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (diffData == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'diffData' is null for cudnnConvolutionBackwardFilter_v3");
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
    if (gradDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'gradDesc' is null for cudnnConvolutionBackwardFilter_v3");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (gradData == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'gradData' is null for cudnnConvolutionBackwardFilter_v3");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnConvolutionBackwardFilter_v3(handle=%p, alpha=%p, srcDesc=%p, srcData=%p, diffDesc=%p, diffData=%p, convDesc=%p, algo=%d, workSpace=%p, workSpaceSizeInBytes=%ld, beta=%p, gradDesc=%p, gradData=%p)\n",
        handle, alpha, srcDesc, srcData, diffDesc, diffData, convDesc, algo, workSpace, workSpaceSizeInBytes, beta, gradDesc, gradData);

    // Native variable declarations
    cudnnHandle_t handle_native;
    void* alpha_native;
    cudnnTensorDescriptor_t srcDesc_native;
    void* srcData_native;
    cudnnTensorDescriptor_t diffDesc_native;
    void* diffData_native;
    cudnnConvolutionDescriptor_t convDesc_native;
    cudnnConvolutionBwdFilterAlgo_t algo_native;
    void* workSpace_native;
    size_t workSpaceSizeInBytes_native = 0;
    void* beta_native;
    cudnnFilterDescriptor_t gradDesc_native;
    void* gradData_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    PointerData *alpha_pointerData = initPointerData(env, alpha);
    if (alpha_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    alpha_native = (void*)alpha_pointerData->getPointer(env);
    srcDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, srcDesc);
    srcData_native = (void*)getPointer(env, srcData);
    diffDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, diffDesc);
    diffData_native = (void*)getPointer(env, diffData);
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
    gradDesc_native = (cudnnFilterDescriptor_t)getNativePointerValue(env, gradDesc);
    gradData_native = (void*)getPointer(env, gradData);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnConvolutionBackwardFilter_v3(handle_native, alpha_native, srcDesc_native, srcData_native, diffDesc_native, diffData_native, convDesc_native, algo_native, workSpace_native, workSpaceSizeInBytes_native, beta_native, gradDesc_native, gradData_native);

    // Write back native variable values
    // handle is a read-only native pointer
    if (!releasePointerData(env, alpha_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // srcDesc is a read-only native pointer
    // srcData is a native pointer
    // diffDesc is a read-only native pointer
    // diffData is a native pointer
    // convDesc is a read-only native pointer
    // algo is primitive
    // workSpace is a native pointer
    // workSpaceSizeInBytes is primitive
    if (!releasePointerData(env, beta_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // gradDesc is a read-only native pointer
    // gradData is a native pointer

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnConvolutionBackwardFilterNative(JNIEnv *env, jclass cls, jobject handle, jobject alpha, jobject srcDesc, jobject srcData, jobject diffDesc, jobject diffData, jobject convDesc, jobject beta, jobject gradDesc, jobject gradData)
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
    if (srcDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'srcDesc' is null for cudnnConvolutionBackwardFilter");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (srcData == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'srcData' is null for cudnnConvolutionBackwardFilter");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (diffDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'diffDesc' is null for cudnnConvolutionBackwardFilter");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (diffData == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'diffData' is null for cudnnConvolutionBackwardFilter");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (convDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'convDesc' is null for cudnnConvolutionBackwardFilter");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (beta == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'beta' is null for cudnnConvolutionBackwardFilter");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (gradDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'gradDesc' is null for cudnnConvolutionBackwardFilter");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (gradData == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'gradData' is null for cudnnConvolutionBackwardFilter");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnConvolutionBackwardFilter(handle=%p, alpha=%p, srcDesc=%p, srcData=%p, diffDesc=%p, diffData=%p, convDesc=%p, beta=%p, gradDesc=%p, gradData=%p)\n",
        handle, alpha, srcDesc, srcData, diffDesc, diffData, convDesc, beta, gradDesc, gradData);

    // Native variable declarations
    cudnnHandle_t handle_native;
    void* alpha_native;
    cudnnTensorDescriptor_t srcDesc_native;
    void* srcData_native;
    cudnnTensorDescriptor_t diffDesc_native;
    void* diffData_native;
    cudnnConvolutionDescriptor_t convDesc_native;
    void* beta_native;
    cudnnFilterDescriptor_t gradDesc_native;
    void* gradData_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    PointerData *alpha_pointerData = initPointerData(env, alpha);
    if (alpha_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    alpha_native = (void*)alpha_pointerData->getPointer(env);
    srcDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, srcDesc);
    srcData_native = (void*)getPointer(env, srcData);
    diffDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, diffDesc);
    diffData_native = (void*)getPointer(env, diffData);
    convDesc_native = (cudnnConvolutionDescriptor_t)getNativePointerValue(env, convDesc);
    PointerData *beta_pointerData = initPointerData(env, beta);
    if (beta_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    beta_native = (void*)beta_pointerData->getPointer(env);
    gradDesc_native = (cudnnFilterDescriptor_t)getNativePointerValue(env, gradDesc);
    gradData_native = (void*)getPointer(env, gradData);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnConvolutionBackwardFilter(handle_native, alpha_native, srcDesc_native, srcData_native, diffDesc_native, diffData_native, convDesc_native, beta_native, gradDesc_native, gradData_native);

    // Write back native variable values
    // handle is a read-only native pointer
    if (!releasePointerData(env, alpha_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // srcDesc is a read-only native pointer
    // srcData is a native pointer
    // diffDesc is a read-only native pointer
    // diffData is a native pointer
    // convDesc is a read-only native pointer
    if (!releasePointerData(env, beta_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // gradDesc is a read-only native pointer
    // gradData is a native pointer

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnFindConvolutionBackwardDataAlgorithmNative(JNIEnv *env, jclass cls, jobject handle, jobject filterDesc, jobject diffDesc, jobject convDesc, jobject gradDesc, jint requestedAlgoCount, jintArray returnedAlgoCount, jobjectArray perfResults)
{
    // Null-checks for non-primitive arguments
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cudnnFindConvolutionBackwardDataAlgorithm");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (filterDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'filterDesc' is null for cudnnFindConvolutionBackwardDataAlgorithm");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (diffDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'diffDesc' is null for cudnnFindConvolutionBackwardDataAlgorithm");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (convDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'convDesc' is null for cudnnFindConvolutionBackwardDataAlgorithm");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (gradDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'gradDesc' is null for cudnnFindConvolutionBackwardDataAlgorithm");
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
    Logger::log(LOG_TRACE, "Executing cudnnFindConvolutionBackwardDataAlgorithm(handle=%p, filterDesc=%p, diffDesc=%p, convDesc=%p, gradDesc=%p, requestedAlgoCount=%d, returnedAlgoCount=%p, perfResults=%p)\n",
        handle, filterDesc, diffDesc, convDesc, gradDesc, requestedAlgoCount, returnedAlgoCount, perfResults);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnFilterDescriptor_t filterDesc_native;
    cudnnTensorDescriptor_t diffDesc_native;
    cudnnConvolutionDescriptor_t convDesc_native;
    cudnnTensorDescriptor_t gradDesc_native;
    int requestedAlgoCount_native = 0;
    int returnedAlgoCount_native = 0;
    cudnnConvolutionBwdDataAlgoPerf_t* perfResults_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    filterDesc_native = (cudnnFilterDescriptor_t)getNativePointerValue(env, filterDesc);
    diffDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, diffDesc);
    convDesc_native = (cudnnConvolutionDescriptor_t)getNativePointerValue(env, convDesc);
    gradDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, gradDesc);
    requestedAlgoCount_native = (int)requestedAlgoCount;
    // returnedAlgoCount is set here
    if (!initNative(env, perfResults, perfResults_native, requestedAlgoCount)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnFindConvolutionBackwardDataAlgorithm(handle_native, filterDesc_native, diffDesc_native, convDesc_native, gradDesc_native, requestedAlgoCount_native, &returnedAlgoCount_native, perfResults_native);

    // Write back native variable values
    // handle is a read-only native pointer
    // filterDesc is a read-only native pointer
    // diffDesc is a read-only native pointer
    // convDesc is a read-only native pointer
    // gradDesc is a read-only native pointer
    // requestedAlgoCount is primitive
    if (!set(env, returnedAlgoCount, 0, (jint)returnedAlgoCount_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!releaseNative(env, perfResults_native, perfResults, returnedAlgoCount_native)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetConvolutionBackwardDataAlgorithmNative(JNIEnv *env, jclass cls, jobject handle, jobject filterDesc, jobject diffDesc, jobject convDesc, jobject gradDesc, jint preference, jlong memoryLimitInbytes, jintArray algo)
{
    // Null-checks for non-primitive arguments
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cudnnGetConvolutionBackwardDataAlgorithm");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (filterDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'filterDesc' is null for cudnnGetConvolutionBackwardDataAlgorithm");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (diffDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'diffDesc' is null for cudnnGetConvolutionBackwardDataAlgorithm");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (convDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'convDesc' is null for cudnnGetConvolutionBackwardDataAlgorithm");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (gradDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'gradDesc' is null for cudnnGetConvolutionBackwardDataAlgorithm");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    // preference is primitive
    // memoryLimitInbytes is primitive
    if (algo == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'algo' is null for cudnnGetConvolutionBackwardDataAlgorithm");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetConvolutionBackwardDataAlgorithm(handle=%p, filterDesc=%p, diffDesc=%p, convDesc=%p, gradDesc=%p, preference=%d, memoryLimitInbytes=%ld, algo=%p)\n",
        handle, filterDesc, diffDesc, convDesc, gradDesc, preference, memoryLimitInbytes, algo);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnFilterDescriptor_t filterDesc_native;
    cudnnTensorDescriptor_t diffDesc_native;
    cudnnConvolutionDescriptor_t convDesc_native;
    cudnnTensorDescriptor_t gradDesc_native;
    cudnnConvolutionBwdDataPreference_t preference_native;
    size_t memoryLimitInbytes_native = 0;
    cudnnConvolutionBwdDataAlgo_t algo_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    filterDesc_native = (cudnnFilterDescriptor_t)getNativePointerValue(env, filterDesc);
    diffDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, diffDesc);
    convDesc_native = (cudnnConvolutionDescriptor_t)getNativePointerValue(env, convDesc);
    gradDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, gradDesc);
    preference_native = (cudnnConvolutionBwdDataPreference_t)preference;
    memoryLimitInbytes_native = (size_t)memoryLimitInbytes;
    // algo is set here

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetConvolutionBackwardDataAlgorithm(handle_native, filterDesc_native, diffDesc_native, convDesc_native, gradDesc_native, preference_native, memoryLimitInbytes_native, &algo_native);

    // Write back native variable values
    // handle is a read-only native pointer
    // filterDesc is a read-only native pointer
    // diffDesc is a read-only native pointer
    // convDesc is a read-only native pointer
    // gradDesc is a read-only native pointer
    // preference is primitive
    // memoryLimitInbytes is primitive
    if (!set(env, algo, 0, (jint)algo_native)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

/** Helper function to return the minimum size of the workspace to be passed to the convolution given an algo*/
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetConvolutionBackwardDataWorkspaceSizeNative(JNIEnv *env, jclass cls, jobject handle, jobject filterDesc, jobject diffDesc, jobject convDesc, jobject gradDesc, jint algo, jlongArray sizeInBytes)
{
    // Null-checks for non-primitive arguments
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cudnnGetConvolutionBackwardDataWorkspaceSize");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (filterDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'filterDesc' is null for cudnnGetConvolutionBackwardDataWorkspaceSize");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (diffDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'diffDesc' is null for cudnnGetConvolutionBackwardDataWorkspaceSize");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (convDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'convDesc' is null for cudnnGetConvolutionBackwardDataWorkspaceSize");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (gradDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'gradDesc' is null for cudnnGetConvolutionBackwardDataWorkspaceSize");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    // algo is primitive
    if (sizeInBytes == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'sizeInBytes' is null for cudnnGetConvolutionBackwardDataWorkspaceSize");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetConvolutionBackwardDataWorkspaceSize(handle=%p, filterDesc=%p, diffDesc=%p, convDesc=%p, gradDesc=%p, algo=%d, sizeInBytes=%p)\n",
        handle, filterDesc, diffDesc, convDesc, gradDesc, algo, sizeInBytes);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnFilterDescriptor_t filterDesc_native;
    cudnnTensorDescriptor_t diffDesc_native;
    cudnnConvolutionDescriptor_t convDesc_native;
    cudnnTensorDescriptor_t gradDesc_native;
    cudnnConvolutionBwdDataAlgo_t algo_native;
    size_t sizeInBytes_native = 0;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    filterDesc_native = (cudnnFilterDescriptor_t)getNativePointerValue(env, filterDesc);
    diffDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, diffDesc);
    convDesc_native = (cudnnConvolutionDescriptor_t)getNativePointerValue(env, convDesc);
    gradDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, gradDesc);
    algo_native = (cudnnConvolutionBwdDataAlgo_t)algo;
    // sizeInBytes is set here

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetConvolutionBackwardDataWorkspaceSize(handle_native, filterDesc_native, diffDesc_native, convDesc_native, gradDesc_native, algo_native, &sizeInBytes_native);

    // Write back native variable values
    // handle is a read-only native pointer
    // filterDesc is a read-only native pointer
    // diffDesc is a read-only native pointer
    // convDesc is a read-only native pointer
    // gradDesc is a read-only native pointer
    // algo is primitive
    if (!set(env, sizeInBytes, 0, (jlong)sizeInBytes_native)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnConvolutionBackwardData_1v3Native(JNIEnv *env, jclass cls, jobject handle, jobject alpha, jobject filterDesc, jobject filterData, jobject diffDesc, jobject diffData, jobject convDesc, jint algo, jobject workSpace, jlong workSpaceSizeInBytes, jobject beta, jobject gradDesc, jobject gradData)
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
    if (filterDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'filterDesc' is null for cudnnConvolutionBackwardData_v3");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (filterData == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'filterData' is null for cudnnConvolutionBackwardData_v3");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (diffDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'diffDesc' is null for cudnnConvolutionBackwardData_v3");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (diffData == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'diffData' is null for cudnnConvolutionBackwardData_v3");
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
    if (gradDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'gradDesc' is null for cudnnConvolutionBackwardData_v3");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (gradData == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'gradData' is null for cudnnConvolutionBackwardData_v3");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnConvolutionBackwardData_v3(handle=%p, alpha=%p, filterDesc=%p, filterData=%p, diffDesc=%p, diffData=%p, convDesc=%p, algo=%d, workSpace=%p, workSpaceSizeInBytes=%ld, beta=%p, gradDesc=%p, gradData=%p)\n",
        handle, alpha, filterDesc, filterData, diffDesc, diffData, convDesc, algo, workSpace, workSpaceSizeInBytes, beta, gradDesc, gradData);

    // Native variable declarations
    cudnnHandle_t handle_native;
    void* alpha_native;
    cudnnFilterDescriptor_t filterDesc_native;
    void* filterData_native;
    cudnnTensorDescriptor_t diffDesc_native;
    void* diffData_native;
    cudnnConvolutionDescriptor_t convDesc_native;
    cudnnConvolutionBwdDataAlgo_t algo_native;
    void* workSpace_native;
    size_t workSpaceSizeInBytes_native = 0;
    void* beta_native;
    cudnnTensorDescriptor_t gradDesc_native;
    void* gradData_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    PointerData *alpha_pointerData = initPointerData(env, alpha);
    if (alpha_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    alpha_native = (void*)alpha_pointerData->getPointer(env);
    filterDesc_native = (cudnnFilterDescriptor_t)getNativePointerValue(env, filterDesc);
    filterData_native = (void*)getPointer(env, filterData);
    diffDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, diffDesc);
    diffData_native = (void*)getPointer(env, diffData);
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
    gradDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, gradDesc);
    gradData_native = (void*)getPointer(env, gradData);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnConvolutionBackwardData_v3(handle_native, alpha_native, filterDesc_native, filterData_native, diffDesc_native, diffData_native, convDesc_native, algo_native, workSpace_native, workSpaceSizeInBytes_native, beta_native, gradDesc_native, gradData_native);

    // Write back native variable values
    // handle is a read-only native pointer
    if (!releasePointerData(env, alpha_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // filterDesc is a read-only native pointer
    // filterData is a native pointer
    // diffDesc is a read-only native pointer
    // diffData is a native pointer
    // convDesc is a read-only native pointer
    // algo is primitive
    // workSpace is a native pointer
    // workSpaceSizeInBytes is primitive
    if (!releasePointerData(env, beta_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // gradDesc is a read-only native pointer
    // gradData is a native pointer

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnConvolutionBackwardDataNative(JNIEnv *env, jclass cls, jobject handle, jobject alpha, jobject filterDesc, jobject filterData, jobject diffDesc, jobject diffData, jobject convDesc, jobject beta, jobject gradDesc, jobject gradData)
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
    if (filterDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'filterDesc' is null for cudnnConvolutionBackwardData");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (filterData == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'filterData' is null for cudnnConvolutionBackwardData");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (diffDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'diffDesc' is null for cudnnConvolutionBackwardData");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (diffData == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'diffData' is null for cudnnConvolutionBackwardData");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (convDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'convDesc' is null for cudnnConvolutionBackwardData");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (beta == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'beta' is null for cudnnConvolutionBackwardData");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (gradDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'gradDesc' is null for cudnnConvolutionBackwardData");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (gradData == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'gradData' is null for cudnnConvolutionBackwardData");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnConvolutionBackwardData(handle=%p, alpha=%p, filterDesc=%p, filterData=%p, diffDesc=%p, diffData=%p, convDesc=%p, beta=%p, gradDesc=%p, gradData=%p)\n",
        handle, alpha, filterDesc, filterData, diffDesc, diffData, convDesc, beta, gradDesc, gradData);

    // Native variable declarations
    cudnnHandle_t handle_native;
    void* alpha_native;
    cudnnFilterDescriptor_t filterDesc_native;
    void* filterData_native;
    cudnnTensorDescriptor_t diffDesc_native;
    void* diffData_native;
    cudnnConvolutionDescriptor_t convDesc_native;
    void* beta_native;
    cudnnTensorDescriptor_t gradDesc_native;
    void* gradData_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    PointerData *alpha_pointerData = initPointerData(env, alpha);
    if (alpha_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    alpha_native = (void*)alpha_pointerData->getPointer(env);
    filterDesc_native = (cudnnFilterDescriptor_t)getNativePointerValue(env, filterDesc);
    filterData_native = (void*)getPointer(env, filterData);
    diffDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, diffDesc);
    diffData_native = (void*)getPointer(env, diffData);
    convDesc_native = (cudnnConvolutionDescriptor_t)getNativePointerValue(env, convDesc);
    PointerData *beta_pointerData = initPointerData(env, beta);
    if (beta_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    beta_native = (void*)beta_pointerData->getPointer(env);
    gradDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, gradDesc);
    gradData_native = (void*)getPointer(env, gradData);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnConvolutionBackwardData(handle_native, alpha_native, filterDesc_native, filterData_native, diffDesc_native, diffData_native, convDesc_native, beta_native, gradDesc_native, gradData_native);

    // Write back native variable values
    // handle is a read-only native pointer
    if (!releasePointerData(env, alpha_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // filterDesc is a read-only native pointer
    // filterData is a native pointer
    // diffDesc is a read-only native pointer
    // diffData is a native pointer
    // convDesc is a read-only native pointer
    if (!releasePointerData(env, beta_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // gradDesc is a read-only native pointer
    // gradData is a native pointer

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnIm2ColNative(JNIEnv *env, jclass cls, jobject handle, jobject srcDesc, jobject srcData, jobject filterDesc, jobject convDesc, jobject colBuffer)
{
    // Null-checks for non-primitive arguments
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cudnnIm2Col");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (srcDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'srcDesc' is null for cudnnIm2Col");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (srcData == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'srcData' is null for cudnnIm2Col");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (filterDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'filterDesc' is null for cudnnIm2Col");
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
    Logger::log(LOG_TRACE, "Executing cudnnIm2Col(handle=%p, srcDesc=%p, srcData=%p, filterDesc=%p, convDesc=%p, colBuffer=%p)\n",
        handle, srcDesc, srcData, filterDesc, convDesc, colBuffer);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnTensorDescriptor_t srcDesc_native;
    void* srcData_native;
    cudnnFilterDescriptor_t filterDesc_native;
    cudnnConvolutionDescriptor_t convDesc_native;
    void* colBuffer_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    srcDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, srcDesc);
    srcData_native = (void*)getPointer(env, srcData);
    filterDesc_native = (cudnnFilterDescriptor_t)getNativePointerValue(env, filterDesc);
    convDesc_native = (cudnnConvolutionDescriptor_t)getNativePointerValue(env, convDesc);
    colBuffer_native = (void*)getPointer(env, colBuffer);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnIm2Col(handle_native, srcDesc_native, srcData_native, filterDesc_native, convDesc_native, colBuffer_native);

    // Write back native variable values
    // handle is a read-only native pointer
    // srcDesc is a read-only native pointer
    // srcData is a native pointer
    // filterDesc is a read-only native pointer
    // convDesc is a read-only native pointer
    // colBuffer is a native pointer

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

/** Softmax functions: All of the form "output = alpha * Op(inputs) + beta * output" */
/** Function to perform forward softmax */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnSoftmaxForwardNative(JNIEnv *env, jclass cls, jobject handle, jint algorithm, jint mode, jobject alpha, jobject srcDesc, jobject srcData, jobject beta, jobject destDesc, jobject destData)
{
    // Null-checks for non-primitive arguments
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cudnnSoftmaxForward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    // algorithm is primitive
    // mode is primitive
    if (alpha == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'alpha' is null for cudnnSoftmaxForward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (srcDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'srcDesc' is null for cudnnSoftmaxForward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (srcData == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'srcData' is null for cudnnSoftmaxForward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (beta == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'beta' is null for cudnnSoftmaxForward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (destDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'destDesc' is null for cudnnSoftmaxForward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (destData == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'destData' is null for cudnnSoftmaxForward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnSoftmaxForward(handle=%p, algorithm=%d, mode=%d, alpha=%p, srcDesc=%p, srcData=%p, beta=%p, destDesc=%p, destData=%p)\n",
        handle, algorithm, mode, alpha, srcDesc, srcData, beta, destDesc, destData);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnSoftmaxAlgorithm_t algorithm_native;
    cudnnSoftmaxMode_t mode_native;
    void* alpha_native;
    cudnnTensorDescriptor_t srcDesc_native;
    void* srcData_native;
    void* beta_native;
    cudnnTensorDescriptor_t destDesc_native;
    void* destData_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    algorithm_native = (cudnnSoftmaxAlgorithm_t)algorithm;
    mode_native = (cudnnSoftmaxMode_t)mode;
    PointerData *alpha_pointerData = initPointerData(env, alpha);
    if (alpha_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    alpha_native = (void*)alpha_pointerData->getPointer(env);
    srcDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, srcDesc);
    srcData_native = (void*)getPointer(env, srcData);
    PointerData *beta_pointerData = initPointerData(env, beta);
    if (beta_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    beta_native = (void*)beta_pointerData->getPointer(env);
    destDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, destDesc);
    destData_native = (void*)getPointer(env, destData);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnSoftmaxForward(handle_native, algorithm_native, mode_native, alpha_native, srcDesc_native, srcData_native, beta_native, destDesc_native, destData_native);

    // Write back native variable values
    // handle is a read-only native pointer
    // algorithm is primitive
    // mode is primitive
    if (!releasePointerData(env, alpha_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // srcDesc is a read-only native pointer
    // srcData is a native pointer
    if (!releasePointerData(env, beta_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // destDesc is a read-only native pointer
    // destData is a native pointer

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

/** Function to perform backward softmax */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnSoftmaxBackwardNative(JNIEnv *env, jclass cls, jobject handle, jint algorithm, jint mode, jobject alpha, jobject srcDesc, jobject srcData, jobject srcDiffDesc, jobject srcDiffData, jobject beta, jobject destDiffDesc, jobject destDiffData)
{
    // Null-checks for non-primitive arguments
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cudnnSoftmaxBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    // algorithm is primitive
    // mode is primitive
    if (alpha == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'alpha' is null for cudnnSoftmaxBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (srcDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'srcDesc' is null for cudnnSoftmaxBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (srcData == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'srcData' is null for cudnnSoftmaxBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (srcDiffDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'srcDiffDesc' is null for cudnnSoftmaxBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (srcDiffData == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'srcDiffData' is null for cudnnSoftmaxBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (beta == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'beta' is null for cudnnSoftmaxBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (destDiffDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'destDiffDesc' is null for cudnnSoftmaxBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (destDiffData == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'destDiffData' is null for cudnnSoftmaxBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnSoftmaxBackward(handle=%p, algorithm=%d, mode=%d, alpha=%p, srcDesc=%p, srcData=%p, srcDiffDesc=%p, srcDiffData=%p, beta=%p, destDiffDesc=%p, destDiffData=%p)\n",
        handle, algorithm, mode, alpha, srcDesc, srcData, srcDiffDesc, srcDiffData, beta, destDiffDesc, destDiffData);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnSoftmaxAlgorithm_t algorithm_native;
    cudnnSoftmaxMode_t mode_native;
    void* alpha_native;
    cudnnTensorDescriptor_t srcDesc_native;
    void* srcData_native;
    cudnnTensorDescriptor_t srcDiffDesc_native;
    void* srcDiffData_native;
    void* beta_native;
    cudnnTensorDescriptor_t destDiffDesc_native;
    void* destDiffData_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    algorithm_native = (cudnnSoftmaxAlgorithm_t)algorithm;
    mode_native = (cudnnSoftmaxMode_t)mode;
    PointerData *alpha_pointerData = initPointerData(env, alpha);
    if (alpha_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    alpha_native = (void*)alpha_pointerData->getPointer(env);
    srcDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, srcDesc);
    srcData_native = (void*)getPointer(env, srcData);
    srcDiffDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, srcDiffDesc);
    srcDiffData_native = (void*)getPointer(env, srcDiffData);
    PointerData *beta_pointerData = initPointerData(env, beta);
    if (beta_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    beta_native = (void*)beta_pointerData->getPointer(env);
    destDiffDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, destDiffDesc);
    destDiffData_native = (void*)getPointer(env, destDiffData);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnSoftmaxBackward(handle_native, algorithm_native, mode_native, alpha_native, srcDesc_native, srcData_native, srcDiffDesc_native, srcDiffData_native, beta_native, destDiffDesc_native, destDiffData_native);

    // Write back native variable values
    // handle is a read-only native pointer
    // algorithm is primitive
    // mode is primitive
    if (!releasePointerData(env, alpha_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // srcDesc is a read-only native pointer
    // srcData is a native pointer
    // srcDiffDesc is a read-only native pointer
    // srcDiffData is a native pointer
    if (!releasePointerData(env, beta_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // destDiffDesc is a read-only native pointer
    // destDiffData is a native pointer

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

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetPooling2dForwardOutputDimNative(JNIEnv *env, jclass cls, jobject poolingDesc, jobject inputTensorDesc, jobject outN, jobject outC, jobject outH, jobject outW)
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
    if (outN == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'outN' is null for cudnnGetPooling2dForwardOutputDim");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (outC == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'outC' is null for cudnnGetPooling2dForwardOutputDim");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (outH == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'outH' is null for cudnnGetPooling2dForwardOutputDim");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (outW == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'outW' is null for cudnnGetPooling2dForwardOutputDim");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetPooling2dForwardOutputDim(poolingDesc=%p, inputTensorDesc=%p, outN=%p, outC=%p, outH=%p, outW=%p)\n",
        poolingDesc, inputTensorDesc, outN, outC, outH, outW);

    // Native variable declarations
    cudnnPoolingDescriptor_t poolingDesc_native;
    cudnnTensorDescriptor_t inputTensorDesc_native;
    int* outN_native;
    int* outC_native;
    int* outH_native;
    int* outW_native;

    // Obtain native variable values
    poolingDesc_native = (cudnnPoolingDescriptor_t)getNativePointerValue(env, poolingDesc);
    inputTensorDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, inputTensorDesc);
    outN_native = (int*)getPointer(env, outN);
    outC_native = (int*)getPointer(env, outC);
    outH_native = (int*)getPointer(env, outH);
    outW_native = (int*)getPointer(env, outW);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetPooling2dForwardOutputDim(poolingDesc_native, inputTensorDesc_native, outN_native, outC_native, outH_native, outW_native);

    // Write back native variable values
    // poolingDesc is a read-only native pointer
    // inputTensorDesc is a read-only native pointer
    // outN is a native pointer
    // outC is a native pointer
    // outH is a native pointer
    // outW is a native pointer

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
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnPoolingForwardNative(JNIEnv *env, jclass cls, jobject handle, jobject poolingDesc, jobject alpha, jobject srcDesc, jobject srcData, jobject beta, jobject destDesc, jobject destData)
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
    if (srcDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'srcDesc' is null for cudnnPoolingForward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (srcData == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'srcData' is null for cudnnPoolingForward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (beta == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'beta' is null for cudnnPoolingForward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (destDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'destDesc' is null for cudnnPoolingForward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (destData == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'destData' is null for cudnnPoolingForward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnPoolingForward(handle=%p, poolingDesc=%p, alpha=%p, srcDesc=%p, srcData=%p, beta=%p, destDesc=%p, destData=%p)\n",
        handle, poolingDesc, alpha, srcDesc, srcData, beta, destDesc, destData);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnPoolingDescriptor_t poolingDesc_native;
    void* alpha_native;
    cudnnTensorDescriptor_t srcDesc_native;
    void* srcData_native;
    void* beta_native;
    cudnnTensorDescriptor_t destDesc_native;
    void* destData_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    poolingDesc_native = (cudnnPoolingDescriptor_t)getNativePointerValue(env, poolingDesc);
    PointerData *alpha_pointerData = initPointerData(env, alpha);
    if (alpha_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    alpha_native = (void*)alpha_pointerData->getPointer(env);
    srcDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, srcDesc);
    srcData_native = (void*)getPointer(env, srcData);
    PointerData *beta_pointerData = initPointerData(env, beta);
    if (beta_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    beta_native = (void*)beta_pointerData->getPointer(env);
    destDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, destDesc);
    destData_native = (void*)getPointer(env, destData);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnPoolingForward(handle_native, poolingDesc_native, alpha_native, srcDesc_native, srcData_native, beta_native, destDesc_native, destData_native);

    // Write back native variable values
    // handle is a read-only native pointer
    // poolingDesc is a read-only native pointer
    if (!releasePointerData(env, alpha_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // srcDesc is a read-only native pointer
    // srcData is a native pointer
    if (!releasePointerData(env, beta_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // destDesc is a read-only native pointer
    // destData is a native pointer

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

/** Function to perform backward pooling */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnPoolingBackwardNative(JNIEnv *env, jclass cls, jobject handle, jobject poolingDesc, jobject alpha, jobject srcDesc, jobject srcData, jobject srcDiffDesc, jobject srcDiffData, jobject destDesc, jobject destData, jobject beta, jobject destDiffDesc, jobject destDiffData)
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
    if (srcDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'srcDesc' is null for cudnnPoolingBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (srcData == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'srcData' is null for cudnnPoolingBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (srcDiffDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'srcDiffDesc' is null for cudnnPoolingBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (srcDiffData == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'srcDiffData' is null for cudnnPoolingBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (destDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'destDesc' is null for cudnnPoolingBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (destData == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'destData' is null for cudnnPoolingBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (beta == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'beta' is null for cudnnPoolingBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (destDiffDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'destDiffDesc' is null for cudnnPoolingBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (destDiffData == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'destDiffData' is null for cudnnPoolingBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnPoolingBackward(handle=%p, poolingDesc=%p, alpha=%p, srcDesc=%p, srcData=%p, srcDiffDesc=%p, srcDiffData=%p, destDesc=%p, destData=%p, beta=%p, destDiffDesc=%p, destDiffData=%p)\n",
        handle, poolingDesc, alpha, srcDesc, srcData, srcDiffDesc, srcDiffData, destDesc, destData, beta, destDiffDesc, destDiffData);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnPoolingDescriptor_t poolingDesc_native;
    void* alpha_native;
    cudnnTensorDescriptor_t srcDesc_native;
    void* srcData_native;
    cudnnTensorDescriptor_t srcDiffDesc_native;
    void* srcDiffData_native;
    cudnnTensorDescriptor_t destDesc_native;
    void* destData_native;
    void* beta_native;
    cudnnTensorDescriptor_t destDiffDesc_native;
    void* destDiffData_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    poolingDesc_native = (cudnnPoolingDescriptor_t)getNativePointerValue(env, poolingDesc);
    PointerData *alpha_pointerData = initPointerData(env, alpha);
    if (alpha_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    alpha_native = (void*)alpha_pointerData->getPointer(env);
    srcDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, srcDesc);
    srcData_native = (void*)getPointer(env, srcData);
    srcDiffDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, srcDiffDesc);
    srcDiffData_native = (void*)getPointer(env, srcDiffData);
    destDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, destDesc);
    destData_native = (void*)getPointer(env, destData);
    PointerData *beta_pointerData = initPointerData(env, beta);
    if (beta_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    beta_native = (void*)beta_pointerData->getPointer(env);
    destDiffDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, destDiffDesc);
    destDiffData_native = (void*)getPointer(env, destDiffData);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnPoolingBackward(handle_native, poolingDesc_native, alpha_native, srcDesc_native, srcData_native, srcDiffDesc_native, srcDiffData_native, destDesc_native, destData_native, beta_native, destDiffDesc_native, destDiffData_native);

    // Write back native variable values
    // handle is a read-only native pointer
    // poolingDesc is a read-only native pointer
    if (!releasePointerData(env, alpha_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // srcDesc is a read-only native pointer
    // srcData is a native pointer
    // srcDiffDesc is a read-only native pointer
    // srcDiffData is a native pointer
    // destDesc is a read-only native pointer
    // destData is a native pointer
    if (!releasePointerData(env, beta_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // destDiffDesc is a read-only native pointer
    // destDiffData is a native pointer

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

/** Activation functions: All of the form "output = alpha * Op(inputs) + beta * output" */
/** Function to perform forward activation  */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnActivationForwardNative(JNIEnv *env, jclass cls, jobject handle, jint mode, jobject alpha, jobject srcDesc, jobject srcData, jobject beta, jobject destDesc, jobject destData)
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
    if (srcDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'srcDesc' is null for cudnnActivationForward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (srcData == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'srcData' is null for cudnnActivationForward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (beta == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'beta' is null for cudnnActivationForward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (destDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'destDesc' is null for cudnnActivationForward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (destData == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'destData' is null for cudnnActivationForward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnActivationForward(handle=%p, mode=%d, alpha=%p, srcDesc=%p, srcData=%p, beta=%p, destDesc=%p, destData=%p)\n",
        handle, mode, alpha, srcDesc, srcData, beta, destDesc, destData);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnActivationMode_t mode_native;
    void* alpha_native;
    cudnnTensorDescriptor_t srcDesc_native;
    void* srcData_native;
    void* beta_native;
    cudnnTensorDescriptor_t destDesc_native;
    void* destData_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    mode_native = (cudnnActivationMode_t)mode;
    PointerData *alpha_pointerData = initPointerData(env, alpha);
    if (alpha_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    alpha_native = (void*)alpha_pointerData->getPointer(env);
    srcDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, srcDesc);
    srcData_native = (void*)getPointer(env, srcData);
    PointerData *beta_pointerData = initPointerData(env, beta);
    if (beta_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    beta_native = (void*)beta_pointerData->getPointer(env);
    destDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, destDesc);
    destData_native = (void*)getPointer(env, destData);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnActivationForward(handle_native, mode_native, alpha_native, srcDesc_native, srcData_native, beta_native, destDesc_native, destData_native);

    // Write back native variable values
    // handle is a read-only native pointer
    // mode is primitive
    if (!releasePointerData(env, alpha_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // srcDesc is a read-only native pointer
    // srcData is a native pointer
    if (!releasePointerData(env, beta_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // destDesc is a read-only native pointer
    // destData is a native pointer

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

/** Function to perform backward activation  */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnActivationBackwardNative(JNIEnv *env, jclass cls, jobject handle, jint mode, jobject alpha, jobject srcDesc, jobject srcData, jobject srcDiffDesc, jobject srcDiffData, jobject destDesc, jobject destData, jobject beta, jobject destDiffDesc, jobject destDiffData)
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
    if (srcDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'srcDesc' is null for cudnnActivationBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (srcData == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'srcData' is null for cudnnActivationBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (srcDiffDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'srcDiffDesc' is null for cudnnActivationBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (srcDiffData == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'srcDiffData' is null for cudnnActivationBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (destDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'destDesc' is null for cudnnActivationBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (destData == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'destData' is null for cudnnActivationBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (beta == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'beta' is null for cudnnActivationBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (destDiffDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'destDiffDesc' is null for cudnnActivationBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (destDiffData == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'destDiffData' is null for cudnnActivationBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnActivationBackward(handle=%p, mode=%d, alpha=%p, srcDesc=%p, srcData=%p, srcDiffDesc=%p, srcDiffData=%p, destDesc=%p, destData=%p, beta=%p, destDiffDesc=%p, destDiffData=%p)\n",
        handle, mode, alpha, srcDesc, srcData, srcDiffDesc, srcDiffData, destDesc, destData, beta, destDiffDesc, destDiffData);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnActivationMode_t mode_native;
    void* alpha_native;
    cudnnTensorDescriptor_t srcDesc_native;
    void* srcData_native;
    cudnnTensorDescriptor_t srcDiffDesc_native;
    void* srcDiffData_native;
    cudnnTensorDescriptor_t destDesc_native;
    void* destData_native;
    void* beta_native;
    cudnnTensorDescriptor_t destDiffDesc_native;
    void* destDiffData_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    mode_native = (cudnnActivationMode_t)mode;
    PointerData *alpha_pointerData = initPointerData(env, alpha);
    if (alpha_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    alpha_native = (void*)alpha_pointerData->getPointer(env);
    srcDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, srcDesc);
    srcData_native = (void*)getPointer(env, srcData);
    srcDiffDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, srcDiffDesc);
    srcDiffData_native = (void*)getPointer(env, srcDiffData);
    destDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, destDesc);
    destData_native = (void*)getPointer(env, destData);
    PointerData *beta_pointerData = initPointerData(env, beta);
    if (beta_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    beta_native = (void*)beta_pointerData->getPointer(env);
    destDiffDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, destDiffDesc);
    destDiffData_native = (void*)getPointer(env, destDiffData);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnActivationBackward(handle_native, mode_native, alpha_native, srcDesc_native, srcData_native, srcDiffDesc_native, srcDiffData_native, destDesc_native, destData_native, beta_native, destDiffDesc_native, destDiffData_native);

    // Write back native variable values
    // handle is a read-only native pointer
    // mode is primitive
    if (!releasePointerData(env, alpha_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // srcDesc is a read-only native pointer
    // srcData is a native pointer
    // srcDiffDesc is a read-only native pointer
    // srcDiffData is a native pointer
    // destDesc is a read-only native pointer
    // destData is a native pointer
    if (!releasePointerData(env, beta_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // destDiffDesc is a read-only native pointer
    // destDiffData is a native pointer

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

// Create an instance of LRN (Local Response Normalization) descriptor
// This function will set lrnN=5, lrnAlpha=1e-4, lrnBeta=0.75, lrnK=2.0 as defaults from Krizhevsky'12 ImageNet paper
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

// LRN uses a window [center-lookBehind, center+lookAhead], where
// lookBehind = floor( (lrnN-1)/2 ), lookAhead = lrnN-lookBehind-1.
// So for n=10, the window is [k-4...k...k+5] with a total of 10 samples.
// Values of double parameters will be cast down to tensor data type.
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

// Retrieve the settings currently stored in an LRN layer descriptor
// Any of the provided pointers can be NULL (no corresponding value will be returned)
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

// Destroy an instance of LRN descriptor
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

// LRN functions: of the form "output = alpha * normalize(srcData) + beta * destData"
// Function to perform LRN forward cross-channel computation
// Values of double parameters will be cast down to tensor data type
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnLRNCrossChannelForwardNative(JNIEnv *env, jclass cls, jobject handle, jobject normDesc, jint lrnMode, jobject alpha, jobject srcDesc, jobject srcData, jobject beta, jobject destDesc, jobject destData)
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
    if (srcDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'srcDesc' is null for cudnnLRNCrossChannelForward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (srcData == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'srcData' is null for cudnnLRNCrossChannelForward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (beta == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'beta' is null for cudnnLRNCrossChannelForward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (destDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'destDesc' is null for cudnnLRNCrossChannelForward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (destData == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'destData' is null for cudnnLRNCrossChannelForward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnLRNCrossChannelForward(handle=%p, normDesc=%p, lrnMode=%d, alpha=%p, srcDesc=%p, srcData=%p, beta=%p, destDesc=%p, destData=%p)\n",
        handle, normDesc, lrnMode, alpha, srcDesc, srcData, beta, destDesc, destData);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnLRNDescriptor_t normDesc_native;
    cudnnLRNMode_t lrnMode_native;
    void* alpha_native;
    cudnnTensorDescriptor_t srcDesc_native;
    void* srcData_native;
    void* beta_native;
    cudnnTensorDescriptor_t destDesc_native;
    void* destData_native;

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
    srcDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, srcDesc);
    srcData_native = (void*)getPointer(env, srcData);
    PointerData *beta_pointerData = initPointerData(env, beta);
    if (beta_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    beta_native = (void*)beta_pointerData->getPointer(env);
    destDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, destDesc);
    destData_native = (void*)getPointer(env, destData);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnLRNCrossChannelForward(handle_native, normDesc_native, lrnMode_native, alpha_native, srcDesc_native, srcData_native, beta_native, destDesc_native, destData_native);

    // Write back native variable values
    // handle is a read-only native pointer
    // normDesc is a read-only native pointer
    // lrnMode is primitive
    if (!releasePointerData(env, alpha_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // srcDesc is a read-only native pointer
    // srcData is a native pointer
    if (!releasePointerData(env, beta_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // destDesc is a read-only native pointer
    // destData is a native pointer

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

// Function to perform LRN cross-channel backpropagation
// values of double parameters will be cast down to tensor data type
// src is the front layer, dst is the back layer
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnLRNCrossChannelBackwardNative(JNIEnv *env, jclass cls, jobject handle, jobject normDesc, jint lrnMode, jobject alpha, jobject srcDesc, jobject srcData, jobject srcDiffDesc, jobject srcDiffData, jobject destDesc, jobject destData, jobject beta, jobject destDiffDesc, jobject destDiffData)
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
    if (srcDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'srcDesc' is null for cudnnLRNCrossChannelBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (srcData == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'srcData' is null for cudnnLRNCrossChannelBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (srcDiffDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'srcDiffDesc' is null for cudnnLRNCrossChannelBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (srcDiffData == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'srcDiffData' is null for cudnnLRNCrossChannelBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (destDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'destDesc' is null for cudnnLRNCrossChannelBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (destData == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'destData' is null for cudnnLRNCrossChannelBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (beta == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'beta' is null for cudnnLRNCrossChannelBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (destDiffDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'destDiffDesc' is null for cudnnLRNCrossChannelBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (destDiffData == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'destDiffData' is null for cudnnLRNCrossChannelBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnLRNCrossChannelBackward(handle=%p, normDesc=%p, lrnMode=%d, alpha=%p, srcDesc=%p, srcData=%p, srcDiffDesc=%p, srcDiffData=%p, destDesc=%p, destData=%p, beta=%p, destDiffDesc=%p, destDiffData=%p)\n",
        handle, normDesc, lrnMode, alpha, srcDesc, srcData, srcDiffDesc, srcDiffData, destDesc, destData, beta, destDiffDesc, destDiffData);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnLRNDescriptor_t normDesc_native;
    cudnnLRNMode_t lrnMode_native;
    void* alpha_native;
    cudnnTensorDescriptor_t srcDesc_native;
    void* srcData_native;
    cudnnTensorDescriptor_t srcDiffDesc_native;
    void* srcDiffData_native;
    cudnnTensorDescriptor_t destDesc_native;
    void* destData_native;
    void* beta_native;
    cudnnTensorDescriptor_t destDiffDesc_native;
    void* destDiffData_native;

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
    srcDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, srcDesc);
    srcData_native = (void*)getPointer(env, srcData);
    srcDiffDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, srcDiffDesc);
    srcDiffData_native = (void*)getPointer(env, srcDiffData);
    destDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, destDesc);
    destData_native = (void*)getPointer(env, destData);
    PointerData *beta_pointerData = initPointerData(env, beta);
    if (beta_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    beta_native = (void*)beta_pointerData->getPointer(env);
    destDiffDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, destDiffDesc);
    destDiffData_native = (void*)getPointer(env, destDiffData);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnLRNCrossChannelBackward(handle_native, normDesc_native, lrnMode_native, alpha_native, srcDesc_native, srcData_native, srcDiffDesc_native, srcDiffData_native, destDesc_native, destData_native, beta_native, destDiffDesc_native, destDiffData_native);

    // Write back native variable values
    // handle is a read-only native pointer
    // normDesc is a read-only native pointer
    // lrnMode is primitive
    if (!releasePointerData(env, alpha_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // srcDesc is a read-only native pointer
    // srcData is a native pointer
    // srcDiffDesc is a read-only native pointer
    // srcDiffData is a native pointer
    // destDesc is a read-only native pointer
    // destData is a native pointer
    if (!releasePointerData(env, beta_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // destDiffDesc is a read-only native pointer
    // destDiffData is a native pointer

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

// LCN/divisive normalization functions: of the form "output = alpha * normalize(srcData) + beta * destData"
// srcMeansData can be NULL to reproduce Caffe's LRN within-channel behavior
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnDivisiveNormalizationForwardNative(JNIEnv *env, jclass cls, jobject handle, jobject normDesc, jint mode, jobject alpha, jobject srcDesc, jobject srcData, jobject srcMeansData, jobject tempData, jobject tempData2, jobject beta, jobject destDesc, jobject destData)
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
    if (srcDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'srcDesc' is null for cudnnDivisiveNormalizationForward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (srcData == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'srcData' is null for cudnnDivisiveNormalizationForward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    // srcMeansData may be NULL
    if (tempData == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'tempData' is null for cudnnDivisiveNormalizationForward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (tempData2 == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'tempData2' is null for cudnnDivisiveNormalizationForward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (beta == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'beta' is null for cudnnDivisiveNormalizationForward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (destDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'destDesc' is null for cudnnDivisiveNormalizationForward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (destData == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'destData' is null for cudnnDivisiveNormalizationForward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnDivisiveNormalizationForward(handle=%p, normDesc=%p, mode=%d, alpha=%p, srcDesc=%p, srcData=%p, srcMeansData=%p, tempData=%p, tempData2=%p, beta=%p, destDesc=%p, destData=%p)\n",
        handle, normDesc, mode, alpha, srcDesc, srcData, srcMeansData, tempData, tempData2, beta, destDesc, destData);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnLRNDescriptor_t normDesc_native;
    cudnnDivNormMode_t mode_native;
    void* alpha_native;
    cudnnTensorDescriptor_t srcDesc_native;
    void* srcData_native;
    void* srcMeansData_native;
    void* tempData_native;
    void* tempData2_native;
    void* beta_native;
    cudnnTensorDescriptor_t destDesc_native;
    void* destData_native;

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
    srcDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, srcDesc);
    srcData_native = (void*)getPointer(env, srcData);
    if (srcMeansData != NULL)
    {
        srcMeansData_native = (void*)getNativePointerValue(env, srcMeansData);
    }
    tempData_native = (void*)getPointer(env, tempData);
    tempData2_native = (void*)getPointer(env, tempData2);
    PointerData *beta_pointerData = initPointerData(env, beta);
    if (beta_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    beta_native = (void*)beta_pointerData->getPointer(env);
    destDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, destDesc);
    destData_native = (void*)getPointer(env, destData);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnDivisiveNormalizationForward(handle_native, normDesc_native, mode_native, alpha_native, srcDesc_native, srcData_native, srcMeansData_native, tempData_native, tempData2_native, beta_native, destDesc_native, destData_native);

    // Write back native variable values
    // handle is a read-only native pointer
    // normDesc is a read-only native pointer
    // mode is primitive
    if (!releasePointerData(env, alpha_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // srcDesc is a read-only native pointer
    // srcData is a native pointer
    // srcMeansData is a read-only native pointer
    // tempData is a native pointer
    // tempData2 is a native pointer
    if (!releasePointerData(env, beta_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // destDesc is a read-only native pointer
    // destData is a native pointer

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnDivisiveNormalizationBackwardNative(JNIEnv *env, jclass cls, jobject handle, jobject normDesc, jint mode, jobject alpha, jobject srcDesc, jobject srcData, jobject srcMeansData, jobject srcDiffData, jobject tempData, jobject tempData2, jobject betaData, jobject destDataDesc, jobject destDataDiff, jobject destMeansDiff)
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
    if (srcDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'srcDesc' is null for cudnnDivisiveNormalizationBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (srcData == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'srcData' is null for cudnnDivisiveNormalizationBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (srcMeansData == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'srcMeansData' is null for cudnnDivisiveNormalizationBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (srcDiffData == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'srcDiffData' is null for cudnnDivisiveNormalizationBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (tempData == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'tempData' is null for cudnnDivisiveNormalizationBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (tempData2 == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'tempData2' is null for cudnnDivisiveNormalizationBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (betaData == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'betaData' is null for cudnnDivisiveNormalizationBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (destDataDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'destDataDesc' is null for cudnnDivisiveNormalizationBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (destDataDiff == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'destDataDiff' is null for cudnnDivisiveNormalizationBackward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    // destMeansDiff may be NULL

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnDivisiveNormalizationBackward(handle=%p, normDesc=%p, mode=%d, alpha=%p, srcDesc=%p, srcData=%p, srcMeansData=%p, srcDiffData=%p, tempData=%p, tempData2=%p, betaData=%p, destDataDesc=%p, destDataDiff=%p, destMeansDiff=%p)\n",
        handle, normDesc, mode, alpha, srcDesc, srcData, srcMeansData, srcDiffData, tempData, tempData2, betaData, destDataDesc, destDataDiff, destMeansDiff);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnLRNDescriptor_t normDesc_native;
    cudnnDivNormMode_t mode_native;
    void* alpha_native;
    cudnnTensorDescriptor_t srcDesc_native;
    void* srcData_native;
    void* srcMeansData_native;
    void* srcDiffData_native;
    void* tempData_native;
    void* tempData2_native;
    void* betaData_native;
    cudnnTensorDescriptor_t destDataDesc_native;
    void* destDataDiff_native;
    void* destMeansDiff_native;

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
    srcDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, srcDesc);
    srcData_native = (void*)getPointer(env, srcData);
    srcMeansData_native = (void*)getPointer(env, srcMeansData);
    srcDiffData_native = (void*)getPointer(env, srcDiffData);
    tempData_native = (void*)getPointer(env, tempData);
    tempData2_native = (void*)getPointer(env, tempData2);
    betaData_native = (void*)getPointer(env, betaData);
    destDataDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, destDataDesc);
    destDataDiff_native = (void*)getPointer(env, destDataDiff);
    if (destMeansDiff != NULL)
    {
        destMeansDiff_native = (void*)getNativePointerValue(env, destMeansDiff);
    }

    // Native function call
    cudnnStatus_t jniResult_native = cudnnDivisiveNormalizationBackward(handle_native, normDesc_native, mode_native, alpha_native, srcDesc_native, srcData_native, srcMeansData_native, srcDiffData_native, tempData_native, tempData2_native, betaData_native, destDataDesc_native, destDataDiff_native, destMeansDiff_native);

    // Write back native variable values
    // handle is a read-only native pointer
    // normDesc is a read-only native pointer
    // mode is primitive
    if (!releasePointerData(env, alpha_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // srcDesc is a read-only native pointer
    // srcData is a native pointer
    // srcMeansData is a native pointer
    // srcDiffData is a native pointer
    // tempData is a native pointer
    // tempData2 is a native pointer
    // betaData is a native pointer
    // destDataDesc is a read-only native pointer
    // destDataDiff is a native pointer
    // destMeansDiff is a read-only native pointer

    // Return the result
    jint jniResult;
    jniResult = (jint)jniResult_native;
    return jniResult;
}

