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

#include "JCudnn.hpp"
#include "JCudnn_common.hpp"
#include <iostream>
#include <string>
#include <map>

// (Note: cudnnDebug is only used in callbacks - they are not supported yet)
 // Field IDs for the cudnnDebug class
jfieldID fid_cudnnDebug_cudnn_version; // unsigned int
jfieldID fid_cudnnDebug_cudnnStatus; // cudnnStatus_t
jfieldID fid_cudnnDebug_time_sec; // unsigned int
jfieldID fid_cudnnDebug_time_usec; // unsigned int
jfieldID fid_cudnnDebug_time_delta; // unsigned int
jfieldID fid_cudnnDebug_handle; // cudnnHandle_t
jfieldID fid_cudnnDebug_stream; // cudaStream_t
jfieldID fid_cudnnDebug_pid; // unsigned long long 
jfieldID fid_cudnnDebug_tid; // unsigned long long 
jfieldID fid_cudnnDebug_cudaDeviceId; // int

// Class and method ID for cudnnConvolutionFwdAlgoPerf and its constructor
jclass cudnnConvolutionFwdAlgoPerf_Class;
jmethodID cudnnConvolutionFwdAlgoPerf_Constructor;

// Field IDs for the cudnnConvolutionFwdAlgoPerf class
jfieldID fid_cudnnConvolutionFwdAlgoPerf_algo; // cudnnConvolutionFwdAlgo_t
jfieldID fid_cudnnConvolutionFwdAlgoPerf_status; // cudnnStatus_t
jfieldID fid_cudnnConvolutionFwdAlgoPerf_time; // float
jfieldID fid_cudnnConvolutionFwdAlgoPerf_memory; // size_t
jfieldID fid_cudnnConvolutionFwdAlgoPerf_determinism; // cudnnDeterminism_t
jfieldID fid_cudnnConvolutionFwdAlgoPerf_mathType; // cudnnMathType_t

// Class and method ID for cudnnConvolutionBwdDataAlgoPerf and its constructor
jclass cudnnConvolutionBwdDataAlgoPerf_Class;
jmethodID cudnnConvolutionBwdDataAlgoPerf_Constructor;

// Field IDs for the cudnnConvolutionBwdDataAlgoPerf class
jfieldID fid_cudnnConvolutionBwdDataAlgoPerf_algo; // cudnnConvolutionBwdDataAlgo_t
jfieldID fid_cudnnConvolutionBwdDataAlgoPerf_status; // cudnnStatus_t
jfieldID fid_cudnnConvolutionBwdDataAlgoPerf_time; // float
jfieldID fid_cudnnConvolutionBwdDataAlgoPerf_memory; // size_t
jfieldID fid_cudnnConvolutionBwdDataAlgoPerf_determinism; // cudnnDeterminism_t
jfieldID fid_cudnnConvolutionBwdDataAlgoPerf_mathType; // cudnnMathType_t

// Class and method ID for cudnnConvolutionBwdFilterAlgoPerf and its constructor
jclass cudnnConvolutionBwdFilterAlgoPerf_Class;
jmethodID cudnnConvolutionBwdFilterAlgoPerf_Constructor;

// Field IDs for the cudnnConvolutionBwdFilterAlgoPerf class
jfieldID fid_cudnnConvolutionBwdFilterAlgoPerf_algo; // cudnnConvolutionBwdFilterAlgo_t
jfieldID fid_cudnnConvolutionBwdFilterAlgoPerf_status; // cudnnStatus_t
jfieldID fid_cudnnConvolutionBwdFilterAlgoPerf_time; // float
jfieldID fid_cudnnConvolutionBwdFilterAlgoPerf_memory; // size_t
jfieldID fid_cudnnConvolutionBwdFilterAlgoPerf_determinism; // cudnnDeterminism_t
jfieldID fid_cudnnConvolutionBwdFilterAlgoPerf_mathType; // cudnnMathType_t
jfieldID fid_cudnnConvolutionBwdFilterAlgoPerf_reserved; // int[3]

// Class and method ID for cudnnAlgorithmPerformance and its constructor
jclass cudnnAlgorithmPerformance_Class;
jmethodID cudnnAlgorithmPerformance_Constructor;

// Static method ID for the cudnnCallback#call function
jmethodID cudnnCallback_call; 


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
    if (!init(env, cudnnAlgorithmPerformance_Class,         cudnnAlgorithmPerformance_Constructor,         "jcuda/jcudnn/cudnnAlgorithmPerformance"        )) return JNI_ERR;

    jclass cls = NULL;

    // Initialize the field IDs for the cudnnDebug class 
    if (!init(env, cls, "jcuda/jcudnn/cudnnDebug")) return JNI_ERR;
    if (!init(env, cls, fid_cudnnDebug_cudnn_version , "cudnn_version", "I"                           )) return JNI_ERR;
    if (!init(env, cls, fid_cudnnDebug_cudnnStatus   , "cudnnStatus"  , "I"                           )) return JNI_ERR;
    if (!init(env, cls, fid_cudnnDebug_time_sec      , "time_sec"     , "I"                           )) return JNI_ERR;
    if (!init(env, cls, fid_cudnnDebug_time_usec     , "time_usec"    , "I"                           )) return JNI_ERR;
    if (!init(env, cls, fid_cudnnDebug_time_delta    , "time_delta"   , "I"                           )) return JNI_ERR;
    if (!init(env, cls, fid_cudnnDebug_handle        , "handle"       , "Ljcuda/jcudnn/cudnnHandle;"  )) return JNI_ERR;
    if (!init(env, cls, fid_cudnnDebug_stream        , "stream"       , "Ljcuda/runtime/cudaStream_t;")) return JNI_ERR;
    if (!init(env, cls, fid_cudnnDebug_pid           , "pid"          , "J"                           )) return JNI_ERR;
    if (!init(env, cls, fid_cudnnDebug_tid           , "tid"          , "J"                           )) return JNI_ERR;
    if (!init(env, cls, fid_cudnnDebug_cudaDeviceId  , "cudaDeviceId" , "I"                           )) return JNI_ERR;

    // Initialize the field IDs for the cudnnConvolutionFwdAlgoPerf class
    if (!init(env, cls, "jcuda/jcudnn/cudnnConvolutionFwdAlgoPerf")) return JNI_ERR;
    if (!init(env, cls, fid_cudnnConvolutionFwdAlgoPerf_algo        , "algo"       , "I" )) return JNI_ERR;
    if (!init(env, cls, fid_cudnnConvolutionFwdAlgoPerf_status      , "status"     , "I" )) return JNI_ERR;
    if (!init(env, cls, fid_cudnnConvolutionFwdAlgoPerf_time        , "time"       , "F" )) return JNI_ERR;
    if (!init(env, cls, fid_cudnnConvolutionFwdAlgoPerf_memory      , "memory"     , "J" )) return JNI_ERR;
    if (!init(env, cls, fid_cudnnConvolutionFwdAlgoPerf_determinism , "determinism", "I" )) return JNI_ERR;
    if (!init(env, cls, fid_cudnnConvolutionFwdAlgoPerf_mathType    , "mathType"   , "I" )) return JNI_ERR;

    // Initialize the field IDs for the cudnnConvolutionBwdDataAlgoPerf class
    if (!init(env, cls, "jcuda/jcudnn/cudnnConvolutionBwdDataAlgoPerf")) return JNI_ERR;
    if (!init(env, cls, fid_cudnnConvolutionBwdDataAlgoPerf_algo        , "algo"       , "I" )) return JNI_ERR;
    if (!init(env, cls, fid_cudnnConvolutionBwdDataAlgoPerf_status      , "status"     , "I" )) return JNI_ERR;
    if (!init(env, cls, fid_cudnnConvolutionBwdDataAlgoPerf_time        , "time"       , "F" )) return JNI_ERR;
    if (!init(env, cls, fid_cudnnConvolutionBwdDataAlgoPerf_memory      , "memory"     , "J" )) return JNI_ERR;
    if (!init(env, cls, fid_cudnnConvolutionBwdDataAlgoPerf_determinism , "determinism", "I" )) return JNI_ERR;
    if (!init(env, cls, fid_cudnnConvolutionBwdDataAlgoPerf_mathType    , "mathType"   , "I" )) return JNI_ERR;

    // Initialize the field IDs for the cudnnConvolutionBwdFilterAlgoPerf class
    if (!init(env, cls, "jcuda/jcudnn/cudnnConvolutionBwdFilterAlgoPerf")) return JNI_ERR;
    if (!init(env, cls, fid_cudnnConvolutionBwdFilterAlgoPerf_algo        , "algo"       , "I" )) return JNI_ERR;
    if (!init(env, cls, fid_cudnnConvolutionBwdFilterAlgoPerf_status      , "status"     , "I" )) return JNI_ERR;
    if (!init(env, cls, fid_cudnnConvolutionBwdFilterAlgoPerf_time        , "time"       , "F" )) return JNI_ERR;
    if (!init(env, cls, fid_cudnnConvolutionBwdFilterAlgoPerf_memory      , "memory"     , "J" )) return JNI_ERR;
    if (!init(env, cls, fid_cudnnConvolutionBwdFilterAlgoPerf_determinism , "determinism", "I" )) return JNI_ERR;
    if (!init(env, cls, fid_cudnnConvolutionBwdFilterAlgoPerf_mathType    , "mathType"   , "I" )) return JNI_ERR;

    // Obtain the methodID for jcuda.jcudnn.cudnnCallback#call
    if (!init(env, cls, "jcuda/jcudnn/cudnnCallback")) return JNI_ERR;
    if (!init(env, cls, cudnnCallback_call, "call", "(ILjava/lang/Object;Ljcuda/jcudnn/cudnnDebug;Ljava/lang/String;)V")) return JNI_ERR;

    return JNI_VERSION_1_4;
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
bool releaseNative(JNIEnv *env, cudnnConvolutionFwdAlgoPerf_t &nativeObject, jobject &javaObject)
{
    env->SetIntField(javaObject, fid_cudnnConvolutionFwdAlgoPerf_algo, (jint)nativeObject.algo);
    env->SetIntField(javaObject, fid_cudnnConvolutionFwdAlgoPerf_status, (jint)nativeObject.status);
    env->SetFloatField(javaObject, fid_cudnnConvolutionFwdAlgoPerf_time, (jfloat)nativeObject.time);
    env->SetLongField(javaObject, fid_cudnnConvolutionFwdAlgoPerf_memory, (jlong)nativeObject.memory);
    env->SetIntField(javaObject, fid_cudnnConvolutionFwdAlgoPerf_determinism, (jint)nativeObject.determinism);
    return true;
}

/**
 * Release and delete the given input array, writing the values back
 * to the given java array, creating (up to 'size') objects if
 * necessary
 */
bool releaseNative(JNIEnv *env, cudnnConvolutionFwdAlgoPerf_t* &input, jobjectArray &output, int size)
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
bool initNative(JNIEnv *env, jobjectArray &input, cudnnConvolutionBwdFilterAlgoPerf_t* &output, jint size)
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
bool releaseNative(JNIEnv *env, cudnnConvolutionBwdFilterAlgoPerf_t &nativeObject, jobject &javaObject)
{
    env->SetIntField(javaObject, fid_cudnnConvolutionBwdFilterAlgoPerf_algo, (jint)nativeObject.algo);
    env->SetIntField(javaObject, fid_cudnnConvolutionBwdFilterAlgoPerf_status, (jint)nativeObject.status);
    env->SetFloatField(javaObject, fid_cudnnConvolutionBwdFilterAlgoPerf_time, (jfloat)nativeObject.time);
    env->SetLongField(javaObject, fid_cudnnConvolutionBwdFilterAlgoPerf_memory, (jlong)nativeObject.memory);
    env->SetIntField(javaObject, fid_cudnnConvolutionBwdFilterAlgoPerf_determinism, (jint)nativeObject.determinism);
    env->SetIntField(javaObject, fid_cudnnConvolutionBwdFilterAlgoPerf_mathType, (jint)nativeObject.mathType);
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
bool initNative(JNIEnv *env, jobjectArray &input, cudnnConvolutionBwdDataAlgoPerf_t* &output, jint size)
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
bool releaseNative(JNIEnv *env, cudnnConvolutionBwdDataAlgoPerf_t &nativeObject, jobject &javaObject)
{
    env->SetIntField(javaObject, fid_cudnnConvolutionBwdDataAlgoPerf_algo, (jint)nativeObject.algo);
    env->SetIntField(javaObject, fid_cudnnConvolutionBwdDataAlgoPerf_status, (jint)nativeObject.status);
    env->SetFloatField(javaObject, fid_cudnnConvolutionBwdDataAlgoPerf_time, (jfloat)nativeObject.time);
    env->SetLongField(javaObject, fid_cudnnConvolutionBwdDataAlgoPerf_memory, (jlong)nativeObject.memory);
    env->SetIntField(javaObject, fid_cudnnConvolutionBwdDataAlgoPerf_determinism, (jint)nativeObject.determinism);
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

/**
* Initialize the given native output array with the pointers that
* are obtained from the Java objects in the given input array.
*/
bool initNative(JNIEnv *env, jobjectArray input, cudnnTensorDescriptor_t* &output, bool fillTarget)
{
    jsize arraySize = env->GetArrayLength(input);
    output = new cudnnTensorDescriptor_t[arraySize];
    for (jsize i = 0; i < arraySize; i++) 
    {
        jobject element = env->GetObjectArrayElement(input, i);
        output[i] = (cudnnTensorDescriptor_t)getNativePointerValue(env, element);
    }
    return true;
}

/**
* Release and delete the given input array. The writeBack flag
* is ignored in this implementation.
*/
bool releaseNative(JNIEnv *env, cudnnTensorDescriptor_t* &input, jobjectArray output, bool writeBack)
{
    delete[] input;
    input = NULL;
    return true;
}





/**
* Initialize the given native output array with the given size.
* The input array will only be checked to have a size that is
* at least as large as the given size, but not be used otherwise
*/
bool initNative(JNIEnv *env, jobjectArray input, cudnnAlgorithmPerformance_t* &output, jint size)
{
    jsize arraySize = env->GetArrayLength(input);
    if (arraySize < size)
    {
        ThrowByName(env, "java/lang/ArrayIndexOutOfBoundsException",
            "Array parameter has insufficient size");
        return false;
    }
    output = new cudnnAlgorithmPerformance_t[arraySize];
    return true;
}

/**
* Release and delete the given input array, writing the values back
* to the given java array, creating (up to 'size') objects if
* necessary
*/
bool releaseNative(JNIEnv *env, cudnnAlgorithmPerformance_t* &input, jobjectArray output, int size)
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
            outputElement = env->NewObject(cudnnAlgorithmPerformance_Class, cudnnAlgorithmPerformance_Constructor);
            env->SetObjectArrayElement(output, i, outputElement);
        }
        setNativePointerValue(env, outputElement, (jlong)input[i]);
    }
    return true;
}

bool initNative(JNIEnv *env, jintArray javaObject, cudnnSeqDataAxis_t* &nativeObject, bool fill)
{
    return initNativeGeneric<jintArray, jint, cudnnSeqDataAxis_t>(env, javaObject, nativeObject, fill);
}
bool releaseNative(JNIEnv *env, cudnnSeqDataAxis_t* &nativeObject, jintArray javaObject, bool writeBack)
{
    return releaseNativeGeneric<jint, jintArray, cudnnSeqDataAxis_t>(env, nativeObject, javaObject, writeBack);
}









/*
* Set the log level
*
* Class:     jcuda_jcudnn_JCudnn
* Method:    setLogLevelNative
* Signature: (I)V
*/
JNIEXPORT void JNICALL Java_jcuda_jcudnn_JCudnn_setLogLevelNative
(JNIEnv *env, jclass cla, jint logLevel)
{
    Logger::setLogLevel((LogLevel)logLevel);
}



JNIEXPORT jlong JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetVersionNative(JNIEnv* env, jclass cls)
{
    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetVersion()\n");

    // Native function call
    size_t jniResult_native = cudnnGetVersion();

    // Return the result
    jlong jniResult = (jlong)jniResult_native;
    return jniResult;
}

/** Returns CUDA Runtime version statically linked against cudnn */
JNIEXPORT jlong JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetCudartVersionNative(JNIEnv* env, jclass cls)
{
    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetCudartVersion()\n");

    // Native function call
    size_t jniResult_native = cudnnGetCudartVersion();

    // Return the result
    jlong jniResult = (jlong)jniResult_native;
    return jniResult;
}

/** human-readable error messages */
JNIEXPORT jstring JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetErrorStringNative(JNIEnv* env, jclass cls, jint status)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetErrorString(status=%d)\n",
        status);

    // Native variable declarations
    cudnnStatus_t status_native = CUDNN_STATUS_SUCCESS;

    // Obtain native variable values
    status_native = (cudnnStatus_t)status;

    // Native function call
    char const* jniResult_native = cudnnGetErrorString(status_native);

    // Write back native variable values
    // status is primitive

    // Return the result
    return env->NewStringUTF(jniResult_native);
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnQueryRuntimeErrorNative(JNIEnv *env, jclass cls, jobject handle, jintArray rstatus, jint mode, jobject tag)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnQueryRuntimeError(handle=%p, rstatus=%p, mode=%d, tag=%p)\n",
        handle, rstatus, mode, tag);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnStatus_t rstatus_native;
    cudnnErrQueryMode_t mode_native;
    cudnnRuntimeTag_t * tag_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    // rstatus is write-only
    mode_native = (cudnnErrQueryMode_t)mode;
    tag_native = (cudnnRuntimeTag_t *)getNativePointerValue(env, tag);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnQueryRuntimeError(handle_native, &rstatus_native, mode_native, tag_native);

    // Write back native variable values
    // handle is read-only
    if (!set(env, rstatus, 0, (jint)rstatus_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // mode is primitive
    // tag is read-only

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetPropertyNative(JNIEnv *env, jclass cls, jint type, jintArray value)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetProperty(type=%d, value=%p)\n",
        type, value);

    // Native variable declarations
    libraryPropertyType type_native;
    int value_native;

    // Obtain native variable values
    type_native = (libraryPropertyType)type;
    // value is write-only

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetProperty(type_native, &value_native);

    // Write back native variable values
    // type is primitive
    if (!set(env, value, 0, (jint)value_native)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnCreateNative(JNIEnv *env, jclass cls, jobject handle)
{
    // Null-checks for non-primitive arguments

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
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnDestroyNative(JNIEnv *env, jclass cls, jobject handle)
{
    // Null-checks for non-primitive arguments

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
    // handle is read-only

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnSetStreamNative(JNIEnv *env, jclass cls, jobject handle, jobject streamId)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnSetStream(handle=%p, streamId=%p)\n",
        handle, streamId);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudaStream_t streamId_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    streamId_native = (cudaStream_t)getNativePointerValue(env, streamId);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnSetStream(handle_native, streamId_native);

    // Write back native variable values
    // handle is read-only
    // streamId is read-only

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetStreamNative(JNIEnv *env, jclass cls, jobject handle, jobject streamId)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetStream(handle=%p, streamId=%p)\n",
        handle, streamId);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudaStream_t streamId_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    // streamId is write-only

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetStream(handle_native, &streamId_native);

    // Write back native variable values
    // handle is read-only
    setNativePointerValue(env, streamId, (jlong)streamId_native);

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

/** Create an instance of a generic Tensor descriptor */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnCreateTensorDescriptorNative(JNIEnv *env, jclass cls, jobject tensorDesc)
{
    // Null-checks for non-primitive arguments

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
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnSetTensor4dDescriptorNative(JNIEnv *env, jclass cls, jobject tensorDesc, jint format, jint dataType, jint n, jint c, jint h, jint w)
{
    // Null-checks for non-primitive arguments

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
    // tensorDesc is read-only
    // format is primitive
    // dataType is primitive
    // n is primitive
    // c is primitive
    // h is primitive
    // w is primitive

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnSetTensor4dDescriptorExNative(JNIEnv *env, jclass cls, jobject tensorDesc, jint dataType, jint n, jint c, jint h, jint w, jint nStride, jint cStride, jint hStride, jint wStride)
{
    // Null-checks for non-primitive arguments

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
    // tensorDesc is read-only
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
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetTensor4dDescriptorNative(JNIEnv *env, jclass cls, jobject tensorDesc, jintArray dataType, jintArray n, jintArray c, jintArray h, jintArray w, jintArray nStride, jintArray cStride, jintArray hStride, jintArray wStride)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetTensor4dDescriptor(tensorDesc=%p, dataType=%p, n=%p, c=%p, h=%p, w=%p, nStride=%p, cStride=%p, hStride=%p, wStride=%p)\n",
        tensorDesc, dataType, n, c, h, w, nStride, cStride, hStride, wStride);

    // Native variable declarations
    cudnnTensorDescriptor_t tensorDesc_native;
    cudnnDataType_t dataType_native;
    int n_native;
    int c_native;
    int h_native;
    int w_native;
    int nStride_native;
    int cStride_native;
    int hStride_native;
    int wStride_native;

    // Obtain native variable values
    tensorDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, tensorDesc);
    // dataType is write-only
    // n is write-only
    // c is write-only
    // h is write-only
    // w is write-only
    // nStride is write-only
    // cStride is write-only
    // hStride is write-only
    // wStride is write-only

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetTensor4dDescriptor(tensorDesc_native, &dataType_native, &n_native, &c_native, &h_native, &w_native, &nStride_native, &cStride_native, &hStride_native, &wStride_native);

    // Write back native variable values
    // tensorDesc is read-only
    if (!set(env, dataType, 0, (jint)dataType_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, n, 0, (jint)n_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, c, 0, (jint)c_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, h, 0, (jint)h_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, w, 0, (jint)w_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, nStride, 0, (jint)nStride_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, cStride, 0, (jint)cStride_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, hStride, 0, (jint)hStride_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, wStride, 0, (jint)wStride_native)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnSetTensorNdDescriptorNative(JNIEnv *env, jclass cls, jobject tensorDesc, jint dataType, jint nbDims, jintArray dimA, jintArray strideA)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnSetTensorNdDescriptor(tensorDesc=%p, dataType=%d, nbDims=%d, dimA=%p, strideA=%p)\n",
        tensorDesc, dataType, nbDims, dimA, strideA);

    // Native variable declarations
    cudnnTensorDescriptor_t tensorDesc_native;
    cudnnDataType_t dataType_native;
    int nbDims_native = 0;
    int * dimA_native = NULL;
    int * strideA_native = NULL;

    // Obtain native variable values
    tensorDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, tensorDesc);
    dataType_native = (cudnnDataType_t)dataType;
    nbDims_native = (int)nbDims;
    if (!initNative(env, dimA, dimA_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!initNative(env, strideA, strideA_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnSetTensorNdDescriptor(tensorDesc_native, dataType_native, nbDims_native, dimA_native, strideA_native);

    // Write back native variable values
    // tensorDesc is read-only
    // dataType is primitive
    // nbDims is primitive
    if (!releaseNative(env, dimA_native, dimA, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!releaseNative(env, strideA_native, strideA, true)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnSetTensorNdDescriptorExNative(JNIEnv *env, jclass cls, jobject tensorDesc, jint format, jint dataType, jint nbDims, jintArray dimA)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnSetTensorNdDescriptorEx(tensorDesc=%p, format=%d, dataType=%d, nbDims=%d, dimA=%p)\n",
        tensorDesc, format, dataType, nbDims, dimA);

    // Native variable declarations
    cudnnTensorDescriptor_t tensorDesc_native;
    cudnnTensorFormat_t format_native;
    cudnnDataType_t dataType_native;
    int nbDims_native = 0;
    int * dimA_native = NULL;

    // Obtain native variable values
    tensorDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, tensorDesc);
    format_native = (cudnnTensorFormat_t)format;
    dataType_native = (cudnnDataType_t)dataType;
    nbDims_native = (int)nbDims;
    if (!initNative(env, dimA, dimA_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnSetTensorNdDescriptorEx(tensorDesc_native, format_native, dataType_native, nbDims_native, dimA_native);

    // Write back native variable values
    // tensorDesc is read-only
    // format is primitive
    // dataType is primitive
    // nbDims is primitive
    if (!releaseNative(env, dimA_native, dimA, true)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetTensorNdDescriptorNative(JNIEnv *env, jclass cls, jobject tensorDesc, jint nbDimsRequested, jintArray dataType, jintArray nbDims, jintArray dimA, jintArray strideA)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetTensorNdDescriptor(tensorDesc=%p, nbDimsRequested=%d, dataType=%p, nbDims=%p, dimA=%p, strideA=%p)\n",
        tensorDesc, nbDimsRequested, dataType, nbDims, dimA, strideA);

    // Native variable declarations
    cudnnTensorDescriptor_t tensorDesc_native;
    int nbDimsRequested_native = 0;
    cudnnDataType_t dataType_native;
    int nbDims_native;
    int * dimA_native = NULL;
    int * strideA_native = NULL;

    // Obtain native variable values
    tensorDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, tensorDesc);
    nbDimsRequested_native = (int)nbDimsRequested;
    // dataType is write-only
    // nbDims is write-only
    if (!initNative(env, dimA, dimA_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!initNative(env, strideA, strideA_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetTensorNdDescriptor(tensorDesc_native, nbDimsRequested_native, &dataType_native, &nbDims_native, dimA_native, strideA_native);

    // Write back native variable values
    // tensorDesc is read-only
    // nbDimsRequested is primitive
    if (!set(env, dataType, 0, (jint)dataType_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, nbDims, 0, (jint)nbDims_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!releaseNative(env, dimA_native, dimA, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!releaseNative(env, strideA_native, strideA, true)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetTensorSizeInBytesNative(JNIEnv *env, jclass cls, jobject tensorDesc, jlongArray size)
{
    // Null-checks for non-primitive arguments
    if (tensorDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'tensorDesc' is null for cudnnGetTensorSizeInBytes");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (size == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'size' is null for cudnnGetTensorSizeInBytes");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetTensorSizeInBytes(tensorDesc=%p, size=%p)\n",
        tensorDesc, size);

    // Native variable declarations
    cudnnTensorDescriptor_t tensorDesc_native;
    size_t size_native;

    // Obtain native variable values
    tensorDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, tensorDesc);
    // size is write-only

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetTensorSizeInBytes(tensorDesc_native, &size_native);

    // Write back native variable values
    // tensorDesc is read-only
    if (!set(env, size, 0, (jlong)size_native)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult = (jint)jniResult_native;
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
    // tensorDesc is read-only

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

/** Create a destination descriptor for cudnnTransformTensor */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnInitTransformDestNative(JNIEnv *env, jclass cls, jobject transformDesc, jobject srcDesc, jobject destDesc, jlongArray destSizeInBytes)
{
    // Null-checks for non-primitive arguments
    if (destSizeInBytes == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'destSizeInBytes' is null for cudnnInitTransformDest");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnInitTransformDest(transformDesc=%p, srcDesc=%p, destDesc=%p, destSizeInBytes=%p)\n",
        transformDesc, srcDesc, destDesc, destSizeInBytes);

    // Native variable declarations
    cudnnTensorTransformDescriptor_t transformDesc_native;
    cudnnTensorDescriptor_t srcDesc_native;
    cudnnTensorDescriptor_t destDesc_native;
    size_t destSizeInBytes_native;

    // Obtain native variable values
    transformDesc_native = (cudnnTensorTransformDescriptor_t)getNativePointerValue(env, transformDesc);
    srcDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, srcDesc);
    destDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, destDesc);
    // destSizeInBytes is write-only

    // Native function call
    cudnnStatus_t jniResult_native = cudnnInitTransformDest(transformDesc_native, srcDesc_native, destDesc_native, &destSizeInBytes_native);

    // Write back native variable values
    // transformDesc is read-only
    // srcDesc is read-only
    // destDesc is read-only
    if (!set(env, destSizeInBytes, 0, (jlong)destSizeInBytes_native)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

/** Create an empty tensor transform descriptor */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnCreateTensorTransformDescriptorNative(JNIEnv *env, jclass cls, jobject transformDesc)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnCreateTensorTransformDescriptor(transformDesc=%p)\n",
        transformDesc);

    // Native variable declarations
    cudnnTensorTransformDescriptor_t transformDesc_native;

    // Obtain native variable values
    // transformDesc is write-only

    // Native function call
    cudnnStatus_t jniResult_native = cudnnCreateTensorTransformDescriptor(&transformDesc_native);

    // Write back native variable values
    setNativePointerValue(env, transformDesc, (jlong)transformDesc_native);

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

/** Initialize a previously created tensor transform descriptor. */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnSetTensorTransformDescriptorNative(JNIEnv *env, jclass cls, jobject transformDesc, jint nbDims, jint destFormat, jintArray padBeforeA, jintArray padAfterA, jintArray foldA, jint direction)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnSetTensorTransformDescriptor(transformDesc=%p, nbDims=%d, destFormat=%d, padBeforeA=%p, padAfterA=%p, foldA=%p, direction=%d)\n",
        transformDesc, nbDims, destFormat, padBeforeA, padAfterA, foldA, direction);

    // Native variable declarations
    cudnnTensorTransformDescriptor_t transformDesc_native;
    uint32_t nbDims_native = 0;
    cudnnTensorFormat_t destFormat_native;
    int32_t * padBeforeA_native = NULL;
    int32_t * padAfterA_native = NULL;
    uint32_t * foldA_native = NULL;
    cudnnFoldingDirection_t direction_native;

    // Obtain native variable values
    transformDesc_native = (cudnnTensorTransformDescriptor_t)getNativePointerValue(env, transformDesc);
    nbDims_native = (uint32_t)nbDims;
    destFormat_native = (cudnnTensorFormat_t)destFormat;
    if (!initNative(env, padBeforeA, padBeforeA_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!initNative(env, padAfterA, padAfterA_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!initNative(env, foldA, foldA_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    direction_native = (cudnnFoldingDirection_t)direction;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnSetTensorTransformDescriptor(transformDesc_native, nbDims_native, destFormat_native, padBeforeA_native, padAfterA_native, foldA_native, direction_native);

    // Write back native variable values
    // transformDesc is read-only
    // nbDims is primitive
    // destFormat is primitive
    if (!releaseNative(env, padBeforeA_native, padBeforeA, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!releaseNative(env, padAfterA_native, padAfterA, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!releaseNative(env, foldA_native, foldA, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // direction is primitive

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

/**
 * <pre>
 * Retrieves the values stored in a previously initialized tensor transform
 * descriptor.
 * </pre>
 */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetTensorTransformDescriptorNative(JNIEnv *env, jclass cls, jobject transformDesc, jint nbDimsRequested, jintArray destFormat, jintArray padBeforeA, jintArray padAfterA, jintArray foldA, jintArray direction)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetTensorTransformDescriptor(transformDesc=%p, nbDimsRequested=%d, destFormat=%p, padBeforeA=%p, padAfterA=%p, foldA=%p, direction=%p)\n",
        transformDesc, nbDimsRequested, destFormat, padBeforeA, padAfterA, foldA, direction);

    // Native variable declarations
    cudnnTensorTransformDescriptor_t transformDesc_native;
    uint32_t nbDimsRequested_native = 0;
    cudnnTensorFormat_t destFormat_native;
    int32_t * padBeforeA_native = NULL;
    int32_t * padAfterA_native = NULL;
    uint32_t * foldA_native = NULL;
    cudnnFoldingDirection_t direction_native;

    // Obtain native variable values
    transformDesc_native = (cudnnTensorTransformDescriptor_t)getNativePointerValue(env, transformDesc);
    nbDimsRequested_native = (uint32_t)nbDimsRequested;
    // destFormat is write-only
    if (!initNative(env, padBeforeA, padBeforeA_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!initNative(env, padAfterA, padAfterA_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!initNative(env, foldA, foldA_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // direction is write-only

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetTensorTransformDescriptor(transformDesc_native, nbDimsRequested_native, &destFormat_native, padBeforeA_native, padAfterA_native, foldA_native, &direction_native);

    // Write back native variable values
    // transformDesc is read-only
    // nbDimsRequested is primitive
    if (!set(env, destFormat, 0, (jint)destFormat_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!releaseNative(env, padBeforeA_native, padBeforeA, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!releaseNative(env, padAfterA_native, padAfterA, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!releaseNative(env, foldA_native, foldA, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, direction, 0, (jint)direction_native)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

/**
 * Destroys a previously created tensor transform descriptor.
 */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnDestroyTensorTransformDescriptorNative(JNIEnv *env, jclass cls, jobject transformDesc)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnDestroyTensorTransformDescriptor(transformDesc=%p)\n",
        transformDesc);

    // Native variable declarations
    cudnnTensorTransformDescriptor_t transformDesc_native;

    // Obtain native variable values
    transformDesc_native = (cudnnTensorTransformDescriptor_t)getNativePointerValue(env, transformDesc);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnDestroyTensorTransformDescriptor(transformDesc_native);

    // Write back native variable values
    // transformDesc is read-only

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

/** Tensor layout conversion helper (y = alpha * x + beta * y) */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnTransformTensorNative(JNIEnv *env, jclass cls, jobject handle, jobject alpha, jobject xDesc, jobject x, jobject beta, jobject yDesc, jobject y)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnTransformTensor(handle=%p, alpha=%p, xDesc=%p, x=%p, beta=%p, yDesc=%p, y=%p)\n",
        handle, alpha, xDesc, x, beta, yDesc, y);

    // Native variable declarations
    cudnnHandle_t handle_native;
    void * alpha_native = NULL;
    cudnnTensorDescriptor_t xDesc_native;
    void * x_native = NULL;
    void * beta_native = NULL;
    cudnnTensorDescriptor_t yDesc_native;
    void * y_native = NULL;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    PointerData *alpha_pointerData = initPointerData(env, alpha);
    if (alpha_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    alpha_native = (void *)alpha_pointerData->getPointer(env);
    xDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, xDesc);
    x_native = (void *)getPointer(env, x);
    PointerData *beta_pointerData = initPointerData(env, beta);
    if (beta_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    beta_native = (void *)beta_pointerData->getPointer(env);
    yDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, yDesc);
    y_native = (void *)getPointer(env, y);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnTransformTensor(handle_native, alpha_native, xDesc_native, x_native, beta_native, yDesc_native, y_native);

    // Write back native variable values
    // handle is read-only
    if (!releasePointerData(env, alpha_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // xDesc is read-only
    // x is a native pointer
    if (!releasePointerData(env, beta_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // yDesc is read-only
    // y is a native pointer

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnTransformTensorExNative(JNIEnv *env, jclass cls, jobject handle, jobject transDesc, jobject alpha, jobject srcDesc, jobject srcData, jobject beta, jobject destDesc, jobject destData)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnTransformTensorEx(handle=%p, transDesc=%p, alpha=%p, srcDesc=%p, srcData=%p, beta=%p, destDesc=%p, destData=%p)\n",
        handle, transDesc, alpha, srcDesc, srcData, beta, destDesc, destData);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnTensorTransformDescriptor_t transDesc_native;
    void * alpha_native = NULL;
    cudnnTensorDescriptor_t srcDesc_native;
    void * srcData_native = NULL;
    void * beta_native = NULL;
    cudnnTensorDescriptor_t destDesc_native;
    void * destData_native = NULL;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    transDesc_native = (cudnnTensorTransformDescriptor_t)getNativePointerValue(env, transDesc);
    PointerData *alpha_pointerData = initPointerData(env, alpha);
    if (alpha_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    alpha_native = (void *)alpha_pointerData->getPointer(env);
    srcDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, srcDesc);
    srcData_native = (void *)getPointer(env, srcData);
    PointerData *beta_pointerData = initPointerData(env, beta);
    if (beta_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    beta_native = (void *)beta_pointerData->getPointer(env);
    destDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, destDesc);
    destData_native = (void *)getPointer(env, destData);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnTransformTensorEx(handle_native, transDesc_native, alpha_native, srcDesc_native, srcData_native, beta_native, destDesc_native, destData_native);

    // Write back native variable values
    // handle is read-only
    // transDesc is read-only
    if (!releasePointerData(env, alpha_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // srcDesc is read-only
    // srcData is a native pointer
    if (!releasePointerData(env, beta_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // destDesc is read-only
    // destData is a native pointer

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

/** Tensor Bias addition : C = alpha * A + beta * C  */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnAddTensorNative(JNIEnv *env, jclass cls, jobject handle, jobject alpha, jobject aDesc, jobject A, jobject beta, jobject cDesc, jobject C)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnAddTensor(handle=%p, alpha=%p, aDesc=%p, A=%p, beta=%p, cDesc=%p, C=%p)\n",
        handle, alpha, aDesc, A, beta, cDesc, C);

    // Native variable declarations
    cudnnHandle_t handle_native;
    void * alpha_native = NULL;
    cudnnTensorDescriptor_t aDesc_native;
    void * A_native = NULL;
    void * beta_native = NULL;
    cudnnTensorDescriptor_t cDesc_native;
    void * C_native = NULL;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    PointerData *alpha_pointerData = initPointerData(env, alpha);
    if (alpha_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    alpha_native = (void *)alpha_pointerData->getPointer(env);
    aDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, aDesc);
    A_native = (void *)getPointer(env, A);
    PointerData *beta_pointerData = initPointerData(env, beta);
    if (beta_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    beta_native = (void *)beta_pointerData->getPointer(env);
    cDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, cDesc);
    C_native = (void *)getPointer(env, C);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnAddTensor(handle_native, alpha_native, aDesc_native, A_native, beta_native, cDesc_native, C_native);

    // Write back native variable values
    // handle is read-only
    if (!releasePointerData(env, alpha_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // aDesc is read-only
    // A is a native pointer
    if (!releasePointerData(env, beta_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // cDesc is read-only
    // C is a native pointer

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnCreateOpTensorDescriptorNative(JNIEnv *env, jclass cls, jobject opTensorDesc)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnCreateOpTensorDescriptor(opTensorDesc=%p)\n",
        opTensorDesc);

    // Native variable declarations
    cudnnOpTensorDescriptor_t opTensorDesc_native;

    // Obtain native variable values
    // opTensorDesc is write-only

    // Native function call
    cudnnStatus_t jniResult_native = cudnnCreateOpTensorDescriptor(&opTensorDesc_native);

    // Write back native variable values
    setNativePointerValue(env, opTensorDesc, (jlong)opTensorDesc_native);

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnSetOpTensorDescriptorNative(JNIEnv *env, jclass cls, jobject opTensorDesc, jint opTensorOp, jint opTensorCompType, jint opTensorNanOpt)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnSetOpTensorDescriptor(opTensorDesc=%p, opTensorOp=%d, opTensorCompType=%d, opTensorNanOpt=%d)\n",
        opTensorDesc, opTensorOp, opTensorCompType, opTensorNanOpt);

    // Native variable declarations
    cudnnOpTensorDescriptor_t opTensorDesc_native;
    cudnnOpTensorOp_t opTensorOp_native;
    cudnnDataType_t opTensorCompType_native;
    cudnnNanPropagation_t opTensorNanOpt_native;

    // Obtain native variable values
    opTensorDesc_native = (cudnnOpTensorDescriptor_t)getNativePointerValue(env, opTensorDesc);
    opTensorOp_native = (cudnnOpTensorOp_t)opTensorOp;
    opTensorCompType_native = (cudnnDataType_t)opTensorCompType;
    opTensorNanOpt_native = (cudnnNanPropagation_t)opTensorNanOpt;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnSetOpTensorDescriptor(opTensorDesc_native, opTensorOp_native, opTensorCompType_native, opTensorNanOpt_native);

    // Write back native variable values
    // opTensorDesc is read-only
    // opTensorOp is primitive
    // opTensorCompType is primitive
    // opTensorNanOpt is primitive

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetOpTensorDescriptorNative(JNIEnv *env, jclass cls, jobject opTensorDesc, jintArray opTensorOp, jintArray opTensorCompType, jintArray opTensorNanOpt)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetOpTensorDescriptor(opTensorDesc=%p, opTensorOp=%p, opTensorCompType=%p, opTensorNanOpt=%p)\n",
        opTensorDesc, opTensorOp, opTensorCompType, opTensorNanOpt);

    // Native variable declarations
    cudnnOpTensorDescriptor_t opTensorDesc_native;
    cudnnOpTensorOp_t opTensorOp_native;
    cudnnDataType_t opTensorCompType_native;
    cudnnNanPropagation_t opTensorNanOpt_native;

    // Obtain native variable values
    opTensorDesc_native = (cudnnOpTensorDescriptor_t)getNativePointerValue(env, opTensorDesc);
    // opTensorOp is write-only
    // opTensorCompType is write-only
    // opTensorNanOpt is write-only

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetOpTensorDescriptor(opTensorDesc_native, &opTensorOp_native, &opTensorCompType_native, &opTensorNanOpt_native);

    // Write back native variable values
    // opTensorDesc is read-only
    if (!set(env, opTensorOp, 0, (jint)opTensorOp_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, opTensorCompType, 0, (jint)opTensorCompType_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, opTensorNanOpt, 0, (jint)opTensorNanOpt_native)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnDestroyOpTensorDescriptorNative(JNIEnv *env, jclass cls, jobject opTensorDesc)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnDestroyOpTensorDescriptor(opTensorDesc=%p)\n",
        opTensorDesc);

    // Native variable declarations
    cudnnOpTensorDescriptor_t opTensorDesc_native;

    // Obtain native variable values
    opTensorDesc_native = (cudnnOpTensorDescriptor_t)getNativePointerValue(env, opTensorDesc);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnDestroyOpTensorDescriptor(opTensorDesc_native);

    // Write back native variable values
    // opTensorDesc is read-only

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

/** Tensor operation : C = op( alpha1 * A, alpha2 * B ) + beta * C */
/** B tensor is ignored for CUDNN_OP_TENSOR_SQRT, CUDNN_OP_TENSOR_NOT. */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnOpTensorNative(JNIEnv *env, jclass cls, jobject handle, jobject opTensorDesc, jobject alpha1, jobject aDesc, jobject A, jobject alpha2, jobject bDesc, jobject B, jobject beta, jobject cDesc, jobject C)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnOpTensor(handle=%p, opTensorDesc=%p, alpha1=%p, aDesc=%p, A=%p, alpha2=%p, bDesc=%p, B=%p, beta=%p, cDesc=%p, C=%p)\n",
        handle, opTensorDesc, alpha1, aDesc, A, alpha2, bDesc, B, beta, cDesc, C);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnOpTensorDescriptor_t opTensorDesc_native;
    void * alpha1_native = NULL;
    cudnnTensorDescriptor_t aDesc_native;
    void * A_native = NULL;
    void * alpha2_native = NULL;
    cudnnTensorDescriptor_t bDesc_native;
    void * B_native = NULL;
    void * beta_native = NULL;
    cudnnTensorDescriptor_t cDesc_native;
    void * C_native = NULL;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    opTensorDesc_native = (cudnnOpTensorDescriptor_t)getNativePointerValue(env, opTensorDesc);
    PointerData *alpha1_pointerData = initPointerData(env, alpha1);
    if (alpha1_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    alpha1_native = (void *)alpha1_pointerData->getPointer(env);
    aDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, aDesc);
    A_native = (void *)getPointer(env, A);
    PointerData *alpha2_pointerData = initPointerData(env, alpha2);
    if (alpha2_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    alpha2_native = (void *)alpha2_pointerData->getPointer(env);
    bDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, bDesc);
    B_native = (void *)getPointer(env, B);
    PointerData *beta_pointerData = initPointerData(env, beta);
    if (beta_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    beta_native = (void *)beta_pointerData->getPointer(env);
    cDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, cDesc);
    C_native = (void *)getPointer(env, C);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnOpTensor(handle_native, opTensorDesc_native, alpha1_native, aDesc_native, A_native, alpha2_native, bDesc_native, B_native, beta_native, cDesc_native, C_native);

    // Write back native variable values
    // handle is read-only
    // opTensorDesc is read-only
    if (!releasePointerData(env, alpha1_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // aDesc is read-only
    // A is a native pointer
    if (!releasePointerData(env, alpha2_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // bDesc is read-only
    // B is a native pointer
    if (!releasePointerData(env, beta_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // cDesc is read-only
    // C is a native pointer

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnCreateReduceTensorDescriptorNative(JNIEnv *env, jclass cls, jobject reduceTensorDesc)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnCreateReduceTensorDescriptor(reduceTensorDesc=%p)\n",
        reduceTensorDesc);

    // Native variable declarations
    cudnnReduceTensorDescriptor_t reduceTensorDesc_native;

    // Obtain native variable values
    // reduceTensorDesc is write-only

    // Native function call
    cudnnStatus_t jniResult_native = cudnnCreateReduceTensorDescriptor(&reduceTensorDesc_native);

    // Write back native variable values
    setNativePointerValue(env, reduceTensorDesc, (jlong)reduceTensorDesc_native);

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnSetReduceTensorDescriptorNative(JNIEnv *env, jclass cls, jobject reduceTensorDesc, jint reduceTensorOp, jint reduceTensorCompType, jint reduceTensorNanOpt, jint reduceTensorIndices, jint reduceTensorIndicesType)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnSetReduceTensorDescriptor(reduceTensorDesc=%p, reduceTensorOp=%d, reduceTensorCompType=%d, reduceTensorNanOpt=%d, reduceTensorIndices=%d, reduceTensorIndicesType=%d)\n",
        reduceTensorDesc, reduceTensorOp, reduceTensorCompType, reduceTensorNanOpt, reduceTensorIndices, reduceTensorIndicesType);

    // Native variable declarations
    cudnnReduceTensorDescriptor_t reduceTensorDesc_native;
    cudnnReduceTensorOp_t reduceTensorOp_native;
    cudnnDataType_t reduceTensorCompType_native;
    cudnnNanPropagation_t reduceTensorNanOpt_native;
    cudnnReduceTensorIndices_t reduceTensorIndices_native;
    cudnnIndicesType_t reduceTensorIndicesType_native;

    // Obtain native variable values
    reduceTensorDesc_native = (cudnnReduceTensorDescriptor_t)getNativePointerValue(env, reduceTensorDesc);
    reduceTensorOp_native = (cudnnReduceTensorOp_t)reduceTensorOp;
    reduceTensorCompType_native = (cudnnDataType_t)reduceTensorCompType;
    reduceTensorNanOpt_native = (cudnnNanPropagation_t)reduceTensorNanOpt;
    reduceTensorIndices_native = (cudnnReduceTensorIndices_t)reduceTensorIndices;
    reduceTensorIndicesType_native = (cudnnIndicesType_t)reduceTensorIndicesType;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnSetReduceTensorDescriptor(reduceTensorDesc_native, reduceTensorOp_native, reduceTensorCompType_native, reduceTensorNanOpt_native, reduceTensorIndices_native, reduceTensorIndicesType_native);

    // Write back native variable values
    // reduceTensorDesc is read-only
    // reduceTensorOp is primitive
    // reduceTensorCompType is primitive
    // reduceTensorNanOpt is primitive
    // reduceTensorIndices is primitive
    // reduceTensorIndicesType is primitive

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetReduceTensorDescriptorNative(JNIEnv *env, jclass cls, jobject reduceTensorDesc, jintArray reduceTensorOp, jintArray reduceTensorCompType, jintArray reduceTensorNanOpt, jintArray reduceTensorIndices, jintArray reduceTensorIndicesType)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetReduceTensorDescriptor(reduceTensorDesc=%p, reduceTensorOp=%p, reduceTensorCompType=%p, reduceTensorNanOpt=%p, reduceTensorIndices=%p, reduceTensorIndicesType=%p)\n",
        reduceTensorDesc, reduceTensorOp, reduceTensorCompType, reduceTensorNanOpt, reduceTensorIndices, reduceTensorIndicesType);

    // Native variable declarations
    cudnnReduceTensorDescriptor_t reduceTensorDesc_native;
    cudnnReduceTensorOp_t reduceTensorOp_native;
    cudnnDataType_t reduceTensorCompType_native;
    cudnnNanPropagation_t reduceTensorNanOpt_native;
    cudnnReduceTensorIndices_t reduceTensorIndices_native;
    cudnnIndicesType_t reduceTensorIndicesType_native;

    // Obtain native variable values
    reduceTensorDesc_native = (cudnnReduceTensorDescriptor_t)getNativePointerValue(env, reduceTensorDesc);
    // reduceTensorOp is write-only
    // reduceTensorCompType is write-only
    // reduceTensorNanOpt is write-only
    // reduceTensorIndices is write-only
    // reduceTensorIndicesType is write-only

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetReduceTensorDescriptor(reduceTensorDesc_native, &reduceTensorOp_native, &reduceTensorCompType_native, &reduceTensorNanOpt_native, &reduceTensorIndices_native, &reduceTensorIndicesType_native);

    // Write back native variable values
    // reduceTensorDesc is read-only
    if (!set(env, reduceTensorOp, 0, (jint)reduceTensorOp_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, reduceTensorCompType, 0, (jint)reduceTensorCompType_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, reduceTensorNanOpt, 0, (jint)reduceTensorNanOpt_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, reduceTensorIndices, 0, (jint)reduceTensorIndices_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, reduceTensorIndicesType, 0, (jint)reduceTensorIndicesType_native)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnDestroyReduceTensorDescriptorNative(JNIEnv *env, jclass cls, jobject reduceTensorDesc)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnDestroyReduceTensorDescriptor(reduceTensorDesc=%p)\n",
        reduceTensorDesc);

    // Native variable declarations
    cudnnReduceTensorDescriptor_t reduceTensorDesc_native;

    // Obtain native variable values
    reduceTensorDesc_native = (cudnnReduceTensorDescriptor_t)getNativePointerValue(env, reduceTensorDesc);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnDestroyReduceTensorDescriptor(reduceTensorDesc_native);

    // Write back native variable values
    // reduceTensorDesc is read-only

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

/** Helper function to return the minimum size of the index space to be passed to the reduction given the input and
 * output tensors */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetReductionIndicesSizeNative(JNIEnv *env, jclass cls, jobject handle, jobject reduceTensorDesc, jobject aDesc, jobject cDesc, jlongArray sizeInBytes)
{
    // Null-checks for non-primitive arguments
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cudnnGetReductionIndicesSize");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (reduceTensorDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'reduceTensorDesc' is null for cudnnGetReductionIndicesSize");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (aDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'aDesc' is null for cudnnGetReductionIndicesSize");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (cDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'cDesc' is null for cudnnGetReductionIndicesSize");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (sizeInBytes == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'sizeInBytes' is null for cudnnGetReductionIndicesSize");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetReductionIndicesSize(handle=%p, reduceTensorDesc=%p, aDesc=%p, cDesc=%p, sizeInBytes=%p)\n",
        handle, reduceTensorDesc, aDesc, cDesc, sizeInBytes);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnReduceTensorDescriptor_t reduceTensorDesc_native;
    cudnnTensorDescriptor_t aDesc_native;
    cudnnTensorDescriptor_t cDesc_native;
    size_t sizeInBytes_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    reduceTensorDesc_native = (cudnnReduceTensorDescriptor_t)getNativePointerValue(env, reduceTensorDesc);
    aDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, aDesc);
    cDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, cDesc);
    // sizeInBytes is write-only

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetReductionIndicesSize(handle_native, reduceTensorDesc_native, aDesc_native, cDesc_native, &sizeInBytes_native);

    // Write back native variable values
    // handle is read-only
    // reduceTensorDesc is read-only
    // aDesc is read-only
    // cDesc is read-only
    if (!set(env, sizeInBytes, 0, (jlong)sizeInBytes_native)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

/** Helper function to return the minimum size of the workspace to be passed to the reduction given the input and output
 * tensors */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetReductionWorkspaceSizeNative(JNIEnv *env, jclass cls, jobject handle, jobject reduceTensorDesc, jobject aDesc, jobject cDesc, jlongArray sizeInBytes)
{
    // Null-checks for non-primitive arguments
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cudnnGetReductionWorkspaceSize");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (reduceTensorDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'reduceTensorDesc' is null for cudnnGetReductionWorkspaceSize");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (aDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'aDesc' is null for cudnnGetReductionWorkspaceSize");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (cDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'cDesc' is null for cudnnGetReductionWorkspaceSize");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (sizeInBytes == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'sizeInBytes' is null for cudnnGetReductionWorkspaceSize");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetReductionWorkspaceSize(handle=%p, reduceTensorDesc=%p, aDesc=%p, cDesc=%p, sizeInBytes=%p)\n",
        handle, reduceTensorDesc, aDesc, cDesc, sizeInBytes);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnReduceTensorDescriptor_t reduceTensorDesc_native;
    cudnnTensorDescriptor_t aDesc_native;
    cudnnTensorDescriptor_t cDesc_native;
    size_t sizeInBytes_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    reduceTensorDesc_native = (cudnnReduceTensorDescriptor_t)getNativePointerValue(env, reduceTensorDesc);
    aDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, aDesc);
    cDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, cDesc);
    // sizeInBytes is write-only

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetReductionWorkspaceSize(handle_native, reduceTensorDesc_native, aDesc_native, cDesc_native, &sizeInBytes_native);

    // Write back native variable values
    // handle is read-only
    // reduceTensorDesc is read-only
    // aDesc is read-only
    // cDesc is read-only
    if (!set(env, sizeInBytes, 0, (jlong)sizeInBytes_native)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

/** Tensor operation : C = reduce op( alpha * A ) + beta * C */
/** The NaN propagation enum applies to only the min and max reduce ops; the other reduce ops propagate NaN as usual. */
/** The indices space is ignored for reduce ops other than min or max. */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnReduceTensorNative(JNIEnv *env, jclass cls, jobject handle, jobject reduceTensorDesc, jobject indices, jlong indicesSizeInBytes, jobject workspace, jlong workspaceSizeInBytes, jobject alpha, jobject aDesc, jobject A, jobject beta, jobject cDesc, jobject C)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnReduceTensor(handle=%p, reduceTensorDesc=%p, indices=%p, indicesSizeInBytes=%ld, workspace=%p, workspaceSizeInBytes=%ld, alpha=%p, aDesc=%p, A=%p, beta=%p, cDesc=%p, C=%p)\n",
        handle, reduceTensorDesc, indices, indicesSizeInBytes, workspace, workspaceSizeInBytes, alpha, aDesc, A, beta, cDesc, C);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnReduceTensorDescriptor_t reduceTensorDesc_native;
    void * indices_native = NULL;
    size_t indicesSizeInBytes_native = 0;
    void * workspace_native = NULL;
    size_t workspaceSizeInBytes_native = 0;
    void * alpha_native = NULL;
    cudnnTensorDescriptor_t aDesc_native;
    void * A_native = NULL;
    void * beta_native = NULL;
    cudnnTensorDescriptor_t cDesc_native;
    void * C_native = NULL;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    reduceTensorDesc_native = (cudnnReduceTensorDescriptor_t)getNativePointerValue(env, reduceTensorDesc);
    indices_native = (void *)getPointer(env, indices);
    indicesSizeInBytes_native = (size_t)indicesSizeInBytes;
    workspace_native = (void *)getPointer(env, workspace);
    workspaceSizeInBytes_native = (size_t)workspaceSizeInBytes;
    PointerData *alpha_pointerData = initPointerData(env, alpha);
    if (alpha_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    alpha_native = (void *)alpha_pointerData->getPointer(env);
    aDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, aDesc);
    A_native = (void *)getPointer(env, A);
    PointerData *beta_pointerData = initPointerData(env, beta);
    if (beta_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    beta_native = (void *)beta_pointerData->getPointer(env);
    cDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, cDesc);
    C_native = (void *)getPointer(env, C);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnReduceTensor(handle_native, reduceTensorDesc_native, indices_native, indicesSizeInBytes_native, workspace_native, workspaceSizeInBytes_native, alpha_native, aDesc_native, A_native, beta_native, cDesc_native, C_native);

    // Write back native variable values
    // handle is read-only
    // reduceTensorDesc is read-only
    // indices is a native pointer
    // indicesSizeInBytes is primitive
    // workspace is a native pointer
    // workspaceSizeInBytes is primitive
    if (!releasePointerData(env, alpha_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // aDesc is read-only
    // A is a native pointer
    if (!releasePointerData(env, beta_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // cDesc is read-only
    // C is a native pointer

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

/** Set all values of a tensor to a given value : y[i] = value[0] */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnSetTensorNative(JNIEnv *env, jclass cls, jobject handle, jobject yDesc, jobject y, jobject valuePtr)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnSetTensor(handle=%p, yDesc=%p, y=%p, valuePtr=%p)\n",
        handle, yDesc, y, valuePtr);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnTensorDescriptor_t yDesc_native;
    void * y_native = NULL;
    void * valuePtr_native = NULL;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    yDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, yDesc);
    y_native = (void *)getPointer(env, y);
    PointerData *valuePtr_pointerData = initPointerData(env, valuePtr);
    if (valuePtr_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    valuePtr_native = (void *)valuePtr_pointerData->getPointer(env);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnSetTensor(handle_native, yDesc_native, y_native, valuePtr_native);

    // Write back native variable values
    // handle is read-only
    // yDesc is read-only
    // y is a native pointer
    if (!releasePointerData(env, valuePtr_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

/** Scale all values of a tensor by a given factor : y[i] = alpha * y[i] */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnScaleTensorNative(JNIEnv *env, jclass cls, jobject handle, jobject yDesc, jobject y, jobject alpha)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnScaleTensor(handle=%p, yDesc=%p, y=%p, alpha=%p)\n",
        handle, yDesc, y, alpha);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnTensorDescriptor_t yDesc_native;
    void * y_native = NULL;
    void * alpha_native = NULL;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    yDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, yDesc);
    y_native = (void *)getPointer(env, y);
    PointerData *alpha_pointerData = initPointerData(env, alpha);
    if (alpha_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    alpha_native = (void *)alpha_pointerData->getPointer(env);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnScaleTensor(handle_native, yDesc_native, y_native, alpha_native);

    // Write back native variable values
    // handle is read-only
    // yDesc is read-only
    // y is a native pointer
    if (!releasePointerData(env, alpha_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

/** Create an instance of FilterStruct */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnCreateFilterDescriptorNative(JNIEnv *env, jclass cls, jobject filterDesc)
{
    // Null-checks for non-primitive arguments

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
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnSetFilter4dDescriptorNative(JNIEnv *env, jclass cls, jobject filterDesc, jint dataType, jint format, jint k, jint c, jint h, jint w)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnSetFilter4dDescriptor(filterDesc=%p, dataType=%d, format=%d, k=%d, c=%d, h=%d, w=%d)\n",
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
    cudnnStatus_t jniResult_native = cudnnSetFilter4dDescriptor(filterDesc_native, dataType_native, format_native, k_native, c_native, h_native, w_native);

    // Write back native variable values
    // filterDesc is read-only
    // dataType is primitive
    // format is primitive
    // k is primitive
    // c is primitive
    // h is primitive
    // w is primitive

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetFilter4dDescriptorNative(JNIEnv *env, jclass cls, jobject filterDesc, jintArray dataType, jintArray format, jintArray k, jintArray c, jintArray h, jintArray w)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetFilter4dDescriptor(filterDesc=%p, dataType=%p, format=%p, k=%p, c=%p, h=%p, w=%p)\n",
        filterDesc, dataType, format, k, c, h, w);

    // Native variable declarations
    cudnnFilterDescriptor_t filterDesc_native;
    cudnnDataType_t dataType_native;
    cudnnTensorFormat_t format_native;
    int k_native;
    int c_native;
    int h_native;
    int w_native;

    // Obtain native variable values
    filterDesc_native = (cudnnFilterDescriptor_t)getNativePointerValue(env, filterDesc);
    // dataType is write-only
    // format is write-only
    // k is write-only
    // c is write-only
    // h is write-only
    // w is write-only

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetFilter4dDescriptor(filterDesc_native, &dataType_native, &format_native, &k_native, &c_native, &h_native, &w_native);

    // Write back native variable values
    // filterDesc is read-only
    if (!set(env, dataType, 0, (jint)dataType_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, format, 0, (jint)format_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, k, 0, (jint)k_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, c, 0, (jint)c_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, h, 0, (jint)h_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, w, 0, (jint)w_native)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnSetFilterNdDescriptorNative(JNIEnv *env, jclass cls, jobject filterDesc, jint dataType, jint format, jint nbDims, jintArray filterDimA)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnSetFilterNdDescriptor(filterDesc=%p, dataType=%d, format=%d, nbDims=%d, filterDimA=%p)\n",
        filterDesc, dataType, format, nbDims, filterDimA);

    // Native variable declarations
    cudnnFilterDescriptor_t filterDesc_native;
    cudnnDataType_t dataType_native;
    cudnnTensorFormat_t format_native;
    int nbDims_native = 0;
    int * filterDimA_native = NULL;

    // Obtain native variable values
    filterDesc_native = (cudnnFilterDescriptor_t)getNativePointerValue(env, filterDesc);
    dataType_native = (cudnnDataType_t)dataType;
    format_native = (cudnnTensorFormat_t)format;
    nbDims_native = (int)nbDims;
    if (!initNative(env, filterDimA, filterDimA_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnSetFilterNdDescriptor(filterDesc_native, dataType_native, format_native, nbDims_native, filterDimA_native);

    // Write back native variable values
    // filterDesc is read-only
    // dataType is primitive
    // format is primitive
    // nbDims is primitive
    if (!releaseNative(env, filterDimA_native, filterDimA, true)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetFilterNdDescriptorNative(JNIEnv *env, jclass cls, jobject filterDesc, jint nbDimsRequested, jintArray dataType, jintArray format, jintArray nbDims, jintArray filterDimA)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetFilterNdDescriptor(filterDesc=%p, nbDimsRequested=%d, dataType=%p, format=%p, nbDims=%p, filterDimA=%p)\n",
        filterDesc, nbDimsRequested, dataType, format, nbDims, filterDimA);

    // Native variable declarations
    cudnnFilterDescriptor_t filterDesc_native;
    int nbDimsRequested_native = 0;
    cudnnDataType_t dataType_native;
    cudnnTensorFormat_t format_native;
    int nbDims_native;
    int * filterDimA_native = NULL;

    // Obtain native variable values
    filterDesc_native = (cudnnFilterDescriptor_t)getNativePointerValue(env, filterDesc);
    nbDimsRequested_native = (int)nbDimsRequested;
    // dataType is write-only
    // format is write-only
    // nbDims is write-only
    if (!initNative(env, filterDimA, filterDimA_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetFilterNdDescriptor(filterDesc_native, nbDimsRequested_native, &dataType_native, &format_native, &nbDims_native, filterDimA_native);

    // Write back native variable values
    // filterDesc is read-only
    // nbDimsRequested is primitive
    if (!set(env, dataType, 0, (jint)dataType_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, format, 0, (jint)format_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, nbDims, 0, (jint)nbDims_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!releaseNative(env, filterDimA_native, filterDimA, true)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetFilterSizeInBytesNative(JNIEnv *env, jclass cls, jobject filterDesc, jlongArray size)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetFilterSizeInBytes(filterDesc=%p, size=%p)\n",
        filterDesc, size);

    // Native variable declarations
    cudnnFilterDescriptor_t filterDesc_native;
    size_t size_native;

    // Obtain native variable values
    filterDesc_native = (cudnnFilterDescriptor_t)getNativePointerValue(env, filterDesc);
    // size is write-only

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetFilterSizeInBytes(filterDesc_native, &size_native);

    // Write back native variable values
    // filterDesc is read-only
    if (!set(env, size, 0, (jlong)size_native)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnTransformFilterNative(JNIEnv *env, jclass cls, jobject handle, jobject transDesc, jobject alpha, jobject srcDesc, jobject srcData, jobject beta, jobject destDesc, jobject destData)
{
    // Null-checks for non-primitive arguments
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cudnnTransformFilter");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (transDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'transDesc' is null for cudnnTransformFilter");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (alpha == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'alpha' is null for cudnnTransformFilter");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (srcDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'srcDesc' is null for cudnnTransformFilter");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (srcData == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'srcData' is null for cudnnTransformFilter");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (beta == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'beta' is null for cudnnTransformFilter");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (destDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'destDesc' is null for cudnnTransformFilter");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (destData == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'destData' is null for cudnnTransformFilter");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnTransformFilter(handle=%p, transDesc=%p, alpha=%p, srcDesc=%p, srcData=%p, beta=%p, destDesc=%p, destData=%p)\n",
        handle, transDesc, alpha, srcDesc, srcData, beta, destDesc, destData);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnTensorTransformDescriptor_t transDesc_native;
    void * alpha_native = NULL;
    cudnnFilterDescriptor_t srcDesc_native;
    void * srcData_native = NULL;
    void * beta_native = NULL;
    cudnnFilterDescriptor_t destDesc_native;
    void * destData_native = NULL;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    transDesc_native = (cudnnTensorTransformDescriptor_t)getNativePointerValue(env, transDesc);
    PointerData *alpha_pointerData = initPointerData(env, alpha);
    if (alpha_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    alpha_native = (void *)alpha_pointerData->getPointer(env);
    srcDesc_native = (cudnnFilterDescriptor_t)getNativePointerValue(env, srcDesc);
    srcData_native = (void *)getPointer(env, srcData);
    PointerData *beta_pointerData = initPointerData(env, beta);
    if (beta_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    beta_native = (void *)beta_pointerData->getPointer(env);
    destDesc_native = (cudnnFilterDescriptor_t)getNativePointerValue(env, destDesc);
    destData_native = (void *)getPointer(env, destData);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnTransformFilter(handle_native, transDesc_native, alpha_native, srcDesc_native, srcData_native, beta_native, destDesc_native, destData_native);

    // Write back native variable values
    // handle is read-only
    // transDesc is read-only
    if (!releasePointerData(env, alpha_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // srcDesc is read-only
    // srcData is a native pointer
    if (!releasePointerData(env, beta_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // destDesc is read-only
    // destData is a native pointer

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnDestroyFilterDescriptorNative(JNIEnv *env, jclass cls, jobject filterDesc)
{
    // Null-checks for non-primitive arguments

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
    // filterDesc is read-only

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

/** Softmax functions: All of the form "output = alpha * Op(inputs) + beta * output" */
/** Function to perform forward softmax */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnSoftmaxForwardNative(JNIEnv *env, jclass cls, jobject handle, jint algo, jint mode, jobject alpha, jobject xDesc, jobject x, jobject beta, jobject yDesc, jobject y)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnSoftmaxForward(handle=%p, algo=%d, mode=%d, alpha=%p, xDesc=%p, x=%p, beta=%p, yDesc=%p, y=%p)\n",
        handle, algo, mode, alpha, xDesc, x, beta, yDesc, y);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnSoftmaxAlgorithm_t algo_native;
    cudnnSoftmaxMode_t mode_native;
    void * alpha_native = NULL;
    cudnnTensorDescriptor_t xDesc_native;
    void * x_native = NULL;
    void * beta_native = NULL;
    cudnnTensorDescriptor_t yDesc_native;
    void * y_native = NULL;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    algo_native = (cudnnSoftmaxAlgorithm_t)algo;
    mode_native = (cudnnSoftmaxMode_t)mode;
    PointerData *alpha_pointerData = initPointerData(env, alpha);
    if (alpha_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    alpha_native = (void *)alpha_pointerData->getPointer(env);
    xDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, xDesc);
    x_native = (void *)getPointer(env, x);
    PointerData *beta_pointerData = initPointerData(env, beta);
    if (beta_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    beta_native = (void *)beta_pointerData->getPointer(env);
    yDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, yDesc);
    y_native = (void *)getPointer(env, y);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnSoftmaxForward(handle_native, algo_native, mode_native, alpha_native, xDesc_native, x_native, beta_native, yDesc_native, y_native);

    // Write back native variable values
    // handle is read-only
    // algo is primitive
    // mode is primitive
    if (!releasePointerData(env, alpha_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // xDesc is read-only
    // x is a native pointer
    if (!releasePointerData(env, beta_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // yDesc is read-only
    // y is a native pointer

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

/** Create an instance of pooling descriptor */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnCreatePoolingDescriptorNative(JNIEnv *env, jclass cls, jobject poolingDesc)
{
    // Null-checks for non-primitive arguments

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
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnSetPooling2dDescriptorNative(JNIEnv *env, jclass cls, jobject poolingDesc, jint mode, jint maxpoolingNanOpt, jint windowHeight, jint windowWidth, jint verticalPadding, jint horizontalPadding, jint verticalStride, jint horizontalStride)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnSetPooling2dDescriptor(poolingDesc=%p, mode=%d, maxpoolingNanOpt=%d, windowHeight=%d, windowWidth=%d, verticalPadding=%d, horizontalPadding=%d, verticalStride=%d, horizontalStride=%d)\n",
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
    cudnnStatus_t jniResult_native = cudnnSetPooling2dDescriptor(poolingDesc_native, mode_native, maxpoolingNanOpt_native, windowHeight_native, windowWidth_native, verticalPadding_native, horizontalPadding_native, verticalStride_native, horizontalStride_native);

    // Write back native variable values
    // poolingDesc is read-only
    // mode is primitive
    // maxpoolingNanOpt is primitive
    // windowHeight is primitive
    // windowWidth is primitive
    // verticalPadding is primitive
    // horizontalPadding is primitive
    // verticalStride is primitive
    // horizontalStride is primitive

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetPooling2dDescriptorNative(JNIEnv *env, jclass cls, jobject poolingDesc, jintArray mode, jintArray maxpoolingNanOpt, jintArray windowHeight, jintArray windowWidth, jintArray verticalPadding, jintArray horizontalPadding, jintArray verticalStride, jintArray horizontalStride)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetPooling2dDescriptor(poolingDesc=%p, mode=%p, maxpoolingNanOpt=%p, windowHeight=%p, windowWidth=%p, verticalPadding=%p, horizontalPadding=%p, verticalStride=%p, horizontalStride=%p)\n",
        poolingDesc, mode, maxpoolingNanOpt, windowHeight, windowWidth, verticalPadding, horizontalPadding, verticalStride, horizontalStride);

    // Native variable declarations
    cudnnPoolingDescriptor_t poolingDesc_native;
    cudnnPoolingMode_t mode_native;
    cudnnNanPropagation_t maxpoolingNanOpt_native;
    int windowHeight_native;
    int windowWidth_native;
    int verticalPadding_native;
    int horizontalPadding_native;
    int verticalStride_native;
    int horizontalStride_native;

    // Obtain native variable values
    poolingDesc_native = (cudnnPoolingDescriptor_t)getNativePointerValue(env, poolingDesc);
    // mode is write-only
    // maxpoolingNanOpt is write-only
    // windowHeight is write-only
    // windowWidth is write-only
    // verticalPadding is write-only
    // horizontalPadding is write-only
    // verticalStride is write-only
    // horizontalStride is write-only

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetPooling2dDescriptor(poolingDesc_native, &mode_native, &maxpoolingNanOpt_native, &windowHeight_native, &windowWidth_native, &verticalPadding_native, &horizontalPadding_native, &verticalStride_native, &horizontalStride_native);

    // Write back native variable values
    // poolingDesc is read-only
    if (!set(env, mode, 0, (jint)mode_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, maxpoolingNanOpt, 0, (jint)maxpoolingNanOpt_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, windowHeight, 0, (jint)windowHeight_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, windowWidth, 0, (jint)windowWidth_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, verticalPadding, 0, (jint)verticalPadding_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, horizontalPadding, 0, (jint)horizontalPadding_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, verticalStride, 0, (jint)verticalStride_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, horizontalStride, 0, (jint)horizontalStride_native)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnSetPoolingNdDescriptorNative(JNIEnv *env, jclass cls, jobject poolingDesc, jint mode, jint maxpoolingNanOpt, jint nbDims, jintArray windowDimA, jintArray paddingA, jintArray strideA)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnSetPoolingNdDescriptor(poolingDesc=%p, mode=%d, maxpoolingNanOpt=%d, nbDims=%d, windowDimA=%p, paddingA=%p, strideA=%p)\n",
        poolingDesc, mode, maxpoolingNanOpt, nbDims, windowDimA, paddingA, strideA);

    // Native variable declarations
    cudnnPoolingDescriptor_t poolingDesc_native;
    cudnnPoolingMode_t mode_native;
    cudnnNanPropagation_t maxpoolingNanOpt_native;
    int nbDims_native = 0;
    int * windowDimA_native = NULL;
    int * paddingA_native = NULL;
    int * strideA_native = NULL;

    // Obtain native variable values
    poolingDesc_native = (cudnnPoolingDescriptor_t)getNativePointerValue(env, poolingDesc);
    mode_native = (cudnnPoolingMode_t)mode;
    maxpoolingNanOpt_native = (cudnnNanPropagation_t)maxpoolingNanOpt;
    nbDims_native = (int)nbDims;
    if (!initNative(env, windowDimA, windowDimA_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!initNative(env, paddingA, paddingA_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!initNative(env, strideA, strideA_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnSetPoolingNdDescriptor(poolingDesc_native, mode_native, maxpoolingNanOpt_native, nbDims_native, windowDimA_native, paddingA_native, strideA_native);

    // Write back native variable values
    // poolingDesc is read-only
    // mode is primitive
    // maxpoolingNanOpt is primitive
    // nbDims is primitive
    if (!releaseNative(env, windowDimA_native, windowDimA, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!releaseNative(env, paddingA_native, paddingA, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!releaseNative(env, strideA_native, strideA, true)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetPoolingNdDescriptorNative(JNIEnv *env, jclass cls, jobject poolingDesc, jint nbDimsRequested, jintArray mode, jintArray maxpoolingNanOpt, jintArray nbDims, jintArray windowDimA, jintArray paddingA, jintArray strideA)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetPoolingNdDescriptor(poolingDesc=%p, nbDimsRequested=%d, mode=%p, maxpoolingNanOpt=%p, nbDims=%p, windowDimA=%p, paddingA=%p, strideA=%p)\n",
        poolingDesc, nbDimsRequested, mode, maxpoolingNanOpt, nbDims, windowDimA, paddingA, strideA);

    // Native variable declarations
    cudnnPoolingDescriptor_t poolingDesc_native;
    int nbDimsRequested_native = 0;
    cudnnPoolingMode_t mode_native;
    cudnnNanPropagation_t maxpoolingNanOpt_native;
    int nbDims_native;
    int * windowDimA_native = NULL;
    int * paddingA_native = NULL;
    int * strideA_native = NULL;

    // Obtain native variable values
    poolingDesc_native = (cudnnPoolingDescriptor_t)getNativePointerValue(env, poolingDesc);
    nbDimsRequested_native = (int)nbDimsRequested;
    // mode is write-only
    // maxpoolingNanOpt is write-only
    // nbDims is write-only
    if (!initNative(env, windowDimA, windowDimA_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!initNative(env, paddingA, paddingA_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!initNative(env, strideA, strideA_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetPoolingNdDescriptor(poolingDesc_native, nbDimsRequested_native, &mode_native, &maxpoolingNanOpt_native, &nbDims_native, windowDimA_native, paddingA_native, strideA_native);

    // Write back native variable values
    // poolingDesc is read-only
    // nbDimsRequested is primitive
    if (!set(env, mode, 0, (jint)mode_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, maxpoolingNanOpt, 0, (jint)maxpoolingNanOpt_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, nbDims, 0, (jint)nbDims_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!releaseNative(env, windowDimA_native, windowDimA, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!releaseNative(env, paddingA_native, paddingA, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!releaseNative(env, strideA_native, strideA, true)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetPoolingNdForwardOutputDimNative(JNIEnv *env, jclass cls, jobject poolingDesc, jobject inputTensorDesc, jint nbDims, jintArray outputTensorDimA)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetPoolingNdForwardOutputDim(poolingDesc=%p, inputTensorDesc=%p, nbDims=%d, outputTensorDimA=%p)\n",
        poolingDesc, inputTensorDesc, nbDims, outputTensorDimA);

    // Native variable declarations
    cudnnPoolingDescriptor_t poolingDesc_native;
    cudnnTensorDescriptor_t inputTensorDesc_native;
    int nbDims_native = 0;
    int * outputTensorDimA_native = NULL;

    // Obtain native variable values
    poolingDesc_native = (cudnnPoolingDescriptor_t)getNativePointerValue(env, poolingDesc);
    inputTensorDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, inputTensorDesc);
    nbDims_native = (int)nbDims;
    if (!initNative(env, outputTensorDimA, outputTensorDimA_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetPoolingNdForwardOutputDim(poolingDesc_native, inputTensorDesc_native, nbDims_native, outputTensorDimA_native);

    // Write back native variable values
    // poolingDesc is read-only
    // inputTensorDesc is read-only
    // nbDims is primitive
    if (!releaseNative(env, outputTensorDimA_native, outputTensorDimA, true)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetPooling2dForwardOutputDimNative(JNIEnv *env, jclass cls, jobject poolingDesc, jobject inputTensorDesc, jintArray n, jintArray c, jintArray h, jintArray w)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetPooling2dForwardOutputDim(poolingDesc=%p, inputTensorDesc=%p, n=%p, c=%p, h=%p, w=%p)\n",
        poolingDesc, inputTensorDesc, n, c, h, w);

    // Native variable declarations
    cudnnPoolingDescriptor_t poolingDesc_native;
    cudnnTensorDescriptor_t inputTensorDesc_native;
    int n_native;
    int c_native;
    int h_native;
    int w_native;

    // Obtain native variable values
    poolingDesc_native = (cudnnPoolingDescriptor_t)getNativePointerValue(env, poolingDesc);
    inputTensorDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, inputTensorDesc);
    // n is write-only
    // c is write-only
    // h is write-only
    // w is write-only

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetPooling2dForwardOutputDim(poolingDesc_native, inputTensorDesc_native, &n_native, &c_native, &h_native, &w_native);

    // Write back native variable values
    // poolingDesc is read-only
    // inputTensorDesc is read-only
    if (!set(env, n, 0, (jint)n_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, c, 0, (jint)c_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, h, 0, (jint)h_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, w, 0, (jint)w_native)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

/** Destroy an instance of pooling descriptor */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnDestroyPoolingDescriptorNative(JNIEnv *env, jclass cls, jobject poolingDesc)
{
    // Null-checks for non-primitive arguments

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
    // poolingDesc is read-only

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

/** Pooling functions: All of the form "output = alpha * Op(inputs) + beta * output" */
/** Function to perform forward pooling */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnPoolingForwardNative(JNIEnv *env, jclass cls, jobject handle, jobject poolingDesc, jobject alpha, jobject xDesc, jobject x, jobject beta, jobject yDesc, jobject y)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnPoolingForward(handle=%p, poolingDesc=%p, alpha=%p, xDesc=%p, x=%p, beta=%p, yDesc=%p, y=%p)\n",
        handle, poolingDesc, alpha, xDesc, x, beta, yDesc, y);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnPoolingDescriptor_t poolingDesc_native;
    void * alpha_native = NULL;
    cudnnTensorDescriptor_t xDesc_native;
    void * x_native = NULL;
    void * beta_native = NULL;
    cudnnTensorDescriptor_t yDesc_native;
    void * y_native = NULL;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    poolingDesc_native = (cudnnPoolingDescriptor_t)getNativePointerValue(env, poolingDesc);
    PointerData *alpha_pointerData = initPointerData(env, alpha);
    if (alpha_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    alpha_native = (void *)alpha_pointerData->getPointer(env);
    xDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, xDesc);
    x_native = (void *)getPointer(env, x);
    PointerData *beta_pointerData = initPointerData(env, beta);
    if (beta_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    beta_native = (void *)beta_pointerData->getPointer(env);
    yDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, yDesc);
    y_native = (void *)getPointer(env, y);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnPoolingForward(handle_native, poolingDesc_native, alpha_native, xDesc_native, x_native, beta_native, yDesc_native, y_native);

    // Write back native variable values
    // handle is read-only
    // poolingDesc is read-only
    if (!releasePointerData(env, alpha_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // xDesc is read-only
    // x is a native pointer
    if (!releasePointerData(env, beta_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // yDesc is read-only
    // y is a native pointer

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

/** Activation functions: All of the form "output = alpha * Op(inputs) + beta * output" */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnCreateActivationDescriptorNative(JNIEnv *env, jclass cls, jobject activationDesc)
{
    // Null-checks for non-primitive arguments

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
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnSetActivationDescriptorNative(JNIEnv *env, jclass cls, jobject activationDesc, jint mode, jint reluNanOpt, jdouble coef)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnSetActivationDescriptor(activationDesc=%p, mode=%d, reluNanOpt=%d, coef=%lf)\n",
        activationDesc, mode, reluNanOpt, coef);

    // Native variable declarations
    cudnnActivationDescriptor_t activationDesc_native;
    cudnnActivationMode_t mode_native;
    cudnnNanPropagation_t reluNanOpt_native;
    double coef_native = 0.0;

    // Obtain native variable values
    activationDesc_native = (cudnnActivationDescriptor_t)getNativePointerValue(env, activationDesc);
    mode_native = (cudnnActivationMode_t)mode;
    reluNanOpt_native = (cudnnNanPropagation_t)reluNanOpt;
    coef_native = (double)coef;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnSetActivationDescriptor(activationDesc_native, mode_native, reluNanOpt_native, coef_native);

    // Write back native variable values
    // activationDesc is read-only
    // mode is primitive
    // reluNanOpt is primitive
    // coef is primitive

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetActivationDescriptorNative(JNIEnv *env, jclass cls, jobject activationDesc, jintArray mode, jintArray reluNanOpt, jdoubleArray coef)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetActivationDescriptor(activationDesc=%p, mode=%p, reluNanOpt=%p, coef=%p)\n",
        activationDesc, mode, reluNanOpt, coef);

    // Native variable declarations
    cudnnActivationDescriptor_t activationDesc_native;
    cudnnActivationMode_t mode_native;
    cudnnNanPropagation_t reluNanOpt_native;
    double coef_native;

    // Obtain native variable values
    activationDesc_native = (cudnnActivationDescriptor_t)getNativePointerValue(env, activationDesc);
    // mode is write-only
    // reluNanOpt is write-only
    // coef is write-only

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetActivationDescriptor(activationDesc_native, &mode_native, &reluNanOpt_native, &coef_native);

    // Write back native variable values
    // activationDesc is read-only
    if (!set(env, mode, 0, (jint)mode_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, reluNanOpt, 0, (jint)reluNanOpt_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, coef, 0, (jdouble)coef_native)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnSetActivationDescriptorSwishBetaNative(JNIEnv *env, jclass cls, jobject activationDesc, jdouble swish_beta)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnSetActivationDescriptorSwishBeta(activationDesc=%p, swish_beta=%lf)\n",
        activationDesc, swish_beta);

    // Native variable declarations
    cudnnActivationDescriptor_t activationDesc_native;
    double swish_beta_native = 0.0;

    // Obtain native variable values
    activationDesc_native = (cudnnActivationDescriptor_t)getNativePointerValue(env, activationDesc);
    swish_beta_native = (double)swish_beta;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnSetActivationDescriptorSwishBeta(activationDesc_native, swish_beta_native);

    // Write back native variable values
    // activationDesc is read-only
    // swish_beta is primitive

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetActivationDescriptorSwishBetaNative(JNIEnv *env, jclass cls, jobject activationDesc, jdoubleArray swish_beta)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetActivationDescriptorSwishBeta(activationDesc=%p, swish_beta=%p)\n",
        activationDesc, swish_beta);

    // Native variable declarations
    cudnnActivationDescriptor_t activationDesc_native;
    double swish_beta_native;

    // Obtain native variable values
    activationDesc_native = (cudnnActivationDescriptor_t)getNativePointerValue(env, activationDesc);
    // swish_beta is write-only

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetActivationDescriptorSwishBeta(activationDesc_native, &swish_beta_native);

    // Write back native variable values
    // activationDesc is read-only
    if (!set(env, swish_beta, 0, (jdouble)swish_beta_native)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnDestroyActivationDescriptorNative(JNIEnv *env, jclass cls, jobject activationDesc)
{
    // Null-checks for non-primitive arguments

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
    // activationDesc is read-only

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

/** Function to perform forward activation  */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnActivationForwardNative(JNIEnv *env, jclass cls, jobject handle, jobject activationDesc, jobject alpha, jobject xDesc, jobject x, jobject beta, jobject yDesc, jobject y)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnActivationForward(handle=%p, activationDesc=%p, alpha=%p, xDesc=%p, x=%p, beta=%p, yDesc=%p, y=%p)\n",
        handle, activationDesc, alpha, xDesc, x, beta, yDesc, y);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnActivationDescriptor_t activationDesc_native;
    void * alpha_native = NULL;
    cudnnTensorDescriptor_t xDesc_native;
    void * x_native = NULL;
    void * beta_native = NULL;
    cudnnTensorDescriptor_t yDesc_native;
    void * y_native = NULL;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    activationDesc_native = (cudnnActivationDescriptor_t)getNativePointerValue(env, activationDesc);
    PointerData *alpha_pointerData = initPointerData(env, alpha);
    if (alpha_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    alpha_native = (void *)alpha_pointerData->getPointer(env);
    xDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, xDesc);
    x_native = (void *)getPointer(env, x);
    PointerData *beta_pointerData = initPointerData(env, beta);
    if (beta_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    beta_native = (void *)beta_pointerData->getPointer(env);
    yDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, yDesc);
    y_native = (void *)getPointer(env, y);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnActivationForward(handle_native, activationDesc_native, alpha_native, xDesc_native, x_native, beta_native, yDesc_native, y_native);

    // Write back native variable values
    // handle is read-only
    // activationDesc is read-only
    if (!releasePointerData(env, alpha_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // xDesc is read-only
    // x is a native pointer
    if (!releasePointerData(env, beta_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // yDesc is read-only
    // y is a native pointer

    // Return the result
    jint jniResult = (jint)jniResult_native;
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
    jint jniResult = (jint)jniResult_native;
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

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnSetLRNDescriptor(normDesc=%p, lrnN=%d, lrnAlpha=%lf, lrnBeta=%lf, lrnK=%lf)\n",
        normDesc, lrnN, lrnAlpha, lrnBeta, lrnK);

    // Native variable declarations
    cudnnLRNDescriptor_t normDesc_native;
    unsigned int lrnN_native;
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
    // normDesc is read-only
    // lrnN is primitive
    // lrnAlpha is primitive
    // lrnBeta is primitive
    // lrnK is primitive

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

/**
 * <pre>
 * Retrieve the settings currently stored in an LRN layer descriptor
 * Any of the provided pointers can be NULL (no corresponding value will be returned)
 * </pre>
 */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetLRNDescriptorNative(JNIEnv *env, jclass cls, jobject normDesc, jintArray lrnN, jdoubleArray lrnAlpha, jdoubleArray lrnBeta, jdoubleArray lrnK)
{
    // Null-checks for non-primitive arguments
    if (normDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'normDesc' is null for cudnnGetLRNDescriptor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetLRNDescriptor(normDesc=%p, lrnN=%p, lrnAlpha=%p, lrnBeta=%p, lrnK=%p)\n",
        normDesc, lrnN, lrnAlpha, lrnBeta, lrnK);

    // Native variable declarations
    cudnnLRNDescriptor_t normDesc_native;
    unsigned int lrnN_native;
    double lrnAlpha_native;
    double lrnBeta_native;
    double lrnK_native;

    // Obtain native variable values
    normDesc_native = (cudnnLRNDescriptor_t)getNativePointerValue(env, normDesc);
    // lrnN is write-only
    // lrnAlpha is write-only
    // lrnBeta is write-only
    // lrnK is write-only

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetLRNDescriptor(normDesc_native, &lrnN_native, &lrnAlpha_native, &lrnBeta_native, &lrnK_native);

    // Write back native variable values
    // normDesc is read-only
    if (!set(env, lrnN, 0, (jint)lrnN_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, lrnAlpha, 0, (jdouble)lrnAlpha_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, lrnBeta, 0, (jdouble)lrnBeta_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, lrnK, 0, (jdouble)lrnK_native)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

/** Destroy an instance of LRN descriptor */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnDestroyLRNDescriptorNative(JNIEnv *env, jclass cls, jobject lrnDesc)
{
    // Null-checks for non-primitive arguments

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
    // lrnDesc is read-only

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

/** LRN functions: output = alpha * normalize(x) + beta * old_y */
/** LRN cross-channel forward computation. Double parameters cast to tensor data type */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnLRNCrossChannelForwardNative(JNIEnv *env, jclass cls, jobject handle, jobject normDesc, jint lrnMode, jobject alpha, jobject xDesc, jobject x, jobject beta, jobject yDesc, jobject y)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnLRNCrossChannelForward(handle=%p, normDesc=%p, lrnMode=%d, alpha=%p, xDesc=%p, x=%p, beta=%p, yDesc=%p, y=%p)\n",
        handle, normDesc, lrnMode, alpha, xDesc, x, beta, yDesc, y);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnLRNDescriptor_t normDesc_native;
    cudnnLRNMode_t lrnMode_native;
    void * alpha_native = NULL;
    cudnnTensorDescriptor_t xDesc_native;
    void * x_native = NULL;
    void * beta_native = NULL;
    cudnnTensorDescriptor_t yDesc_native;
    void * y_native = NULL;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    normDesc_native = (cudnnLRNDescriptor_t)getNativePointerValue(env, normDesc);
    lrnMode_native = (cudnnLRNMode_t)lrnMode;
    PointerData *alpha_pointerData = initPointerData(env, alpha);
    if (alpha_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    alpha_native = (void *)alpha_pointerData->getPointer(env);
    xDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, xDesc);
    x_native = (void *)getPointer(env, x);
    PointerData *beta_pointerData = initPointerData(env, beta);
    if (beta_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    beta_native = (void *)beta_pointerData->getPointer(env);
    yDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, yDesc);
    y_native = (void *)getPointer(env, y);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnLRNCrossChannelForward(handle_native, normDesc_native, lrnMode_native, alpha_native, xDesc_native, x_native, beta_native, yDesc_native, y_native);

    // Write back native variable values
    // handle is read-only
    // normDesc is read-only
    // lrnMode is primitive
    if (!releasePointerData(env, alpha_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // xDesc is read-only
    // x is a native pointer
    if (!releasePointerData(env, beta_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // yDesc is read-only
    // y is a native pointer

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

/** LCN/divisive normalization functions: y = alpha * normalize(x) + beta * y */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnDivisiveNormalizationForwardNative(JNIEnv *env, jclass cls, jobject handle, jobject normDesc, jint mode, jobject alpha, jobject xDesc, jobject x, jobject means, jobject temp, jobject temp2, jobject beta, jobject yDesc, jobject y)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnDivisiveNormalizationForward(handle=%p, normDesc=%p, mode=%d, alpha=%p, xDesc=%p, x=%p, means=%p, temp=%p, temp2=%p, beta=%p, yDesc=%p, y=%p)\n",
        handle, normDesc, mode, alpha, xDesc, x, means, temp, temp2, beta, yDesc, y);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnLRNDescriptor_t normDesc_native;
    cudnnDivNormMode_t mode_native;
    void * alpha_native = NULL;
    cudnnTensorDescriptor_t xDesc_native;
    void * x_native = NULL;
    void * means_native = NULL;
    void * temp_native = NULL;
    void * temp2_native = NULL;
    void * beta_native = NULL;
    cudnnTensorDescriptor_t yDesc_native;
    void * y_native = NULL;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    normDesc_native = (cudnnLRNDescriptor_t)getNativePointerValue(env, normDesc);
    mode_native = (cudnnDivNormMode_t)mode;
    PointerData *alpha_pointerData = initPointerData(env, alpha);
    if (alpha_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    alpha_native = (void *)alpha_pointerData->getPointer(env);
    xDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, xDesc);
    x_native = (void *)getPointer(env, x);
    means_native = (void *)getPointer(env, means);
    temp_native = (void *)getPointer(env, temp);
    temp2_native = (void *)getPointer(env, temp2);
    PointerData *beta_pointerData = initPointerData(env, beta);
    if (beta_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    beta_native = (void *)beta_pointerData->getPointer(env);
    yDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, yDesc);
    y_native = (void *)getPointer(env, y);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnDivisiveNormalizationForward(handle_native, normDesc_native, mode_native, alpha_native, xDesc_native, x_native, means_native, temp_native, temp2_native, beta_native, yDesc_native, y_native);

    // Write back native variable values
    // handle is read-only
    // normDesc is read-only
    // mode is primitive
    if (!releasePointerData(env, alpha_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // xDesc is read-only
    // x is a native pointer
    // means is a native pointer
    // temp is a native pointer
    // temp2 is a native pointer
    if (!releasePointerData(env, beta_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // yDesc is read-only
    // y is a native pointer

    // Return the result
    jint jniResult = (jint)jniResult_native;
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
    // derivedBnDesc is read-only
    // xDesc is read-only
    // mode is primitive

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

/**
 * <pre>
 * Performs Batch Normalization during Inference:
 * y[i] = bnScale[k]*(x[i]-estimatedMean[k])/sqrt(epsilon+estimatedVariance[k]) + bnBias[k]
 * with bnScale, bnBias, runningMean, runningInvVariance tensors indexed
 * according to spatial or per-activation mode. Refer to cudnnBatchNormalizationForwardTraining
 * above for notes on function arguments.
 * </pre>
 */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnBatchNormalizationForwardInferenceNative(JNIEnv *env, jclass cls, jobject handle, jint mode, jobject alpha, jobject beta, jobject xDesc, jobject x, jobject yDesc, jobject y, jobject bnScaleBiasMeanVarDesc, jobject bnScale, jobject bnBias, jobject estimatedMean, jobject estimatedVariance, jdouble epsilon)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnBatchNormalizationForwardInference(handle=%p, mode=%d, alpha=%p, beta=%p, xDesc=%p, x=%p, yDesc=%p, y=%p, bnScaleBiasMeanVarDesc=%p, bnScale=%p, bnBias=%p, estimatedMean=%p, estimatedVariance=%p, epsilon=%lf)\n",
        handle, mode, alpha, beta, xDesc, x, yDesc, y, bnScaleBiasMeanVarDesc, bnScale, bnBias, estimatedMean, estimatedVariance, epsilon);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnBatchNormMode_t mode_native;
    void * alpha_native = NULL;
    void * beta_native = NULL;
    cudnnTensorDescriptor_t xDesc_native;
    void * x_native = NULL;
    cudnnTensorDescriptor_t yDesc_native;
    void * y_native = NULL;
    cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc_native;
    void * bnScale_native = NULL;
    void * bnBias_native = NULL;
    void * estimatedMean_native = NULL;
    void * estimatedVariance_native = NULL;
    double epsilon_native = 0.0;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    mode_native = (cudnnBatchNormMode_t)mode;
    PointerData *alpha_pointerData = initPointerData(env, alpha);
    if (alpha_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    alpha_native = (void *)alpha_pointerData->getPointer(env);
    PointerData *beta_pointerData = initPointerData(env, beta);
    if (beta_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    beta_native = (void *)beta_pointerData->getPointer(env);
    xDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, xDesc);
    x_native = (void *)getPointer(env, x);
    yDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, yDesc);
    y_native = (void *)getPointer(env, y);
    bnScaleBiasMeanVarDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, bnScaleBiasMeanVarDesc);
    bnScale_native = (void *)getPointer(env, bnScale);
    bnBias_native = (void *)getPointer(env, bnBias);
    estimatedMean_native = (void *)getPointer(env, estimatedMean);
    estimatedVariance_native = (void *)getPointer(env, estimatedVariance);
    epsilon_native = (double)epsilon;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnBatchNormalizationForwardInference(handle_native, mode_native, alpha_native, beta_native, xDesc_native, x_native, yDesc_native, y_native, bnScaleBiasMeanVarDesc_native, bnScale_native, bnBias_native, estimatedMean_native, estimatedVariance_native, epsilon_native);

    // Write back native variable values
    // handle is read-only
    // mode is primitive
    if (!releasePointerData(env, alpha_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!releasePointerData(env, beta_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // xDesc is read-only
    // x is a native pointer
    // yDesc is read-only
    // y is a native pointer
    // bnScaleBiasMeanVarDesc is read-only
    // bnScale is a native pointer
    // bnBias is a native pointer
    // estimatedMean is a native pointer
    // estimatedVariance is a native pointer
    // epsilon is primitive

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

/**
 * <pre>
 * Derives a tensor descriptor from layer data descriptor for Normalization
 * scale, invVariance, bnBias, bnScale tensors. Use this tensor desc for
 * normScaleBiasMeanVarDesc and normScaleBiasDiffDesc in Normalization forward and backward functions.
 * </pre>
 */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnDeriveNormTensorDescriptorNative(JNIEnv *env, jclass cls, jobject derivedNormScaleBiasDesc, jobject derivedNormMeanVarDesc, jobject xDesc, jint mode, jint groupCnt)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnDeriveNormTensorDescriptor(derivedNormScaleBiasDesc=%p, derivedNormMeanVarDesc=%p, xDesc=%p, mode=%d, groupCnt=%d)\n",
        derivedNormScaleBiasDesc, derivedNormMeanVarDesc, xDesc, mode, groupCnt);

    // Native variable declarations
    cudnnTensorDescriptor_t derivedNormScaleBiasDesc_native;
    cudnnTensorDescriptor_t derivedNormMeanVarDesc_native;
    cudnnTensorDescriptor_t xDesc_native;
    cudnnNormMode_t mode_native;
    int groupCnt_native = 0;

    // Obtain native variable values
    derivedNormScaleBiasDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, derivedNormScaleBiasDesc);
    derivedNormMeanVarDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, derivedNormMeanVarDesc);
    xDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, xDesc);
    mode_native = (cudnnNormMode_t)mode;
    groupCnt_native = (int)groupCnt;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnDeriveNormTensorDescriptor(derivedNormScaleBiasDesc_native, derivedNormMeanVarDesc_native, xDesc_native, mode_native, groupCnt_native);

    // Write back native variable values
    // derivedNormScaleBiasDesc is read-only
    // derivedNormMeanVarDesc is read-only
    // xDesc is read-only
    // mode is primitive
    // groupCnt is primitive

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

/**
 * <pre>
 * Performs Normalization during Inference:
 * y[i] = normScale[k]*(x[i]-estimatedMean[k])/sqrt(epsilon+estimatedVariance[k]) + normBias[k]
 * with normScale, normBias, runningMean, runningInvVariance tensors indexed
 * according to per-channel or per-activation mode. Refer to cudnnNormalizationForwardTraining
 * above for notes on function arguments.
 * </pre>
 */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnNormalizationForwardInferenceNative(JNIEnv *env, jclass cls, jobject handle, jint mode, jint normOps, jint algo, jobject alpha, jobject beta, jobject xDesc, jobject x, jobject normScaleBiasDesc, jobject normScale, jobject normBias, jobject normMeanVarDesc, jobject estimatedMean, jobject estimatedVariance, jobject zDesc, jobject z, jobject activationDesc, jobject yDesc, jobject y, jdouble epsilon, jint groupCnt)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnNormalizationForwardInference(handle=%p, mode=%d, normOps=%d, algo=%d, alpha=%p, beta=%p, xDesc=%p, x=%p, normScaleBiasDesc=%p, normScale=%p, normBias=%p, normMeanVarDesc=%p, estimatedMean=%p, estimatedVariance=%p, zDesc=%p, z=%p, activationDesc=%p, yDesc=%p, y=%p, epsilon=%lf, groupCnt=%d)\n",
        handle, mode, normOps, algo, alpha, beta, xDesc, x, normScaleBiasDesc, normScale, normBias, normMeanVarDesc, estimatedMean, estimatedVariance, zDesc, z, activationDesc, yDesc, y, epsilon, groupCnt);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnNormMode_t mode_native;
    cudnnNormOps_t normOps_native;
    cudnnNormAlgo_t algo_native;
    void * alpha_native = NULL;
    void * beta_native = NULL;
    cudnnTensorDescriptor_t xDesc_native;
    void * x_native = NULL;
    cudnnTensorDescriptor_t normScaleBiasDesc_native;
    void * normScale_native = NULL;
    void * normBias_native = NULL;
    cudnnTensorDescriptor_t normMeanVarDesc_native;
    void * estimatedMean_native = NULL;
    void * estimatedVariance_native = NULL;
    cudnnTensorDescriptor_t zDesc_native;
    void * z_native = NULL;
    cudnnActivationDescriptor_t activationDesc_native;
    cudnnTensorDescriptor_t yDesc_native;
    void * y_native = NULL;
    double epsilon_native = 0.0;
    int groupCnt_native = 0;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    mode_native = (cudnnNormMode_t)mode;
    normOps_native = (cudnnNormOps_t)normOps;
    algo_native = (cudnnNormAlgo_t)algo;
    PointerData *alpha_pointerData = initPointerData(env, alpha);
    if (alpha_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    alpha_native = (void *)alpha_pointerData->getPointer(env);
    PointerData *beta_pointerData = initPointerData(env, beta);
    if (beta_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    beta_native = (void *)beta_pointerData->getPointer(env);
    xDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, xDesc);
    x_native = (void *)getPointer(env, x);
    normScaleBiasDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, normScaleBiasDesc);
    normScale_native = (void *)getPointer(env, normScale);
    normBias_native = (void *)getPointer(env, normBias);
    normMeanVarDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, normMeanVarDesc);
    estimatedMean_native = (void *)getPointer(env, estimatedMean);
    estimatedVariance_native = (void *)getPointer(env, estimatedVariance);
    zDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, zDesc);
    z_native = (void *)getPointer(env, z);
    activationDesc_native = (cudnnActivationDescriptor_t)getNativePointerValue(env, activationDesc);
    yDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, yDesc);
    y_native = (void *)getPointer(env, y);
    epsilon_native = (double)epsilon;
    groupCnt_native = (int)groupCnt;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnNormalizationForwardInference(handle_native, mode_native, normOps_native, algo_native, alpha_native, beta_native, xDesc_native, x_native, normScaleBiasDesc_native, normScale_native, normBias_native, normMeanVarDesc_native, estimatedMean_native, estimatedVariance_native, zDesc_native, z_native, activationDesc_native, yDesc_native, y_native, epsilon_native, groupCnt_native);

    // Write back native variable values
    // handle is read-only
    // mode is primitive
    // normOps is primitive
    // algo is primitive
    if (!releasePointerData(env, alpha_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!releasePointerData(env, beta_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // xDesc is read-only
    // x is a native pointer
    // normScaleBiasDesc is read-only
    // normScale is a native pointer
    // normBias is a native pointer
    // normMeanVarDesc is read-only
    // estimatedMean is a native pointer
    // estimatedVariance is a native pointer
    // zDesc is read-only
    // z is a native pointer
    // activationDesc is read-only
    // yDesc is read-only
    // y is a native pointer
    // epsilon is primitive
    // groupCnt is primitive

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnCreateSpatialTransformerDescriptorNative(JNIEnv *env, jclass cls, jobject stDesc)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnCreateSpatialTransformerDescriptor(stDesc=%p)\n",
        stDesc);

    // Native variable declarations
    cudnnSpatialTransformerDescriptor_t stDesc_native;

    // Obtain native variable values
    // stDesc is write-only

    // Native function call
    cudnnStatus_t jniResult_native = cudnnCreateSpatialTransformerDescriptor(&stDesc_native);

    // Write back native variable values
    setNativePointerValue(env, stDesc, (jlong)stDesc_native);

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnSetSpatialTransformerNdDescriptorNative(JNIEnv *env, jclass cls, jobject stDesc, jint samplerType, jint dataType, jint nbDims, jintArray dimA)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnSetSpatialTransformerNdDescriptor(stDesc=%p, samplerType=%d, dataType=%d, nbDims=%d, dimA=%p)\n",
        stDesc, samplerType, dataType, nbDims, dimA);

    // Native variable declarations
    cudnnSpatialTransformerDescriptor_t stDesc_native;
    cudnnSamplerType_t samplerType_native;
    cudnnDataType_t dataType_native;
    int nbDims_native = 0;
    int * dimA_native = NULL;

    // Obtain native variable values
    stDesc_native = (cudnnSpatialTransformerDescriptor_t)getNativePointerValue(env, stDesc);
    samplerType_native = (cudnnSamplerType_t)samplerType;
    dataType_native = (cudnnDataType_t)dataType;
    nbDims_native = (int)nbDims;
    if (!initNative(env, dimA, dimA_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnSetSpatialTransformerNdDescriptor(stDesc_native, samplerType_native, dataType_native, nbDims_native, dimA_native);

    // Write back native variable values
    // stDesc is read-only
    // samplerType is primitive
    // dataType is primitive
    // nbDims is primitive
    if (!releaseNative(env, dimA_native, dimA, true)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnDestroySpatialTransformerDescriptorNative(JNIEnv *env, jclass cls, jobject stDesc)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnDestroySpatialTransformerDescriptor(stDesc=%p)\n",
        stDesc);

    // Native variable declarations
    cudnnSpatialTransformerDescriptor_t stDesc_native;

    // Obtain native variable values
    stDesc_native = (cudnnSpatialTransformerDescriptor_t)getNativePointerValue(env, stDesc);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnDestroySpatialTransformerDescriptor(stDesc_native);

    // Write back native variable values
    // stDesc is read-only

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnSpatialTfGridGeneratorForwardNative(JNIEnv *env, jclass cls, jobject handle, jobject stDesc, jobject theta, jobject grid)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnSpatialTfGridGeneratorForward(handle=%p, stDesc=%p, theta=%p, grid=%p)\n",
        handle, stDesc, theta, grid);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnSpatialTransformerDescriptor_t stDesc_native;
    void * theta_native = NULL;
    void * grid_native = NULL;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    stDesc_native = (cudnnSpatialTransformerDescriptor_t)getNativePointerValue(env, stDesc);
    theta_native = (void *)getPointer(env, theta);
    grid_native = (void *)getPointer(env, grid);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnSpatialTfGridGeneratorForward(handle_native, stDesc_native, theta_native, grid_native);

    // Write back native variable values
    // handle is read-only
    // stDesc is read-only
    // theta is a native pointer
    // grid is a native pointer

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnSpatialTfSamplerForwardNative(JNIEnv *env, jclass cls, jobject handle, jobject stDesc, jobject alpha, jobject xDesc, jobject x, jobject grid, jobject beta, jobject yDesc, jobject y)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnSpatialTfSamplerForward(handle=%p, stDesc=%p, alpha=%p, xDesc=%p, x=%p, grid=%p, beta=%p, yDesc=%p, y=%p)\n",
        handle, stDesc, alpha, xDesc, x, grid, beta, yDesc, y);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnSpatialTransformerDescriptor_t stDesc_native;
    void * alpha_native = NULL;
    cudnnTensorDescriptor_t xDesc_native;
    void * x_native = NULL;
    void * grid_native = NULL;
    void * beta_native = NULL;
    cudnnTensorDescriptor_t yDesc_native;
    void * y_native = NULL;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    stDesc_native = (cudnnSpatialTransformerDescriptor_t)getNativePointerValue(env, stDesc);
    PointerData *alpha_pointerData = initPointerData(env, alpha);
    if (alpha_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    alpha_native = (void *)alpha_pointerData->getPointer(env);
    xDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, xDesc);
    x_native = (void *)getPointer(env, x);
    grid_native = (void *)getPointer(env, grid);
    PointerData *beta_pointerData = initPointerData(env, beta);
    if (beta_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    beta_native = (void *)beta_pointerData->getPointer(env);
    yDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, yDesc);
    y_native = (void *)getPointer(env, y);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnSpatialTfSamplerForward(handle_native, stDesc_native, alpha_native, xDesc_native, x_native, grid_native, beta_native, yDesc_native, y_native);

    // Write back native variable values
    // handle is read-only
    // stDesc is read-only
    if (!releasePointerData(env, alpha_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // xDesc is read-only
    // x is a native pointer
    // grid is a native pointer
    if (!releasePointerData(env, beta_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // yDesc is read-only
    // y is a native pointer

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnCreateDropoutDescriptorNative(JNIEnv *env, jclass cls, jobject dropoutDesc)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnCreateDropoutDescriptor(dropoutDesc=%p)\n",
        dropoutDesc);

    // Native variable declarations
    cudnnDropoutDescriptor_t dropoutDesc_native;

    // Obtain native variable values
    // dropoutDesc is write-only

    // Native function call
    cudnnStatus_t jniResult_native = cudnnCreateDropoutDescriptor(&dropoutDesc_native);

    // Write back native variable values
    setNativePointerValue(env, dropoutDesc, (jlong)dropoutDesc_native);

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnDestroyDropoutDescriptorNative(JNIEnv *env, jclass cls, jobject dropoutDesc)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnDestroyDropoutDescriptor(dropoutDesc=%p)\n",
        dropoutDesc);

    // Native variable declarations
    cudnnDropoutDescriptor_t dropoutDesc_native;

    // Obtain native variable values
    dropoutDesc_native = (cudnnDropoutDescriptor_t)getNativePointerValue(env, dropoutDesc);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnDestroyDropoutDescriptor(dropoutDesc_native);

    // Write back native variable values
    // dropoutDesc is read-only

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

/**helper function to determine size of the states to be passed to cudnnSetDropoutDescriptor */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnDropoutGetStatesSizeNative(JNIEnv *env, jclass cls, jobject handle, jlongArray sizeInBytes)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnDropoutGetStatesSize(handle=%p, sizeInBytes=%p)\n",
        handle, sizeInBytes);

    // Native variable declarations
    cudnnHandle_t handle_native;
    size_t sizeInBytes_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    // sizeInBytes is write-only

    // Native function call
    cudnnStatus_t jniResult_native = cudnnDropoutGetStatesSize(handle_native, &sizeInBytes_native);

    // Write back native variable values
    // handle is read-only
    if (!set(env, sizeInBytes, 0, (jlong)sizeInBytes_native)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

/**helper function to determine size of the reserve space to be passed to dropout forward/backward calls */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnDropoutGetReserveSpaceSizeNative(JNIEnv *env, jclass cls, jobject xdesc, jlongArray sizeInBytes)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnDropoutGetReserveSpaceSize(xdesc=%p, sizeInBytes=%p)\n",
        xdesc, sizeInBytes);

    // Native variable declarations
    cudnnTensorDescriptor_t xdesc_native;
    size_t sizeInBytes_native;

    // Obtain native variable values
    xdesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, xdesc);
    // sizeInBytes is write-only

    // Native function call
    cudnnStatus_t jniResult_native = cudnnDropoutGetReserveSpaceSize(xdesc_native, &sizeInBytes_native);

    // Write back native variable values
    // xdesc is read-only
    if (!set(env, sizeInBytes, 0, (jlong)sizeInBytes_native)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnSetDropoutDescriptorNative(JNIEnv *env, jclass cls, jobject dropoutDesc, jobject handle, jfloat dropout, jobject states, jlong stateSizeInBytes, jlong seed)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnSetDropoutDescriptor(dropoutDesc=%p, handle=%p, dropout=%f, states=%p, stateSizeInBytes=%ld, seed=%ld)\n",
        dropoutDesc, handle, dropout, states, stateSizeInBytes, seed);

    // Native variable declarations
    cudnnDropoutDescriptor_t dropoutDesc_native;
    cudnnHandle_t handle_native;
    float dropout_native = 0.0f;
    void * states_native = NULL;
    size_t stateSizeInBytes_native = 0;
    unsigned long long  seed_native;

    // Obtain native variable values
    dropoutDesc_native = (cudnnDropoutDescriptor_t)getNativePointerValue(env, dropoutDesc);
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    dropout_native = (float)dropout;
    states_native = (void *)getPointer(env, states);
    stateSizeInBytes_native = (size_t)stateSizeInBytes;
    seed_native = (unsigned long long )seed;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnSetDropoutDescriptor(dropoutDesc_native, handle_native, dropout_native, states_native, stateSizeInBytes_native, seed_native);

    // Write back native variable values
    // dropoutDesc is read-only
    // handle is read-only
    // dropout is primitive
    // states is a native pointer
    // stateSizeInBytes is primitive
    // seed is primitive

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

/** Restores the dropout descriptor to a previously saved-off state */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnRestoreDropoutDescriptorNative(JNIEnv *env, jclass cls, jobject dropoutDesc, jobject handle, jfloat dropout, jobject states, jlong stateSizeInBytes, jlong seed)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnRestoreDropoutDescriptor(dropoutDesc=%p, handle=%p, dropout=%f, states=%p, stateSizeInBytes=%ld, seed=%ld)\n",
        dropoutDesc, handle, dropout, states, stateSizeInBytes, seed);

    // Native variable declarations
    cudnnDropoutDescriptor_t dropoutDesc_native;
    cudnnHandle_t handle_native;
    float dropout_native = 0.0f;
    void * states_native = NULL;
    size_t stateSizeInBytes_native = 0;
    unsigned long long  seed_native;

    // Obtain native variable values
    dropoutDesc_native = (cudnnDropoutDescriptor_t)getNativePointerValue(env, dropoutDesc);
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    dropout_native = (float)dropout;
    states_native = (void *)getPointer(env, states);
    stateSizeInBytes_native = (size_t)stateSizeInBytes;
    seed_native = (unsigned long long )seed;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnRestoreDropoutDescriptor(dropoutDesc_native, handle_native, dropout_native, states_native, stateSizeInBytes_native, seed_native);

    // Write back native variable values
    // dropoutDesc is read-only
    // handle is read-only
    // dropout is primitive
    // states is a native pointer
    // stateSizeInBytes is primitive
    // seed is primitive

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetDropoutDescriptorNative(JNIEnv *env, jclass cls, jobject dropoutDesc, jobject handle, jfloatArray dropout, jobject states, jlongArray seed)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetDropoutDescriptor(dropoutDesc=%p, handle=%p, dropout=%p, states=%p, seed=%p)\n",
        dropoutDesc, handle, dropout, states, seed);

    // Native variable declarations
    cudnnDropoutDescriptor_t dropoutDesc_native;
    cudnnHandle_t handle_native;
    float dropout_native;
    void * * states_native = NULL;
    unsigned long long  seed_native;

    // Obtain native variable values
    dropoutDesc_native = (cudnnDropoutDescriptor_t)getNativePointerValue(env, dropoutDesc);
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    // dropout is write-only
    states_native = (void * *)getPointer(env, states);
    // seed is write-only

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetDropoutDescriptor(dropoutDesc_native, handle_native, &dropout_native, states_native, &seed_native);

    // Write back native variable values
    // dropoutDesc is read-only
    // handle is read-only
    if (!set(env, dropout, 0, (jfloat)dropout_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // states is a native pointer
    if (!set(env, seed, 0, (jlong)seed_native)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnDropoutForwardNative(JNIEnv *env, jclass cls, jobject handle, jobject dropoutDesc, jobject xdesc, jobject x, jobject ydesc, jobject y, jobject reserveSpace, jlong reserveSpaceSizeInBytes)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnDropoutForward(handle=%p, dropoutDesc=%p, xdesc=%p, x=%p, ydesc=%p, y=%p, reserveSpace=%p, reserveSpaceSizeInBytes=%ld)\n",
        handle, dropoutDesc, xdesc, x, ydesc, y, reserveSpace, reserveSpaceSizeInBytes);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnDropoutDescriptor_t dropoutDesc_native;
    cudnnTensorDescriptor_t xdesc_native;
    void * x_native = NULL;
    cudnnTensorDescriptor_t ydesc_native;
    void * y_native = NULL;
    void * reserveSpace_native = NULL;
    size_t reserveSpaceSizeInBytes_native = 0;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    dropoutDesc_native = (cudnnDropoutDescriptor_t)getNativePointerValue(env, dropoutDesc);
    xdesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, xdesc);
    x_native = (void *)getPointer(env, x);
    ydesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, ydesc);
    y_native = (void *)getPointer(env, y);
    reserveSpace_native = (void *)getPointer(env, reserveSpace);
    reserveSpaceSizeInBytes_native = (size_t)reserveSpaceSizeInBytes;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnDropoutForward(handle_native, dropoutDesc_native, xdesc_native, x_native, ydesc_native, y_native, reserveSpace_native, reserveSpaceSizeInBytes_native);

    // Write back native variable values
    // handle is read-only
    // dropoutDesc is read-only
    // xdesc is read-only
    // x is a native pointer
    // ydesc is read-only
    // y is a native pointer
    // reserveSpace is a native pointer
    // reserveSpaceSizeInBytes is primitive

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnCreateAlgorithmDescriptorNative(JNIEnv *env, jclass cls, jobject algoDesc)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnCreateAlgorithmDescriptor(algoDesc=%p)\n",
        algoDesc);

    // Native variable declarations
    cudnnAlgorithmDescriptor_t algoDesc_native;

    // Obtain native variable values
    // algoDesc is write-only

    // Native function call
    cudnnStatus_t jniResult_native = cudnnCreateAlgorithmDescriptor(&algoDesc_native);

    // Write back native variable values
    setNativePointerValue(env, algoDesc, (jlong)algoDesc_native);

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnSetAlgorithmDescriptorNative(JNIEnv *env, jclass cls, jobject algoDesc, jint algorithm)
{
    // Null-checks for non-primitive arguments
    if (algoDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'algoDesc' is null for cudnnSetAlgorithmDescriptor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    // algorithm is primitive

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnSetAlgorithmDescriptor(algoDesc=%p, algorithm=%d)\n",
        algoDesc, algorithm);

    // Native variable declarations
    cudnnAlgorithmDescriptor_t algoDesc_native;
    cudnnAlgorithm_t algorithm_native;

    // Obtain native variable values
    algoDesc_native = (cudnnAlgorithmDescriptor_t)getNativePointerValue(env, algoDesc);
    algorithm_native.algo.convFwdAlgo = (cudnnConvolutionFwdAlgo_t)algorithm;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnSetAlgorithmDescriptor(algoDesc_native, algorithm_native);

    // Write back native variable values
    // algoDesc is read-only
    // algorithm is primitive

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetAlgorithmDescriptorNative(JNIEnv *env, jclass cls, jobject algoDesc, jintArray algorithm)
{
    // Null-checks for non-primitive arguments
    if (algoDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'algoDesc' is null for cudnnGetAlgorithmDescriptor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (algorithm == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'algorithm' is null for cudnnGetAlgorithmDescriptor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetAlgorithmDescriptor(algoDesc=%p, algorithm=%p)\n",
        algoDesc, algorithm);

    // Native variable declarations
    cudnnAlgorithmDescriptor_t algoDesc_native;
    cudnnAlgorithm_t algorithm_native;

    // Obtain native variable values
    algoDesc_native = (cudnnAlgorithmDescriptor_t)getNativePointerValue(env, algoDesc);
    // algorithm is write-only

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetAlgorithmDescriptor(algoDesc_native, &algorithm_native);

    // Write back native variable values
    // algoDesc is read-only
    if (!set(env, algorithm, 0, (jint)algorithm_native.algo.convFwdAlgo)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnCopyAlgorithmDescriptorNative(JNIEnv *env, jclass cls, jobject src, jobject dest)
{
    // Deprecated as of cuDNN 8
    ThrowByName(env, "java/lang/UnsupportedOperationException", "This function is no longer supported");
    return JCUDNN_STATUS_INTERNAL_ERROR;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnDestroyAlgorithmDescriptorNative(JNIEnv *env, jclass cls, jobject algoDesc)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnDestroyAlgorithmDescriptor(algoDesc=%p)\n",
        algoDesc);

    // Native variable declarations
    cudnnAlgorithmDescriptor_t algoDesc_native;

    // Obtain native variable values
    algoDesc_native = (cudnnAlgorithmDescriptor_t)getNativePointerValue(env, algoDesc);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnDestroyAlgorithmDescriptor(algoDesc_native);

    // Write back native variable values
    // algoDesc is read-only

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnCreateAlgorithmPerformanceNative(JNIEnv *env, jclass cls, jobjectArray algoPerf, jint numberToCreate)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnCreateAlgorithmPerformance(algoPerf=%p, numberToCreate=%d)\n",
        algoPerf, numberToCreate);

    // Native variable declarations
    cudnnAlgorithmPerformance_t algoPerf_native;
    int numberToCreate_native = 0;

    // Obtain native variable values
    // algoPerf is write-only
    numberToCreate_native = (int)numberToCreate;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnCreateAlgorithmPerformance(&algoPerf_native, numberToCreate_native);

    // Write back native variable values
    setNativePointerValue(env, algoPerf, (jlong)algoPerf_native);
    // numberToCreate is primitive

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnSetAlgorithmPerformanceNative(JNIEnv *env, jclass cls, jobject algoPerf, jobject algoDesc, jint status, jfloat time, jlong memory)
{
    // Null-checks for non-primitive arguments
    if (algoPerf == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'algoPerf' is null for cudnnSetAlgorithmPerformance");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (algoDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'algoDesc' is null for cudnnSetAlgorithmPerformance");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    // status is primitive
    // time is primitive
    // memory is primitive

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnSetAlgorithmPerformance(algoPerf=%p, algoDesc=%p, status=%d, time=%f, memory=%ld)\n",
        algoPerf, algoDesc, status, time, memory);

    // Native variable declarations
    cudnnAlgorithmPerformance_t algoPerf_native;
    cudnnAlgorithmDescriptor_t algoDesc_native;
    cudnnStatus_t status_native = CUDNN_STATUS_SUCCESS;
    float time_native = 0.0f;
    size_t memory_native = 0;

    // Obtain native variable values
    algoPerf_native = (cudnnAlgorithmPerformance_t)getNativePointerValue(env, algoPerf);
    algoDesc_native = (cudnnAlgorithmDescriptor_t)getNativePointerValue(env, algoDesc);
    status_native = (cudnnStatus_t)status;
    time_native = (float)time;
    memory_native = (size_t)memory;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnSetAlgorithmPerformance(algoPerf_native, algoDesc_native, status_native, time_native, memory_native);

    // Write back native variable values
    // algoPerf is read-only
    // algoDesc is read-only
    // status is primitive
    // time is primitive
    // memory is primitive

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetAlgorithmPerformanceNative(JNIEnv *env, jclass cls, jobject algoPerf, jobject algoDesc, jintArray status, jfloatArray time, jlongArray memory)
{
    // Null-checks for non-primitive arguments
    if (algoPerf == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'algoPerf' is null for cudnnGetAlgorithmPerformance");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (algoDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'algoDesc' is null for cudnnGetAlgorithmPerformance");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (status == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'status' is null for cudnnGetAlgorithmPerformance");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (time == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'time' is null for cudnnGetAlgorithmPerformance");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (memory == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'memory' is null for cudnnGetAlgorithmPerformance");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetAlgorithmPerformance(algoPerf=%p, algoDesc=%p, status=%p, time=%p, memory=%p)\n",
        algoPerf, algoDesc, status, time, memory);

    // Native variable declarations
    cudnnAlgorithmPerformance_t algoPerf_native;
    cudnnAlgorithmDescriptor_t * algoDesc_native;
    cudnnStatus_t status_native;
    float time_native;
    size_t memory_native;

    // Obtain native variable values
    algoPerf_native = (cudnnAlgorithmPerformance_t)getNativePointerValue(env, algoPerf);
    algoDesc_native = (cudnnAlgorithmDescriptor_t *)getNativePointerValue(env, algoDesc);
    // status is write-only
    // time is write-only
    // memory is write-only

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetAlgorithmPerformance(algoPerf_native, algoDesc_native, &status_native, &time_native, &memory_native);

    // Write back native variable values
    // algoPerf is read-only
    // algoDesc is read-only
    if (!set(env, status, 0, (jint)status_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, time, 0, (jfloat)time_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, memory, 0, (jlong)memory_native)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnDestroyAlgorithmPerformanceNative(JNIEnv *env, jclass cls, jobjectArray algoPerf, jint numberToDestroy)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnDestroyAlgorithmPerformance(algoPerf=%p, numberToDestroy=%d)\n",
        algoPerf, numberToDestroy);

    // Native variable declarations
    cudnnAlgorithmPerformance_t * algoPerf_native;
    int numberToDestroy_native = 0;

    // Obtain native variable values
    algoPerf_native = (cudnnAlgorithmPerformance_t *)getNativePointerValue(env, algoPerf);
    numberToDestroy_native = (int)numberToDestroy;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnDestroyAlgorithmPerformance(algoPerf_native, numberToDestroy_native);

    // Write back native variable values
    // algoPerf is read-only
    // numberToDestroy is primitive

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetAlgorithmSpaceSizeNative(JNIEnv *env, jclass cls, jobject handle, jobject algoDesc, jlongArray algoSpaceSizeInBytes)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetAlgorithmSpaceSize(handle=%p, algoDesc=%p, algoSpaceSizeInBytes=%p)\n",
        handle, algoDesc, algoSpaceSizeInBytes);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnAlgorithmDescriptor_t algoDesc_native;
    size_t algoSpaceSizeInBytes_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    algoDesc_native = (cudnnAlgorithmDescriptor_t)getNativePointerValue(env, algoDesc);
    // algoSpaceSizeInBytes is write-only

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetAlgorithmSpaceSize(handle_native, algoDesc_native, &algoSpaceSizeInBytes_native);

    // Write back native variable values
    // handle is read-only
    // algoDesc is read-only
    if (!set(env, algoSpaceSizeInBytes, 0, (jlong)algoSpaceSizeInBytes_native)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnSaveAlgorithmNative(JNIEnv *env, jclass cls, jobject handle, jobject algoDesc, jobject algoSpace, jlong algoSpaceSizeInBytes)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnSaveAlgorithm(handle=%p, algoDesc=%p, algoSpace=%p, algoSpaceSizeInBytes=%ld)\n",
        handle, algoDesc, algoSpace, algoSpaceSizeInBytes);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnAlgorithmDescriptor_t algoDesc_native;
    void * algoSpace_native = NULL;
    size_t algoSpaceSizeInBytes_native = 0;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    algoDesc_native = (cudnnAlgorithmDescriptor_t)getNativePointerValue(env, algoDesc);
    algoSpace_native = (void *)getPointer(env, algoSpace);
    algoSpaceSizeInBytes_native = (size_t)algoSpaceSizeInBytes;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnSaveAlgorithm(handle_native, algoDesc_native, algoSpace_native, algoSpaceSizeInBytes_native);

    // Write back native variable values
    // handle is read-only
    // algoDesc is read-only
    // algoSpace is a native pointer
    // algoSpaceSizeInBytes is primitive

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnRestoreAlgorithmNative(JNIEnv *env, jclass cls, jobject handle, jobject algoSpace, jlong algoSpaceSizeInBytes, jobject algoDesc)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnRestoreAlgorithm(handle=%p, algoSpace=%p, algoSpaceSizeInBytes=%ld, algoDesc=%p)\n",
        handle, algoSpace, algoSpaceSizeInBytes, algoDesc);

    // Native variable declarations
    cudnnHandle_t handle_native;
    void * algoSpace_native = NULL;
    size_t algoSpaceSizeInBytes_native = 0;
    cudnnAlgorithmDescriptor_t algoDesc_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    algoSpace_native = (void *)getPointer(env, algoSpace);
    algoSpaceSizeInBytes_native = (size_t)algoSpaceSizeInBytes;
    algoDesc_native = (cudnnAlgorithmDescriptor_t)getNativePointerValue(env, algoDesc);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnRestoreAlgorithm(handle_native, algoSpace_native, algoSpaceSizeInBytes_native, algoDesc_native);

    // Write back native variable values
    // handle is read-only
    // algoSpace is a native pointer
    // algoSpaceSizeInBytes is primitive
    // algoDesc is read-only

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnSetCallbackNative(JNIEnv *env, jclass cls, jint mask, jobject udata, jobject fptr)
{
    // XXX Callbacks are not supported yet
    ThrowByName(env, "java/lang/UnsupportedOperationException", "This function is not supported yet");
    return JCUDNN_STATUS_INTERNAL_ERROR;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetCallbackNative(JNIEnv *env, jclass cls, jintArray mask, jobject udata, jobjectArray fptr)
{
    // XXX Callbacks are not supported yet
    ThrowByName(env, "java/lang/UnsupportedOperationException", "This function is not supported yet");
    return JCUDNN_STATUS_INTERNAL_ERROR;
}

/**
 * <pre>
 * Cross-library version checker..
 * This function is implemented differently in each sub-library. Each sublib
 * checks whether its own version matches that of its dependencies.
 * @return CUDNN_STATUS_SUCCESS if the version check passes,
 *          CUDNN_STATUS_VERSION_MISMATCH if the versions are inconsistent.
 * </pre>
 */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnOpsInferVersionCheckNative(JNIEnv *env, jclass cls)
{
    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnOpsInferVersionCheck()\n");

    // Native function call
    cudnnStatus_t jniResult_native = cudnnOpsInferVersionCheck();

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

/** Function to perform backward softmax */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnSoftmaxBackwardNative(JNIEnv *env, jclass cls, jobject handle, jint algo, jint mode, jobject alpha, jobject yDesc, jobject y, jobject dyDesc, jobject dy, jobject beta, jobject dxDesc, jobject dx)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnSoftmaxBackward(handle=%p, algo=%d, mode=%d, alpha=%p, yDesc=%p, y=%p, dyDesc=%p, dy=%p, beta=%p, dxDesc=%p, dx=%p)\n",
        handle, algo, mode, alpha, yDesc, y, dyDesc, dy, beta, dxDesc, dx);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnSoftmaxAlgorithm_t algo_native;
    cudnnSoftmaxMode_t mode_native;
    void * alpha_native = NULL;
    cudnnTensorDescriptor_t yDesc_native;
    void * y_native = NULL;
    cudnnTensorDescriptor_t dyDesc_native;
    void * dy_native = NULL;
    void * beta_native = NULL;
    cudnnTensorDescriptor_t dxDesc_native;
    void * dx_native = NULL;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    algo_native = (cudnnSoftmaxAlgorithm_t)algo;
    mode_native = (cudnnSoftmaxMode_t)mode;
    PointerData *alpha_pointerData = initPointerData(env, alpha);
    if (alpha_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    alpha_native = (void *)alpha_pointerData->getPointer(env);
    yDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, yDesc);
    y_native = (void *)getPointer(env, y);
    dyDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, dyDesc);
    dy_native = (void *)getPointer(env, dy);
    PointerData *beta_pointerData = initPointerData(env, beta);
    if (beta_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    beta_native = (void *)beta_pointerData->getPointer(env);
    dxDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, dxDesc);
    dx_native = (void *)getPointer(env, dx);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnSoftmaxBackward(handle_native, algo_native, mode_native, alpha_native, yDesc_native, y_native, dyDesc_native, dy_native, beta_native, dxDesc_native, dx_native);

    // Write back native variable values
    // handle is read-only
    // algo is primitive
    // mode is primitive
    if (!releasePointerData(env, alpha_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // yDesc is read-only
    // y is a native pointer
    // dyDesc is read-only
    // dy is a native pointer
    if (!releasePointerData(env, beta_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // dxDesc is read-only
    // dx is a native pointer

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

/** Function to perform backward pooling */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnPoolingBackwardNative(JNIEnv *env, jclass cls, jobject handle, jobject poolingDesc, jobject alpha, jobject yDesc, jobject y, jobject dyDesc, jobject dy, jobject xDesc, jobject x, jobject beta, jobject dxDesc, jobject dx)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnPoolingBackward(handle=%p, poolingDesc=%p, alpha=%p, yDesc=%p, y=%p, dyDesc=%p, dy=%p, xDesc=%p, x=%p, beta=%p, dxDesc=%p, dx=%p)\n",
        handle, poolingDesc, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnPoolingDescriptor_t poolingDesc_native;
    void * alpha_native = NULL;
    cudnnTensorDescriptor_t yDesc_native;
    void * y_native = NULL;
    cudnnTensorDescriptor_t dyDesc_native;
    void * dy_native = NULL;
    cudnnTensorDescriptor_t xDesc_native;
    void * x_native = NULL;
    void * beta_native = NULL;
    cudnnTensorDescriptor_t dxDesc_native;
    void * dx_native = NULL;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    poolingDesc_native = (cudnnPoolingDescriptor_t)getNativePointerValue(env, poolingDesc);
    PointerData *alpha_pointerData = initPointerData(env, alpha);
    if (alpha_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    alpha_native = (void *)alpha_pointerData->getPointer(env);
    yDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, yDesc);
    y_native = (void *)getPointer(env, y);
    dyDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, dyDesc);
    dy_native = (void *)getPointer(env, dy);
    xDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, xDesc);
    x_native = (void *)getPointer(env, x);
    PointerData *beta_pointerData = initPointerData(env, beta);
    if (beta_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    beta_native = (void *)beta_pointerData->getPointer(env);
    dxDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, dxDesc);
    dx_native = (void *)getPointer(env, dx);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnPoolingBackward(handle_native, poolingDesc_native, alpha_native, yDesc_native, y_native, dyDesc_native, dy_native, xDesc_native, x_native, beta_native, dxDesc_native, dx_native);

    // Write back native variable values
    // handle is read-only
    // poolingDesc is read-only
    if (!releasePointerData(env, alpha_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // yDesc is read-only
    // y is a native pointer
    // dyDesc is read-only
    // dy is a native pointer
    // xDesc is read-only
    // x is a native pointer
    if (!releasePointerData(env, beta_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // dxDesc is read-only
    // dx is a native pointer

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

/** Function to perform backward activation  */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnActivationBackwardNative(JNIEnv *env, jclass cls, jobject handle, jobject activationDesc, jobject alpha, jobject yDesc, jobject y, jobject dyDesc, jobject dy, jobject xDesc, jobject x, jobject beta, jobject dxDesc, jobject dx)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnActivationBackward(handle=%p, activationDesc=%p, alpha=%p, yDesc=%p, y=%p, dyDesc=%p, dy=%p, xDesc=%p, x=%p, beta=%p, dxDesc=%p, dx=%p)\n",
        handle, activationDesc, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnActivationDescriptor_t activationDesc_native;
    void * alpha_native = NULL;
    cudnnTensorDescriptor_t yDesc_native;
    void * y_native = NULL;
    cudnnTensorDescriptor_t dyDesc_native;
    void * dy_native = NULL;
    cudnnTensorDescriptor_t xDesc_native;
    void * x_native = NULL;
    void * beta_native = NULL;
    cudnnTensorDescriptor_t dxDesc_native;
    void * dx_native = NULL;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    activationDesc_native = (cudnnActivationDescriptor_t)getNativePointerValue(env, activationDesc);
    PointerData *alpha_pointerData = initPointerData(env, alpha);
    if (alpha_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    alpha_native = (void *)alpha_pointerData->getPointer(env);
    yDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, yDesc);
    y_native = (void *)getPointer(env, y);
    dyDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, dyDesc);
    dy_native = (void *)getPointer(env, dy);
    xDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, xDesc);
    x_native = (void *)getPointer(env, x);
    PointerData *beta_pointerData = initPointerData(env, beta);
    if (beta_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    beta_native = (void *)beta_pointerData->getPointer(env);
    dxDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, dxDesc);
    dx_native = (void *)getPointer(env, dx);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnActivationBackward(handle_native, activationDesc_native, alpha_native, yDesc_native, y_native, dyDesc_native, dy_native, xDesc_native, x_native, beta_native, dxDesc_native, dx_native);

    // Write back native variable values
    // handle is read-only
    // activationDesc is read-only
    if (!releasePointerData(env, alpha_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // yDesc is read-only
    // y is a native pointer
    // dyDesc is read-only
    // dy is a native pointer
    // xDesc is read-only
    // x is a native pointer
    if (!releasePointerData(env, beta_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // dxDesc is read-only
    // dx is a native pointer

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

/** LRN cross-channel backward computation. Double parameters cast to tensor data type */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnLRNCrossChannelBackwardNative(JNIEnv *env, jclass cls, jobject handle, jobject normDesc, jint lrnMode, jobject alpha, jobject yDesc, jobject y, jobject dyDesc, jobject dy, jobject xDesc, jobject x, jobject beta, jobject dxDesc, jobject dx)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnLRNCrossChannelBackward(handle=%p, normDesc=%p, lrnMode=%d, alpha=%p, yDesc=%p, y=%p, dyDesc=%p, dy=%p, xDesc=%p, x=%p, beta=%p, dxDesc=%p, dx=%p)\n",
        handle, normDesc, lrnMode, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnLRNDescriptor_t normDesc_native;
    cudnnLRNMode_t lrnMode_native;
    void * alpha_native = NULL;
    cudnnTensorDescriptor_t yDesc_native;
    void * y_native = NULL;
    cudnnTensorDescriptor_t dyDesc_native;
    void * dy_native = NULL;
    cudnnTensorDescriptor_t xDesc_native;
    void * x_native = NULL;
    void * beta_native = NULL;
    cudnnTensorDescriptor_t dxDesc_native;
    void * dx_native = NULL;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    normDesc_native = (cudnnLRNDescriptor_t)getNativePointerValue(env, normDesc);
    lrnMode_native = (cudnnLRNMode_t)lrnMode;
    PointerData *alpha_pointerData = initPointerData(env, alpha);
    if (alpha_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    alpha_native = (void *)alpha_pointerData->getPointer(env);
    yDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, yDesc);
    y_native = (void *)getPointer(env, y);
    dyDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, dyDesc);
    dy_native = (void *)getPointer(env, dy);
    xDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, xDesc);
    x_native = (void *)getPointer(env, x);
    PointerData *beta_pointerData = initPointerData(env, beta);
    if (beta_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    beta_native = (void *)beta_pointerData->getPointer(env);
    dxDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, dxDesc);
    dx_native = (void *)getPointer(env, dx);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnLRNCrossChannelBackward(handle_native, normDesc_native, lrnMode_native, alpha_native, yDesc_native, y_native, dyDesc_native, dy_native, xDesc_native, x_native, beta_native, dxDesc_native, dx_native);

    // Write back native variable values
    // handle is read-only
    // normDesc is read-only
    // lrnMode is primitive
    if (!releasePointerData(env, alpha_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // yDesc is read-only
    // y is a native pointer
    // dyDesc is read-only
    // dy is a native pointer
    // xDesc is read-only
    // x is a native pointer
    if (!releasePointerData(env, beta_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // dxDesc is read-only
    // dx is a native pointer

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnDivisiveNormalizationBackwardNative(JNIEnv *env, jclass cls, jobject handle, jobject normDesc, jint mode, jobject alpha, jobject xDesc, jobject x, jobject means, jobject dy, jobject temp, jobject temp2, jobject beta, jobject dXdMeansDesc, jobject dx, jobject dMeans)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnDivisiveNormalizationBackward(handle=%p, normDesc=%p, mode=%d, alpha=%p, xDesc=%p, x=%p, means=%p, dy=%p, temp=%p, temp2=%p, beta=%p, dXdMeansDesc=%p, dx=%p, dMeans=%p)\n",
        handle, normDesc, mode, alpha, xDesc, x, means, dy, temp, temp2, beta, dXdMeansDesc, dx, dMeans);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnLRNDescriptor_t normDesc_native;
    cudnnDivNormMode_t mode_native;
    void * alpha_native = NULL;
    cudnnTensorDescriptor_t xDesc_native;
    void * x_native = NULL;
    void * means_native = NULL;
    void * dy_native = NULL;
    void * temp_native = NULL;
    void * temp2_native = NULL;
    void * beta_native = NULL;
    cudnnTensorDescriptor_t dXdMeansDesc_native;
    void * dx_native = NULL;
    void * dMeans_native = NULL;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    normDesc_native = (cudnnLRNDescriptor_t)getNativePointerValue(env, normDesc);
    mode_native = (cudnnDivNormMode_t)mode;
    PointerData *alpha_pointerData = initPointerData(env, alpha);
    if (alpha_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    alpha_native = (void *)alpha_pointerData->getPointer(env);
    xDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, xDesc);
    x_native = (void *)getPointer(env, x);
    means_native = (void *)getPointer(env, means);
    dy_native = (void *)getPointer(env, dy);
    temp_native = (void *)getPointer(env, temp);
    temp2_native = (void *)getPointer(env, temp2);
    PointerData *beta_pointerData = initPointerData(env, beta);
    if (beta_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    beta_native = (void *)beta_pointerData->getPointer(env);
    dXdMeansDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, dXdMeansDesc);
    dx_native = (void *)getPointer(env, dx);
    dMeans_native = (void *)getPointer(env, dMeans);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnDivisiveNormalizationBackward(handle_native, normDesc_native, mode_native, alpha_native, xDesc_native, x_native, means_native, dy_native, temp_native, temp2_native, beta_native, dXdMeansDesc_native, dx_native, dMeans_native);

    // Write back native variable values
    // handle is read-only
    // normDesc is read-only
    // mode is primitive
    if (!releasePointerData(env, alpha_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // xDesc is read-only
    // x is a native pointer
    // means is a native pointer
    // dy is a native pointer
    // temp is a native pointer
    // temp2 is a native pointer
    if (!releasePointerData(env, beta_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // dXdMeansDesc is read-only
    // dx is a native pointer
    // dMeans is a native pointer

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetBatchNormalizationForwardTrainingExWorkspaceSizeNative(JNIEnv *env, jclass cls, jobject handle, jint mode, jint bnOps, jobject xDesc, jobject zDesc, jobject yDesc, jobject bnScaleBiasMeanVarDesc, jobject activationDesc, jlongArray sizeInBytes)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(handle=%p, mode=%d, bnOps=%d, xDesc=%p, zDesc=%p, yDesc=%p, bnScaleBiasMeanVarDesc=%p, activationDesc=%p, sizeInBytes=%p)\n",
        handle, mode, bnOps, xDesc, zDesc, yDesc, bnScaleBiasMeanVarDesc, activationDesc, sizeInBytes);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnBatchNormMode_t mode_native;
    cudnnBatchNormOps_t bnOps_native;
    cudnnTensorDescriptor_t xDesc_native;
    cudnnTensorDescriptor_t zDesc_native;
    cudnnTensorDescriptor_t yDesc_native;
    cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc_native;
    cudnnActivationDescriptor_t activationDesc_native;
    size_t sizeInBytes_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    mode_native = (cudnnBatchNormMode_t)mode;
    bnOps_native = (cudnnBatchNormOps_t)bnOps;
    xDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, xDesc);
    zDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, zDesc);
    yDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, yDesc);
    bnScaleBiasMeanVarDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, bnScaleBiasMeanVarDesc);
    activationDesc_native = (cudnnActivationDescriptor_t)getNativePointerValue(env, activationDesc);
    // sizeInBytes is write-only

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(handle_native, mode_native, bnOps_native, xDesc_native, zDesc_native, yDesc_native, bnScaleBiasMeanVarDesc_native, activationDesc_native, &sizeInBytes_native);

    // Write back native variable values
    // handle is read-only
    // mode is primitive
    // bnOps is primitive
    // xDesc is read-only
    // zDesc is read-only
    // yDesc is read-only
    // bnScaleBiasMeanVarDesc is read-only
    // activationDesc is read-only
    if (!set(env, sizeInBytes, 0, (jlong)sizeInBytes_native)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetBatchNormalizationBackwardExWorkspaceSizeNative(JNIEnv *env, jclass cls, jobject handle, jint mode, jint bnOps, jobject xDesc, jobject yDesc, jobject dyDesc, jobject dzDesc, jobject dxDesc, jobject dBnScaleBiasDesc, jobject activationDesc, jlongArray sizeInBytes)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetBatchNormalizationBackwardExWorkspaceSize(handle=%p, mode=%d, bnOps=%d, xDesc=%p, yDesc=%p, dyDesc=%p, dzDesc=%p, dxDesc=%p, dBnScaleBiasDesc=%p, activationDesc=%p, sizeInBytes=%p)\n",
        handle, mode, bnOps, xDesc, yDesc, dyDesc, dzDesc, dxDesc, dBnScaleBiasDesc, activationDesc, sizeInBytes);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnBatchNormMode_t mode_native;
    cudnnBatchNormOps_t bnOps_native;
    cudnnTensorDescriptor_t xDesc_native;
    cudnnTensorDescriptor_t yDesc_native;
    cudnnTensorDescriptor_t dyDesc_native;
    cudnnTensorDescriptor_t dzDesc_native;
    cudnnTensorDescriptor_t dxDesc_native;
    cudnnTensorDescriptor_t dBnScaleBiasDesc_native;
    cudnnActivationDescriptor_t activationDesc_native;
    size_t sizeInBytes_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    mode_native = (cudnnBatchNormMode_t)mode;
    bnOps_native = (cudnnBatchNormOps_t)bnOps;
    xDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, xDesc);
    yDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, yDesc);
    dyDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, dyDesc);
    dzDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, dzDesc);
    dxDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, dxDesc);
    dBnScaleBiasDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, dBnScaleBiasDesc);
    activationDesc_native = (cudnnActivationDescriptor_t)getNativePointerValue(env, activationDesc);
    // sizeInBytes is write-only

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetBatchNormalizationBackwardExWorkspaceSize(handle_native, mode_native, bnOps_native, xDesc_native, yDesc_native, dyDesc_native, dzDesc_native, dxDesc_native, dBnScaleBiasDesc_native, activationDesc_native, &sizeInBytes_native);

    // Write back native variable values
    // handle is read-only
    // mode is primitive
    // bnOps is primitive
    // xDesc is read-only
    // yDesc is read-only
    // dyDesc is read-only
    // dzDesc is read-only
    // dxDesc is read-only
    // dBnScaleBiasDesc is read-only
    // activationDesc is read-only
    if (!set(env, sizeInBytes, 0, (jlong)sizeInBytes_native)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetBatchNormalizationTrainingExReserveSpaceSizeNative(JNIEnv *env, jclass cls, jobject handle, jint mode, jint bnOps, jobject activationDesc, jobject xDesc, jlongArray sizeInBytes)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetBatchNormalizationTrainingExReserveSpaceSize(handle=%p, mode=%d, bnOps=%d, activationDesc=%p, xDesc=%p, sizeInBytes=%p)\n",
        handle, mode, bnOps, activationDesc, xDesc, sizeInBytes);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnBatchNormMode_t mode_native;
    cudnnBatchNormOps_t bnOps_native;
    cudnnActivationDescriptor_t activationDesc_native;
    cudnnTensorDescriptor_t xDesc_native;
    size_t sizeInBytes_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    mode_native = (cudnnBatchNormMode_t)mode;
    bnOps_native = (cudnnBatchNormOps_t)bnOps;
    activationDesc_native = (cudnnActivationDescriptor_t)getNativePointerValue(env, activationDesc);
    xDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, xDesc);
    // sizeInBytes is write-only

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetBatchNormalizationTrainingExReserveSpaceSize(handle_native, mode_native, bnOps_native, activationDesc_native, xDesc_native, &sizeInBytes_native);

    // Write back native variable values
    // handle is read-only
    // mode is primitive
    // bnOps is primitive
    // activationDesc is read-only
    // xDesc is read-only
    if (!set(env, sizeInBytes, 0, (jlong)sizeInBytes_native)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

/** Computes y = BN(x). Also accumulates moving averages of mean and inverse variances */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnBatchNormalizationForwardTrainingNative(JNIEnv *env, jclass cls, jobject handle, jint mode, jobject alpha, jobject beta, jobject xDesc, jobject x, jobject yDesc, jobject y, jobject bnScaleBiasMeanVarDesc, jobject bnScale, jobject bnBias, jdouble exponentialAverageFactor, jobject resultRunningMean, jobject resultRunningVariance, jdouble epsilon, jobject resultSaveMean, jobject resultSaveInvVariance)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnBatchNormalizationForwardTraining(handle=%p, mode=%d, alpha=%p, beta=%p, xDesc=%p, x=%p, yDesc=%p, y=%p, bnScaleBiasMeanVarDesc=%p, bnScale=%p, bnBias=%p, exponentialAverageFactor=%lf, resultRunningMean=%p, resultRunningVariance=%p, epsilon=%lf, resultSaveMean=%p, resultSaveInvVariance=%p)\n",
        handle, mode, alpha, beta, xDesc, x, yDesc, y, bnScaleBiasMeanVarDesc, bnScale, bnBias, exponentialAverageFactor, resultRunningMean, resultRunningVariance, epsilon, resultSaveMean, resultSaveInvVariance);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnBatchNormMode_t mode_native;
    void * alpha_native = NULL;
    void * beta_native = NULL;
    cudnnTensorDescriptor_t xDesc_native;
    void * x_native = NULL;
    cudnnTensorDescriptor_t yDesc_native;
    void * y_native = NULL;
    cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc_native;
    void * bnScale_native = NULL;
    void * bnBias_native = NULL;
    double exponentialAverageFactor_native = 0.0;
    void * resultRunningMean_native = NULL;
    void * resultRunningVariance_native = NULL;
    double epsilon_native = 0.0;
    void * resultSaveMean_native = NULL;
    void * resultSaveInvVariance_native = NULL;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    mode_native = (cudnnBatchNormMode_t)mode;
    PointerData *alpha_pointerData = initPointerData(env, alpha);
    if (alpha_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    alpha_native = (void *)alpha_pointerData->getPointer(env);
    PointerData *beta_pointerData = initPointerData(env, beta);
    if (beta_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    beta_native = (void *)beta_pointerData->getPointer(env);
    xDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, xDesc);
    x_native = (void *)getPointer(env, x);
    yDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, yDesc);
    y_native = (void *)getPointer(env, y);
    bnScaleBiasMeanVarDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, bnScaleBiasMeanVarDesc);
    bnScale_native = (void *)getPointer(env, bnScale);
    bnBias_native = (void *)getPointer(env, bnBias);
    exponentialAverageFactor_native = (double)exponentialAverageFactor;
    resultRunningMean_native = (void *)getPointer(env, resultRunningMean);
    resultRunningVariance_native = (void *)getPointer(env, resultRunningVariance);
    epsilon_native = (double)epsilon;
    resultSaveMean_native = (void *)getPointer(env, resultSaveMean);
    resultSaveInvVariance_native = (void *)getPointer(env, resultSaveInvVariance);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnBatchNormalizationForwardTraining(handle_native, mode_native, alpha_native, beta_native, xDesc_native, x_native, yDesc_native, y_native, bnScaleBiasMeanVarDesc_native, bnScale_native, bnBias_native, exponentialAverageFactor_native, resultRunningMean_native, resultRunningVariance_native, epsilon_native, resultSaveMean_native, resultSaveInvVariance_native);

    // Write back native variable values
    // handle is read-only
    // mode is primitive
    if (!releasePointerData(env, alpha_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!releasePointerData(env, beta_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // xDesc is read-only
    // x is a native pointer
    // yDesc is read-only
    // y is a native pointer
    // bnScaleBiasMeanVarDesc is read-only
    // bnScale is a native pointer
    // bnBias is a native pointer
    // exponentialAverageFactor is primitive
    // resultRunningMean is a native pointer
    // resultRunningVariance is a native pointer
    // epsilon is primitive
    // resultSaveMean is a native pointer
    // resultSaveInvVariance is a native pointer

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

/** Computes y = relu(BN(x) + z). Also accumulates moving averages of mean and inverse variances */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnBatchNormalizationForwardTrainingExNative(JNIEnv *env, jclass cls, jobject handle, jint mode, jint bnOps, jobject alpha, jobject beta, jobject xDesc, jobject xData, jobject zDesc, jobject zData, jobject yDesc, jobject yData, jobject bnScaleBiasMeanVarDesc, jobject bnScale, jobject bnBias, jdouble exponentialAverageFactor, jobject resultRunningMean, jobject resultRunningVariance, jdouble epsilon, jobject resultSaveMean, jobject resultSaveInvVariance, jobject activationDesc, jobject workspace, jlong workSpaceSizeInBytes, jobject reserveSpace, jlong reserveSpaceSizeInBytes)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnBatchNormalizationForwardTrainingEx(handle=%p, mode=%d, bnOps=%d, alpha=%p, beta=%p, xDesc=%p, xData=%p, zDesc=%p, zData=%p, yDesc=%p, yData=%p, bnScaleBiasMeanVarDesc=%p, bnScale=%p, bnBias=%p, exponentialAverageFactor=%lf, resultRunningMean=%p, resultRunningVariance=%p, epsilon=%lf, resultSaveMean=%p, resultSaveInvVariance=%p, activationDesc=%p, workspace=%p, workSpaceSizeInBytes=%ld, reserveSpace=%p, reserveSpaceSizeInBytes=%ld)\n",
        handle, mode, bnOps, alpha, beta, xDesc, xData, zDesc, zData, yDesc, yData, bnScaleBiasMeanVarDesc, bnScale, bnBias, exponentialAverageFactor, resultRunningMean, resultRunningVariance, epsilon, resultSaveMean, resultSaveInvVariance, activationDesc, workspace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnBatchNormMode_t mode_native;
    cudnnBatchNormOps_t bnOps_native;
    void * alpha_native = NULL;
    void * beta_native = NULL;
    cudnnTensorDescriptor_t xDesc_native;
    void * xData_native = NULL;
    cudnnTensorDescriptor_t zDesc_native;
    void * zData_native = NULL;
    cudnnTensorDescriptor_t yDesc_native;
    void * yData_native = NULL;
    cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc_native;
    void * bnScale_native = NULL;
    void * bnBias_native = NULL;
    double exponentialAverageFactor_native = 0.0;
    void * resultRunningMean_native = NULL;
    void * resultRunningVariance_native = NULL;
    double epsilon_native = 0.0;
    void * resultSaveMean_native = NULL;
    void * resultSaveInvVariance_native = NULL;
    cudnnActivationDescriptor_t activationDesc_native;
    void * workspace_native = NULL;
    size_t workSpaceSizeInBytes_native = 0;
    void * reserveSpace_native = NULL;
    size_t reserveSpaceSizeInBytes_native = 0;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    mode_native = (cudnnBatchNormMode_t)mode;
    bnOps_native = (cudnnBatchNormOps_t)bnOps;
    PointerData *alpha_pointerData = initPointerData(env, alpha);
    if (alpha_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    alpha_native = (void *)alpha_pointerData->getPointer(env);
    PointerData *beta_pointerData = initPointerData(env, beta);
    if (beta_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    beta_native = (void *)beta_pointerData->getPointer(env);
    xDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, xDesc);
    xData_native = (void *)getPointer(env, xData);
    zDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, zDesc);
    zData_native = (void *)getPointer(env, zData);
    yDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, yDesc);
    yData_native = (void *)getPointer(env, yData);
    bnScaleBiasMeanVarDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, bnScaleBiasMeanVarDesc);
    bnScale_native = (void *)getPointer(env, bnScale);
    bnBias_native = (void *)getPointer(env, bnBias);
    exponentialAverageFactor_native = (double)exponentialAverageFactor;
    resultRunningMean_native = (void *)getPointer(env, resultRunningMean);
    resultRunningVariance_native = (void *)getPointer(env, resultRunningVariance);
    epsilon_native = (double)epsilon;
    resultSaveMean_native = (void *)getPointer(env, resultSaveMean);
    resultSaveInvVariance_native = (void *)getPointer(env, resultSaveInvVariance);
    activationDesc_native = (cudnnActivationDescriptor_t)getNativePointerValue(env, activationDesc);
    workspace_native = (void *)getPointer(env, workspace);
    workSpaceSizeInBytes_native = (size_t)workSpaceSizeInBytes;
    reserveSpace_native = (void *)getPointer(env, reserveSpace);
    reserveSpaceSizeInBytes_native = (size_t)reserveSpaceSizeInBytes;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnBatchNormalizationForwardTrainingEx(handle_native, mode_native, bnOps_native, alpha_native, beta_native, xDesc_native, xData_native, zDesc_native, zData_native, yDesc_native, yData_native, bnScaleBiasMeanVarDesc_native, bnScale_native, bnBias_native, exponentialAverageFactor_native, resultRunningMean_native, resultRunningVariance_native, epsilon_native, resultSaveMean_native, resultSaveInvVariance_native, activationDesc_native, workspace_native, workSpaceSizeInBytes_native, reserveSpace_native, reserveSpaceSizeInBytes_native);

    // Write back native variable values
    // handle is read-only
    // mode is primitive
    // bnOps is primitive
    if (!releasePointerData(env, alpha_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!releasePointerData(env, beta_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // xDesc is read-only
    // xData is a native pointer
    // zDesc is read-only
    // zData is a native pointer
    // yDesc is read-only
    // yData is a native pointer
    // bnScaleBiasMeanVarDesc is read-only
    // bnScale is a native pointer
    // bnBias is a native pointer
    // exponentialAverageFactor is primitive
    // resultRunningMean is a native pointer
    // resultRunningVariance is a native pointer
    // epsilon is primitive
    // resultSaveMean is a native pointer
    // resultSaveInvVariance is a native pointer
    // activationDesc is read-only
    // workspace is a native pointer
    // workSpaceSizeInBytes is primitive
    // reserveSpace is a native pointer
    // reserveSpaceSizeInBytes is primitive

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

/** Performs backward pass of Batch Normalization layer. Returns x gradient,
* bnScale gradient and bnBias gradient */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnBatchNormalizationBackwardNative(JNIEnv *env, jclass cls, jobject handle, jint mode, jobject alphaDataDiff, jobject betaDataDiff, jobject alphaParamDiff, jobject betaParamDiff, jobject xDesc, jobject x, jobject dyDesc, jobject dy, jobject dxDesc, jobject dx, jobject dBnScaleBiasDesc, jobject bnScale, jobject dBnScaleResult, jobject dBnBiasResult, jdouble epsilon, jobject savedMean, jobject savedInvVariance)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnBatchNormalizationBackward(handle=%p, mode=%d, alphaDataDiff=%p, betaDataDiff=%p, alphaParamDiff=%p, betaParamDiff=%p, xDesc=%p, x=%p, dyDesc=%p, dy=%p, dxDesc=%p, dx=%p, dBnScaleBiasDesc=%p, bnScale=%p, dBnScaleResult=%p, dBnBiasResult=%p, epsilon=%lf, savedMean=%p, savedInvVariance=%p)\n",
        handle, mode, alphaDataDiff, betaDataDiff, alphaParamDiff, betaParamDiff, xDesc, x, dyDesc, dy, dxDesc, dx, dBnScaleBiasDesc, bnScale, dBnScaleResult, dBnBiasResult, epsilon, savedMean, savedInvVariance);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnBatchNormMode_t mode_native;
    void * alphaDataDiff_native = NULL;
    void * betaDataDiff_native = NULL;
    void * alphaParamDiff_native = NULL;
    void * betaParamDiff_native = NULL;
    cudnnTensorDescriptor_t xDesc_native;
    void * x_native = NULL;
    cudnnTensorDescriptor_t dyDesc_native;
    void * dy_native = NULL;
    cudnnTensorDescriptor_t dxDesc_native;
    void * dx_native = NULL;
    cudnnTensorDescriptor_t dBnScaleBiasDesc_native;
    void * bnScale_native = NULL;
    void * dBnScaleResult_native = NULL;
    void * dBnBiasResult_native = NULL;
    double epsilon_native = 0.0;
    void * savedMean_native = NULL;
    void * savedInvVariance_native = NULL;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    mode_native = (cudnnBatchNormMode_t)mode;
    PointerData *alphaDataDiff_pointerData = initPointerData(env, alphaDataDiff);
    if (alphaDataDiff_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    alphaDataDiff_native = (void *)alphaDataDiff_pointerData->getPointer(env);
    PointerData *betaDataDiff_pointerData = initPointerData(env, betaDataDiff);
    if (betaDataDiff_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    betaDataDiff_native = (void *)betaDataDiff_pointerData->getPointer(env);
    PointerData *alphaParamDiff_pointerData = initPointerData(env, alphaParamDiff);
    if (alphaParamDiff_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    alphaParamDiff_native = (void *)alphaParamDiff_pointerData->getPointer(env);
    PointerData *betaParamDiff_pointerData = initPointerData(env, betaParamDiff);
    if (betaParamDiff_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    betaParamDiff_native = (void *)betaParamDiff_pointerData->getPointer(env);
    xDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, xDesc);
    x_native = (void *)getPointer(env, x);
    dyDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, dyDesc);
    dy_native = (void *)getPointer(env, dy);
    dxDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, dxDesc);
    dx_native = (void *)getPointer(env, dx);
    dBnScaleBiasDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, dBnScaleBiasDesc);
    bnScale_native = (void *)getPointer(env, bnScale);
    dBnScaleResult_native = (void *)getPointer(env, dBnScaleResult);
    dBnBiasResult_native = (void *)getPointer(env, dBnBiasResult);
    epsilon_native = (double)epsilon;
    savedMean_native = (void *)getPointer(env, savedMean);
    savedInvVariance_native = (void *)getPointer(env, savedInvVariance);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnBatchNormalizationBackward(handle_native, mode_native, alphaDataDiff_native, betaDataDiff_native, alphaParamDiff_native, betaParamDiff_native, xDesc_native, x_native, dyDesc_native, dy_native, dxDesc_native, dx_native, dBnScaleBiasDesc_native, bnScale_native, dBnScaleResult_native, dBnBiasResult_native, epsilon_native, savedMean_native, savedInvVariance_native);

    // Write back native variable values
    // handle is read-only
    // mode is primitive
    if (!releasePointerData(env, alphaDataDiff_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!releasePointerData(env, betaDataDiff_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!releasePointerData(env, alphaParamDiff_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!releasePointerData(env, betaParamDiff_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // xDesc is read-only
    // x is a native pointer
    // dyDesc is read-only
    // dy is a native pointer
    // dxDesc is read-only
    // dx is a native pointer
    // dBnScaleBiasDesc is read-only
    // bnScale is a native pointer
    // dBnScaleResult is a native pointer
    // dBnBiasResult is a native pointer
    // epsilon is primitive
    // savedMean is a native pointer
    // savedInvVariance is a native pointer

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnBatchNormalizationBackwardExNative(JNIEnv *env, jclass cls, jobject handle, jint mode, jint bnOps, jobject alphaDataDiff, jobject betaDataDiff, jobject alphaParamDiff, jobject betaParamDiff, jobject xDesc, jobject xData, jobject yDesc, jobject yData, jobject dyDesc, jobject dyData, jobject dzDesc, jobject dzData, jobject dxDesc, jobject dxData, jobject dBnScaleBiasDesc, jobject bnScaleData, jobject bnBiasData, jobject dBnScaleData, jobject dBnBiasData, jdouble epsilon, jobject savedMean, jobject savedInvVariance, jobject activationDesc, jobject workSpace, jlong workSpaceSizeInBytes, jobject reserveSpace, jlong reserveSpaceSizeInBytes)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnBatchNormalizationBackwardEx(handle=%p, mode=%d, bnOps=%d, alphaDataDiff=%p, betaDataDiff=%p, alphaParamDiff=%p, betaParamDiff=%p, xDesc=%p, xData=%p, yDesc=%p, yData=%p, dyDesc=%p, dyData=%p, dzDesc=%p, dzData=%p, dxDesc=%p, dxData=%p, dBnScaleBiasDesc=%p, bnScaleData=%p, bnBiasData=%p, dBnScaleData=%p, dBnBiasData=%p, epsilon=%lf, savedMean=%p, savedInvVariance=%p, activationDesc=%p, workSpace=%p, workSpaceSizeInBytes=%ld, reserveSpace=%p, reserveSpaceSizeInBytes=%ld)\n",
        handle, mode, bnOps, alphaDataDiff, betaDataDiff, alphaParamDiff, betaParamDiff, xDesc, xData, yDesc, yData, dyDesc, dyData, dzDesc, dzData, dxDesc, dxData, dBnScaleBiasDesc, bnScaleData, bnBiasData, dBnScaleData, dBnBiasData, epsilon, savedMean, savedInvVariance, activationDesc, workSpace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnBatchNormMode_t mode_native;
    cudnnBatchNormOps_t bnOps_native;
    void * alphaDataDiff_native = NULL;
    void * betaDataDiff_native = NULL;
    void * alphaParamDiff_native = NULL;
    void * betaParamDiff_native = NULL;
    cudnnTensorDescriptor_t xDesc_native;
    void * xData_native = NULL;
    cudnnTensorDescriptor_t yDesc_native;
    void * yData_native = NULL;
    cudnnTensorDescriptor_t dyDesc_native;
    void * dyData_native = NULL;
    cudnnTensorDescriptor_t dzDesc_native;
    void * dzData_native = NULL;
    cudnnTensorDescriptor_t dxDesc_native;
    void * dxData_native = NULL;
    cudnnTensorDescriptor_t dBnScaleBiasDesc_native;
    void * bnScaleData_native = NULL;
    void * bnBiasData_native = NULL;
    void * dBnScaleData_native = NULL;
    void * dBnBiasData_native = NULL;
    double epsilon_native = 0.0;
    void * savedMean_native = NULL;
    void * savedInvVariance_native = NULL;
    cudnnActivationDescriptor_t activationDesc_native;
    void * workSpace_native = NULL;
    size_t workSpaceSizeInBytes_native = 0;
    void * reserveSpace_native = NULL;
    size_t reserveSpaceSizeInBytes_native = 0;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    mode_native = (cudnnBatchNormMode_t)mode;
    bnOps_native = (cudnnBatchNormOps_t)bnOps;
    PointerData *alphaDataDiff_pointerData = initPointerData(env, alphaDataDiff);
    if (alphaDataDiff_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    alphaDataDiff_native = (void *)alphaDataDiff_pointerData->getPointer(env);
    PointerData *betaDataDiff_pointerData = initPointerData(env, betaDataDiff);
    if (betaDataDiff_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    betaDataDiff_native = (void *)betaDataDiff_pointerData->getPointer(env);
    PointerData *alphaParamDiff_pointerData = initPointerData(env, alphaParamDiff);
    if (alphaParamDiff_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    alphaParamDiff_native = (void *)alphaParamDiff_pointerData->getPointer(env);
    PointerData *betaParamDiff_pointerData = initPointerData(env, betaParamDiff);
    if (betaParamDiff_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    betaParamDiff_native = (void *)betaParamDiff_pointerData->getPointer(env);
    xDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, xDesc);
    xData_native = (void *)getPointer(env, xData);
    yDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, yDesc);
    yData_native = (void *)getPointer(env, yData);
    dyDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, dyDesc);
    dyData_native = (void *)getPointer(env, dyData);
    dzDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, dzDesc);
    dzData_native = (void *)getPointer(env, dzData);
    dxDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, dxDesc);
    dxData_native = (void *)getPointer(env, dxData);
    dBnScaleBiasDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, dBnScaleBiasDesc);
    bnScaleData_native = (void *)getPointer(env, bnScaleData);
    bnBiasData_native = (void *)getPointer(env, bnBiasData);
    dBnScaleData_native = (void *)getPointer(env, dBnScaleData);
    dBnBiasData_native = (void *)getPointer(env, dBnBiasData);
    epsilon_native = (double)epsilon;
    savedMean_native = (void *)getPointer(env, savedMean);
    savedInvVariance_native = (void *)getPointer(env, savedInvVariance);
    activationDesc_native = (cudnnActivationDescriptor_t)getNativePointerValue(env, activationDesc);
    workSpace_native = (void *)getPointer(env, workSpace);
    workSpaceSizeInBytes_native = (size_t)workSpaceSizeInBytes;
    reserveSpace_native = (void *)getPointer(env, reserveSpace);
    reserveSpaceSizeInBytes_native = (size_t)reserveSpaceSizeInBytes;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnBatchNormalizationBackwardEx(handle_native, mode_native, bnOps_native, alphaDataDiff_native, betaDataDiff_native, alphaParamDiff_native, betaParamDiff_native, xDesc_native, xData_native, yDesc_native, yData_native, dyDesc_native, dyData_native, dzDesc_native, dzData_native, dxDesc_native, dxData_native, dBnScaleBiasDesc_native, bnScaleData_native, bnBiasData_native, dBnScaleData_native, dBnBiasData_native, epsilon_native, savedMean_native, savedInvVariance_native, activationDesc_native, workSpace_native, workSpaceSizeInBytes_native, reserveSpace_native, reserveSpaceSizeInBytes_native);

    // Write back native variable values
    // handle is read-only
    // mode is primitive
    // bnOps is primitive
    if (!releasePointerData(env, alphaDataDiff_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!releasePointerData(env, betaDataDiff_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!releasePointerData(env, alphaParamDiff_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!releasePointerData(env, betaParamDiff_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // xDesc is read-only
    // xData is a native pointer
    // yDesc is read-only
    // yData is a native pointer
    // dyDesc is read-only
    // dyData is a native pointer
    // dzDesc is read-only
    // dzData is a native pointer
    // dxDesc is read-only
    // dxData is a native pointer
    // dBnScaleBiasDesc is read-only
    // bnScaleData is a native pointer
    // bnBiasData is a native pointer
    // dBnScaleData is a native pointer
    // dBnBiasData is a native pointer
    // epsilon is primitive
    // savedMean is a native pointer
    // savedInvVariance is a native pointer
    // activationDesc is read-only
    // workSpace is a native pointer
    // workSpaceSizeInBytes is primitive
    // reserveSpace is a native pointer
    // reserveSpaceSizeInBytes is primitive

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetNormalizationForwardTrainingWorkspaceSizeNative(JNIEnv *env, jclass cls, jobject handle, jint mode, jint normOps, jint algo, jobject xDesc, jobject zDesc, jobject yDesc, jobject normScaleBiasDesc, jobject activationDesc, jobject normMeanVarDesc, jlongArray sizeInBytes, jint groupCnt)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetNormalizationForwardTrainingWorkspaceSize(handle=%p, mode=%d, normOps=%d, algo=%d, xDesc=%p, zDesc=%p, yDesc=%p, normScaleBiasDesc=%p, activationDesc=%p, normMeanVarDesc=%p, sizeInBytes=%p, groupCnt=%d)\n",
        handle, mode, normOps, algo, xDesc, zDesc, yDesc, normScaleBiasDesc, activationDesc, normMeanVarDesc, sizeInBytes, groupCnt);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnNormMode_t mode_native;
    cudnnNormOps_t normOps_native;
    cudnnNormAlgo_t algo_native;
    cudnnTensorDescriptor_t xDesc_native;
    cudnnTensorDescriptor_t zDesc_native;
    cudnnTensorDescriptor_t yDesc_native;
    cudnnTensorDescriptor_t normScaleBiasDesc_native;
    cudnnActivationDescriptor_t activationDesc_native;
    cudnnTensorDescriptor_t normMeanVarDesc_native;
    size_t sizeInBytes_native;
    int groupCnt_native = 0;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    mode_native = (cudnnNormMode_t)mode;
    normOps_native = (cudnnNormOps_t)normOps;
    algo_native = (cudnnNormAlgo_t)algo;
    xDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, xDesc);
    zDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, zDesc);
    yDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, yDesc);
    normScaleBiasDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, normScaleBiasDesc);
    activationDesc_native = (cudnnActivationDescriptor_t)getNativePointerValue(env, activationDesc);
    normMeanVarDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, normMeanVarDesc);
    // sizeInBytes is write-only
    groupCnt_native = (int)groupCnt;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetNormalizationForwardTrainingWorkspaceSize(handle_native, mode_native, normOps_native, algo_native, xDesc_native, zDesc_native, yDesc_native, normScaleBiasDesc_native, activationDesc_native, normMeanVarDesc_native, &sizeInBytes_native, groupCnt_native);

    // Write back native variable values
    // handle is read-only
    // mode is primitive
    // normOps is primitive
    // algo is primitive
    // xDesc is read-only
    // zDesc is read-only
    // yDesc is read-only
    // normScaleBiasDesc is read-only
    // activationDesc is read-only
    // normMeanVarDesc is read-only
    if (!set(env, sizeInBytes, 0, (jlong)sizeInBytes_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // groupCnt is primitive

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetNormalizationBackwardWorkspaceSizeNative(JNIEnv *env, jclass cls, jobject handle, jint mode, jint normOps, jint algo, jobject xDesc, jobject yDesc, jobject dyDesc, jobject dzDesc, jobject dxDesc, jobject dNormScaleBiasDesc, jobject activationDesc, jobject normMeanVarDesc, jlongArray sizeInBytes, jint groupCnt)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetNormalizationBackwardWorkspaceSize(handle=%p, mode=%d, normOps=%d, algo=%d, xDesc=%p, yDesc=%p, dyDesc=%p, dzDesc=%p, dxDesc=%p, dNormScaleBiasDesc=%p, activationDesc=%p, normMeanVarDesc=%p, sizeInBytes=%p, groupCnt=%d)\n",
        handle, mode, normOps, algo, xDesc, yDesc, dyDesc, dzDesc, dxDesc, dNormScaleBiasDesc, activationDesc, normMeanVarDesc, sizeInBytes, groupCnt);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnNormMode_t mode_native;
    cudnnNormOps_t normOps_native;
    cudnnNormAlgo_t algo_native;
    cudnnTensorDescriptor_t xDesc_native;
    cudnnTensorDescriptor_t yDesc_native;
    cudnnTensorDescriptor_t dyDesc_native;
    cudnnTensorDescriptor_t dzDesc_native;
    cudnnTensorDescriptor_t dxDesc_native;
    cudnnTensorDescriptor_t dNormScaleBiasDesc_native;
    cudnnActivationDescriptor_t activationDesc_native;
    cudnnTensorDescriptor_t normMeanVarDesc_native;
    size_t sizeInBytes_native;
    int groupCnt_native = 0;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    mode_native = (cudnnNormMode_t)mode;
    normOps_native = (cudnnNormOps_t)normOps;
    algo_native = (cudnnNormAlgo_t)algo;
    xDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, xDesc);
    yDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, yDesc);
    dyDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, dyDesc);
    dzDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, dzDesc);
    dxDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, dxDesc);
    dNormScaleBiasDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, dNormScaleBiasDesc);
    activationDesc_native = (cudnnActivationDescriptor_t)getNativePointerValue(env, activationDesc);
    normMeanVarDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, normMeanVarDesc);
    // sizeInBytes is write-only
    groupCnt_native = (int)groupCnt;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetNormalizationBackwardWorkspaceSize(handle_native, mode_native, normOps_native, algo_native, xDesc_native, yDesc_native, dyDesc_native, dzDesc_native, dxDesc_native, dNormScaleBiasDesc_native, activationDesc_native, normMeanVarDesc_native, &sizeInBytes_native, groupCnt_native);

    // Write back native variable values
    // handle is read-only
    // mode is primitive
    // normOps is primitive
    // algo is primitive
    // xDesc is read-only
    // yDesc is read-only
    // dyDesc is read-only
    // dzDesc is read-only
    // dxDesc is read-only
    // dNormScaleBiasDesc is read-only
    // activationDesc is read-only
    // normMeanVarDesc is read-only
    if (!set(env, sizeInBytes, 0, (jlong)sizeInBytes_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // groupCnt is primitive

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetNormalizationTrainingReserveSpaceSizeNative(JNIEnv *env, jclass cls, jobject handle, jint mode, jint normOps, jint algo, jobject activationDesc, jobject xDesc, jlongArray sizeInBytes, jint groupCnt)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetNormalizationTrainingReserveSpaceSize(handle=%p, mode=%d, normOps=%d, algo=%d, activationDesc=%p, xDesc=%p, sizeInBytes=%p, groupCnt=%d)\n",
        handle, mode, normOps, algo, activationDesc, xDesc, sizeInBytes, groupCnt);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnNormMode_t mode_native;
    cudnnNormOps_t normOps_native;
    cudnnNormAlgo_t algo_native;
    cudnnActivationDescriptor_t activationDesc_native;
    cudnnTensorDescriptor_t xDesc_native;
    size_t sizeInBytes_native;
    int groupCnt_native = 0;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    mode_native = (cudnnNormMode_t)mode;
    normOps_native = (cudnnNormOps_t)normOps;
    algo_native = (cudnnNormAlgo_t)algo;
    activationDesc_native = (cudnnActivationDescriptor_t)getNativePointerValue(env, activationDesc);
    xDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, xDesc);
    // sizeInBytes is write-only
    groupCnt_native = (int)groupCnt;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetNormalizationTrainingReserveSpaceSize(handle_native, mode_native, normOps_native, algo_native, activationDesc_native, xDesc_native, &sizeInBytes_native, groupCnt_native);

    // Write back native variable values
    // handle is read-only
    // mode is primitive
    // normOps is primitive
    // algo is primitive
    // activationDesc is read-only
    // xDesc is read-only
    if (!set(env, sizeInBytes, 0, (jlong)sizeInBytes_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // groupCnt is primitive

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

/** Computes y = relu(Norm(x) + z). Also accumulates moving averages of mean and inverse variances */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnNormalizationForwardTrainingNative(JNIEnv *env, jclass cls, jobject handle, jint mode, jint normOps, jint algo, jobject alpha, jobject beta, jobject xDesc, jobject xData, jobject normScaleBiasDesc, jobject normScale, jobject normBias, jdouble exponentialAverageFactor, jobject normMeanVarDesc, jobject resultRunningMean, jobject resultRunningVariance, jdouble epsilon, jobject resultSaveMean, jobject resultSaveInvVariance, jobject activationDesc, jobject zDesc, jobject zData, jobject yDesc, jobject yData, jobject workspace, jlong workSpaceSizeInBytes, jobject reserveSpace, jlong reserveSpaceSizeInBytes, jint groupCnt)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnNormalizationForwardTraining(handle=%p, mode=%d, normOps=%d, algo=%d, alpha=%p, beta=%p, xDesc=%p, xData=%p, normScaleBiasDesc=%p, normScale=%p, normBias=%p, exponentialAverageFactor=%lf, normMeanVarDesc=%p, resultRunningMean=%p, resultRunningVariance=%p, epsilon=%lf, resultSaveMean=%p, resultSaveInvVariance=%p, activationDesc=%p, zDesc=%p, zData=%p, yDesc=%p, yData=%p, workspace=%p, workSpaceSizeInBytes=%ld, reserveSpace=%p, reserveSpaceSizeInBytes=%ld, groupCnt=%d)\n",
        handle, mode, normOps, algo, alpha, beta, xDesc, xData, normScaleBiasDesc, normScale, normBias, exponentialAverageFactor, normMeanVarDesc, resultRunningMean, resultRunningVariance, epsilon, resultSaveMean, resultSaveInvVariance, activationDesc, zDesc, zData, yDesc, yData, workspace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes, groupCnt);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnNormMode_t mode_native;
    cudnnNormOps_t normOps_native;
    cudnnNormAlgo_t algo_native;
    void * alpha_native = NULL;
    void * beta_native = NULL;
    cudnnTensorDescriptor_t xDesc_native;
    void * xData_native = NULL;
    cudnnTensorDescriptor_t normScaleBiasDesc_native;
    void * normScale_native = NULL;
    void * normBias_native = NULL;
    double exponentialAverageFactor_native = 0.0;
    cudnnTensorDescriptor_t normMeanVarDesc_native;
    void * resultRunningMean_native = NULL;
    void * resultRunningVariance_native = NULL;
    double epsilon_native = 0.0;
    void * resultSaveMean_native = NULL;
    void * resultSaveInvVariance_native = NULL;
    cudnnActivationDescriptor_t activationDesc_native;
    cudnnTensorDescriptor_t zDesc_native;
    void * zData_native = NULL;
    cudnnTensorDescriptor_t yDesc_native;
    void * yData_native = NULL;
    void * workspace_native = NULL;
    size_t workSpaceSizeInBytes_native = 0;
    void * reserveSpace_native = NULL;
    size_t reserveSpaceSizeInBytes_native = 0;
    int groupCnt_native = 0;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    mode_native = (cudnnNormMode_t)mode;
    normOps_native = (cudnnNormOps_t)normOps;
    algo_native = (cudnnNormAlgo_t)algo;
    PointerData *alpha_pointerData = initPointerData(env, alpha);
    if (alpha_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    alpha_native = (void *)alpha_pointerData->getPointer(env);
    PointerData *beta_pointerData = initPointerData(env, beta);
    if (beta_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    beta_native = (void *)beta_pointerData->getPointer(env);
    xDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, xDesc);
    xData_native = (void *)getPointer(env, xData);
    normScaleBiasDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, normScaleBiasDesc);
    normScale_native = (void *)getPointer(env, normScale);
    normBias_native = (void *)getPointer(env, normBias);
    exponentialAverageFactor_native = (double)exponentialAverageFactor;
    normMeanVarDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, normMeanVarDesc);
    resultRunningMean_native = (void *)getPointer(env, resultRunningMean);
    resultRunningVariance_native = (void *)getPointer(env, resultRunningVariance);
    epsilon_native = (double)epsilon;
    resultSaveMean_native = (void *)getPointer(env, resultSaveMean);
    resultSaveInvVariance_native = (void *)getPointer(env, resultSaveInvVariance);
    activationDesc_native = (cudnnActivationDescriptor_t)getNativePointerValue(env, activationDesc);
    zDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, zDesc);
    zData_native = (void *)getPointer(env, zData);
    yDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, yDesc);
    yData_native = (void *)getPointer(env, yData);
    workspace_native = (void *)getPointer(env, workspace);
    workSpaceSizeInBytes_native = (size_t)workSpaceSizeInBytes;
    reserveSpace_native = (void *)getPointer(env, reserveSpace);
    reserveSpaceSizeInBytes_native = (size_t)reserveSpaceSizeInBytes;
    groupCnt_native = (int)groupCnt;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnNormalizationForwardTraining(handle_native, mode_native, normOps_native, algo_native, alpha_native, beta_native, xDesc_native, xData_native, normScaleBiasDesc_native, normScale_native, normBias_native, exponentialAverageFactor_native, normMeanVarDesc_native, resultRunningMean_native, resultRunningVariance_native, epsilon_native, resultSaveMean_native, resultSaveInvVariance_native, activationDesc_native, zDesc_native, zData_native, yDesc_native, yData_native, workspace_native, workSpaceSizeInBytes_native, reserveSpace_native, reserveSpaceSizeInBytes_native, groupCnt_native);

    // Write back native variable values
    // handle is read-only
    // mode is primitive
    // normOps is primitive
    // algo is primitive
    if (!releasePointerData(env, alpha_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!releasePointerData(env, beta_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // xDesc is read-only
    // xData is a native pointer
    // normScaleBiasDesc is read-only
    // normScale is a native pointer
    // normBias is a native pointer
    // exponentialAverageFactor is primitive
    // normMeanVarDesc is read-only
    // resultRunningMean is a native pointer
    // resultRunningVariance is a native pointer
    // epsilon is primitive
    // resultSaveMean is a native pointer
    // resultSaveInvVariance is a native pointer
    // activationDesc is read-only
    // zDesc is read-only
    // zData is a native pointer
    // yDesc is read-only
    // yData is a native pointer
    // workspace is a native pointer
    // workSpaceSizeInBytes is primitive
    // reserveSpace is a native pointer
    // reserveSpaceSizeInBytes is primitive
    // groupCnt is primitive

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnNormalizationBackwardNative(JNIEnv *env, jclass cls, jobject handle, jint mode, jint normOps, jint algo, jobject alphaDataDiff, jobject betaDataDiff, jobject alphaParamDiff, jobject betaParamDiff, jobject xDesc, jobject xData, jobject yDesc, jobject yData, jobject dyDesc, jobject dyData, jobject dzDesc, jobject dzData, jobject dxDesc, jobject dxData, jobject dNormScaleBiasDesc, jobject normScaleData, jobject normBiasData, jobject dNormScaleData, jobject dNormBiasData, jdouble epsilon, jobject normMeanVarDesc, jobject savedMean, jobject savedInvVariance, jobject activationDesc, jobject workSpace, jlong workSpaceSizeInBytes, jobject reserveSpace, jlong reserveSpaceSizeInBytes, jint groupCnt)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnNormalizationBackward(handle=%p, mode=%d, normOps=%d, algo=%d, alphaDataDiff=%p, betaDataDiff=%p, alphaParamDiff=%p, betaParamDiff=%p, xDesc=%p, xData=%p, yDesc=%p, yData=%p, dyDesc=%p, dyData=%p, dzDesc=%p, dzData=%p, dxDesc=%p, dxData=%p, dNormScaleBiasDesc=%p, normScaleData=%p, normBiasData=%p, dNormScaleData=%p, dNormBiasData=%p, epsilon=%lf, normMeanVarDesc=%p, savedMean=%p, savedInvVariance=%p, activationDesc=%p, workSpace=%p, workSpaceSizeInBytes=%ld, reserveSpace=%p, reserveSpaceSizeInBytes=%ld, groupCnt=%d)\n",
        handle, mode, normOps, algo, alphaDataDiff, betaDataDiff, alphaParamDiff, betaParamDiff, xDesc, xData, yDesc, yData, dyDesc, dyData, dzDesc, dzData, dxDesc, dxData, dNormScaleBiasDesc, normScaleData, normBiasData, dNormScaleData, dNormBiasData, epsilon, normMeanVarDesc, savedMean, savedInvVariance, activationDesc, workSpace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes, groupCnt);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnNormMode_t mode_native;
    cudnnNormOps_t normOps_native;
    cudnnNormAlgo_t algo_native;
    void * alphaDataDiff_native = NULL;
    void * betaDataDiff_native = NULL;
    void * alphaParamDiff_native = NULL;
    void * betaParamDiff_native = NULL;
    cudnnTensorDescriptor_t xDesc_native;
    void * xData_native = NULL;
    cudnnTensorDescriptor_t yDesc_native;
    void * yData_native = NULL;
    cudnnTensorDescriptor_t dyDesc_native;
    void * dyData_native = NULL;
    cudnnTensorDescriptor_t dzDesc_native;
    void * dzData_native = NULL;
    cudnnTensorDescriptor_t dxDesc_native;
    void * dxData_native = NULL;
    cudnnTensorDescriptor_t dNormScaleBiasDesc_native;
    void * normScaleData_native = NULL;
    void * normBiasData_native = NULL;
    void * dNormScaleData_native = NULL;
    void * dNormBiasData_native = NULL;
    double epsilon_native = 0.0;
    cudnnTensorDescriptor_t normMeanVarDesc_native;
    void * savedMean_native = NULL;
    void * savedInvVariance_native = NULL;
    cudnnActivationDescriptor_t activationDesc_native;
    void * workSpace_native = NULL;
    size_t workSpaceSizeInBytes_native = 0;
    void * reserveSpace_native = NULL;
    size_t reserveSpaceSizeInBytes_native = 0;
    int groupCnt_native = 0;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    mode_native = (cudnnNormMode_t)mode;
    normOps_native = (cudnnNormOps_t)normOps;
    algo_native = (cudnnNormAlgo_t)algo;
    PointerData *alphaDataDiff_pointerData = initPointerData(env, alphaDataDiff);
    if (alphaDataDiff_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    alphaDataDiff_native = (void *)alphaDataDiff_pointerData->getPointer(env);
    PointerData *betaDataDiff_pointerData = initPointerData(env, betaDataDiff);
    if (betaDataDiff_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    betaDataDiff_native = (void *)betaDataDiff_pointerData->getPointer(env);
    PointerData *alphaParamDiff_pointerData = initPointerData(env, alphaParamDiff);
    if (alphaParamDiff_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    alphaParamDiff_native = (void *)alphaParamDiff_pointerData->getPointer(env);
    PointerData *betaParamDiff_pointerData = initPointerData(env, betaParamDiff);
    if (betaParamDiff_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    betaParamDiff_native = (void *)betaParamDiff_pointerData->getPointer(env);
    xDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, xDesc);
    xData_native = (void *)getPointer(env, xData);
    yDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, yDesc);
    yData_native = (void *)getPointer(env, yData);
    dyDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, dyDesc);
    dyData_native = (void *)getPointer(env, dyData);
    dzDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, dzDesc);
    dzData_native = (void *)getPointer(env, dzData);
    dxDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, dxDesc);
    dxData_native = (void *)getPointer(env, dxData);
    dNormScaleBiasDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, dNormScaleBiasDesc);
    normScaleData_native = (void *)getPointer(env, normScaleData);
    normBiasData_native = (void *)getPointer(env, normBiasData);
    dNormScaleData_native = (void *)getPointer(env, dNormScaleData);
    dNormBiasData_native = (void *)getPointer(env, dNormBiasData);
    epsilon_native = (double)epsilon;
    normMeanVarDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, normMeanVarDesc);
    savedMean_native = (void *)getPointer(env, savedMean);
    savedInvVariance_native = (void *)getPointer(env, savedInvVariance);
    activationDesc_native = (cudnnActivationDescriptor_t)getNativePointerValue(env, activationDesc);
    workSpace_native = (void *)getPointer(env, workSpace);
    workSpaceSizeInBytes_native = (size_t)workSpaceSizeInBytes;
    reserveSpace_native = (void *)getPointer(env, reserveSpace);
    reserveSpaceSizeInBytes_native = (size_t)reserveSpaceSizeInBytes;
    groupCnt_native = (int)groupCnt;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnNormalizationBackward(handle_native, mode_native, normOps_native, algo_native, alphaDataDiff_native, betaDataDiff_native, alphaParamDiff_native, betaParamDiff_native, xDesc_native, xData_native, yDesc_native, yData_native, dyDesc_native, dyData_native, dzDesc_native, dzData_native, dxDesc_native, dxData_native, dNormScaleBiasDesc_native, normScaleData_native, normBiasData_native, dNormScaleData_native, dNormBiasData_native, epsilon_native, normMeanVarDesc_native, savedMean_native, savedInvVariance_native, activationDesc_native, workSpace_native, workSpaceSizeInBytes_native, reserveSpace_native, reserveSpaceSizeInBytes_native, groupCnt_native);

    // Write back native variable values
    // handle is read-only
    // mode is primitive
    // normOps is primitive
    // algo is primitive
    if (!releasePointerData(env, alphaDataDiff_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!releasePointerData(env, betaDataDiff_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!releasePointerData(env, alphaParamDiff_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!releasePointerData(env, betaParamDiff_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // xDesc is read-only
    // xData is a native pointer
    // yDesc is read-only
    // yData is a native pointer
    // dyDesc is read-only
    // dyData is a native pointer
    // dzDesc is read-only
    // dzData is a native pointer
    // dxDesc is read-only
    // dxData is a native pointer
    // dNormScaleBiasDesc is read-only
    // normScaleData is a native pointer
    // normBiasData is a native pointer
    // dNormScaleData is a native pointer
    // dNormBiasData is a native pointer
    // epsilon is primitive
    // normMeanVarDesc is read-only
    // savedMean is a native pointer
    // savedInvVariance is a native pointer
    // activationDesc is read-only
    // workSpace is a native pointer
    // workSpaceSizeInBytes is primitive
    // reserveSpace is a native pointer
    // reserveSpaceSizeInBytes is primitive
    // groupCnt is primitive

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnSpatialTfGridGeneratorBackwardNative(JNIEnv *env, jclass cls, jobject handle, jobject stDesc, jobject dgrid, jobject dtheta)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnSpatialTfGridGeneratorBackward(handle=%p, stDesc=%p, dgrid=%p, dtheta=%p)\n",
        handle, stDesc, dgrid, dtheta);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnSpatialTransformerDescriptor_t stDesc_native;
    void * dgrid_native = NULL;
    void * dtheta_native = NULL;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    stDesc_native = (cudnnSpatialTransformerDescriptor_t)getNativePointerValue(env, stDesc);
    dgrid_native = (void *)getPointer(env, dgrid);
    dtheta_native = (void *)getPointer(env, dtheta);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnSpatialTfGridGeneratorBackward(handle_native, stDesc_native, dgrid_native, dtheta_native);

    // Write back native variable values
    // handle is read-only
    // stDesc is read-only
    // dgrid is a native pointer
    // dtheta is a native pointer

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnSpatialTfSamplerBackwardNative(JNIEnv *env, jclass cls, jobject handle, jobject stDesc, jobject alpha, jobject xDesc, jobject x, jobject beta, jobject dxDesc, jobject dx, jobject alphaDgrid, jobject dyDesc, jobject dy, jobject grid, jobject betaDgrid, jobject dgrid)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnSpatialTfSamplerBackward(handle=%p, stDesc=%p, alpha=%p, xDesc=%p, x=%p, beta=%p, dxDesc=%p, dx=%p, alphaDgrid=%p, dyDesc=%p, dy=%p, grid=%p, betaDgrid=%p, dgrid=%p)\n",
        handle, stDesc, alpha, xDesc, x, beta, dxDesc, dx, alphaDgrid, dyDesc, dy, grid, betaDgrid, dgrid);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnSpatialTransformerDescriptor_t stDesc_native;
    void * alpha_native = NULL;
    cudnnTensorDescriptor_t xDesc_native;
    void * x_native = NULL;
    void * beta_native = NULL;
    cudnnTensorDescriptor_t dxDesc_native;
    void * dx_native = NULL;
    void * alphaDgrid_native = NULL;
    cudnnTensorDescriptor_t dyDesc_native;
    void * dy_native = NULL;
    void * grid_native = NULL;
    void * betaDgrid_native = NULL;
    void * dgrid_native = NULL;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    stDesc_native = (cudnnSpatialTransformerDescriptor_t)getNativePointerValue(env, stDesc);
    PointerData *alpha_pointerData = initPointerData(env, alpha);
    if (alpha_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    alpha_native = (void *)alpha_pointerData->getPointer(env);
    xDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, xDesc);
    x_native = (void *)getPointer(env, x);
    PointerData *beta_pointerData = initPointerData(env, beta);
    if (beta_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    beta_native = (void *)beta_pointerData->getPointer(env);
    dxDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, dxDesc);
    dx_native = (void *)getPointer(env, dx);
    PointerData *alphaDgrid_pointerData = initPointerData(env, alphaDgrid);
    if (alphaDgrid_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    alphaDgrid_native = (void *)alphaDgrid_pointerData->getPointer(env);
    dyDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, dyDesc);
    dy_native = (void *)getPointer(env, dy);
    grid_native = (void *)getPointer(env, grid);
    PointerData *betaDgrid_pointerData = initPointerData(env, betaDgrid);
    if (betaDgrid_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    betaDgrid_native = (void *)betaDgrid_pointerData->getPointer(env);
    dgrid_native = (void *)getPointer(env, dgrid);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnSpatialTfSamplerBackward(handle_native, stDesc_native, alpha_native, xDesc_native, x_native, beta_native, dxDesc_native, dx_native, alphaDgrid_native, dyDesc_native, dy_native, grid_native, betaDgrid_native, dgrid_native);

    // Write back native variable values
    // handle is read-only
    // stDesc is read-only
    if (!releasePointerData(env, alpha_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // xDesc is read-only
    // x is a native pointer
    if (!releasePointerData(env, beta_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // dxDesc is read-only
    // dx is a native pointer
    if (!releasePointerData(env, alphaDgrid_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // dyDesc is read-only
    // dy is a native pointer
    // grid is a native pointer
    if (!releasePointerData(env, betaDgrid_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // dgrid is a native pointer

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnDropoutBackwardNative(JNIEnv *env, jclass cls, jobject handle, jobject dropoutDesc, jobject dydesc, jobject dy, jobject dxdesc, jobject dx, jobject reserveSpace, jlong reserveSpaceSizeInBytes)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnDropoutBackward(handle=%p, dropoutDesc=%p, dydesc=%p, dy=%p, dxdesc=%p, dx=%p, reserveSpace=%p, reserveSpaceSizeInBytes=%ld)\n",
        handle, dropoutDesc, dydesc, dy, dxdesc, dx, reserveSpace, reserveSpaceSizeInBytes);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnDropoutDescriptor_t dropoutDesc_native;
    cudnnTensorDescriptor_t dydesc_native;
    void * dy_native = NULL;
    cudnnTensorDescriptor_t dxdesc_native;
    void * dx_native = NULL;
    void * reserveSpace_native = NULL;
    size_t reserveSpaceSizeInBytes_native = 0;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    dropoutDesc_native = (cudnnDropoutDescriptor_t)getNativePointerValue(env, dropoutDesc);
    dydesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, dydesc);
    dy_native = (void *)getPointer(env, dy);
    dxdesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, dxdesc);
    dx_native = (void *)getPointer(env, dx);
    reserveSpace_native = (void *)getPointer(env, reserveSpace);
    reserveSpaceSizeInBytes_native = (size_t)reserveSpaceSizeInBytes;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnDropoutBackward(handle_native, dropoutDesc_native, dydesc_native, dy_native, dxdesc_native, dx_native, reserveSpace_native, reserveSpaceSizeInBytes_native);

    // Write back native variable values
    // handle is read-only
    // dropoutDesc is read-only
    // dydesc is read-only
    // dy is a native pointer
    // dxdesc is read-only
    // dx is a native pointer
    // reserveSpace is a native pointer
    // reserveSpaceSizeInBytes is primitive

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

/**
 * <pre>
 * Cross-library version checker..
 * This function is implemented differently in each sub-library. Each sublib
 * checks whether its own version matches that of its dependencies.
 * @return CUDNN_STATUS_SUCCESS if the version check passes,
 *          CUDNN_STATUS_VERSION_MISMATCH if the versions are inconsistent.
 * </pre>
 */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnOpsTrainVersionCheckNative(JNIEnv *env, jclass cls)
{
    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnOpsTrainVersionCheck()\n");

    // Native function call
    cudnnStatus_t jniResult_native = cudnnOpsTrainVersionCheck();

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnCreateRNNDescriptorNative(JNIEnv *env, jclass cls, jobject rnnDesc)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnCreateRNNDescriptor(rnnDesc=%p)\n",
        rnnDesc);

    // Native variable declarations
    cudnnRNNDescriptor_t rnnDesc_native;

    // Obtain native variable values
    // rnnDesc is write-only

    // Native function call
    cudnnStatus_t jniResult_native = cudnnCreateRNNDescriptor(&rnnDesc_native);

    // Write back native variable values
    setNativePointerValue(env, rnnDesc, (jlong)rnnDesc_native);

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnDestroyRNNDescriptorNative(JNIEnv *env, jclass cls, jobject rnnDesc)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnDestroyRNNDescriptor(rnnDesc=%p)\n",
        rnnDesc);

    // Native variable declarations
    cudnnRNNDescriptor_t rnnDesc_native;

    // Obtain native variable values
    rnnDesc_native = (cudnnRNNDescriptor_t)getNativePointerValue(env, rnnDesc);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnDestroyRNNDescriptor(rnnDesc_native);

    // Write back native variable values
    // rnnDesc is read-only

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnSetRNNDescriptor_1v8Native(JNIEnv *env, jclass cls, jobject rnnDesc, jint algo, jint cellMode, jint biasMode, jint dirMode, jint inputMode, jint dataType, jint mathPrec, jint mathType, jint inputSize, jint hiddenSize, jint projSize, jint numLayers, jobject dropoutDesc, jint auxFlags)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnSetRNNDescriptor_v8(rnnDesc=%p, algo=%d, cellMode=%d, biasMode=%d, dirMode=%d, inputMode=%d, dataType=%d, mathPrec=%d, mathType=%d, inputSize=%d, hiddenSize=%d, projSize=%d, numLayers=%d, dropoutDesc=%p, auxFlags=%d)\n",
        rnnDesc, algo, cellMode, biasMode, dirMode, inputMode, dataType, mathPrec, mathType, inputSize, hiddenSize, projSize, numLayers, dropoutDesc, auxFlags);

    // Native variable declarations
    cudnnRNNDescriptor_t rnnDesc_native;
    cudnnRNNAlgo_t algo_native;
    cudnnRNNMode_t cellMode_native;
    cudnnRNNBiasMode_t biasMode_native;
    cudnnDirectionMode_t dirMode_native;
    cudnnRNNInputMode_t inputMode_native;
    cudnnDataType_t dataType_native;
    cudnnDataType_t mathPrec_native;
    cudnnMathType_t mathType_native;
    int32_t inputSize_native = 0;
    int32_t hiddenSize_native = 0;
    int32_t projSize_native = 0;
    int32_t numLayers_native = 0;
    cudnnDropoutDescriptor_t dropoutDesc_native;
    uint32_t auxFlags_native = 0;

    // Obtain native variable values
    rnnDesc_native = (cudnnRNNDescriptor_t)getNativePointerValue(env, rnnDesc);
    algo_native = (cudnnRNNAlgo_t)algo;
    cellMode_native = (cudnnRNNMode_t)cellMode;
    biasMode_native = (cudnnRNNBiasMode_t)biasMode;
    dirMode_native = (cudnnDirectionMode_t)dirMode;
    inputMode_native = (cudnnRNNInputMode_t)inputMode;
    dataType_native = (cudnnDataType_t)dataType;
    mathPrec_native = (cudnnDataType_t)mathPrec;
    mathType_native = (cudnnMathType_t)mathType;
    inputSize_native = (int32_t)inputSize;
    hiddenSize_native = (int32_t)hiddenSize;
    projSize_native = (int32_t)projSize;
    numLayers_native = (int32_t)numLayers;
    dropoutDesc_native = (cudnnDropoutDescriptor_t)getNativePointerValue(env, dropoutDesc);
    auxFlags_native = (uint32_t)auxFlags;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnSetRNNDescriptor_v8(rnnDesc_native, algo_native, cellMode_native, biasMode_native, dirMode_native, inputMode_native, dataType_native, mathPrec_native, mathType_native, inputSize_native, hiddenSize_native, projSize_native, numLayers_native, dropoutDesc_native, auxFlags_native);

    // Write back native variable values
    // rnnDesc is read-only
    // algo is primitive
    // cellMode is primitive
    // biasMode is primitive
    // dirMode is primitive
    // inputMode is primitive
    // dataType is primitive
    // mathPrec is primitive
    // mathType is primitive
    // inputSize is primitive
    // hiddenSize is primitive
    // projSize is primitive
    // numLayers is primitive
    // dropoutDesc is read-only
    // auxFlags is primitive

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetRNNDescriptor_1v8Native(JNIEnv *env, jclass cls, jobject rnnDesc, jintArray algo, jintArray cellMode, jintArray biasMode, jintArray dirMode, jintArray inputMode, jintArray dataType, jintArray mathPrec, jintArray mathType, jintArray inputSize, jintArray hiddenSize, jintArray projSize, jintArray numLayers, jobject dropoutDesc, jintArray auxFlags)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetRNNDescriptor_v8(rnnDesc=%p, algo=%p, cellMode=%p, biasMode=%p, dirMode=%p, inputMode=%p, dataType=%p, mathPrec=%p, mathType=%p, inputSize=%p, hiddenSize=%p, projSize=%p, numLayers=%p, dropoutDesc=%p, auxFlags=%p)\n",
        rnnDesc, algo, cellMode, biasMode, dirMode, inputMode, dataType, mathPrec, mathType, inputSize, hiddenSize, projSize, numLayers, dropoutDesc, auxFlags);

    // Native variable declarations
    cudnnRNNDescriptor_t rnnDesc_native;
    cudnnRNNAlgo_t algo_native;
    cudnnRNNMode_t cellMode_native;
    cudnnRNNBiasMode_t biasMode_native;
    cudnnDirectionMode_t dirMode_native;
    cudnnRNNInputMode_t inputMode_native;
    cudnnDataType_t dataType_native;
    cudnnDataType_t mathPrec_native;
    cudnnMathType_t mathType_native;
    int32_t inputSize_native;
    int32_t hiddenSize_native;
    int32_t projSize_native;
    int32_t numLayers_native;
    cudnnDropoutDescriptor_t * dropoutDesc_native;
    uint32_t auxFlags_native;

    // Obtain native variable values
    rnnDesc_native = (cudnnRNNDescriptor_t)getNativePointerValue(env, rnnDesc);
    // algo is write-only
    // cellMode is write-only
    // biasMode is write-only
    // dirMode is write-only
    // inputMode is write-only
    // dataType is write-only
    // mathPrec is write-only
    // mathType is write-only
    // inputSize is write-only
    // hiddenSize is write-only
    // projSize is write-only
    // numLayers is write-only
    dropoutDesc_native = (cudnnDropoutDescriptor_t *)getNativePointerValue(env, dropoutDesc);
    // auxFlags is write-only

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetRNNDescriptor_v8(rnnDesc_native, &algo_native, &cellMode_native, &biasMode_native, &dirMode_native, &inputMode_native, &dataType_native, &mathPrec_native, &mathType_native, &inputSize_native, &hiddenSize_native, &projSize_native, &numLayers_native, dropoutDesc_native, &auxFlags_native);

    // Write back native variable values
    // rnnDesc is read-only
    if (!set(env, algo, 0, (jint)algo_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, cellMode, 0, (jint)cellMode_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, biasMode, 0, (jint)biasMode_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, dirMode, 0, (jint)dirMode_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, inputMode, 0, (jint)inputMode_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, dataType, 0, (jint)dataType_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, mathPrec, 0, (jint)mathPrec_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, mathType, 0, (jint)mathType_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, inputSize, 0, (jint)inputSize_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, hiddenSize, 0, (jint)hiddenSize_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, projSize, 0, (jint)projSize_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, numLayers, 0, (jint)numLayers_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // dropoutDesc is read-only
    if (!set(env, auxFlags, 0, (jint)auxFlags_native)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

/**
 * <pre>
 * mathPrec in cudnnSetRNNDescriptor_v6() specifies compute precision
 * compute precision is further modified by cudnnSetRNNMatrixMathType()
 * dataType in cudnnGetRNNParamsSize() and wDesc specify weight storage
 * dropout is between RNN layers, not between recurrent steps
 * </pre>
 */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnSetRNNDescriptor_1v6Native(JNIEnv *env, jclass cls, jobject handle, jobject rnnDesc, jint hiddenSize, jint numLayers, jobject dropoutDesc, jint inputMode, jint direction, jint cellMode, jint algo, jint mathPrec)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnSetRNNDescriptor_v6(handle=%p, rnnDesc=%p, hiddenSize=%d, numLayers=%d, dropoutDesc=%p, inputMode=%d, direction=%d, cellMode=%d, algo=%d, mathPrec=%d)\n",
        handle, rnnDesc, hiddenSize, numLayers, dropoutDesc, inputMode, direction, cellMode, algo, mathPrec);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnRNNDescriptor_t rnnDesc_native;
    int hiddenSize_native = 0;
    int numLayers_native = 0;
    cudnnDropoutDescriptor_t dropoutDesc_native;
    cudnnRNNInputMode_t inputMode_native;
    cudnnDirectionMode_t direction_native;
    cudnnRNNMode_t cellMode_native;
    cudnnRNNAlgo_t algo_native;
    cudnnDataType_t mathPrec_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    rnnDesc_native = (cudnnRNNDescriptor_t)getNativePointerValue(env, rnnDesc);
    hiddenSize_native = (int)hiddenSize;
    numLayers_native = (int)numLayers;
    dropoutDesc_native = (cudnnDropoutDescriptor_t)getNativePointerValue(env, dropoutDesc);
    inputMode_native = (cudnnRNNInputMode_t)inputMode;
    direction_native = (cudnnDirectionMode_t)direction;
    cellMode_native = (cudnnRNNMode_t)cellMode;
    algo_native = (cudnnRNNAlgo_t)algo;
    mathPrec_native = (cudnnDataType_t)mathPrec;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnSetRNNDescriptor_v6(handle_native, rnnDesc_native, hiddenSize_native, numLayers_native, dropoutDesc_native, inputMode_native, direction_native, cellMode_native, algo_native, mathPrec_native);

    // Write back native variable values
    // handle is read-only
    // rnnDesc is read-only
    // hiddenSize is primitive
    // numLayers is primitive
    // dropoutDesc is read-only
    // inputMode is primitive
    // direction is primitive
    // cellMode is primitive
    // algo is primitive
    // mathPrec is primitive

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetRNNDescriptor_1v6Native(JNIEnv *env, jclass cls, jobject handle, jobject rnnDesc, jintArray hiddenSize, jintArray numLayers, jobject dropoutDesc, jintArray inputMode, jintArray direction, jintArray cellMode, jintArray algo, jintArray mathPrec)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetRNNDescriptor_v6(handle=%p, rnnDesc=%p, hiddenSize=%p, numLayers=%p, dropoutDesc=%p, inputMode=%p, direction=%p, cellMode=%p, algo=%p, mathPrec=%p)\n",
        handle, rnnDesc, hiddenSize, numLayers, dropoutDesc, inputMode, direction, cellMode, algo, mathPrec);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnRNNDescriptor_t rnnDesc_native;
    int hiddenSize_native;
    int numLayers_native;
    cudnnDropoutDescriptor_t * dropoutDesc_native;
    cudnnRNNInputMode_t inputMode_native;
    cudnnDirectionMode_t direction_native;
    cudnnRNNMode_t cellMode_native;
    cudnnRNNAlgo_t algo_native;
    cudnnDataType_t mathPrec_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    rnnDesc_native = (cudnnRNNDescriptor_t)getNativePointerValue(env, rnnDesc);
    // hiddenSize is write-only
    // numLayers is write-only
    dropoutDesc_native = (cudnnDropoutDescriptor_t *)getNativePointerValue(env, dropoutDesc);
    // inputMode is write-only
    // direction is write-only
    // cellMode is write-only
    // algo is write-only
    // mathPrec is write-only

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetRNNDescriptor_v6(handle_native, rnnDesc_native, &hiddenSize_native, &numLayers_native, dropoutDesc_native, &inputMode_native, &direction_native, &cellMode_native, &algo_native, &mathPrec_native);

    // Write back native variable values
    // handle is read-only
    // rnnDesc is read-only
    if (!set(env, hiddenSize, 0, (jint)hiddenSize_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, numLayers, 0, (jint)numLayers_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // dropoutDesc is read-only
    if (!set(env, inputMode, 0, (jint)inputMode_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, direction, 0, (jint)direction_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, cellMode, 0, (jint)cellMode_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, algo, 0, (jint)algo_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, mathPrec, 0, (jint)mathPrec_native)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnSetRNNMatrixMathTypeNative(JNIEnv *env, jclass cls, jobject rnnDesc, jint mType)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnSetRNNMatrixMathType(rnnDesc=%p, mType=%d)\n",
        rnnDesc, mType);

    // Native variable declarations
    cudnnRNNDescriptor_t rnnDesc_native;
    cudnnMathType_t mType_native;

    // Obtain native variable values
    rnnDesc_native = (cudnnRNNDescriptor_t)getNativePointerValue(env, rnnDesc);
    mType_native = (cudnnMathType_t)mType;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnSetRNNMatrixMathType(rnnDesc_native, mType_native);

    // Write back native variable values
    // rnnDesc is read-only
    // mType is primitive

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetRNNMatrixMathTypeNative(JNIEnv *env, jclass cls, jobject rnnDesc, jintArray mType)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetRNNMatrixMathType(rnnDesc=%p, mType=%p)\n",
        rnnDesc, mType);

    // Native variable declarations
    cudnnRNNDescriptor_t rnnDesc_native;
    cudnnMathType_t mType_native;

    // Obtain native variable values
    rnnDesc_native = (cudnnRNNDescriptor_t)getNativePointerValue(env, rnnDesc);
    // mType is write-only

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetRNNMatrixMathType(rnnDesc_native, &mType_native);

    // Write back native variable values
    // rnnDesc is read-only
    if (!set(env, mType, 0, (jint)mType_native)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnSetRNNBiasModeNative(JNIEnv *env, jclass cls, jobject rnnDesc, jint biasMode)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnSetRNNBiasMode(rnnDesc=%p, biasMode=%d)\n",
        rnnDesc, biasMode);

    // Native variable declarations
    cudnnRNNDescriptor_t rnnDesc_native;
    cudnnRNNBiasMode_t biasMode_native;

    // Obtain native variable values
    rnnDesc_native = (cudnnRNNDescriptor_t)getNativePointerValue(env, rnnDesc);
    biasMode_native = (cudnnRNNBiasMode_t)biasMode;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnSetRNNBiasMode(rnnDesc_native, biasMode_native);

    // Write back native variable values
    // rnnDesc is read-only
    // biasMode is primitive

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetRNNBiasModeNative(JNIEnv *env, jclass cls, jobject rnnDesc, jintArray biasMode)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetRNNBiasMode(rnnDesc=%p, biasMode=%p)\n",
        rnnDesc, biasMode);

    // Native variable declarations
    cudnnRNNDescriptor_t rnnDesc_native;
    cudnnRNNBiasMode_t biasMode_native;

    // Obtain native variable values
    rnnDesc_native = (cudnnRNNDescriptor_t)getNativePointerValue(env, rnnDesc);
    // biasMode is write-only

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetRNNBiasMode(rnnDesc_native, &biasMode_native);

    // Write back native variable values
    // rnnDesc is read-only
    if (!set(env, biasMode, 0, (jint)biasMode_native)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnRNNSetClip_1v8Native(JNIEnv *env, jclass cls, jobject rnnDesc, jint clipMode, jint clipNanOpt, jdouble lclip, jdouble rclip)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnRNNSetClip_v8(rnnDesc=%p, clipMode=%d, clipNanOpt=%d, lclip=%lf, rclip=%lf)\n",
        rnnDesc, clipMode, clipNanOpt, lclip, rclip);

    // Native variable declarations
    cudnnRNNDescriptor_t rnnDesc_native;
    cudnnRNNClipMode_t clipMode_native;
    cudnnNanPropagation_t clipNanOpt_native;
    double lclip_native = 0.0;
    double rclip_native = 0.0;

    // Obtain native variable values
    rnnDesc_native = (cudnnRNNDescriptor_t)getNativePointerValue(env, rnnDesc);
    clipMode_native = (cudnnRNNClipMode_t)clipMode;
    clipNanOpt_native = (cudnnNanPropagation_t)clipNanOpt;
    lclip_native = (double)lclip;
    rclip_native = (double)rclip;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnRNNSetClip_v8(rnnDesc_native, clipMode_native, clipNanOpt_native, lclip_native, rclip_native);

    // Write back native variable values
    // rnnDesc is read-only
    // clipMode is primitive
    // clipNanOpt is primitive
    // lclip is primitive
    // rclip is primitive

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnRNNGetClip_1v8Native(JNIEnv *env, jclass cls, jobject rnnDesc, jintArray clipMode, jintArray clipNanOpt, jdoubleArray lclip, jdoubleArray rclip)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnRNNGetClip_v8(rnnDesc=%p, clipMode=%p, clipNanOpt=%p, lclip=%p, rclip=%p)\n",
        rnnDesc, clipMode, clipNanOpt, lclip, rclip);

    // Native variable declarations
    cudnnRNNDescriptor_t rnnDesc_native;
    cudnnRNNClipMode_t clipMode_native;
    cudnnNanPropagation_t clipNanOpt_native;
    double lclip_native;
    double rclip_native;

    // Obtain native variable values
    rnnDesc_native = (cudnnRNNDescriptor_t)getNativePointerValue(env, rnnDesc);
    // clipMode is write-only
    // clipNanOpt is write-only
    // lclip is write-only
    // rclip is write-only

    // Native function call
    cudnnStatus_t jniResult_native = cudnnRNNGetClip_v8(rnnDesc_native, &clipMode_native, &clipNanOpt_native, &lclip_native, &rclip_native);

    // Write back native variable values
    // rnnDesc is read-only
    if (!set(env, clipMode, 0, (jint)clipMode_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, clipNanOpt, 0, (jint)clipNanOpt_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, lclip, 0, (jdouble)lclip_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, rclip, 0, (jdouble)rclip_native)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnRNNSetClipNative(JNIEnv *env, jclass cls, jobject handle, jobject rnnDesc, jint clipMode, jint clipNanOpt, jdouble lclip, jdouble rclip)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnRNNSetClip(handle=%p, rnnDesc=%p, clipMode=%d, clipNanOpt=%d, lclip=%lf, rclip=%lf)\n",
        handle, rnnDesc, clipMode, clipNanOpt, lclip, rclip);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnRNNDescriptor_t rnnDesc_native;
    cudnnRNNClipMode_t clipMode_native;
    cudnnNanPropagation_t clipNanOpt_native;
    double lclip_native = 0.0;
    double rclip_native = 0.0;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    rnnDesc_native = (cudnnRNNDescriptor_t)getNativePointerValue(env, rnnDesc);
    clipMode_native = (cudnnRNNClipMode_t)clipMode;
    clipNanOpt_native = (cudnnNanPropagation_t)clipNanOpt;
    lclip_native = (double)lclip;
    rclip_native = (double)rclip;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnRNNSetClip(handle_native, rnnDesc_native, clipMode_native, clipNanOpt_native, lclip_native, rclip_native);

    // Write back native variable values
    // handle is read-only
    // rnnDesc is read-only
    // clipMode is primitive
    // clipNanOpt is primitive
    // lclip is primitive
    // rclip is primitive

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnRNNGetClipNative(JNIEnv *env, jclass cls, jobject handle, jobject rnnDesc, jintArray clipMode, jintArray clipNanOpt, jdoubleArray lclip, jdoubleArray rclip)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnRNNGetClip(handle=%p, rnnDesc=%p, clipMode=%p, clipNanOpt=%p, lclip=%p, rclip=%p)\n",
        handle, rnnDesc, clipMode, clipNanOpt, lclip, rclip);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnRNNDescriptor_t rnnDesc_native;
    cudnnRNNClipMode_t clipMode_native;
    cudnnNanPropagation_t clipNanOpt_native;
    double lclip_native;
    double rclip_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    rnnDesc_native = (cudnnRNNDescriptor_t)getNativePointerValue(env, rnnDesc);
    // clipMode is write-only
    // clipNanOpt is write-only
    // lclip is write-only
    // rclip is write-only

    // Native function call
    cudnnStatus_t jniResult_native = cudnnRNNGetClip(handle_native, rnnDesc_native, &clipMode_native, &clipNanOpt_native, &lclip_native, &rclip_native);

    // Write back native variable values
    // handle is read-only
    // rnnDesc is read-only
    if (!set(env, clipMode, 0, (jint)clipMode_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, clipNanOpt, 0, (jint)clipNanOpt_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, lclip, 0, (jdouble)lclip_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, rclip, 0, (jdouble)rclip_native)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnSetRNNProjectionLayersNative(JNIEnv *env, jclass cls, jobject handle, jobject rnnDesc, jint recProjSize, jint outProjSize)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnSetRNNProjectionLayers(handle=%p, rnnDesc=%p, recProjSize=%d, outProjSize=%d)\n",
        handle, rnnDesc, recProjSize, outProjSize);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnRNNDescriptor_t rnnDesc_native;
    int recProjSize_native = 0;
    int outProjSize_native = 0;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    rnnDesc_native = (cudnnRNNDescriptor_t)getNativePointerValue(env, rnnDesc);
    recProjSize_native = (int)recProjSize;
    outProjSize_native = (int)outProjSize;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnSetRNNProjectionLayers(handle_native, rnnDesc_native, recProjSize_native, outProjSize_native);

    // Write back native variable values
    // handle is read-only
    // rnnDesc is read-only
    // recProjSize is primitive
    // outProjSize is primitive

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetRNNProjectionLayersNative(JNIEnv *env, jclass cls, jobject handle, jobject rnnDesc, jintArray recProjSize, jintArray outProjSize)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetRNNProjectionLayers(handle=%p, rnnDesc=%p, recProjSize=%p, outProjSize=%p)\n",
        handle, rnnDesc, recProjSize, outProjSize);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnRNNDescriptor_t rnnDesc_native;
    int recProjSize_native;
    int outProjSize_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    rnnDesc_native = (cudnnRNNDescriptor_t)getNativePointerValue(env, rnnDesc);
    // recProjSize is write-only
    // outProjSize is write-only

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetRNNProjectionLayers(handle_native, rnnDesc_native, &recProjSize_native, &outProjSize_native);

    // Write back native variable values
    // handle is read-only
    // rnnDesc is read-only
    if (!set(env, recProjSize, 0, (jint)recProjSize_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, outProjSize, 0, (jint)outProjSize_native)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

/** Expensive. Creates the plan for the specific settings. */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnCreatePersistentRNNPlanNative(JNIEnv *env, jclass cls, jobject rnnDesc, jint minibatch, jint dataType, jobject plan)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnCreatePersistentRNNPlan(rnnDesc=%p, minibatch=%d, dataType=%d, plan=%p)\n",
        rnnDesc, minibatch, dataType, plan);

    // Native variable declarations
    cudnnRNNDescriptor_t rnnDesc_native;
    int minibatch_native = 0;
    cudnnDataType_t dataType_native;
    cudnnPersistentRNNPlan_t plan_native;

    // Obtain native variable values
    rnnDesc_native = (cudnnRNNDescriptor_t)getNativePointerValue(env, rnnDesc);
    minibatch_native = (int)minibatch;
    dataType_native = (cudnnDataType_t)dataType;
    // plan is write-only

    // Native function call
    cudnnStatus_t jniResult_native = cudnnCreatePersistentRNNPlan(rnnDesc_native, minibatch_native, dataType_native, &plan_native);

    // Write back native variable values
    // rnnDesc is read-only
    // minibatch is primitive
    // dataType is primitive
    setNativePointerValue(env, plan, (jlong)plan_native);

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnDestroyPersistentRNNPlanNative(JNIEnv *env, jclass cls, jobject plan)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnDestroyPersistentRNNPlan(plan=%p)\n",
        plan);

    // Native variable declarations
    cudnnPersistentRNNPlan_t plan_native;

    // Obtain native variable values
    plan_native = (cudnnPersistentRNNPlan_t)getNativePointerValue(env, plan);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnDestroyPersistentRNNPlan(plan_native);

    // Write back native variable values
    // plan is read-only

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnSetPersistentRNNPlanNative(JNIEnv *env, jclass cls, jobject rnnDesc, jobject plan)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnSetPersistentRNNPlan(rnnDesc=%p, plan=%p)\n",
        rnnDesc, plan);

    // Native variable declarations
    cudnnRNNDescriptor_t rnnDesc_native;
    cudnnPersistentRNNPlan_t plan_native;

    // Obtain native variable values
    rnnDesc_native = (cudnnRNNDescriptor_t)getNativePointerValue(env, rnnDesc);
    plan_native = (cudnnPersistentRNNPlan_t)getNativePointerValue(env, plan);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnSetPersistentRNNPlan(rnnDesc_native, plan_native);

    // Write back native variable values
    // rnnDesc is read-only
    // plan is read-only

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnBuildRNNDynamicNative(JNIEnv *env, jclass cls, jobject handle, jobject rnnDesc, jint miniBatch)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnBuildRNNDynamic(handle=%p, rnnDesc=%p, miniBatch=%d)\n",
        handle, rnnDesc, miniBatch);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnRNNDescriptor_t rnnDesc_native;
    int miniBatch_native = 0;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    rnnDesc_native = (cudnnRNNDescriptor_t)getNativePointerValue(env, rnnDesc);
    miniBatch_native = (int)miniBatch;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnBuildRNNDynamic(handle_native, rnnDesc_native, miniBatch_native);

    // Write back native variable values
    // handle is read-only
    // rnnDesc is read-only
    // miniBatch is primitive

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

/** dataType in weight descriptors and input descriptors is used to describe storage */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetRNNWorkspaceSizeNative(JNIEnv *env, jclass cls, jobject handle, jobject rnnDesc, jint seqLength, jobjectArray xDesc, jlongArray sizeInBytes)
{
    // Null-checks for non-primitive arguments
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cudnnGetRNNWorkspaceSize");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (rnnDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'rnnDesc' is null for cudnnGetRNNWorkspaceSize");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    // seqLength is primitive
    if (xDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'xDesc' is null for cudnnGetRNNWorkspaceSize");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (sizeInBytes == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'sizeInBytes' is null for cudnnGetRNNWorkspaceSize");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetRNNWorkspaceSize(handle=%p, rnnDesc=%p, seqLength=%d, xDesc=%p, sizeInBytes=%p)\n",
        handle, rnnDesc, seqLength, xDesc, sizeInBytes);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnRNNDescriptor_t rnnDesc_native;
    int seqLength_native = 0;
    cudnnTensorDescriptor_t * xDesc_native;
    size_t sizeInBytes_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    rnnDesc_native = (cudnnRNNDescriptor_t)getNativePointerValue(env, rnnDesc);
    seqLength_native = (int)seqLength;
    if (!initNative(env, xDesc, xDesc_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // sizeInBytes is write-only

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetRNNWorkspaceSize(handle_native, rnnDesc_native, seqLength_native, xDesc_native, &sizeInBytes_native);

    // Write back native variable values
    // handle is read-only
    // rnnDesc is read-only
    // seqLength is primitive
    if (!releaseNative(env, xDesc_native, xDesc, false)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, sizeInBytes, 0, (jlong)sizeInBytes_native)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetRNNTrainingReserveSizeNative(JNIEnv *env, jclass cls, jobject handle, jobject rnnDesc, jint seqLength, jobjectArray xDesc, jlongArray sizeInBytes)
{
    // Null-checks for non-primitive arguments
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cudnnGetRNNTrainingReserveSize");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (rnnDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'rnnDesc' is null for cudnnGetRNNTrainingReserveSize");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    // seqLength is primitive
    if (xDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'xDesc' is null for cudnnGetRNNTrainingReserveSize");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (sizeInBytes == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'sizeInBytes' is null for cudnnGetRNNTrainingReserveSize");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetRNNTrainingReserveSize(handle=%p, rnnDesc=%p, seqLength=%d, xDesc=%p, sizeInBytes=%p)\n",
        handle, rnnDesc, seqLength, xDesc, sizeInBytes);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnRNNDescriptor_t rnnDesc_native;
    int seqLength_native = 0;
    cudnnTensorDescriptor_t * xDesc_native;
    size_t sizeInBytes_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    rnnDesc_native = (cudnnRNNDescriptor_t)getNativePointerValue(env, rnnDesc);
    seqLength_native = (int)seqLength;
    if (!initNative(env, xDesc, xDesc_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // sizeInBytes is write-only

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetRNNTrainingReserveSize(handle_native, rnnDesc_native, seqLength_native, xDesc_native, &sizeInBytes_native);

    // Write back native variable values
    // handle is read-only
    // rnnDesc is read-only
    // seqLength is primitive
    if (!releaseNative(env, xDesc_native, xDesc, false)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, sizeInBytes, 0, (jlong)sizeInBytes_native)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetRNNTempSpaceSizesNative(JNIEnv *env, jclass cls, jobject handle, jobject rnnDesc, jint fMode, jobject xDesc, jlongArray workSpaceSize, jlongArray reserveSpaceSize)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetRNNTempSpaceSizes(handle=%p, rnnDesc=%p, fMode=%d, xDesc=%p, workSpaceSize=%p, reserveSpaceSize=%p)\n",
        handle, rnnDesc, fMode, xDesc, workSpaceSize, reserveSpaceSize);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnRNNDescriptor_t rnnDesc_native;
    cudnnForwardMode_t fMode_native;
    cudnnRNNDataDescriptor_t xDesc_native;
    size_t workSpaceSize_native;
    size_t reserveSpaceSize_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    rnnDesc_native = (cudnnRNNDescriptor_t)getNativePointerValue(env, rnnDesc);
    fMode_native = (cudnnForwardMode_t)fMode;
    xDesc_native = (cudnnRNNDataDescriptor_t)getNativePointerValue(env, xDesc);
    // workSpaceSize is write-only
    // reserveSpaceSize is write-only

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetRNNTempSpaceSizes(handle_native, rnnDesc_native, fMode_native, xDesc_native, &workSpaceSize_native, &reserveSpaceSize_native);

    // Write back native variable values
    // handle is read-only
    // rnnDesc is read-only
    // fMode is primitive
    // xDesc is read-only
    if (!set(env, workSpaceSize, 0, (jlong)workSpaceSize_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, reserveSpaceSize, 0, (jlong)reserveSpaceSize_native)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetRNNParamsSizeNative(JNIEnv *env, jclass cls, jobject handle, jobject rnnDesc, jobject xDesc, jlongArray sizeInBytes, jint dataType)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetRNNParamsSize(handle=%p, rnnDesc=%p, xDesc=%p, sizeInBytes=%p, dataType=%d)\n",
        handle, rnnDesc, xDesc, sizeInBytes, dataType);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnRNNDescriptor_t rnnDesc_native;
    cudnnTensorDescriptor_t xDesc_native;
    size_t sizeInBytes_native;
    cudnnDataType_t dataType_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    rnnDesc_native = (cudnnRNNDescriptor_t)getNativePointerValue(env, rnnDesc);
    xDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, xDesc);
    // sizeInBytes is write-only
    dataType_native = (cudnnDataType_t)dataType;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetRNNParamsSize(handle_native, rnnDesc_native, xDesc_native, &sizeInBytes_native, dataType_native);

    // Write back native variable values
    // handle is read-only
    // rnnDesc is read-only
    // xDesc is read-only
    if (!set(env, sizeInBytes, 0, (jlong)sizeInBytes_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // dataType is primitive

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetRNNWeightSpaceSizeNative(JNIEnv *env, jclass cls, jobject handle, jobject rnnDesc, jlongArray weightSpaceSize)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetRNNWeightSpaceSize(handle=%p, rnnDesc=%p, weightSpaceSize=%p)\n",
        handle, rnnDesc, weightSpaceSize);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnRNNDescriptor_t rnnDesc_native;
    size_t weightSpaceSize_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    rnnDesc_native = (cudnnRNNDescriptor_t)getNativePointerValue(env, rnnDesc);
    // weightSpaceSize is write-only

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetRNNWeightSpaceSize(handle_native, rnnDesc_native, &weightSpaceSize_native);

    // Write back native variable values
    // handle is read-only
    // rnnDesc is read-only
    if (!set(env, weightSpaceSize, 0, (jlong)weightSpaceSize_native)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetRNNLinLayerMatrixParamsNative(JNIEnv *env, jclass cls, jobject handle, jobject rnnDesc, jint pseudoLayer, jobject xDesc, jobject wDesc, jobject w, jint linLayerID, jobject linLayerMatDesc, jobject linLayerMat)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetRNNLinLayerMatrixParams(handle=%p, rnnDesc=%p, pseudoLayer=%d, xDesc=%p, wDesc=%p, w=%p, linLayerID=%d, linLayerMatDesc=%p, linLayerMat=%p)\n",
        handle, rnnDesc, pseudoLayer, xDesc, wDesc, w, linLayerID, linLayerMatDesc, linLayerMat);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnRNNDescriptor_t rnnDesc_native;
    int pseudoLayer_native = 0;
    cudnnTensorDescriptor_t xDesc_native;
    cudnnFilterDescriptor_t wDesc_native;
    void * w_native = NULL;
    int linLayerID_native = 0;
    cudnnFilterDescriptor_t linLayerMatDesc_native;
    void * linLayerMat_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    rnnDesc_native = (cudnnRNNDescriptor_t)getNativePointerValue(env, rnnDesc);
    pseudoLayer_native = (int)pseudoLayer;
    xDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, xDesc);
    wDesc_native = (cudnnFilterDescriptor_t)getNativePointerValue(env, wDesc);
    w_native = (void *)getPointer(env, w);
    linLayerID_native = (int)linLayerID;
    linLayerMatDesc_native = (cudnnFilterDescriptor_t)getNativePointerValue(env, linLayerMatDesc);
    // linLayerMat is write-only

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetRNNLinLayerMatrixParams(handle_native, rnnDesc_native, pseudoLayer_native, xDesc_native, wDesc_native, w_native, linLayerID_native, linLayerMatDesc_native, &linLayerMat_native);

    // Write back native variable values
    // handle is read-only
    // rnnDesc is read-only
    // pseudoLayer is primitive
    // xDesc is read-only
    // wDesc is read-only
    // w is a native pointer
    // linLayerID is primitive
    // linLayerMatDesc is read-only
    setNativePointerValue(env, linLayerMat, (jlong)linLayerMat_native);

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetRNNLinLayerBiasParamsNative(JNIEnv *env, jclass cls, jobject handle, jobject rnnDesc, jint pseudoLayer, jobject xDesc, jobject wDesc, jobject w, jint linLayerID, jobject linLayerBiasDesc, jobject linLayerBias)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetRNNLinLayerBiasParams(handle=%p, rnnDesc=%p, pseudoLayer=%d, xDesc=%p, wDesc=%p, w=%p, linLayerID=%d, linLayerBiasDesc=%p, linLayerBias=%p)\n",
        handle, rnnDesc, pseudoLayer, xDesc, wDesc, w, linLayerID, linLayerBiasDesc, linLayerBias);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnRNNDescriptor_t rnnDesc_native;
    int pseudoLayer_native = 0;
    cudnnTensorDescriptor_t xDesc_native;
    cudnnFilterDescriptor_t wDesc_native;
    void * w_native = NULL;
    int linLayerID_native = 0;
    cudnnFilterDescriptor_t linLayerBiasDesc_native;
    void * linLayerBias_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    rnnDesc_native = (cudnnRNNDescriptor_t)getNativePointerValue(env, rnnDesc);
    pseudoLayer_native = (int)pseudoLayer;
    xDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, xDesc);
    wDesc_native = (cudnnFilterDescriptor_t)getNativePointerValue(env, wDesc);
    w_native = (void *)getPointer(env, w);
    linLayerID_native = (int)linLayerID;
    linLayerBiasDesc_native = (cudnnFilterDescriptor_t)getNativePointerValue(env, linLayerBiasDesc);
    // linLayerBias is write-only

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetRNNLinLayerBiasParams(handle_native, rnnDesc_native, pseudoLayer_native, xDesc_native, wDesc_native, w_native, linLayerID_native, linLayerBiasDesc_native, &linLayerBias_native);

    // Write back native variable values
    // handle is read-only
    // rnnDesc is read-only
    // pseudoLayer is primitive
    // xDesc is read-only
    // wDesc is read-only
    // w is a native pointer
    // linLayerID is primitive
    // linLayerBiasDesc is read-only
    setNativePointerValue(env, linLayerBias, (jlong)linLayerBias_native);

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetRNNWeightParamsNative(JNIEnv *env, jclass cls, jobject handle, jobject rnnDesc, jint pseudoLayer, jlong weightSpaceSize, jobject weightSpace, jint linLayerID, jobject mDesc, jobject mAddr, jobject bDesc, jobject bAddr)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetRNNWeightParams(handle=%p, rnnDesc=%p, pseudoLayer=%d, weightSpaceSize=%ld, weightSpace=%p, linLayerID=%d, mDesc=%p, mAddr=%p, bDesc=%p, bAddr=%p)\n",
        handle, rnnDesc, pseudoLayer, weightSpaceSize, weightSpace, linLayerID, mDesc, mAddr, bDesc, bAddr);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnRNNDescriptor_t rnnDesc_native;
    int32_t pseudoLayer_native = 0;
    size_t weightSpaceSize_native = 0;
    void * weightSpace_native = NULL;
    int32_t linLayerID_native = 0;
    cudnnTensorDescriptor_t mDesc_native;
    void * mAddr_native;
    cudnnTensorDescriptor_t bDesc_native;
    void * bAddr_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    rnnDesc_native = (cudnnRNNDescriptor_t)getNativePointerValue(env, rnnDesc);
    pseudoLayer_native = (int32_t)pseudoLayer;
    weightSpaceSize_native = (size_t)weightSpaceSize;
    weightSpace_native = (void *)getPointer(env, weightSpace);
    linLayerID_native = (int32_t)linLayerID;
    mDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, mDesc);
    // mAddr is write-only
    bDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, bDesc);
    // bAddr is write-only

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetRNNWeightParams(handle_native, rnnDesc_native, pseudoLayer_native, weightSpaceSize_native, weightSpace_native, linLayerID_native, mDesc_native, &mAddr_native, bDesc_native, &bAddr_native);

    // Write back native variable values
    // handle is read-only
    // rnnDesc is read-only
    // pseudoLayer is primitive
    // weightSpaceSize is primitive
    // weightSpace is a native pointer
    // linLayerID is primitive
    // mDesc is read-only
    setNativePointerValue(env, mAddr, (jlong)mAddr_native);
    // bDesc is read-only
    setNativePointerValue(env, bAddr, (jlong)bAddr_native);

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnRNNForwardInferenceNative(JNIEnv *env, jclass cls, jobject handle, jobject rnnDesc, jint seqLength, jobjectArray xDesc, jobject x, jobject hxDesc, jobject hx, jobject cxDesc, jobject cx, jobject wDesc, jobject w, jobjectArray yDesc, jobject y, jobject hyDesc, jobject hy, jobject cyDesc, jobject cy, jobject workSpace, jlong workSpaceSizeInBytes)
{
    // Null-checks for non-primitive arguments
    if (workSpace == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'workSpace' is null for cudnnRNNForwardInference");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnRNNForwardInference(handle=%p, rnnDesc=%p, seqLength=%d, xDesc=%p, x=%p, hxDesc=%p, hx=%p, cxDesc=%p, cx=%p, wDesc=%p, w=%p, yDesc=%p, y=%p, hyDesc=%p, hy=%p, cyDesc=%p, cy=%p, workSpace=%p, workSpaceSizeInBytes=%ld)\n",
        handle, rnnDesc, seqLength, xDesc, x, hxDesc, hx, cxDesc, cx, wDesc, w, yDesc, y, hyDesc, hy, cyDesc, cy, workSpace, workSpaceSizeInBytes);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnRNNDescriptor_t rnnDesc_native;
    int seqLength_native = 0;
    cudnnTensorDescriptor_t * xDesc_native;
    void * x_native = NULL;
    cudnnTensorDescriptor_t hxDesc_native;
    void * hx_native = NULL;
    cudnnTensorDescriptor_t cxDesc_native;
    void * cx_native = NULL;
    cudnnFilterDescriptor_t wDesc_native;
    void * w_native = NULL;
    cudnnTensorDescriptor_t * yDesc_native;
    void * y_native = NULL;
    cudnnTensorDescriptor_t hyDesc_native;
    void * hy_native = NULL;
    cudnnTensorDescriptor_t cyDesc_native;
    void * cy_native = NULL;
    void * workSpace_native = NULL;
    size_t workSpaceSizeInBytes_native = 0;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    rnnDesc_native = (cudnnRNNDescriptor_t)getNativePointerValue(env, rnnDesc);
    seqLength_native = (int)seqLength;
    if (!initNative(env, xDesc, xDesc_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    x_native = (void *)getPointer(env, x);
    hxDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, hxDesc);
    hx_native = (void *)getPointer(env, hx);
    cxDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, cxDesc);
    cx_native = (void *)getPointer(env, cx);
    wDesc_native = (cudnnFilterDescriptor_t)getNativePointerValue(env, wDesc);
    w_native = (void *)getPointer(env, w);
    if (!initNative(env, yDesc, yDesc_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    y_native = (void *)getPointer(env, y);
    hyDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, hyDesc);
    hy_native = (void *)getPointer(env, hy);
    cyDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, cyDesc);
    cy_native = (void *)getPointer(env, cy);
    workSpace_native = (void *)getPointer(env, workSpace);
    workSpaceSizeInBytes_native = (size_t)workSpaceSizeInBytes;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnRNNForwardInference(handle_native, rnnDesc_native, seqLength_native, xDesc_native, x_native, hxDesc_native, hx_native, cxDesc_native, cx_native, wDesc_native, w_native, yDesc_native, y_native, hyDesc_native, hy_native, cyDesc_native, cy_native, workSpace_native, workSpaceSizeInBytes_native);

    // Write back native variable values
    // handle is read-only
    // rnnDesc is read-only
    // seqLength is primitive
    if (!releaseNative(env, xDesc_native, xDesc, false)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // x is a native pointer
    // hxDesc is read-only
    // hx is a native pointer
    // cxDesc is read-only
    // cx is a native pointer
    // wDesc is read-only
    // w is a native pointer
    if (!releaseNative(env, yDesc_native, yDesc, false)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // y is a native pointer
    // hyDesc is read-only
    // hy is a native pointer
    // cyDesc is read-only
    // cy is a native pointer
    // workSpace is a native pointer
    // workSpaceSizeInBytes is primitive

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

/** RNN EX API */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnSetRNNPaddingModeNative(JNIEnv *env, jclass cls, jobject rnnDesc, jint paddingMode)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnSetRNNPaddingMode(rnnDesc=%p, paddingMode=%d)\n",
        rnnDesc, paddingMode);

    // Native variable declarations
    cudnnRNNDescriptor_t rnnDesc_native;
    unsigned int paddingMode_native;

    // Obtain native variable values
    rnnDesc_native = (cudnnRNNDescriptor_t)getNativePointerValue(env, rnnDesc);
    paddingMode_native = (unsigned int)paddingMode;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnSetRNNPaddingMode(rnnDesc_native, paddingMode_native);

    // Write back native variable values
    // rnnDesc is read-only
    // paddingMode is primitive

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetRNNPaddingModeNative(JNIEnv *env, jclass cls, jobject rnnDesc, jintArray paddingMode)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetRNNPaddingMode(rnnDesc=%p, paddingMode=%p)\n",
        rnnDesc, paddingMode);

    // Native variable declarations
    cudnnRNNDescriptor_t rnnDesc_native;
    unsigned int paddingMode_native;

    // Obtain native variable values
    rnnDesc_native = (cudnnRNNDescriptor_t)getNativePointerValue(env, rnnDesc);
    // paddingMode is write-only

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetRNNPaddingMode(rnnDesc_native, &paddingMode_native);

    // Write back native variable values
    // rnnDesc is read-only
    if (!set(env, paddingMode, 0, (jint)paddingMode_native)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnCreateRNNDataDescriptorNative(JNIEnv *env, jclass cls, jobject rnnDataDesc)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnCreateRNNDataDescriptor(rnnDataDesc=%p)\n",
        rnnDataDesc);

    // Native variable declarations
    cudnnRNNDataDescriptor_t rnnDataDesc_native;

    // Obtain native variable values
    // rnnDataDesc is write-only

    // Native function call
    cudnnStatus_t jniResult_native = cudnnCreateRNNDataDescriptor(&rnnDataDesc_native);

    // Write back native variable values
    setNativePointerValue(env, rnnDataDesc, (jlong)rnnDataDesc_native);

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnDestroyRNNDataDescriptorNative(JNIEnv *env, jclass cls, jobject rnnDataDesc)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnDestroyRNNDataDescriptor(rnnDataDesc=%p)\n",
        rnnDataDesc);

    // Native variable declarations
    cudnnRNNDataDescriptor_t rnnDataDesc_native;

    // Obtain native variable values
    rnnDataDesc_native = (cudnnRNNDataDescriptor_t)getNativePointerValue(env, rnnDataDesc);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnDestroyRNNDataDescriptor(rnnDataDesc_native);

    // Write back native variable values
    // rnnDataDesc is read-only

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnSetRNNDataDescriptorNative(JNIEnv *env, jclass cls, jobject rnnDataDesc, jint dataType, jint layout, jint maxSeqLength, jint batchSize, jint vectorSize, jintArray seqLengthArray, jobject paddingFill)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnSetRNNDataDescriptor(rnnDataDesc=%p, dataType=%d, layout=%d, maxSeqLength=%d, batchSize=%d, vectorSize=%d, seqLengthArray=%p, paddingFill=%p)\n",
        rnnDataDesc, dataType, layout, maxSeqLength, batchSize, vectorSize, seqLengthArray, paddingFill);

    // Native variable declarations
    cudnnRNNDataDescriptor_t rnnDataDesc_native;
    cudnnDataType_t dataType_native;
    cudnnRNNDataLayout_t layout_native;
    int maxSeqLength_native = 0;
    int batchSize_native = 0;
    int vectorSize_native = 0;
    int * seqLengthArray_native = NULL;
    void * paddingFill_native = NULL;

    // Obtain native variable values
    rnnDataDesc_native = (cudnnRNNDataDescriptor_t)getNativePointerValue(env, rnnDataDesc);
    dataType_native = (cudnnDataType_t)dataType;
    layout_native = (cudnnRNNDataLayout_t)layout;
    maxSeqLength_native = (int)maxSeqLength;
    batchSize_native = (int)batchSize;
    vectorSize_native = (int)vectorSize;
    if (!initNative(env, seqLengthArray, seqLengthArray_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    paddingFill_native = (void *)getPointer(env, paddingFill);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnSetRNNDataDescriptor(rnnDataDesc_native, dataType_native, layout_native, maxSeqLength_native, batchSize_native, vectorSize_native, seqLengthArray_native, paddingFill_native);

    // Write back native variable values
    // rnnDataDesc is read-only
    // dataType is primitive
    // layout is primitive
    // maxSeqLength is primitive
    // batchSize is primitive
    // vectorSize is primitive
    if (!releaseNative(env, seqLengthArray_native, seqLengthArray, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // paddingFill is a native pointer

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetRNNDataDescriptorNative(JNIEnv *env, jclass cls, jobject rnnDataDesc, jintArray dataType, jintArray layout, jintArray maxSeqLength, jintArray batchSize, jintArray vectorSize, jint arrayLengthRequested, jintArray seqLengthArray, jobject paddingFill)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetRNNDataDescriptor(rnnDataDesc=%p, dataType=%p, layout=%p, maxSeqLength=%p, batchSize=%p, vectorSize=%p, arrayLengthRequested=%d, seqLengthArray=%p, paddingFill=%p)\n",
        rnnDataDesc, dataType, layout, maxSeqLength, batchSize, vectorSize, arrayLengthRequested, seqLengthArray, paddingFill);

    // Native variable declarations
    cudnnRNNDataDescriptor_t rnnDataDesc_native;
    cudnnDataType_t dataType_native;
    cudnnRNNDataLayout_t layout_native;
    int maxSeqLength_native;
    int batchSize_native;
    int vectorSize_native;
    int arrayLengthRequested_native = 0;
    int * seqLengthArray_native = NULL;
    void * paddingFill_native = NULL;

    // Obtain native variable values
    rnnDataDesc_native = (cudnnRNNDataDescriptor_t)getNativePointerValue(env, rnnDataDesc);
    // dataType is write-only
    // layout is write-only
    // maxSeqLength is write-only
    // batchSize is write-only
    // vectorSize is write-only
    arrayLengthRequested_native = (int)arrayLengthRequested;
    if (!initNative(env, seqLengthArray, seqLengthArray_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    paddingFill_native = (void *)getPointer(env, paddingFill);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetRNNDataDescriptor(rnnDataDesc_native, &dataType_native, &layout_native, &maxSeqLength_native, &batchSize_native, &vectorSize_native, arrayLengthRequested_native, seqLengthArray_native, paddingFill_native);

    // Write back native variable values
    // rnnDataDesc is read-only
    if (!set(env, dataType, 0, (jint)dataType_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, layout, 0, (jint)layout_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, maxSeqLength, 0, (jint)maxSeqLength_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, batchSize, 0, (jint)batchSize_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, vectorSize, 0, (jint)vectorSize_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // arrayLengthRequested is primitive
    if (!releaseNative(env, seqLengthArray_native, seqLengthArray, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // paddingFill is a native pointer

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnRNNForwardInferenceExNative(JNIEnv *env, jclass cls, jobject handle, jobject rnnDesc, jobject xDesc, jobject x, jobject hxDesc, jobject hx, jobject cxDesc, jobject cx, jobject wDesc, jobject w, jobject yDesc, jobject y, jobject hyDesc, jobject hy, jobject cyDesc, jobject cy, jobject kDesc, jobject keys, jobject cDesc, jobject cAttn, jobject iDesc, jobject iAttn, jobject qDesc, jobject queries, jobject workSpace, jlong workSpaceSizeInBytes)
{
    // Null-checks for non-primitive arguments
    if (workSpace == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'workSpace' is null for cudnnRNNForwardInferenceEx");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnRNNForwardInferenceEx(handle=%p, rnnDesc=%p, xDesc=%p, x=%p, hxDesc=%p, hx=%p, cxDesc=%p, cx=%p, wDesc=%p, w=%p, yDesc=%p, y=%p, hyDesc=%p, hy=%p, cyDesc=%p, cy=%p, kDesc=%p, keys=%p, cDesc=%p, cAttn=%p, iDesc=%p, iAttn=%p, qDesc=%p, queries=%p, workSpace=%p, workSpaceSizeInBytes=%ld)\n",
        handle, rnnDesc, xDesc, x, hxDesc, hx, cxDesc, cx, wDesc, w, yDesc, y, hyDesc, hy, cyDesc, cy, kDesc, keys, cDesc, cAttn, iDesc, iAttn, qDesc, queries, workSpace, workSpaceSizeInBytes);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnRNNDescriptor_t rnnDesc_native;
    cudnnRNNDataDescriptor_t xDesc_native;
    void * x_native = NULL;
    cudnnTensorDescriptor_t hxDesc_native;
    void * hx_native = NULL;
    cudnnTensorDescriptor_t cxDesc_native;
    void * cx_native = NULL;
    cudnnFilterDescriptor_t wDesc_native;
    void * w_native = NULL;
    cudnnRNNDataDescriptor_t yDesc_native;
    void * y_native = NULL;
    cudnnTensorDescriptor_t hyDesc_native;
    void * hy_native = NULL;
    cudnnTensorDescriptor_t cyDesc_native;
    void * cy_native = NULL;
    cudnnRNNDataDescriptor_t kDesc_native;
    void * keys_native = NULL;
    cudnnRNNDataDescriptor_t cDesc_native;
    void * cAttn_native = NULL;
    cudnnRNNDataDescriptor_t iDesc_native;
    void * iAttn_native = NULL;
    cudnnRNNDataDescriptor_t qDesc_native;
    void * queries_native = NULL;
    void * workSpace_native = NULL;
    size_t workSpaceSizeInBytes_native = 0;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    rnnDesc_native = (cudnnRNNDescriptor_t)getNativePointerValue(env, rnnDesc);
    xDesc_native = (cudnnRNNDataDescriptor_t)getNativePointerValue(env, xDesc);
    x_native = (void *)getPointer(env, x);
    hxDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, hxDesc);
    hx_native = (void *)getPointer(env, hx);
    cxDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, cxDesc);
    cx_native = (void *)getPointer(env, cx);
    wDesc_native = (cudnnFilterDescriptor_t)getNativePointerValue(env, wDesc);
    w_native = (void *)getPointer(env, w);
    yDesc_native = (cudnnRNNDataDescriptor_t)getNativePointerValue(env, yDesc);
    y_native = (void *)getPointer(env, y);
    hyDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, hyDesc);
    hy_native = (void *)getPointer(env, hy);
    cyDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, cyDesc);
    cy_native = (void *)getPointer(env, cy);
    kDesc_native = (cudnnRNNDataDescriptor_t)getNativePointerValue(env, kDesc);
    keys_native = (void *)getPointer(env, keys);
    cDesc_native = (cudnnRNNDataDescriptor_t)getNativePointerValue(env, cDesc);
    cAttn_native = (void *)getPointer(env, cAttn);
    iDesc_native = (cudnnRNNDataDescriptor_t)getNativePointerValue(env, iDesc);
    iAttn_native = (void *)getPointer(env, iAttn);
    qDesc_native = (cudnnRNNDataDescriptor_t)getNativePointerValue(env, qDesc);
    queries_native = (void *)getPointer(env, queries);
    workSpace_native = (void *)getPointer(env, workSpace);
    workSpaceSizeInBytes_native = (size_t)workSpaceSizeInBytes;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnRNNForwardInferenceEx(handle_native, rnnDesc_native, xDesc_native, x_native, hxDesc_native, hx_native, cxDesc_native, cx_native, wDesc_native, w_native, yDesc_native, y_native, hyDesc_native, hy_native, cyDesc_native, cy_native, kDesc_native, keys_native, cDesc_native, cAttn_native, iDesc_native, iAttn_native, qDesc_native, queries_native, workSpace_native, workSpaceSizeInBytes_native);

    // Write back native variable values
    // handle is read-only
    // rnnDesc is read-only
    // xDesc is read-only
    // x is a native pointer
    // hxDesc is read-only
    // hx is a native pointer
    // cxDesc is read-only
    // cx is a native pointer
    // wDesc is read-only
    // w is a native pointer
    // yDesc is read-only
    // y is a native pointer
    // hyDesc is read-only
    // hy is a native pointer
    // cyDesc is read-only
    // cy is a native pointer
    // kDesc is read-only
    // keys is a native pointer
    // cDesc is read-only
    // cAttn is a native pointer
    // iDesc is read-only
    // iAttn is a native pointer
    // qDesc is read-only
    // queries is a native pointer
    // workSpace is a native pointer
    // workSpaceSizeInBytes is primitive

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnRNNForwardNative(JNIEnv *env, jclass cls, jobject handle, jobject rnnDesc, jint fwdMode, jintArray devSeqLengths, jobject xDesc, jobject x, jobject yDesc, jobject y, jobject hDesc, jobject hx, jobject hy, jobject cDesc, jobject cx, jobject cy, jlong weightSpaceSize, jobject weightSpace, jlong workSpaceSize, jobject workSpace, jlong reserveSpaceSize, jobject reserveSpace)
{
    // Null-checks for non-primitive arguments
    if (workSpace == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'workSpace' is null for cudnnRNNForward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (reserveSpace == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'reserveSpace' is null for cudnnRNNForward");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnRNNForward(handle=%p, rnnDesc=%p, fwdMode=%d, devSeqLengths=%p, xDesc=%p, x=%p, yDesc=%p, y=%p, hDesc=%p, hx=%p, hy=%p, cDesc=%p, cx=%p, cy=%p, weightSpaceSize=%ld, weightSpace=%p, workSpaceSize=%ld, workSpace=%p, reserveSpaceSize=%ld, reserveSpace=%p)\n",
        handle, rnnDesc, fwdMode, devSeqLengths, xDesc, x, yDesc, y, hDesc, hx, hy, cDesc, cx, cy, weightSpaceSize, weightSpace, workSpaceSize, workSpace, reserveSpaceSize, reserveSpace);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnRNNDescriptor_t rnnDesc_native;
    cudnnForwardMode_t fwdMode_native;
    int32_t * devSeqLengths_native = NULL;
    cudnnRNNDataDescriptor_t xDesc_native;
    void * x_native = NULL;
    cudnnRNNDataDescriptor_t yDesc_native;
    void * y_native = NULL;
    cudnnTensorDescriptor_t hDesc_native;
    void * hx_native = NULL;
    void * hy_native = NULL;
    cudnnTensorDescriptor_t cDesc_native;
    void * cx_native = NULL;
    void * cy_native = NULL;
    size_t weightSpaceSize_native = 0;
    void * weightSpace_native = NULL;
    size_t workSpaceSize_native = 0;
    void * workSpace_native = NULL;
    size_t reserveSpaceSize_native = 0;
    void * reserveSpace_native = NULL;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    rnnDesc_native = (cudnnRNNDescriptor_t)getNativePointerValue(env, rnnDesc);
    fwdMode_native = (cudnnForwardMode_t)fwdMode;
    devSeqLengths_native = (int32_t *)getPointer(env, devSeqLengths);
    xDesc_native = (cudnnRNNDataDescriptor_t)getNativePointerValue(env, xDesc);
    x_native = (void *)getPointer(env, x);
    yDesc_native = (cudnnRNNDataDescriptor_t)getNativePointerValue(env, yDesc);
    y_native = (void *)getPointer(env, y);
    hDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, hDesc);
    hx_native = (void *)getPointer(env, hx);
    hy_native = (void *)getPointer(env, hy);
    cDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, cDesc);
    cx_native = (void *)getPointer(env, cx);
    cy_native = (void *)getPointer(env, cy);
    weightSpaceSize_native = (size_t)weightSpaceSize;
    weightSpace_native = (void *)getPointer(env, weightSpace);
    workSpaceSize_native = (size_t)workSpaceSize;
    workSpace_native = (void *)getPointer(env, workSpace);
    reserveSpaceSize_native = (size_t)reserveSpaceSize;
    reserveSpace_native = (void *)getPointer(env, reserveSpace);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnRNNForward(handle_native, rnnDesc_native, fwdMode_native, devSeqLengths_native, xDesc_native, x_native, yDesc_native, y_native, hDesc_native, hx_native, hy_native, cDesc_native, cx_native, cy_native, weightSpaceSize_native, weightSpace_native, workSpaceSize_native, workSpace_native, reserveSpaceSize_native, reserveSpace_native);

    // Write back native variable values
    // handle is read-only
    // rnnDesc is read-only
    // fwdMode is primitive
    // devSeqLengths is a native pointer
    // xDesc is read-only
    // x is a native pointer
    // yDesc is read-only
    // y is a native pointer
    // hDesc is read-only
    // hx is a native pointer
    // hy is a native pointer
    // cDesc is read-only
    // cx is a native pointer
    // cy is a native pointer
    // weightSpaceSize is primitive
    // weightSpace is a native pointer
    // workSpaceSize is primitive
    // workSpace is a native pointer
    // reserveSpaceSize is primitive
    // reserveSpace is a native pointer

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

/** RNN FIND API */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnSetRNNAlgorithmDescriptorNative(JNIEnv *env, jclass cls, jobject handle, jobject rnnDesc, jobject algoDesc)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnSetRNNAlgorithmDescriptor(handle=%p, rnnDesc=%p, algoDesc=%p)\n",
        handle, rnnDesc, algoDesc);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnRNNDescriptor_t rnnDesc_native;
    cudnnAlgorithmDescriptor_t algoDesc_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    rnnDesc_native = (cudnnRNNDescriptor_t)getNativePointerValue(env, rnnDesc);
    algoDesc_native = (cudnnAlgorithmDescriptor_t)getNativePointerValue(env, algoDesc);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnSetRNNAlgorithmDescriptor(handle_native, rnnDesc_native, algoDesc_native);

    // Write back native variable values
    // handle is read-only
    // rnnDesc is read-only
    // algoDesc is read-only

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetRNNForwardInferenceAlgorithmMaxCountNative(JNIEnv *env, jclass cls, jobject handle, jobject rnnDesc, jintArray count)
{
    // Null-checks for non-primitive arguments
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cudnnGetRNNForwardInferenceAlgorithmMaxCount");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (rnnDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'rnnDesc' is null for cudnnGetRNNForwardInferenceAlgorithmMaxCount");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (count == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'count' is null for cudnnGetRNNForwardInferenceAlgorithmMaxCount");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetRNNForwardInferenceAlgorithmMaxCount(handle=%p, rnnDesc=%p, count=%p)\n",
        handle, rnnDesc, count);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnRNNDescriptor_t rnnDesc_native;
    int count_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    rnnDesc_native = (cudnnRNNDescriptor_t)getNativePointerValue(env, rnnDesc);
    // count is write-only

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetRNNForwardInferenceAlgorithmMaxCount(handle_native, rnnDesc_native, &count_native);

    // Write back native variable values
    // handle is read-only
    // rnnDesc is read-only
    if (!set(env, count, 0, (jint)count_native)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnFindRNNForwardInferenceAlgorithmExNative(JNIEnv *env, jclass cls, jobject handle, jobject rnnDesc, jint seqLength, jobjectArray xDesc, jobject x, jobject hxDesc, jobject hx, jobject cxDesc, jobject cx, jobject wDesc, jobject w, jobjectArray yDesc, jobject y, jobject hyDesc, jobject hy, jobject cyDesc, jobject cy, jfloat findIntensity, jint requestedAlgoCount, jintArray returnedAlgoCount, jobjectArray perfResults, jobject workspace, jlong workSpaceSizeInBytes)
{
    // Null-checks for non-primitive arguments
    if (perfResults == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'perfResults' is null for cudnnFindRNNForwardInferenceAlgorithmEx");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnFindRNNForwardInferenceAlgorithmEx(handle=%p, rnnDesc=%p, seqLength=%d, xDesc=%p, x=%p, hxDesc=%p, hx=%p, cxDesc=%p, cx=%p, wDesc=%p, w=%p, yDesc=%p, y=%p, hyDesc=%p, hy=%p, cyDesc=%p, cy=%p, findIntensity=%f, requestedAlgoCount=%d, returnedAlgoCount=%p, perfResults=%p, workspace=%p, workSpaceSizeInBytes=%ld)\n",
        handle, rnnDesc, seqLength, xDesc, x, hxDesc, hx, cxDesc, cx, wDesc, w, yDesc, y, hyDesc, hy, cyDesc, cy, findIntensity, requestedAlgoCount, returnedAlgoCount, perfResults, workspace, workSpaceSizeInBytes);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnRNNDescriptor_t rnnDesc_native;
    int seqLength_native = 0;
    cudnnTensorDescriptor_t * xDesc_native;
    void * x_native = NULL;
    cudnnTensorDescriptor_t hxDesc_native;
    void * hx_native = NULL;
    cudnnTensorDescriptor_t cxDesc_native;
    void * cx_native = NULL;
    cudnnFilterDescriptor_t wDesc_native;
    void * w_native = NULL;
    cudnnTensorDescriptor_t * yDesc_native;
    void * y_native = NULL;
    cudnnTensorDescriptor_t hyDesc_native;
    void * hy_native = NULL;
    cudnnTensorDescriptor_t cyDesc_native;
    void * cy_native = NULL;
    float findIntensity_native = 0.0f;
    int requestedAlgoCount_native = 0;
    int returnedAlgoCount_native;
    cudnnAlgorithmPerformance_t * perfResults_native;
    void * workspace_native = NULL;
    size_t workSpaceSizeInBytes_native = 0;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    rnnDesc_native = (cudnnRNNDescriptor_t)getNativePointerValue(env, rnnDesc);
    seqLength_native = (int)seqLength;
    if (!initNative(env, xDesc, xDesc_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    x_native = (void *)getPointer(env, x);
    hxDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, hxDesc);
    hx_native = (void *)getPointer(env, hx);
    cxDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, cxDesc);
    cx_native = (void *)getPointer(env, cx);
    wDesc_native = (cudnnFilterDescriptor_t)getNativePointerValue(env, wDesc);
    w_native = (void *)getPointer(env, w);
    if (!initNative(env, yDesc, yDesc_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    y_native = (void *)getPointer(env, y);
    hyDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, hyDesc);
    hy_native = (void *)getPointer(env, hy);
    cyDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, cyDesc);
    cy_native = (void *)getPointer(env, cy);
    findIntensity_native = (float)findIntensity;
    requestedAlgoCount_native = (int)requestedAlgoCount;
    // returnedAlgoCount is write-only
    if (!initNative(env, perfResults, perfResults_native, requestedAlgoCount)) return JCUDNN_STATUS_INTERNAL_ERROR;
    workspace_native = (void *)getPointer(env, workspace);
    workSpaceSizeInBytes_native = (size_t)workSpaceSizeInBytes;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnFindRNNForwardInferenceAlgorithmEx(handle_native, rnnDesc_native, seqLength_native, xDesc_native, x_native, hxDesc_native, hx_native, cxDesc_native, cx_native, wDesc_native, w_native, yDesc_native, y_native, hyDesc_native, hy_native, cyDesc_native, cy_native, findIntensity_native, requestedAlgoCount_native, &returnedAlgoCount_native, perfResults_native, workspace_native, workSpaceSizeInBytes_native);

    // Write back native variable values
    // handle is read-only
    // rnnDesc is read-only
    // seqLength is primitive
    if (!releaseNative(env, xDesc_native, xDesc, false)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // x is a native pointer
    // hxDesc is read-only
    // hx is a native pointer
    // cxDesc is read-only
    // cx is a native pointer
    // wDesc is read-only
    // w is a native pointer
    if (!releaseNative(env, yDesc_native, yDesc, false)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // y is a native pointer
    // hyDesc is read-only
    // hy is a native pointer
    // cyDesc is read-only
    // cy is a native pointer
    // findIntensity is primitive
    // requestedAlgoCount is primitive
    if (!set(env, returnedAlgoCount, 0, (jint)returnedAlgoCount_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!releaseNative(env, perfResults_native, perfResults, returnedAlgoCount_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // workspace is a native pointer
    // workSpaceSizeInBytes is primitive

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnCreateSeqDataDescriptorNative(JNIEnv *env, jclass cls, jobject seqDataDesc)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnCreateSeqDataDescriptor(seqDataDesc=%p)\n",
        seqDataDesc);

    // Native variable declarations
    cudnnSeqDataDescriptor_t seqDataDesc_native;

    // Obtain native variable values
    // seqDataDesc is write-only

    // Native function call
    cudnnStatus_t jniResult_native = cudnnCreateSeqDataDescriptor(&seqDataDesc_native);

    // Write back native variable values
    setNativePointerValue(env, seqDataDesc, (jlong)seqDataDesc_native);

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnDestroySeqDataDescriptorNative(JNIEnv *env, jclass cls, jobject seqDataDesc)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnDestroySeqDataDescriptor(seqDataDesc=%p)\n",
        seqDataDesc);

    // Native variable declarations
    cudnnSeqDataDescriptor_t seqDataDesc_native;

    // Obtain native variable values
    seqDataDesc_native = (cudnnSeqDataDescriptor_t)getNativePointerValue(env, seqDataDesc);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnDestroySeqDataDescriptor(seqDataDesc_native);

    // Write back native variable values
    // seqDataDesc is read-only

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnSetSeqDataDescriptorNative(JNIEnv *env, jclass cls, jobject seqDataDesc, jint dataType, jint nbDims, jintArray dimA, jintArray axes, jlong seqLengthArraySize, jintArray seqLengthArray, jobject paddingFill)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnSetSeqDataDescriptor(seqDataDesc=%p, dataType=%d, nbDims=%d, dimA=%p, axes=%p, seqLengthArraySize=%ld, seqLengthArray=%p, paddingFill=%p)\n",
        seqDataDesc, dataType, nbDims, dimA, axes, seqLengthArraySize, seqLengthArray, paddingFill);

    // Native variable declarations
    cudnnSeqDataDescriptor_t seqDataDesc_native;
    cudnnDataType_t dataType_native;
    int nbDims_native = 0;
    int * dimA_native = NULL;
    cudnnSeqDataAxis_t * axes_native = NULL;
    size_t seqLengthArraySize_native = 0;
    int * seqLengthArray_native = NULL;
    void * paddingFill_native = NULL;

    // Obtain native variable values
    seqDataDesc_native = (cudnnSeqDataDescriptor_t)getNativePointerValue(env, seqDataDesc);
    dataType_native = (cudnnDataType_t)dataType;
    nbDims_native = (int)nbDims;
    if (!initNative(env, dimA, dimA_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!initNative(env, axes, axes_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    seqLengthArraySize_native = (size_t)seqLengthArraySize;
    if (!initNative(env, seqLengthArray, seqLengthArray_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    paddingFill_native = (void *)getPointer(env, paddingFill);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnSetSeqDataDescriptor(seqDataDesc_native, dataType_native, nbDims_native, dimA_native, axes_native, seqLengthArraySize_native, seqLengthArray_native, paddingFill_native);

    // Write back native variable values
    // seqDataDesc is read-only
    // dataType is primitive
    // nbDims is primitive
    if (!releaseNative(env, dimA_native, dimA, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!releaseNative(env, axes_native, axes, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // seqLengthArraySize is primitive
    if (!releaseNative(env, seqLengthArray_native, seqLengthArray, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // paddingFill is a native pointer

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetSeqDataDescriptorNative(JNIEnv *env, jclass cls, jobject seqDataDesc, jintArray dataType, jintArray nbDims, jint nbDimsRequested, jintArray dimA, jintArray axes, jlongArray seqLengthArraySize, jlong seqLengthSizeRequested, jintArray seqLengthArray, jobject paddingFill)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetSeqDataDescriptor(seqDataDesc=%p, dataType=%p, nbDims=%p, nbDimsRequested=%d, dimA=%p, axes=%p, seqLengthArraySize=%p, seqLengthSizeRequested=%ld, seqLengthArray=%p, paddingFill=%p)\n",
        seqDataDesc, dataType, nbDims, nbDimsRequested, dimA, axes, seqLengthArraySize, seqLengthSizeRequested, seqLengthArray, paddingFill);

    // Native variable declarations
    cudnnSeqDataDescriptor_t seqDataDesc_native;
    cudnnDataType_t dataType_native;
    int nbDims_native;
    int nbDimsRequested_native = 0;
    int * dimA_native = NULL;
    cudnnSeqDataAxis_t * axes_native = NULL;
    size_t seqLengthArraySize_native;
    size_t seqLengthSizeRequested_native = 0;
    int * seqLengthArray_native = NULL;
    void * paddingFill_native = NULL;

    // Obtain native variable values
    seqDataDesc_native = (cudnnSeqDataDescriptor_t)getNativePointerValue(env, seqDataDesc);
    // dataType is write-only
    // nbDims is write-only
    nbDimsRequested_native = (int)nbDimsRequested;
    if (!initNative(env, dimA, dimA_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!initNative(env, axes, axes_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // seqLengthArraySize is write-only
    seqLengthSizeRequested_native = (size_t)seqLengthSizeRequested;
    if (!initNative(env, seqLengthArray, seqLengthArray_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    paddingFill_native = (void *)getPointer(env, paddingFill);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetSeqDataDescriptor(seqDataDesc_native, &dataType_native, &nbDims_native, nbDimsRequested_native, dimA_native, axes_native, &seqLengthArraySize_native, seqLengthSizeRequested_native, seqLengthArray_native, paddingFill_native);

    // Write back native variable values
    // seqDataDesc is read-only
    if (!set(env, dataType, 0, (jint)dataType_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, nbDims, 0, (jint)nbDims_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // nbDimsRequested is primitive
    if (!releaseNative(env, dimA_native, dimA, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!releaseNative(env, axes_native, axes, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, seqLengthArraySize, 0, (jlong)seqLengthArraySize_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // seqLengthSizeRequested is primitive
    if (!releaseNative(env, seqLengthArray_native, seqLengthArray, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // paddingFill is a native pointer

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnCreateAttnDescriptorNative(JNIEnv *env, jclass cls, jobject attnDesc)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnCreateAttnDescriptor(attnDesc=%p)\n",
        attnDesc);

    // Native variable declarations
    cudnnAttnDescriptor_t attnDesc_native;

    // Obtain native variable values
    // attnDesc is write-only

    // Native function call
    cudnnStatus_t jniResult_native = cudnnCreateAttnDescriptor(&attnDesc_native);

    // Write back native variable values
    setNativePointerValue(env, attnDesc, (jlong)attnDesc_native);

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnDestroyAttnDescriptorNative(JNIEnv *env, jclass cls, jobject attnDesc)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnDestroyAttnDescriptor(attnDesc=%p)\n",
        attnDesc);

    // Native variable declarations
    cudnnAttnDescriptor_t attnDesc_native;

    // Obtain native variable values
    attnDesc_native = (cudnnAttnDescriptor_t)getNativePointerValue(env, attnDesc);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnDestroyAttnDescriptor(attnDesc_native);

    // Write back native variable values
    // attnDesc is read-only

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnSetAttnDescriptorNative(JNIEnv *env, jclass cls, jobject attnDesc, jint attnMode, jint nHeads, jdouble smScaler, jint dataType, jint computePrec, jint mathType, jobject attnDropoutDesc, jobject postDropoutDesc, jint qSize, jint kSize, jint vSize, jint qProjSize, jint kProjSize, jint vProjSize, jint oProjSize, jint qoMaxSeqLength, jint kvMaxSeqLength, jint maxBatchSize, jint maxBeamSize)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnSetAttnDescriptor(attnDesc=%p, attnMode=%d, nHeads=%d, smScaler=%lf, dataType=%d, computePrec=%d, mathType=%d, attnDropoutDesc=%p, postDropoutDesc=%p, qSize=%d, kSize=%d, vSize=%d, qProjSize=%d, kProjSize=%d, vProjSize=%d, oProjSize=%d, qoMaxSeqLength=%d, kvMaxSeqLength=%d, maxBatchSize=%d, maxBeamSize=%d)\n",
        attnDesc, attnMode, nHeads, smScaler, dataType, computePrec, mathType, attnDropoutDesc, postDropoutDesc, qSize, kSize, vSize, qProjSize, kProjSize, vProjSize, oProjSize, qoMaxSeqLength, kvMaxSeqLength, maxBatchSize, maxBeamSize);

    // Native variable declarations
    cudnnAttnDescriptor_t attnDesc_native;
    unsigned int attnMode_native;
    int nHeads_native = 0;
    double smScaler_native = 0.0;
    cudnnDataType_t dataType_native;
    cudnnDataType_t computePrec_native;
    cudnnMathType_t mathType_native;
    cudnnDropoutDescriptor_t attnDropoutDesc_native;
    cudnnDropoutDescriptor_t postDropoutDesc_native;
    int qSize_native = 0;
    int kSize_native = 0;
    int vSize_native = 0;
    int qProjSize_native = 0;
    int kProjSize_native = 0;
    int vProjSize_native = 0;
    int oProjSize_native = 0;
    int qoMaxSeqLength_native = 0;
    int kvMaxSeqLength_native = 0;
    int maxBatchSize_native = 0;
    int maxBeamSize_native = 0;

    // Obtain native variable values
    attnDesc_native = (cudnnAttnDescriptor_t)getNativePointerValue(env, attnDesc);
    attnMode_native = (unsigned int)attnMode;
    nHeads_native = (int)nHeads;
    smScaler_native = (double)smScaler;
    dataType_native = (cudnnDataType_t)dataType;
    computePrec_native = (cudnnDataType_t)computePrec;
    mathType_native = (cudnnMathType_t)mathType;
    attnDropoutDesc_native = (cudnnDropoutDescriptor_t)getNativePointerValue(env, attnDropoutDesc);
    postDropoutDesc_native = (cudnnDropoutDescriptor_t)getNativePointerValue(env, postDropoutDesc);
    qSize_native = (int)qSize;
    kSize_native = (int)kSize;
    vSize_native = (int)vSize;
    qProjSize_native = (int)qProjSize;
    kProjSize_native = (int)kProjSize;
    vProjSize_native = (int)vProjSize;
    oProjSize_native = (int)oProjSize;
    qoMaxSeqLength_native = (int)qoMaxSeqLength;
    kvMaxSeqLength_native = (int)kvMaxSeqLength;
    maxBatchSize_native = (int)maxBatchSize;
    maxBeamSize_native = (int)maxBeamSize;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnSetAttnDescriptor(attnDesc_native, attnMode_native, nHeads_native, smScaler_native, dataType_native, computePrec_native, mathType_native, attnDropoutDesc_native, postDropoutDesc_native, qSize_native, kSize_native, vSize_native, qProjSize_native, kProjSize_native, vProjSize_native, oProjSize_native, qoMaxSeqLength_native, kvMaxSeqLength_native, maxBatchSize_native, maxBeamSize_native);

    // Write back native variable values
    // attnDesc is read-only
    // attnMode is primitive
    // nHeads is primitive
    // smScaler is primitive
    // dataType is primitive
    // computePrec is primitive
    // mathType is primitive
    // attnDropoutDesc is read-only
    // postDropoutDesc is read-only
    // qSize is primitive
    // kSize is primitive
    // vSize is primitive
    // qProjSize is primitive
    // kProjSize is primitive
    // vProjSize is primitive
    // oProjSize is primitive
    // qoMaxSeqLength is primitive
    // kvMaxSeqLength is primitive
    // maxBatchSize is primitive
    // maxBeamSize is primitive

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetAttnDescriptorNative(JNIEnv *env, jclass cls, jobject attnDesc, jintArray attnMode, jintArray nHeads, jdoubleArray smScaler, jintArray dataType, jintArray computePrec, jintArray mathType, jobject attnDropoutDesc, jobject postDropoutDesc, jintArray qSize, jintArray kSize, jintArray vSize, jintArray qProjSize, jintArray kProjSize, jintArray vProjSize, jintArray oProjSize, jintArray qoMaxSeqLength, jintArray kvMaxSeqLength, jintArray maxBatchSize, jintArray maxBeamSize)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetAttnDescriptor(attnDesc=%p, attnMode=%p, nHeads=%p, smScaler=%p, dataType=%p, computePrec=%p, mathType=%p, attnDropoutDesc=%p, postDropoutDesc=%p, qSize=%p, kSize=%p, vSize=%p, qProjSize=%p, kProjSize=%p, vProjSize=%p, oProjSize=%p, qoMaxSeqLength=%p, kvMaxSeqLength=%p, maxBatchSize=%p, maxBeamSize=%p)\n",
        attnDesc, attnMode, nHeads, smScaler, dataType, computePrec, mathType, attnDropoutDesc, postDropoutDesc, qSize, kSize, vSize, qProjSize, kProjSize, vProjSize, oProjSize, qoMaxSeqLength, kvMaxSeqLength, maxBatchSize, maxBeamSize);

    // Native variable declarations
    cudnnAttnDescriptor_t attnDesc_native;
    unsigned int attnMode_native;
    int nHeads_native;
    double smScaler_native;
    cudnnDataType_t dataType_native;
    cudnnDataType_t computePrec_native;
    cudnnMathType_t mathType_native;
    cudnnDropoutDescriptor_t * attnDropoutDesc_native;
    cudnnDropoutDescriptor_t * postDropoutDesc_native;
    int qSize_native;
    int kSize_native;
    int vSize_native;
    int qProjSize_native;
    int kProjSize_native;
    int vProjSize_native;
    int oProjSize_native;
    int qoMaxSeqLength_native;
    int kvMaxSeqLength_native;
    int maxBatchSize_native;
    int maxBeamSize_native;

    // Obtain native variable values
    attnDesc_native = (cudnnAttnDescriptor_t)getNativePointerValue(env, attnDesc);
    // attnMode is write-only
    // nHeads is write-only
    // smScaler is write-only
    // dataType is write-only
    // computePrec is write-only
    // mathType is write-only
    attnDropoutDesc_native = (cudnnDropoutDescriptor_t *)getNativePointerValue(env, attnDropoutDesc);
    postDropoutDesc_native = (cudnnDropoutDescriptor_t *)getNativePointerValue(env, postDropoutDesc);
    // qSize is write-only
    // kSize is write-only
    // vSize is write-only
    // qProjSize is write-only
    // kProjSize is write-only
    // vProjSize is write-only
    // oProjSize is write-only
    // qoMaxSeqLength is write-only
    // kvMaxSeqLength is write-only
    // maxBatchSize is write-only
    // maxBeamSize is write-only

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetAttnDescriptor(attnDesc_native, &attnMode_native, &nHeads_native, &smScaler_native, &dataType_native, &computePrec_native, &mathType_native, attnDropoutDesc_native, postDropoutDesc_native, &qSize_native, &kSize_native, &vSize_native, &qProjSize_native, &kProjSize_native, &vProjSize_native, &oProjSize_native, &qoMaxSeqLength_native, &kvMaxSeqLength_native, &maxBatchSize_native, &maxBeamSize_native);

    // Write back native variable values
    // attnDesc is read-only
    if (!set(env, attnMode, 0, (jint)attnMode_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, nHeads, 0, (jint)nHeads_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, smScaler, 0, (jdouble)smScaler_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, dataType, 0, (jint)dataType_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, computePrec, 0, (jint)computePrec_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, mathType, 0, (jint)mathType_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // attnDropoutDesc is read-only
    // postDropoutDesc is read-only
    if (!set(env, qSize, 0, (jint)qSize_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, kSize, 0, (jint)kSize_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, vSize, 0, (jint)vSize_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, qProjSize, 0, (jint)qProjSize_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, kProjSize, 0, (jint)kProjSize_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, vProjSize, 0, (jint)vProjSize_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, oProjSize, 0, (jint)oProjSize_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, qoMaxSeqLength, 0, (jint)qoMaxSeqLength_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, kvMaxSeqLength, 0, (jint)kvMaxSeqLength_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, maxBatchSize, 0, (jint)maxBatchSize_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, maxBeamSize, 0, (jint)maxBeamSize_native)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetMultiHeadAttnBuffersNative(JNIEnv *env, jclass cls, jobject handle, jobject attnDesc, jlongArray weightSizeInBytes, jlongArray workSpaceSizeInBytes, jlongArray reserveSpaceSizeInBytes)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetMultiHeadAttnBuffers(handle=%p, attnDesc=%p, weightSizeInBytes=%p, workSpaceSizeInBytes=%p, reserveSpaceSizeInBytes=%p)\n",
        handle, attnDesc, weightSizeInBytes, workSpaceSizeInBytes, reserveSpaceSizeInBytes);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnAttnDescriptor_t attnDesc_native;
    size_t weightSizeInBytes_native;
    size_t workSpaceSizeInBytes_native;
    size_t reserveSpaceSizeInBytes_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    attnDesc_native = (cudnnAttnDescriptor_t)getNativePointerValue(env, attnDesc);
    // weightSizeInBytes is write-only
    // workSpaceSizeInBytes is write-only
    // reserveSpaceSizeInBytes is write-only

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetMultiHeadAttnBuffers(handle_native, attnDesc_native, &weightSizeInBytes_native, &workSpaceSizeInBytes_native, &reserveSpaceSizeInBytes_native);

    // Write back native variable values
    // handle is read-only
    // attnDesc is read-only
    if (!set(env, weightSizeInBytes, 0, (jlong)weightSizeInBytes_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, workSpaceSizeInBytes, 0, (jlong)workSpaceSizeInBytes_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, reserveSpaceSizeInBytes, 0, (jlong)reserveSpaceSizeInBytes_native)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetMultiHeadAttnWeightsNative(JNIEnv *env, jclass cls, jobject handle, jobject attnDesc, jint wKind, jlong weightSizeInBytes, jobject weights, jobject wDesc, jobject wAddr)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetMultiHeadAttnWeights(handle=%p, attnDesc=%p, wKind=%d, weightSizeInBytes=%ld, weights=%p, wDesc=%p, wAddr=%p)\n",
        handle, attnDesc, wKind, weightSizeInBytes, weights, wDesc, wAddr);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnAttnDescriptor_t attnDesc_native;
    cudnnMultiHeadAttnWeightKind_t wKind_native;
    size_t weightSizeInBytes_native = 0;
    void * weights_native = NULL;
    cudnnTensorDescriptor_t wDesc_native;
    void * wAddr_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    attnDesc_native = (cudnnAttnDescriptor_t)getNativePointerValue(env, attnDesc);
    wKind_native = (cudnnMultiHeadAttnWeightKind_t)wKind;
    weightSizeInBytes_native = (size_t)weightSizeInBytes;
    weights_native = (void *)getPointer(env, weights);
    wDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, wDesc);
    // wAddr is write-only

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetMultiHeadAttnWeights(handle_native, attnDesc_native, wKind_native, weightSizeInBytes_native, weights_native, wDesc_native, &wAddr_native);

    // Write back native variable values
    // handle is read-only
    // attnDesc is read-only
    // wKind is primitive
    // weightSizeInBytes is primitive
    // weights is a native pointer
    // wDesc is read-only
    setNativePointerValue(env, wAddr, (jlong)wAddr_native);

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnMultiHeadAttnForwardNative(JNIEnv *env, jclass cls, jobject handle, jobject attnDesc, jint currIdx, jintArray loWinIdx, jintArray hiWinIdx, jintArray devSeqLengthsQO, jintArray devSeqLengthsKV, jobject qDesc, jobject queries, jobject residuals, jobject kDesc, jobject keys, jobject vDesc, jobject values, jobject oDesc, jobject out, jlong weightSizeInBytes, jobject weights, jlong workSpaceSizeInBytes, jobject workSpace, jlong reserveSpaceSizeInBytes, jobject reserveSpace)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnMultiHeadAttnForward(handle=%p, attnDesc=%p, currIdx=%d, loWinIdx=%p, hiWinIdx=%p, devSeqLengthsQO=%p, devSeqLengthsKV=%p, qDesc=%p, queries=%p, residuals=%p, kDesc=%p, keys=%p, vDesc=%p, values=%p, oDesc=%p, out=%p, weightSizeInBytes=%ld, weights=%p, workSpaceSizeInBytes=%ld, workSpace=%p, reserveSpaceSizeInBytes=%ld, reserveSpace=%p)\n",
        handle, attnDesc, currIdx, loWinIdx, hiWinIdx, devSeqLengthsQO, devSeqLengthsKV, qDesc, queries, residuals, kDesc, keys, vDesc, values, oDesc, out, weightSizeInBytes, weights, workSpaceSizeInBytes, workSpace, reserveSpaceSizeInBytes, reserveSpace);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnAttnDescriptor_t attnDesc_native;
    int currIdx_native = 0;
    int * loWinIdx_native = NULL;
    int * hiWinIdx_native = NULL;
    int * devSeqLengthsQO_native = NULL;
    int * devSeqLengthsKV_native = NULL;
    cudnnSeqDataDescriptor_t qDesc_native;
    void * queries_native = NULL;
    void * residuals_native = NULL;
    cudnnSeqDataDescriptor_t kDesc_native;
    void * keys_native = NULL;
    cudnnSeqDataDescriptor_t vDesc_native;
    void * values_native = NULL;
    cudnnSeqDataDescriptor_t oDesc_native;
    void * out_native = NULL;
    size_t weightSizeInBytes_native = 0;
    void * weights_native = NULL;
    size_t workSpaceSizeInBytes_native = 0;
    void * workSpace_native = NULL;
    size_t reserveSpaceSizeInBytes_native = 0;
    void * reserveSpace_native = NULL;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    attnDesc_native = (cudnnAttnDescriptor_t)getNativePointerValue(env, attnDesc);
    currIdx_native = (int)currIdx;
    if (!initNative(env, loWinIdx, loWinIdx_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!initNative(env, hiWinIdx, hiWinIdx_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    devSeqLengthsQO_native = (int *)getPointer(env, devSeqLengthsQO);
    devSeqLengthsKV_native = (int *)getPointer(env, devSeqLengthsKV);
    qDesc_native = (cudnnSeqDataDescriptor_t)getNativePointerValue(env, qDesc);
    queries_native = (void *)getPointer(env, queries);
    residuals_native = (void *)getPointer(env, residuals);
    kDesc_native = (cudnnSeqDataDescriptor_t)getNativePointerValue(env, kDesc);
    keys_native = (void *)getPointer(env, keys);
    vDesc_native = (cudnnSeqDataDescriptor_t)getNativePointerValue(env, vDesc);
    values_native = (void *)getPointer(env, values);
    oDesc_native = (cudnnSeqDataDescriptor_t)getNativePointerValue(env, oDesc);
    out_native = (void *)getPointer(env, out);
    weightSizeInBytes_native = (size_t)weightSizeInBytes;
    weights_native = (void *)getPointer(env, weights);
    workSpaceSizeInBytes_native = (size_t)workSpaceSizeInBytes;
    workSpace_native = (void *)getPointer(env, workSpace);
    reserveSpaceSizeInBytes_native = (size_t)reserveSpaceSizeInBytes;
    reserveSpace_native = (void *)getPointer(env, reserveSpace);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnMultiHeadAttnForward(handle_native, attnDesc_native, currIdx_native, loWinIdx_native, hiWinIdx_native, devSeqLengthsQO_native, devSeqLengthsKV_native, qDesc_native, queries_native, residuals_native, kDesc_native, keys_native, vDesc_native, values_native, oDesc_native, out_native, weightSizeInBytes_native, weights_native, workSpaceSizeInBytes_native, workSpace_native, reserveSpaceSizeInBytes_native, reserveSpace_native);

    // Write back native variable values
    // handle is read-only
    // attnDesc is read-only
    // currIdx is primitive
    if (!releaseNative(env, loWinIdx_native, loWinIdx, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!releaseNative(env, hiWinIdx_native, hiWinIdx, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // devSeqLengthsQO is a native pointer
    // devSeqLengthsKV is a native pointer
    // qDesc is read-only
    // queries is a native pointer
    // residuals is a native pointer
    // kDesc is read-only
    // keys is a native pointer
    // vDesc is read-only
    // values is a native pointer
    // oDesc is read-only
    // out is a native pointer
    // weightSizeInBytes is primitive
    // weights is a native pointer
    // workSpaceSizeInBytes is primitive
    // workSpace is a native pointer
    // reserveSpaceSizeInBytes is primitive
    // reserveSpace is a native pointer

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

/**
 * <pre>
 * Cross-library version checker..
 * This function is implemented differently in each sub-library. Each sublib
 * checks whether its own version matches that of its dependencies.
 * @return CUDNN_STATUS_SUCCESS if the version check passes,
 *          CUDNN_STATUS_VERSION_MISMATCH if the versions are inconsistent.
 * </pre>
 */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnAdvInferVersionCheckNative(JNIEnv *env, jclass cls)
{
    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnAdvInferVersionCheck()\n");

    // Native function call
    cudnnStatus_t jniResult_native = cudnnAdvInferVersionCheck();

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnRNNForwardTrainingNative(JNIEnv *env, jclass cls, jobject handle, jobject rnnDesc, jint seqLength, jobjectArray xDesc, jobject x, jobject hxDesc, jobject hx, jobject cxDesc, jobject cx, jobject wDesc, jobject w, jobjectArray yDesc, jobject y, jobject hyDesc, jobject hy, jobject cyDesc, jobject cy, jobject workSpace, jlong workSpaceSizeInBytes, jobject reserveSpace, jlong reserveSpaceSizeInBytes)
{
    // Null-checks for non-primitive arguments
    if (workSpace == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'workSpace' is null for cudnnRNNForwardTraining");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (reserveSpace == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'reserveSpace' is null for cudnnRNNForwardTraining");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnRNNForwardTraining(handle=%p, rnnDesc=%p, seqLength=%d, xDesc=%p, x=%p, hxDesc=%p, hx=%p, cxDesc=%p, cx=%p, wDesc=%p, w=%p, yDesc=%p, y=%p, hyDesc=%p, hy=%p, cyDesc=%p, cy=%p, workSpace=%p, workSpaceSizeInBytes=%ld, reserveSpace=%p, reserveSpaceSizeInBytes=%ld)\n",
        handle, rnnDesc, seqLength, xDesc, x, hxDesc, hx, cxDesc, cx, wDesc, w, yDesc, y, hyDesc, hy, cyDesc, cy, workSpace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnRNNDescriptor_t rnnDesc_native;
    int seqLength_native = 0;
    cudnnTensorDescriptor_t * xDesc_native;
    void * x_native = NULL;
    cudnnTensorDescriptor_t hxDesc_native;
    void * hx_native = NULL;
    cudnnTensorDescriptor_t cxDesc_native;
    void * cx_native = NULL;
    cudnnFilterDescriptor_t wDesc_native;
    void * w_native = NULL;
    cudnnTensorDescriptor_t * yDesc_native;
    void * y_native = NULL;
    cudnnTensorDescriptor_t hyDesc_native;
    void * hy_native = NULL;
    cudnnTensorDescriptor_t cyDesc_native;
    void * cy_native = NULL;
    void * workSpace_native = NULL;
    size_t workSpaceSizeInBytes_native = 0;
    void * reserveSpace_native = NULL;
    size_t reserveSpaceSizeInBytes_native = 0;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    rnnDesc_native = (cudnnRNNDescriptor_t)getNativePointerValue(env, rnnDesc);
    seqLength_native = (int)seqLength;
    if (!initNative(env, xDesc, xDesc_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    x_native = (void *)getPointer(env, x);
    hxDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, hxDesc);
    hx_native = (void *)getPointer(env, hx);
    cxDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, cxDesc);
    cx_native = (void *)getPointer(env, cx);
    wDesc_native = (cudnnFilterDescriptor_t)getNativePointerValue(env, wDesc);
    w_native = (void *)getPointer(env, w);
    if (!initNative(env, yDesc, yDesc_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    y_native = (void *)getPointer(env, y);
    hyDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, hyDesc);
    hy_native = (void *)getPointer(env, hy);
    cyDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, cyDesc);
    cy_native = (void *)getPointer(env, cy);
    workSpace_native = (void *)getPointer(env, workSpace);
    workSpaceSizeInBytes_native = (size_t)workSpaceSizeInBytes;
    reserveSpace_native = (void *)getPointer(env, reserveSpace);
    reserveSpaceSizeInBytes_native = (size_t)reserveSpaceSizeInBytes;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnRNNForwardTraining(handle_native, rnnDesc_native, seqLength_native, xDesc_native, x_native, hxDesc_native, hx_native, cxDesc_native, cx_native, wDesc_native, w_native, yDesc_native, y_native, hyDesc_native, hy_native, cyDesc_native, cy_native, workSpace_native, workSpaceSizeInBytes_native, reserveSpace_native, reserveSpaceSizeInBytes_native);

    // Write back native variable values
    // handle is read-only
    // rnnDesc is read-only
    // seqLength is primitive
    if (!releaseNative(env, xDesc_native, xDesc, false)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // x is a native pointer
    // hxDesc is read-only
    // hx is a native pointer
    // cxDesc is read-only
    // cx is a native pointer
    // wDesc is read-only
    // w is a native pointer
    if (!releaseNative(env, yDesc_native, yDesc, false)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // y is a native pointer
    // hyDesc is read-only
    // hy is a native pointer
    // cyDesc is read-only
    // cy is a native pointer
    // workSpace is a native pointer
    // workSpaceSizeInBytes is primitive
    // reserveSpace is a native pointer
    // reserveSpaceSizeInBytes is primitive

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnRNNBackwardDataNative(JNIEnv *env, jclass cls, jobject handle, jobject rnnDesc, jint seqLength, jobjectArray yDesc, jobject y, jobjectArray dyDesc, jobject dy, jobject dhyDesc, jobject dhy, jobject dcyDesc, jobject dcy, jobject wDesc, jobject w, jobject hxDesc, jobject hx, jobject cxDesc, jobject cx, jobjectArray dxDesc, jobject dx, jobject dhxDesc, jobject dhx, jobject dcxDesc, jobject dcx, jobject workSpace, jlong workSpaceSizeInBytes, jobject reserveSpace, jlong reserveSpaceSizeInBytes)
{
    // Null-checks for non-primitive arguments
    if (workSpace == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'workSpace' is null for cudnnRNNBackwardData");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (reserveSpace == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'reserveSpace' is null for cudnnRNNBackwardData");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnRNNBackwardData(handle=%p, rnnDesc=%p, seqLength=%d, yDesc=%p, y=%p, dyDesc=%p, dy=%p, dhyDesc=%p, dhy=%p, dcyDesc=%p, dcy=%p, wDesc=%p, w=%p, hxDesc=%p, hx=%p, cxDesc=%p, cx=%p, dxDesc=%p, dx=%p, dhxDesc=%p, dhx=%p, dcxDesc=%p, dcx=%p, workSpace=%p, workSpaceSizeInBytes=%ld, reserveSpace=%p, reserveSpaceSizeInBytes=%ld)\n",
        handle, rnnDesc, seqLength, yDesc, y, dyDesc, dy, dhyDesc, dhy, dcyDesc, dcy, wDesc, w, hxDesc, hx, cxDesc, cx, dxDesc, dx, dhxDesc, dhx, dcxDesc, dcx, workSpace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnRNNDescriptor_t rnnDesc_native;
    int seqLength_native = 0;
    cudnnTensorDescriptor_t * yDesc_native;
    void * y_native = NULL;
    cudnnTensorDescriptor_t * dyDesc_native;
    void * dy_native = NULL;
    cudnnTensorDescriptor_t dhyDesc_native;
    void * dhy_native = NULL;
    cudnnTensorDescriptor_t dcyDesc_native;
    void * dcy_native = NULL;
    cudnnFilterDescriptor_t wDesc_native;
    void * w_native = NULL;
    cudnnTensorDescriptor_t hxDesc_native;
    void * hx_native = NULL;
    cudnnTensorDescriptor_t cxDesc_native;
    void * cx_native = NULL;
    cudnnTensorDescriptor_t * dxDesc_native;
    void * dx_native = NULL;
    cudnnTensorDescriptor_t dhxDesc_native;
    void * dhx_native = NULL;
    cudnnTensorDescriptor_t dcxDesc_native;
    void * dcx_native = NULL;
    void * workSpace_native = NULL;
    size_t workSpaceSizeInBytes_native = 0;
    void * reserveSpace_native = NULL;
    size_t reserveSpaceSizeInBytes_native = 0;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    rnnDesc_native = (cudnnRNNDescriptor_t)getNativePointerValue(env, rnnDesc);
    seqLength_native = (int)seqLength;
    if (!initNative(env, yDesc, yDesc_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    y_native = (void *)getPointer(env, y);
    if (!initNative(env, dyDesc, dyDesc_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    dy_native = (void *)getPointer(env, dy);
    dhyDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, dhyDesc);
    dhy_native = (void *)getPointer(env, dhy);
    dcyDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, dcyDesc);
    dcy_native = (void *)getPointer(env, dcy);
    wDesc_native = (cudnnFilterDescriptor_t)getNativePointerValue(env, wDesc);
    w_native = (void *)getPointer(env, w);
    hxDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, hxDesc);
    hx_native = (void *)getPointer(env, hx);
    cxDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, cxDesc);
    cx_native = (void *)getPointer(env, cx);
    if (!initNative(env, dxDesc, dxDesc_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    dx_native = (void *)getPointer(env, dx);
    dhxDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, dhxDesc);
    dhx_native = (void *)getPointer(env, dhx);
    dcxDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, dcxDesc);
    dcx_native = (void *)getPointer(env, dcx);
    workSpace_native = (void *)getPointer(env, workSpace);
    workSpaceSizeInBytes_native = (size_t)workSpaceSizeInBytes;
    reserveSpace_native = (void *)getPointer(env, reserveSpace);
    reserveSpaceSizeInBytes_native = (size_t)reserveSpaceSizeInBytes;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnRNNBackwardData(handle_native, rnnDesc_native, seqLength_native, yDesc_native, y_native, dyDesc_native, dy_native, dhyDesc_native, dhy_native, dcyDesc_native, dcy_native, wDesc_native, w_native, hxDesc_native, hx_native, cxDesc_native, cx_native, dxDesc_native, dx_native, dhxDesc_native, dhx_native, dcxDesc_native, dcx_native, workSpace_native, workSpaceSizeInBytes_native, reserveSpace_native, reserveSpaceSizeInBytes_native);

    // Write back native variable values
    // handle is read-only
    // rnnDesc is read-only
    // seqLength is primitive
    if (!releaseNative(env, yDesc_native, yDesc, false)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // y is a native pointer
    if (!releaseNative(env, dyDesc_native, dyDesc, false)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // dy is a native pointer
    // dhyDesc is read-only
    // dhy is a native pointer
    // dcyDesc is read-only
    // dcy is a native pointer
    // wDesc is read-only
    // w is a native pointer
    // hxDesc is read-only
    // hx is a native pointer
    // cxDesc is read-only
    // cx is a native pointer
    if (!releaseNative(env, dxDesc_native, dxDesc, false)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // dx is a native pointer
    // dhxDesc is read-only
    // dhx is a native pointer
    // dcxDesc is read-only
    // dcx is a native pointer
    // workSpace is a native pointer
    // workSpaceSizeInBytes is primitive
    // reserveSpace is a native pointer
    // reserveSpaceSizeInBytes is primitive

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnRNNBackwardData_1v8Native(JNIEnv *env, jclass cls, jobject handle, jobject rnnDesc, jintArray devSeqLengths, jobject yDesc, jobject y, jobject dy, jobject xDesc, jobject dx, jobject hDesc, jobject hx, jobject dhy, jobject dhx, jobject cDesc, jobject cx, jobject dcy, jobject dcx, jlong weightSpaceSize, jobject weightSpace, jlong workSpaceSize, jobject workSpace, jlong reserveSpaceSize, jobject reserveSpace)
{
    // Null-checks for non-primitive arguments
    if (workSpace == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'workSpace' is null for cudnnRNNBackwardData_v8");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (reserveSpace == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'reserveSpace' is null for cudnnRNNBackwardData_v8");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnRNNBackwardData_v8(handle=%p, rnnDesc=%p, devSeqLengths=%p, yDesc=%p, y=%p, dy=%p, xDesc=%p, dx=%p, hDesc=%p, hx=%p, dhy=%p, dhx=%p, cDesc=%p, cx=%p, dcy=%p, dcx=%p, weightSpaceSize=%ld, weightSpace=%p, workSpaceSize=%ld, workSpace=%p, reserveSpaceSize=%ld, reserveSpace=%p)\n",
        handle, rnnDesc, devSeqLengths, yDesc, y, dy, xDesc, dx, hDesc, hx, dhy, dhx, cDesc, cx, dcy, dcx, weightSpaceSize, weightSpace, workSpaceSize, workSpace, reserveSpaceSize, reserveSpace);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnRNNDescriptor_t rnnDesc_native;
    int32_t * devSeqLengths_native = NULL;
    cudnnRNNDataDescriptor_t yDesc_native;
    void * y_native = NULL;
    void * dy_native = NULL;
    cudnnRNNDataDescriptor_t xDesc_native;
    void * dx_native = NULL;
    cudnnTensorDescriptor_t hDesc_native;
    void * hx_native = NULL;
    void * dhy_native = NULL;
    void * dhx_native = NULL;
    cudnnTensorDescriptor_t cDesc_native;
    void * cx_native = NULL;
    void * dcy_native = NULL;
    void * dcx_native = NULL;
    size_t weightSpaceSize_native = 0;
    void * weightSpace_native = NULL;
    size_t workSpaceSize_native = 0;
    void * workSpace_native = NULL;
    size_t reserveSpaceSize_native = 0;
    void * reserveSpace_native = NULL;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    rnnDesc_native = (cudnnRNNDescriptor_t)getNativePointerValue(env, rnnDesc);
    devSeqLengths_native = (int32_t *)getPointer(env, devSeqLengths);
    yDesc_native = (cudnnRNNDataDescriptor_t)getNativePointerValue(env, yDesc);
    y_native = (void *)getPointer(env, y);
    dy_native = (void *)getPointer(env, dy);
    xDesc_native = (cudnnRNNDataDescriptor_t)getNativePointerValue(env, xDesc);
    dx_native = (void *)getPointer(env, dx);
    hDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, hDesc);
    hx_native = (void *)getPointer(env, hx);
    dhy_native = (void *)getPointer(env, dhy);
    dhx_native = (void *)getPointer(env, dhx);
    cDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, cDesc);
    cx_native = (void *)getPointer(env, cx);
    dcy_native = (void *)getPointer(env, dcy);
    dcx_native = (void *)getPointer(env, dcx);
    weightSpaceSize_native = (size_t)weightSpaceSize;
    weightSpace_native = (void *)getPointer(env, weightSpace);
    workSpaceSize_native = (size_t)workSpaceSize;
    workSpace_native = (void *)getPointer(env, workSpace);
    reserveSpaceSize_native = (size_t)reserveSpaceSize;
    reserveSpace_native = (void *)getPointer(env, reserveSpace);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnRNNBackwardData_v8(handle_native, rnnDesc_native, devSeqLengths_native, yDesc_native, y_native, dy_native, xDesc_native, dx_native, hDesc_native, hx_native, dhy_native, dhx_native, cDesc_native, cx_native, dcy_native, dcx_native, weightSpaceSize_native, weightSpace_native, workSpaceSize_native, workSpace_native, reserveSpaceSize_native, reserveSpace_native);

    // Write back native variable values
    // handle is read-only
    // rnnDesc is read-only
    // devSeqLengths is a native pointer
    // yDesc is read-only
    // y is a native pointer
    // dy is a native pointer
    // xDesc is read-only
    // dx is a native pointer
    // hDesc is read-only
    // hx is a native pointer
    // dhy is a native pointer
    // dhx is a native pointer
    // cDesc is read-only
    // cx is a native pointer
    // dcy is a native pointer
    // dcx is a native pointer
    // weightSpaceSize is primitive
    // weightSpace is a native pointer
    // workSpaceSize is primitive
    // workSpace is a native pointer
    // reserveSpaceSize is primitive
    // reserveSpace is a native pointer

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnRNNBackwardWeightsNative(JNIEnv *env, jclass cls, jobject handle, jobject rnnDesc, jint seqLength, jobjectArray xDesc, jobject x, jobject hxDesc, jobject hx, jobjectArray yDesc, jobject y, jobject workSpace, jlong workSpaceSizeInBytes, jobject dwDesc, jobject dw, jobject reserveSpace, jlong reserveSpaceSizeInBytes)
{
    // Null-checks for non-primitive arguments
    if (workSpace == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'workSpace' is null for cudnnRNNBackwardWeights");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (reserveSpace == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'reserveSpace' is null for cudnnRNNBackwardWeights");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnRNNBackwardWeights(handle=%p, rnnDesc=%p, seqLength=%d, xDesc=%p, x=%p, hxDesc=%p, hx=%p, yDesc=%p, y=%p, workSpace=%p, workSpaceSizeInBytes=%ld, dwDesc=%p, dw=%p, reserveSpace=%p, reserveSpaceSizeInBytes=%ld)\n",
        handle, rnnDesc, seqLength, xDesc, x, hxDesc, hx, yDesc, y, workSpace, workSpaceSizeInBytes, dwDesc, dw, reserveSpace, reserveSpaceSizeInBytes);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnRNNDescriptor_t rnnDesc_native;
    int seqLength_native = 0;
    cudnnTensorDescriptor_t * xDesc_native;
    void * x_native = NULL;
    cudnnTensorDescriptor_t hxDesc_native;
    void * hx_native = NULL;
    cudnnTensorDescriptor_t * yDesc_native;
    void * y_native = NULL;
    void * workSpace_native = NULL;
    size_t workSpaceSizeInBytes_native = 0;
    cudnnFilterDescriptor_t dwDesc_native;
    void * dw_native = NULL;
    void * reserveSpace_native = NULL;
    size_t reserveSpaceSizeInBytes_native = 0;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    rnnDesc_native = (cudnnRNNDescriptor_t)getNativePointerValue(env, rnnDesc);
    seqLength_native = (int)seqLength;
    if (!initNative(env, xDesc, xDesc_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    x_native = (void *)getPointer(env, x);
    hxDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, hxDesc);
    hx_native = (void *)getPointer(env, hx);
    if (!initNative(env, yDesc, yDesc_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    y_native = (void *)getPointer(env, y);
    workSpace_native = (void *)getPointer(env, workSpace);
    workSpaceSizeInBytes_native = (size_t)workSpaceSizeInBytes;
    dwDesc_native = (cudnnFilterDescriptor_t)getNativePointerValue(env, dwDesc);
    dw_native = (void *)getPointer(env, dw);
    reserveSpace_native = (void *)getPointer(env, reserveSpace);
    reserveSpaceSizeInBytes_native = (size_t)reserveSpaceSizeInBytes;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnRNNBackwardWeights(handle_native, rnnDesc_native, seqLength_native, xDesc_native, x_native, hxDesc_native, hx_native, yDesc_native, y_native, workSpace_native, workSpaceSizeInBytes_native, dwDesc_native, dw_native, reserveSpace_native, reserveSpaceSizeInBytes_native);

    // Write back native variable values
    // handle is read-only
    // rnnDesc is read-only
    // seqLength is primitive
    if (!releaseNative(env, xDesc_native, xDesc, false)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // x is a native pointer
    // hxDesc is read-only
    // hx is a native pointer
    if (!releaseNative(env, yDesc_native, yDesc, false)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // y is a native pointer
    // workSpace is a native pointer
    // workSpaceSizeInBytes is primitive
    // dwDesc is read-only
    // dw is a native pointer
    // reserveSpace is a native pointer
    // reserveSpaceSizeInBytes is primitive

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnRNNBackwardWeights_1v8Native(JNIEnv *env, jclass cls, jobject handle, jobject rnnDesc, jint addGrad, jintArray devSeqLengths, jobject xDesc, jobject x, jobject hDesc, jobject hx, jobject yDesc, jobject y, jlong weightSpaceSize, jobject dweightSpace, jlong workSpaceSize, jobject workSpace, jlong reserveSpaceSize, jobject reserveSpace)
{
    // Null-checks for non-primitive arguments
    if (workSpace == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'workSpace' is null for cudnnRNNBackwardWeights_v8");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (reserveSpace == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'reserveSpace' is null for cudnnRNNBackwardWeights_v8");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnRNNBackwardWeights_v8(handle=%p, rnnDesc=%p, addGrad=%d, devSeqLengths=%p, xDesc=%p, x=%p, hDesc=%p, hx=%p, yDesc=%p, y=%p, weightSpaceSize=%ld, dweightSpace=%p, workSpaceSize=%ld, workSpace=%p, reserveSpaceSize=%ld, reserveSpace=%p)\n",
        handle, rnnDesc, addGrad, devSeqLengths, xDesc, x, hDesc, hx, yDesc, y, weightSpaceSize, dweightSpace, workSpaceSize, workSpace, reserveSpaceSize, reserveSpace);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnRNNDescriptor_t rnnDesc_native;
    cudnnWgradMode_t addGrad_native;
    int32_t * devSeqLengths_native = NULL;
    cudnnRNNDataDescriptor_t xDesc_native;
    void * x_native = NULL;
    cudnnTensorDescriptor_t hDesc_native;
    void * hx_native = NULL;
    cudnnRNNDataDescriptor_t yDesc_native;
    void * y_native = NULL;
    size_t weightSpaceSize_native = 0;
    void * dweightSpace_native = NULL;
    size_t workSpaceSize_native = 0;
    void * workSpace_native = NULL;
    size_t reserveSpaceSize_native = 0;
    void * reserveSpace_native = NULL;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    rnnDesc_native = (cudnnRNNDescriptor_t)getNativePointerValue(env, rnnDesc);
    addGrad_native = (cudnnWgradMode_t)addGrad;
    devSeqLengths_native = (int32_t *)getPointer(env, devSeqLengths);
    xDesc_native = (cudnnRNNDataDescriptor_t)getNativePointerValue(env, xDesc);
    x_native = (void *)getPointer(env, x);
    hDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, hDesc);
    hx_native = (void *)getPointer(env, hx);
    yDesc_native = (cudnnRNNDataDescriptor_t)getNativePointerValue(env, yDesc);
    y_native = (void *)getPointer(env, y);
    weightSpaceSize_native = (size_t)weightSpaceSize;
    dweightSpace_native = (void *)getPointer(env, dweightSpace);
    workSpaceSize_native = (size_t)workSpaceSize;
    workSpace_native = (void *)getPointer(env, workSpace);
    reserveSpaceSize_native = (size_t)reserveSpaceSize;
    reserveSpace_native = (void *)getPointer(env, reserveSpace);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnRNNBackwardWeights_v8(handle_native, rnnDesc_native, addGrad_native, devSeqLengths_native, xDesc_native, x_native, hDesc_native, hx_native, yDesc_native, y_native, weightSpaceSize_native, dweightSpace_native, workSpaceSize_native, workSpace_native, reserveSpaceSize_native, reserveSpace_native);

    // Write back native variable values
    // handle is read-only
    // rnnDesc is read-only
    // addGrad is primitive
    // devSeqLengths is a native pointer
    // xDesc is read-only
    // x is a native pointer
    // hDesc is read-only
    // hx is a native pointer
    // yDesc is read-only
    // y is a native pointer
    // weightSpaceSize is primitive
    // dweightSpace is a native pointer
    // workSpaceSize is primitive
    // workSpace is a native pointer
    // reserveSpaceSize is primitive
    // reserveSpace is a native pointer

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

/** RNN EX API */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnRNNForwardTrainingExNative(JNIEnv *env, jclass cls, jobject handle, jobject rnnDesc, jobject xDesc, jobject x, jobject hxDesc, jobject hx, jobject cxDesc, jobject cx, jobject wDesc, jobject w, jobject yDesc, jobject y, jobject hyDesc, jobject hy, jobject cyDesc, jobject cy, jobject kDesc, jobject keys, jobject cDesc, jobject cAttn, jobject iDesc, jobject iAttn, jobject qDesc, jobject queries, jobject workSpace, jlong workSpaceSizeInBytes, jobject reserveSpace, jlong reserveSpaceSizeInBytes)
{
    // Null-checks for non-primitive arguments
    if (workSpace == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'workSpace' is null for cudnnRNNForwardTrainingEx");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (reserveSpace == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'reserveSpace' is null for cudnnRNNForwardTrainingEx");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnRNNForwardTrainingEx(handle=%p, rnnDesc=%p, xDesc=%p, x=%p, hxDesc=%p, hx=%p, cxDesc=%p, cx=%p, wDesc=%p, w=%p, yDesc=%p, y=%p, hyDesc=%p, hy=%p, cyDesc=%p, cy=%p, kDesc=%p, keys=%p, cDesc=%p, cAttn=%p, iDesc=%p, iAttn=%p, qDesc=%p, queries=%p, workSpace=%p, workSpaceSizeInBytes=%ld, reserveSpace=%p, reserveSpaceSizeInBytes=%ld)\n",
        handle, rnnDesc, xDesc, x, hxDesc, hx, cxDesc, cx, wDesc, w, yDesc, y, hyDesc, hy, cyDesc, cy, kDesc, keys, cDesc, cAttn, iDesc, iAttn, qDesc, queries, workSpace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnRNNDescriptor_t rnnDesc_native;
    cudnnRNNDataDescriptor_t xDesc_native;
    void * x_native = NULL;
    cudnnTensorDescriptor_t hxDesc_native;
    void * hx_native = NULL;
    cudnnTensorDescriptor_t cxDesc_native;
    void * cx_native = NULL;
    cudnnFilterDescriptor_t wDesc_native;
    void * w_native = NULL;
    cudnnRNNDataDescriptor_t yDesc_native;
    void * y_native = NULL;
    cudnnTensorDescriptor_t hyDesc_native;
    void * hy_native = NULL;
    cudnnTensorDescriptor_t cyDesc_native;
    void * cy_native = NULL;
    cudnnRNNDataDescriptor_t kDesc_native;
    void * keys_native = NULL;
    cudnnRNNDataDescriptor_t cDesc_native;
    void * cAttn_native = NULL;
    cudnnRNNDataDescriptor_t iDesc_native;
    void * iAttn_native = NULL;
    cudnnRNNDataDescriptor_t qDesc_native;
    void * queries_native = NULL;
    void * workSpace_native = NULL;
    size_t workSpaceSizeInBytes_native = 0;
    void * reserveSpace_native = NULL;
    size_t reserveSpaceSizeInBytes_native = 0;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    rnnDesc_native = (cudnnRNNDescriptor_t)getNativePointerValue(env, rnnDesc);
    xDesc_native = (cudnnRNNDataDescriptor_t)getNativePointerValue(env, xDesc);
    x_native = (void *)getPointer(env, x);
    hxDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, hxDesc);
    hx_native = (void *)getPointer(env, hx);
    cxDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, cxDesc);
    cx_native = (void *)getPointer(env, cx);
    wDesc_native = (cudnnFilterDescriptor_t)getNativePointerValue(env, wDesc);
    w_native = (void *)getPointer(env, w);
    yDesc_native = (cudnnRNNDataDescriptor_t)getNativePointerValue(env, yDesc);
    y_native = (void *)getPointer(env, y);
    hyDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, hyDesc);
    hy_native = (void *)getPointer(env, hy);
    cyDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, cyDesc);
    cy_native = (void *)getPointer(env, cy);
    kDesc_native = (cudnnRNNDataDescriptor_t)getNativePointerValue(env, kDesc);
    keys_native = (void *)getPointer(env, keys);
    cDesc_native = (cudnnRNNDataDescriptor_t)getNativePointerValue(env, cDesc);
    cAttn_native = (void *)getPointer(env, cAttn);
    iDesc_native = (cudnnRNNDataDescriptor_t)getNativePointerValue(env, iDesc);
    iAttn_native = (void *)getPointer(env, iAttn);
    qDesc_native = (cudnnRNNDataDescriptor_t)getNativePointerValue(env, qDesc);
    queries_native = (void *)getPointer(env, queries);
    workSpace_native = (void *)getPointer(env, workSpace);
    workSpaceSizeInBytes_native = (size_t)workSpaceSizeInBytes;
    reserveSpace_native = (void *)getPointer(env, reserveSpace);
    reserveSpaceSizeInBytes_native = (size_t)reserveSpaceSizeInBytes;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnRNNForwardTrainingEx(handle_native, rnnDesc_native, xDesc_native, x_native, hxDesc_native, hx_native, cxDesc_native, cx_native, wDesc_native, w_native, yDesc_native, y_native, hyDesc_native, hy_native, cyDesc_native, cy_native, kDesc_native, keys_native, cDesc_native, cAttn_native, iDesc_native, iAttn_native, qDesc_native, queries_native, workSpace_native, workSpaceSizeInBytes_native, reserveSpace_native, reserveSpaceSizeInBytes_native);

    // Write back native variable values
    // handle is read-only
    // rnnDesc is read-only
    // xDesc is read-only
    // x is a native pointer
    // hxDesc is read-only
    // hx is a native pointer
    // cxDesc is read-only
    // cx is a native pointer
    // wDesc is read-only
    // w is a native pointer
    // yDesc is read-only
    // y is a native pointer
    // hyDesc is read-only
    // hy is a native pointer
    // cyDesc is read-only
    // cy is a native pointer
    // kDesc is read-only
    // keys is a native pointer
    // cDesc is read-only
    // cAttn is a native pointer
    // iDesc is read-only
    // iAttn is a native pointer
    // qDesc is read-only
    // queries is a native pointer
    // workSpace is a native pointer
    // workSpaceSizeInBytes is primitive
    // reserveSpace is a native pointer
    // reserveSpaceSizeInBytes is primitive

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnRNNBackwardDataExNative(JNIEnv *env, jclass cls, jobject handle, jobject rnnDesc, jobject yDesc, jobject y, jobject dyDesc, jobject dy, jobject dcDesc, jobject dcAttn, jobject dhyDesc, jobject dhy, jobject dcyDesc, jobject dcy, jobject wDesc, jobject w, jobject hxDesc, jobject hx, jobject cxDesc, jobject cx, jobject dxDesc, jobject dx, jobject dhxDesc, jobject dhx, jobject dcxDesc, jobject dcx, jobject dkDesc, jobject dkeys, jobject workSpace, jlong workSpaceSizeInBytes, jobject reserveSpace, jlong reserveSpaceSizeInBytes)
{
    // Null-checks for non-primitive arguments
    if (workSpace == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'workSpace' is null for cudnnRNNBackwardDataEx");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (reserveSpace == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'reserveSpace' is null for cudnnRNNBackwardDataEx");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnRNNBackwardDataEx(handle=%p, rnnDesc=%p, yDesc=%p, y=%p, dyDesc=%p, dy=%p, dcDesc=%p, dcAttn=%p, dhyDesc=%p, dhy=%p, dcyDesc=%p, dcy=%p, wDesc=%p, w=%p, hxDesc=%p, hx=%p, cxDesc=%p, cx=%p, dxDesc=%p, dx=%p, dhxDesc=%p, dhx=%p, dcxDesc=%p, dcx=%p, dkDesc=%p, dkeys=%p, workSpace=%p, workSpaceSizeInBytes=%ld, reserveSpace=%p, reserveSpaceSizeInBytes=%ld)\n",
        handle, rnnDesc, yDesc, y, dyDesc, dy, dcDesc, dcAttn, dhyDesc, dhy, dcyDesc, dcy, wDesc, w, hxDesc, hx, cxDesc, cx, dxDesc, dx, dhxDesc, dhx, dcxDesc, dcx, dkDesc, dkeys, workSpace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnRNNDescriptor_t rnnDesc_native;
    cudnnRNNDataDescriptor_t yDesc_native;
    void * y_native = NULL;
    cudnnRNNDataDescriptor_t dyDesc_native;
    void * dy_native = NULL;
    cudnnRNNDataDescriptor_t dcDesc_native;
    void * dcAttn_native = NULL;
    cudnnTensorDescriptor_t dhyDesc_native;
    void * dhy_native = NULL;
    cudnnTensorDescriptor_t dcyDesc_native;
    void * dcy_native = NULL;
    cudnnFilterDescriptor_t wDesc_native;
    void * w_native = NULL;
    cudnnTensorDescriptor_t hxDesc_native;
    void * hx_native = NULL;
    cudnnTensorDescriptor_t cxDesc_native;
    void * cx_native = NULL;
    cudnnRNNDataDescriptor_t dxDesc_native;
    void * dx_native = NULL;
    cudnnTensorDescriptor_t dhxDesc_native;
    void * dhx_native = NULL;
    cudnnTensorDescriptor_t dcxDesc_native;
    void * dcx_native = NULL;
    cudnnRNNDataDescriptor_t dkDesc_native;
    void * dkeys_native = NULL;
    void * workSpace_native = NULL;
    size_t workSpaceSizeInBytes_native = 0;
    void * reserveSpace_native = NULL;
    size_t reserveSpaceSizeInBytes_native = 0;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    rnnDesc_native = (cudnnRNNDescriptor_t)getNativePointerValue(env, rnnDesc);
    yDesc_native = (cudnnRNNDataDescriptor_t)getNativePointerValue(env, yDesc);
    y_native = (void *)getPointer(env, y);
    dyDesc_native = (cudnnRNNDataDescriptor_t)getNativePointerValue(env, dyDesc);
    dy_native = (void *)getPointer(env, dy);
    dcDesc_native = (cudnnRNNDataDescriptor_t)getNativePointerValue(env, dcDesc);
    dcAttn_native = (void *)getPointer(env, dcAttn);
    dhyDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, dhyDesc);
    dhy_native = (void *)getPointer(env, dhy);
    dcyDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, dcyDesc);
    dcy_native = (void *)getPointer(env, dcy);
    wDesc_native = (cudnnFilterDescriptor_t)getNativePointerValue(env, wDesc);
    w_native = (void *)getPointer(env, w);
    hxDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, hxDesc);
    hx_native = (void *)getPointer(env, hx);
    cxDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, cxDesc);
    cx_native = (void *)getPointer(env, cx);
    dxDesc_native = (cudnnRNNDataDescriptor_t)getNativePointerValue(env, dxDesc);
    dx_native = (void *)getPointer(env, dx);
    dhxDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, dhxDesc);
    dhx_native = (void *)getPointer(env, dhx);
    dcxDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, dcxDesc);
    dcx_native = (void *)getPointer(env, dcx);
    dkDesc_native = (cudnnRNNDataDescriptor_t)getNativePointerValue(env, dkDesc);
    dkeys_native = (void *)getPointer(env, dkeys);
    workSpace_native = (void *)getPointer(env, workSpace);
    workSpaceSizeInBytes_native = (size_t)workSpaceSizeInBytes;
    reserveSpace_native = (void *)getPointer(env, reserveSpace);
    reserveSpaceSizeInBytes_native = (size_t)reserveSpaceSizeInBytes;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnRNNBackwardDataEx(handle_native, rnnDesc_native, yDesc_native, y_native, dyDesc_native, dy_native, dcDesc_native, dcAttn_native, dhyDesc_native, dhy_native, dcyDesc_native, dcy_native, wDesc_native, w_native, hxDesc_native, hx_native, cxDesc_native, cx_native, dxDesc_native, dx_native, dhxDesc_native, dhx_native, dcxDesc_native, dcx_native, dkDesc_native, dkeys_native, workSpace_native, workSpaceSizeInBytes_native, reserveSpace_native, reserveSpaceSizeInBytes_native);

    // Write back native variable values
    // handle is read-only
    // rnnDesc is read-only
    // yDesc is read-only
    // y is a native pointer
    // dyDesc is read-only
    // dy is a native pointer
    // dcDesc is read-only
    // dcAttn is a native pointer
    // dhyDesc is read-only
    // dhy is a native pointer
    // dcyDesc is read-only
    // dcy is a native pointer
    // wDesc is read-only
    // w is a native pointer
    // hxDesc is read-only
    // hx is a native pointer
    // cxDesc is read-only
    // cx is a native pointer
    // dxDesc is read-only
    // dx is a native pointer
    // dhxDesc is read-only
    // dhx is a native pointer
    // dcxDesc is read-only
    // dcx is a native pointer
    // dkDesc is read-only
    // dkeys is a native pointer
    // workSpace is a native pointer
    // workSpaceSizeInBytes is primitive
    // reserveSpace is a native pointer
    // reserveSpaceSizeInBytes is primitive

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnRNNBackwardWeightsExNative(JNIEnv *env, jclass cls, jobject handle, jobject rnnDesc, jobject xDesc, jobject x, jobject hxDesc, jobject hx, jobject yDesc, jobject y, jobject workSpace, jlong workSpaceSizeInBytes, jobject dwDesc, jobject dw, jobject reserveSpace, jlong reserveSpaceSizeInBytes)
{
    // Null-checks for non-primitive arguments
    if (workSpace == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'workSpace' is null for cudnnRNNBackwardWeightsEx");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (reserveSpace == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'reserveSpace' is null for cudnnRNNBackwardWeightsEx");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnRNNBackwardWeightsEx(handle=%p, rnnDesc=%p, xDesc=%p, x=%p, hxDesc=%p, hx=%p, yDesc=%p, y=%p, workSpace=%p, workSpaceSizeInBytes=%ld, dwDesc=%p, dw=%p, reserveSpace=%p, reserveSpaceSizeInBytes=%ld)\n",
        handle, rnnDesc, xDesc, x, hxDesc, hx, yDesc, y, workSpace, workSpaceSizeInBytes, dwDesc, dw, reserveSpace, reserveSpaceSizeInBytes);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnRNNDescriptor_t rnnDesc_native;
    cudnnRNNDataDescriptor_t xDesc_native;
    void * x_native = NULL;
    cudnnTensorDescriptor_t hxDesc_native;
    void * hx_native = NULL;
    cudnnRNNDataDescriptor_t yDesc_native;
    void * y_native = NULL;
    void * workSpace_native = NULL;
    size_t workSpaceSizeInBytes_native = 0;
    cudnnFilterDescriptor_t dwDesc_native;
    void * dw_native = NULL;
    void * reserveSpace_native = NULL;
    size_t reserveSpaceSizeInBytes_native = 0;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    rnnDesc_native = (cudnnRNNDescriptor_t)getNativePointerValue(env, rnnDesc);
    xDesc_native = (cudnnRNNDataDescriptor_t)getNativePointerValue(env, xDesc);
    x_native = (void *)getPointer(env, x);
    hxDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, hxDesc);
    hx_native = (void *)getPointer(env, hx);
    yDesc_native = (cudnnRNNDataDescriptor_t)getNativePointerValue(env, yDesc);
    y_native = (void *)getPointer(env, y);
    workSpace_native = (void *)getPointer(env, workSpace);
    workSpaceSizeInBytes_native = (size_t)workSpaceSizeInBytes;
    dwDesc_native = (cudnnFilterDescriptor_t)getNativePointerValue(env, dwDesc);
    dw_native = (void *)getPointer(env, dw);
    reserveSpace_native = (void *)getPointer(env, reserveSpace);
    reserveSpaceSizeInBytes_native = (size_t)reserveSpaceSizeInBytes;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnRNNBackwardWeightsEx(handle_native, rnnDesc_native, xDesc_native, x_native, hxDesc_native, hx_native, yDesc_native, y_native, workSpace_native, workSpaceSizeInBytes_native, dwDesc_native, dw_native, reserveSpace_native, reserveSpaceSizeInBytes_native);

    // Write back native variable values
    // handle is read-only
    // rnnDesc is read-only
    // xDesc is read-only
    // x is a native pointer
    // hxDesc is read-only
    // hx is a native pointer
    // yDesc is read-only
    // y is a native pointer
    // workSpace is a native pointer
    // workSpaceSizeInBytes is primitive
    // dwDesc is read-only
    // dw is a native pointer
    // reserveSpace is a native pointer
    // reserveSpaceSizeInBytes is primitive

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

/** RNN FIND API */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetRNNForwardTrainingAlgorithmMaxCountNative(JNIEnv *env, jclass cls, jobject handle, jobject rnnDesc, jintArray count)
{
    // Null-checks for non-primitive arguments
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cudnnGetRNNForwardTrainingAlgorithmMaxCount");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (rnnDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'rnnDesc' is null for cudnnGetRNNForwardTrainingAlgorithmMaxCount");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (count == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'count' is null for cudnnGetRNNForwardTrainingAlgorithmMaxCount");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetRNNForwardTrainingAlgorithmMaxCount(handle=%p, rnnDesc=%p, count=%p)\n",
        handle, rnnDesc, count);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnRNNDescriptor_t rnnDesc_native;
    int count_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    rnnDesc_native = (cudnnRNNDescriptor_t)getNativePointerValue(env, rnnDesc);
    // count is write-only

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetRNNForwardTrainingAlgorithmMaxCount(handle_native, rnnDesc_native, &count_native);

    // Write back native variable values
    // handle is read-only
    // rnnDesc is read-only
    if (!set(env, count, 0, (jint)count_native)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnFindRNNForwardTrainingAlgorithmExNative(JNIEnv *env, jclass cls, jobject handle, jobject rnnDesc, jint seqLength, jobjectArray xDesc, jobject x, jobject hxDesc, jobject hx, jobject cxDesc, jobject cx, jobject wDesc, jobject w, jobjectArray yDesc, jobject y, jobject hyDesc, jobject hy, jobject cyDesc, jobject cy, jfloat findIntensity, jint requestedAlgoCount, jintArray returnedAlgoCount, jobjectArray perfResults, jobject workspace, jlong workSpaceSizeInBytes, jobject reserveSpace, jlong reserveSpaceSizeInBytes)
{
    // Null-checks for non-primitive arguments
    if (perfResults == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'perfResults' is null for cudnnFindRNNForwardTrainingAlgorithmEx");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnFindRNNForwardTrainingAlgorithmEx(handle=%p, rnnDesc=%p, seqLength=%d, xDesc=%p, x=%p, hxDesc=%p, hx=%p, cxDesc=%p, cx=%p, wDesc=%p, w=%p, yDesc=%p, y=%p, hyDesc=%p, hy=%p, cyDesc=%p, cy=%p, findIntensity=%f, requestedAlgoCount=%d, returnedAlgoCount=%p, perfResults=%p, workspace=%p, workSpaceSizeInBytes=%ld, reserveSpace=%p, reserveSpaceSizeInBytes=%ld)\n",
        handle, rnnDesc, seqLength, xDesc, x, hxDesc, hx, cxDesc, cx, wDesc, w, yDesc, y, hyDesc, hy, cyDesc, cy, findIntensity, requestedAlgoCount, returnedAlgoCount, perfResults, workspace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnRNNDescriptor_t rnnDesc_native;
    int seqLength_native = 0;
    cudnnTensorDescriptor_t * xDesc_native;
    void * x_native = NULL;
    cudnnTensorDescriptor_t hxDesc_native;
    void * hx_native = NULL;
    cudnnTensorDescriptor_t cxDesc_native;
    void * cx_native = NULL;
    cudnnFilterDescriptor_t wDesc_native;
    void * w_native = NULL;
    cudnnTensorDescriptor_t * yDesc_native;
    void * y_native = NULL;
    cudnnTensorDescriptor_t hyDesc_native;
    void * hy_native = NULL;
    cudnnTensorDescriptor_t cyDesc_native;
    void * cy_native = NULL;
    float findIntensity_native = 0.0f;
    int requestedAlgoCount_native = 0;
    int returnedAlgoCount_native;
    cudnnAlgorithmPerformance_t * perfResults_native;
    void * workspace_native = NULL;
    size_t workSpaceSizeInBytes_native = 0;
    void * reserveSpace_native = NULL;
    size_t reserveSpaceSizeInBytes_native = 0;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    rnnDesc_native = (cudnnRNNDescriptor_t)getNativePointerValue(env, rnnDesc);
    seqLength_native = (int)seqLength;
    if (!initNative(env, xDesc, xDesc_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    x_native = (void *)getPointer(env, x);
    hxDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, hxDesc);
    hx_native = (void *)getPointer(env, hx);
    cxDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, cxDesc);
    cx_native = (void *)getPointer(env, cx);
    wDesc_native = (cudnnFilterDescriptor_t)getNativePointerValue(env, wDesc);
    w_native = (void *)getPointer(env, w);
    if (!initNative(env, yDesc, yDesc_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    y_native = (void *)getPointer(env, y);
    hyDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, hyDesc);
    hy_native = (void *)getPointer(env, hy);
    cyDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, cyDesc);
    cy_native = (void *)getPointer(env, cy);
    findIntensity_native = (float)findIntensity;
    requestedAlgoCount_native = (int)requestedAlgoCount;
    // returnedAlgoCount is write-only
    if (!initNative(env, perfResults, perfResults_native, requestedAlgoCount)) return JCUDNN_STATUS_INTERNAL_ERROR;
    workspace_native = (void *)getPointer(env, workspace);
    workSpaceSizeInBytes_native = (size_t)workSpaceSizeInBytes;
    reserveSpace_native = (void *)getPointer(env, reserveSpace);
    reserveSpaceSizeInBytes_native = (size_t)reserveSpaceSizeInBytes;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnFindRNNForwardTrainingAlgorithmEx(handle_native, rnnDesc_native, seqLength_native, xDesc_native, x_native, hxDesc_native, hx_native, cxDesc_native, cx_native, wDesc_native, w_native, yDesc_native, y_native, hyDesc_native, hy_native, cyDesc_native, cy_native, findIntensity_native, requestedAlgoCount_native, &returnedAlgoCount_native, perfResults_native, workspace_native, workSpaceSizeInBytes_native, reserveSpace_native, reserveSpaceSizeInBytes_native);

    // Write back native variable values
    // handle is read-only
    // rnnDesc is read-only
    // seqLength is primitive
    if (!releaseNative(env, xDesc_native, xDesc, false)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // x is a native pointer
    // hxDesc is read-only
    // hx is a native pointer
    // cxDesc is read-only
    // cx is a native pointer
    // wDesc is read-only
    // w is a native pointer
    if (!releaseNative(env, yDesc_native, yDesc, false)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // y is a native pointer
    // hyDesc is read-only
    // hy is a native pointer
    // cyDesc is read-only
    // cy is a native pointer
    // findIntensity is primitive
    // requestedAlgoCount is primitive
    if (!set(env, returnedAlgoCount, 0, (jint)returnedAlgoCount_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!releaseNative(env, perfResults_native, perfResults, returnedAlgoCount_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // workspace is a native pointer
    // workSpaceSizeInBytes is primitive
    // reserveSpace is a native pointer
    // reserveSpaceSizeInBytes is primitive

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetRNNBackwardDataAlgorithmMaxCountNative(JNIEnv *env, jclass cls, jobject handle, jobject rnnDesc, jintArray count)
{
    // Null-checks for non-primitive arguments
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cudnnGetRNNBackwardDataAlgorithmMaxCount");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (rnnDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'rnnDesc' is null for cudnnGetRNNBackwardDataAlgorithmMaxCount");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (count == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'count' is null for cudnnGetRNNBackwardDataAlgorithmMaxCount");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetRNNBackwardDataAlgorithmMaxCount(handle=%p, rnnDesc=%p, count=%p)\n",
        handle, rnnDesc, count);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnRNNDescriptor_t rnnDesc_native;
    int count_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    rnnDesc_native = (cudnnRNNDescriptor_t)getNativePointerValue(env, rnnDesc);
    // count is write-only

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetRNNBackwardDataAlgorithmMaxCount(handle_native, rnnDesc_native, &count_native);

    // Write back native variable values
    // handle is read-only
    // rnnDesc is read-only
    if (!set(env, count, 0, (jint)count_native)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnFindRNNBackwardDataAlgorithmExNative(JNIEnv *env, jclass cls, jobject handle, jobject rnnDesc, jint seqLength, jobjectArray yDesc, jobject y, jobjectArray dyDesc, jobject dy, jobject dhyDesc, jobject dhy, jobject dcyDesc, jobject dcy, jobject wDesc, jobject w, jobject hxDesc, jobject hx, jobject cxDesc, jobject cx, jobjectArray dxDesc, jobject dx, jobject dhxDesc, jobject dhx, jobject dcxDesc, jobject dcx, jfloat findIntensity, jint requestedAlgoCount, jintArray returnedAlgoCount, jobjectArray perfResults, jobject workspace, jlong workSpaceSizeInBytes, jobject reserveSpace, jlong reserveSpaceSizeInBytes)
{
    // Null-checks for non-primitive arguments
    if (perfResults == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'perfResults' is null for cudnnFindRNNBackwardDataAlgorithmEx");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnFindRNNBackwardDataAlgorithmEx(handle=%p, rnnDesc=%p, seqLength=%d, yDesc=%p, y=%p, dyDesc=%p, dy=%p, dhyDesc=%p, dhy=%p, dcyDesc=%p, dcy=%p, wDesc=%p, w=%p, hxDesc=%p, hx=%p, cxDesc=%p, cx=%p, dxDesc=%p, dx=%p, dhxDesc=%p, dhx=%p, dcxDesc=%p, dcx=%p, findIntensity=%f, requestedAlgoCount=%d, returnedAlgoCount=%p, perfResults=%p, workspace=%p, workSpaceSizeInBytes=%ld, reserveSpace=%p, reserveSpaceSizeInBytes=%ld)\n",
        handle, rnnDesc, seqLength, yDesc, y, dyDesc, dy, dhyDesc, dhy, dcyDesc, dcy, wDesc, w, hxDesc, hx, cxDesc, cx, dxDesc, dx, dhxDesc, dhx, dcxDesc, dcx, findIntensity, requestedAlgoCount, returnedAlgoCount, perfResults, workspace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnRNNDescriptor_t rnnDesc_native;
    int seqLength_native = 0;
    cudnnTensorDescriptor_t * yDesc_native;
    void * y_native = NULL;
    cudnnTensorDescriptor_t * dyDesc_native;
    void * dy_native = NULL;
    cudnnTensorDescriptor_t dhyDesc_native;
    void * dhy_native = NULL;
    cudnnTensorDescriptor_t dcyDesc_native;
    void * dcy_native = NULL;
    cudnnFilterDescriptor_t wDesc_native;
    void * w_native = NULL;
    cudnnTensorDescriptor_t hxDesc_native;
    void * hx_native = NULL;
    cudnnTensorDescriptor_t cxDesc_native;
    void * cx_native = NULL;
    cudnnTensorDescriptor_t * dxDesc_native;
    void * dx_native = NULL;
    cudnnTensorDescriptor_t dhxDesc_native;
    void * dhx_native = NULL;
    cudnnTensorDescriptor_t dcxDesc_native;
    void * dcx_native = NULL;
    float findIntensity_native = 0.0f;
    int requestedAlgoCount_native = 0;
    int returnedAlgoCount_native;
    cudnnAlgorithmPerformance_t * perfResults_native;
    void * workspace_native = NULL;
    size_t workSpaceSizeInBytes_native = 0;
    void * reserveSpace_native = NULL;
    size_t reserveSpaceSizeInBytes_native = 0;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    rnnDesc_native = (cudnnRNNDescriptor_t)getNativePointerValue(env, rnnDesc);
    seqLength_native = (int)seqLength;
    if (!initNative(env, yDesc, yDesc_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    y_native = (void *)getPointer(env, y);
    if (!initNative(env, dyDesc, dyDesc_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    dy_native = (void *)getPointer(env, dy);
    dhyDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, dhyDesc);
    dhy_native = (void *)getPointer(env, dhy);
    dcyDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, dcyDesc);
    dcy_native = (void *)getPointer(env, dcy);
    wDesc_native = (cudnnFilterDescriptor_t)getNativePointerValue(env, wDesc);
    w_native = (void *)getPointer(env, w);
    hxDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, hxDesc);
    hx_native = (void *)getPointer(env, hx);
    cxDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, cxDesc);
    cx_native = (void *)getPointer(env, cx);
    if (!initNative(env, dxDesc, dxDesc_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    dx_native = (void *)getPointer(env, dx);
    dhxDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, dhxDesc);
    dhx_native = (void *)getPointer(env, dhx);
    dcxDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, dcxDesc);
    dcx_native = (void *)getPointer(env, dcx);
    findIntensity_native = (float)findIntensity;
    requestedAlgoCount_native = (int)requestedAlgoCount;
    // returnedAlgoCount is write-only
    if (!initNative(env, perfResults, perfResults_native, requestedAlgoCount)) return JCUDNN_STATUS_INTERNAL_ERROR;
    workspace_native = (void *)getPointer(env, workspace);
    workSpaceSizeInBytes_native = (size_t)workSpaceSizeInBytes;
    reserveSpace_native = (void *)getPointer(env, reserveSpace);
    reserveSpaceSizeInBytes_native = (size_t)reserveSpaceSizeInBytes;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnFindRNNBackwardDataAlgorithmEx(handle_native, rnnDesc_native, seqLength_native, yDesc_native, y_native, dyDesc_native, dy_native, dhyDesc_native, dhy_native, dcyDesc_native, dcy_native, wDesc_native, w_native, hxDesc_native, hx_native, cxDesc_native, cx_native, dxDesc_native, dx_native, dhxDesc_native, dhx_native, dcxDesc_native, dcx_native, findIntensity_native, requestedAlgoCount_native, &returnedAlgoCount_native, perfResults_native, workspace_native, workSpaceSizeInBytes_native, reserveSpace_native, reserveSpaceSizeInBytes_native);

    // Write back native variable values
    // handle is read-only
    // rnnDesc is read-only
    // seqLength is primitive
    if (!releaseNative(env, yDesc_native, yDesc, false)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // y is a native pointer
    if (!releaseNative(env, dyDesc_native, dyDesc, false)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // dy is a native pointer
    // dhyDesc is read-only
    // dhy is a native pointer
    // dcyDesc is read-only
    // dcy is a native pointer
    // wDesc is read-only
    // w is a native pointer
    // hxDesc is read-only
    // hx is a native pointer
    // cxDesc is read-only
    // cx is a native pointer
    if (!releaseNative(env, dxDesc_native, dxDesc, false)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // dx is a native pointer
    // dhxDesc is read-only
    // dhx is a native pointer
    // dcxDesc is read-only
    // dcx is a native pointer
    // findIntensity is primitive
    // requestedAlgoCount is primitive
    if (!set(env, returnedAlgoCount, 0, (jint)returnedAlgoCount_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!releaseNative(env, perfResults_native, perfResults, returnedAlgoCount_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // workspace is a native pointer
    // workSpaceSizeInBytes is primitive
    // reserveSpace is a native pointer
    // reserveSpaceSizeInBytes is primitive

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetRNNBackwardWeightsAlgorithmMaxCountNative(JNIEnv *env, jclass cls, jobject handle, jobject rnnDesc, jintArray count)
{
    // Null-checks for non-primitive arguments
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cudnnGetRNNBackwardWeightsAlgorithmMaxCount");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (rnnDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'rnnDesc' is null for cudnnGetRNNBackwardWeightsAlgorithmMaxCount");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (count == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'count' is null for cudnnGetRNNBackwardWeightsAlgorithmMaxCount");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetRNNBackwardWeightsAlgorithmMaxCount(handle=%p, rnnDesc=%p, count=%p)\n",
        handle, rnnDesc, count);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnRNNDescriptor_t rnnDesc_native;
    int count_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    rnnDesc_native = (cudnnRNNDescriptor_t)getNativePointerValue(env, rnnDesc);
    // count is write-only

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetRNNBackwardWeightsAlgorithmMaxCount(handle_native, rnnDesc_native, &count_native);

    // Write back native variable values
    // handle is read-only
    // rnnDesc is read-only
    if (!set(env, count, 0, (jint)count_native)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnFindRNNBackwardWeightsAlgorithmExNative(JNIEnv *env, jclass cls, jobject handle, jobject rnnDesc, jint seqLength, jobjectArray xDesc, jobject x, jobject hxDesc, jobject hx, jobjectArray yDesc, jobject y, jfloat findIntensity, jint requestedAlgoCount, jintArray returnedAlgoCount, jobjectArray perfResults, jobject workspace, jlong workSpaceSizeInBytes, jobject dwDesc, jobject dw, jobject reserveSpace, jlong reserveSpaceSizeInBytes)
{
    // Null-checks for non-primitive arguments
    if (perfResults == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'perfResults' is null for cudnnFindRNNBackwardWeightsAlgorithmEx");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnFindRNNBackwardWeightsAlgorithmEx(handle=%p, rnnDesc=%p, seqLength=%d, xDesc=%p, x=%p, hxDesc=%p, hx=%p, yDesc=%p, y=%p, findIntensity=%f, requestedAlgoCount=%d, returnedAlgoCount=%p, perfResults=%p, workspace=%p, workSpaceSizeInBytes=%ld, dwDesc=%p, dw=%p, reserveSpace=%p, reserveSpaceSizeInBytes=%ld)\n",
        handle, rnnDesc, seqLength, xDesc, x, hxDesc, hx, yDesc, y, findIntensity, requestedAlgoCount, returnedAlgoCount, perfResults, workspace, workSpaceSizeInBytes, dwDesc, dw, reserveSpace, reserveSpaceSizeInBytes);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnRNNDescriptor_t rnnDesc_native;
    int seqLength_native = 0;
    cudnnTensorDescriptor_t * xDesc_native;
    void * x_native = NULL;
    cudnnTensorDescriptor_t hxDesc_native;
    void * hx_native = NULL;
    cudnnTensorDescriptor_t * yDesc_native;
    void * y_native = NULL;
    float findIntensity_native = 0.0f;
    int requestedAlgoCount_native = 0;
    int returnedAlgoCount_native;
    cudnnAlgorithmPerformance_t * perfResults_native;
    void * workspace_native = NULL;
    size_t workSpaceSizeInBytes_native = 0;
    cudnnFilterDescriptor_t dwDesc_native;
    void * dw_native = NULL;
    void * reserveSpace_native = NULL;
    size_t reserveSpaceSizeInBytes_native = 0;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    rnnDesc_native = (cudnnRNNDescriptor_t)getNativePointerValue(env, rnnDesc);
    seqLength_native = (int)seqLength;
    if (!initNative(env, xDesc, xDesc_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    x_native = (void *)getPointer(env, x);
    hxDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, hxDesc);
    hx_native = (void *)getPointer(env, hx);
    if (!initNative(env, yDesc, yDesc_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    y_native = (void *)getPointer(env, y);
    findIntensity_native = (float)findIntensity;
    requestedAlgoCount_native = (int)requestedAlgoCount;
    // returnedAlgoCount is write-only
    if (!initNative(env, perfResults, perfResults_native, requestedAlgoCount)) return JCUDNN_STATUS_INTERNAL_ERROR;
    workspace_native = (void *)getPointer(env, workspace);
    workSpaceSizeInBytes_native = (size_t)workSpaceSizeInBytes;
    dwDesc_native = (cudnnFilterDescriptor_t)getNativePointerValue(env, dwDesc);
    dw_native = (void *)getPointer(env, dw);
    reserveSpace_native = (void *)getPointer(env, reserveSpace);
    reserveSpaceSizeInBytes_native = (size_t)reserveSpaceSizeInBytes;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnFindRNNBackwardWeightsAlgorithmEx(handle_native, rnnDesc_native, seqLength_native, xDesc_native, x_native, hxDesc_native, hx_native, yDesc_native, y_native, findIntensity_native, requestedAlgoCount_native, &returnedAlgoCount_native, perfResults_native, workspace_native, workSpaceSizeInBytes_native, dwDesc_native, dw_native, reserveSpace_native, reserveSpaceSizeInBytes_native);

    // Write back native variable values
    // handle is read-only
    // rnnDesc is read-only
    // seqLength is primitive
    if (!releaseNative(env, xDesc_native, xDesc, false)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // x is a native pointer
    // hxDesc is read-only
    // hx is a native pointer
    if (!releaseNative(env, yDesc_native, yDesc, false)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // y is a native pointer
    // findIntensity is primitive
    // requestedAlgoCount is primitive
    if (!set(env, returnedAlgoCount, 0, (jint)returnedAlgoCount_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!releaseNative(env, perfResults_native, perfResults, returnedAlgoCount_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // workspace is a native pointer
    // workSpaceSizeInBytes is primitive
    // dwDesc is read-only
    // dw is a native pointer
    // reserveSpace is a native pointer
    // reserveSpaceSizeInBytes is primitive

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnMultiHeadAttnBackwardDataNative(JNIEnv *env, jclass cls, jobject handle, jobject attnDesc, jintArray loWinIdx, jintArray hiWinIdx, jintArray devSeqLengthsDQDO, jintArray devSeqLengthsDKDV, jobject doDesc, jobject dout, jobject dqDesc, jobject dqueries, jobject queries, jobject dkDesc, jobject dkeys, jobject keys, jobject dvDesc, jobject dvalues, jobject values, jlong weightSizeInBytes, jobject weights, jlong workSpaceSizeInBytes, jobject workSpace, jlong reserveSpaceSizeInBytes, jobject reserveSpace)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnMultiHeadAttnBackwardData(handle=%p, attnDesc=%p, loWinIdx=%p, hiWinIdx=%p, devSeqLengthsDQDO=%p, devSeqLengthsDKDV=%p, doDesc=%p, dout=%p, dqDesc=%p, dqueries=%p, queries=%p, dkDesc=%p, dkeys=%p, keys=%p, dvDesc=%p, dvalues=%p, values=%p, weightSizeInBytes=%ld, weights=%p, workSpaceSizeInBytes=%ld, workSpace=%p, reserveSpaceSizeInBytes=%ld, reserveSpace=%p)\n",
        handle, attnDesc, loWinIdx, hiWinIdx, devSeqLengthsDQDO, devSeqLengthsDKDV, doDesc, dout, dqDesc, dqueries, queries, dkDesc, dkeys, keys, dvDesc, dvalues, values, weightSizeInBytes, weights, workSpaceSizeInBytes, workSpace, reserveSpaceSizeInBytes, reserveSpace);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnAttnDescriptor_t attnDesc_native;
    int * loWinIdx_native = NULL;
    int * hiWinIdx_native = NULL;
    int * devSeqLengthsDQDO_native = NULL;
    int * devSeqLengthsDKDV_native = NULL;
    cudnnSeqDataDescriptor_t doDesc_native;
    void * dout_native = NULL;
    cudnnSeqDataDescriptor_t dqDesc_native;
    void * dqueries_native = NULL;
    void * queries_native = NULL;
    cudnnSeqDataDescriptor_t dkDesc_native;
    void * dkeys_native = NULL;
    void * keys_native = NULL;
    cudnnSeqDataDescriptor_t dvDesc_native;
    void * dvalues_native = NULL;
    void * values_native = NULL;
    size_t weightSizeInBytes_native = 0;
    void * weights_native = NULL;
    size_t workSpaceSizeInBytes_native = 0;
    void * workSpace_native = NULL;
    size_t reserveSpaceSizeInBytes_native = 0;
    void * reserveSpace_native = NULL;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    attnDesc_native = (cudnnAttnDescriptor_t)getNativePointerValue(env, attnDesc);
    if (!initNative(env, loWinIdx, loWinIdx_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!initNative(env, hiWinIdx, hiWinIdx_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    devSeqLengthsDQDO_native = (int *)getPointer(env, devSeqLengthsDQDO);
    devSeqLengthsDKDV_native = (int *)getPointer(env, devSeqLengthsDKDV);
    doDesc_native = (cudnnSeqDataDescriptor_t)getNativePointerValue(env, doDesc);
    dout_native = (void *)getPointer(env, dout);
    dqDesc_native = (cudnnSeqDataDescriptor_t)getNativePointerValue(env, dqDesc);
    dqueries_native = (void *)getPointer(env, dqueries);
    queries_native = (void *)getPointer(env, queries);
    dkDesc_native = (cudnnSeqDataDescriptor_t)getNativePointerValue(env, dkDesc);
    dkeys_native = (void *)getPointer(env, dkeys);
    keys_native = (void *)getPointer(env, keys);
    dvDesc_native = (cudnnSeqDataDescriptor_t)getNativePointerValue(env, dvDesc);
    dvalues_native = (void *)getPointer(env, dvalues);
    values_native = (void *)getPointer(env, values);
    weightSizeInBytes_native = (size_t)weightSizeInBytes;
    weights_native = (void *)getPointer(env, weights);
    workSpaceSizeInBytes_native = (size_t)workSpaceSizeInBytes;
    workSpace_native = (void *)getPointer(env, workSpace);
    reserveSpaceSizeInBytes_native = (size_t)reserveSpaceSizeInBytes;
    reserveSpace_native = (void *)getPointer(env, reserveSpace);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnMultiHeadAttnBackwardData(handle_native, attnDesc_native, loWinIdx_native, hiWinIdx_native, devSeqLengthsDQDO_native, devSeqLengthsDKDV_native, doDesc_native, dout_native, dqDesc_native, dqueries_native, queries_native, dkDesc_native, dkeys_native, keys_native, dvDesc_native, dvalues_native, values_native, weightSizeInBytes_native, weights_native, workSpaceSizeInBytes_native, workSpace_native, reserveSpaceSizeInBytes_native, reserveSpace_native);

    // Write back native variable values
    // handle is read-only
    // attnDesc is read-only
    if (!releaseNative(env, loWinIdx_native, loWinIdx, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!releaseNative(env, hiWinIdx_native, hiWinIdx, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // devSeqLengthsDQDO is a native pointer
    // devSeqLengthsDKDV is a native pointer
    // doDesc is read-only
    // dout is a native pointer
    // dqDesc is read-only
    // dqueries is a native pointer
    // queries is a native pointer
    // dkDesc is read-only
    // dkeys is a native pointer
    // keys is a native pointer
    // dvDesc is read-only
    // dvalues is a native pointer
    // values is a native pointer
    // weightSizeInBytes is primitive
    // weights is a native pointer
    // workSpaceSizeInBytes is primitive
    // workSpace is a native pointer
    // reserveSpaceSizeInBytes is primitive
    // reserveSpace is a native pointer

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnMultiHeadAttnBackwardWeightsNative(JNIEnv *env, jclass cls, jobject handle, jobject attnDesc, jint addGrad, jobject qDesc, jobject queries, jobject kDesc, jobject keys, jobject vDesc, jobject values, jobject doDesc, jobject dout, jlong weightSizeInBytes, jobject weights, jobject dweights, jlong workSpaceSizeInBytes, jobject workSpace, jlong reserveSpaceSizeInBytes, jobject reserveSpace)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnMultiHeadAttnBackwardWeights(handle=%p, attnDesc=%p, addGrad=%d, qDesc=%p, queries=%p, kDesc=%p, keys=%p, vDesc=%p, values=%p, doDesc=%p, dout=%p, weightSizeInBytes=%ld, weights=%p, dweights=%p, workSpaceSizeInBytes=%ld, workSpace=%p, reserveSpaceSizeInBytes=%ld, reserveSpace=%p)\n",
        handle, attnDesc, addGrad, qDesc, queries, kDesc, keys, vDesc, values, doDesc, dout, weightSizeInBytes, weights, dweights, workSpaceSizeInBytes, workSpace, reserveSpaceSizeInBytes, reserveSpace);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnAttnDescriptor_t attnDesc_native;
    cudnnWgradMode_t addGrad_native;
    cudnnSeqDataDescriptor_t qDesc_native;
    void * queries_native = NULL;
    cudnnSeqDataDescriptor_t kDesc_native;
    void * keys_native = NULL;
    cudnnSeqDataDescriptor_t vDesc_native;
    void * values_native = NULL;
    cudnnSeqDataDescriptor_t doDesc_native;
    void * dout_native = NULL;
    size_t weightSizeInBytes_native = 0;
    void * weights_native = NULL;
    void * dweights_native = NULL;
    size_t workSpaceSizeInBytes_native = 0;
    void * workSpace_native = NULL;
    size_t reserveSpaceSizeInBytes_native = 0;
    void * reserveSpace_native = NULL;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    attnDesc_native = (cudnnAttnDescriptor_t)getNativePointerValue(env, attnDesc);
    addGrad_native = (cudnnWgradMode_t)addGrad;
    qDesc_native = (cudnnSeqDataDescriptor_t)getNativePointerValue(env, qDesc);
    queries_native = (void *)getPointer(env, queries);
    kDesc_native = (cudnnSeqDataDescriptor_t)getNativePointerValue(env, kDesc);
    keys_native = (void *)getPointer(env, keys);
    vDesc_native = (cudnnSeqDataDescriptor_t)getNativePointerValue(env, vDesc);
    values_native = (void *)getPointer(env, values);
    doDesc_native = (cudnnSeqDataDescriptor_t)getNativePointerValue(env, doDesc);
    dout_native = (void *)getPointer(env, dout);
    weightSizeInBytes_native = (size_t)weightSizeInBytes;
    weights_native = (void *)getPointer(env, weights);
    dweights_native = (void *)getPointer(env, dweights);
    workSpaceSizeInBytes_native = (size_t)workSpaceSizeInBytes;
    workSpace_native = (void *)getPointer(env, workSpace);
    reserveSpaceSizeInBytes_native = (size_t)reserveSpaceSizeInBytes;
    reserveSpace_native = (void *)getPointer(env, reserveSpace);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnMultiHeadAttnBackwardWeights(handle_native, attnDesc_native, addGrad_native, qDesc_native, queries_native, kDesc_native, keys_native, vDesc_native, values_native, doDesc_native, dout_native, weightSizeInBytes_native, weights_native, dweights_native, workSpaceSizeInBytes_native, workSpace_native, reserveSpaceSizeInBytes_native, reserveSpace_native);

    // Write back native variable values
    // handle is read-only
    // attnDesc is read-only
    // addGrad is primitive
    // qDesc is read-only
    // queries is a native pointer
    // kDesc is read-only
    // keys is a native pointer
    // vDesc is read-only
    // values is a native pointer
    // doDesc is read-only
    // dout is a native pointer
    // weightSizeInBytes is primitive
    // weights is a native pointer
    // dweights is a native pointer
    // workSpaceSizeInBytes is primitive
    // workSpace is a native pointer
    // reserveSpaceSizeInBytes is primitive
    // reserveSpace is a native pointer

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnCreateCTCLossDescriptorNative(JNIEnv *env, jclass cls, jobject ctcLossDesc)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnCreateCTCLossDescriptor(ctcLossDesc=%p)\n",
        ctcLossDesc);

    // Native variable declarations
    cudnnCTCLossDescriptor_t ctcLossDesc_native;

    // Obtain native variable values
    // ctcLossDesc is write-only

    // Native function call
    cudnnStatus_t jniResult_native = cudnnCreateCTCLossDescriptor(&ctcLossDesc_native);

    // Write back native variable values
    setNativePointerValue(env, ctcLossDesc, (jlong)ctcLossDesc_native);

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnSetCTCLossDescriptorNative(JNIEnv *env, jclass cls, jobject ctcLossDesc, jint compType)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnSetCTCLossDescriptor(ctcLossDesc=%p, compType=%d)\n",
        ctcLossDesc, compType);

    // Native variable declarations
    cudnnCTCLossDescriptor_t ctcLossDesc_native;
    cudnnDataType_t compType_native;

    // Obtain native variable values
    ctcLossDesc_native = (cudnnCTCLossDescriptor_t)getNativePointerValue(env, ctcLossDesc);
    compType_native = (cudnnDataType_t)compType;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnSetCTCLossDescriptor(ctcLossDesc_native, compType_native);

    // Write back native variable values
    // ctcLossDesc is read-only
    // compType is primitive

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnSetCTCLossDescriptorExNative(JNIEnv *env, jclass cls, jobject ctcLossDesc, jint compType, jint normMode, jint gradMode)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnSetCTCLossDescriptorEx(ctcLossDesc=%p, compType=%d, normMode=%d, gradMode=%d)\n",
        ctcLossDesc, compType, normMode, gradMode);

    // Native variable declarations
    cudnnCTCLossDescriptor_t ctcLossDesc_native;
    cudnnDataType_t compType_native;
    cudnnLossNormalizationMode_t normMode_native;
    cudnnNanPropagation_t gradMode_native;

    // Obtain native variable values
    ctcLossDesc_native = (cudnnCTCLossDescriptor_t)getNativePointerValue(env, ctcLossDesc);
    compType_native = (cudnnDataType_t)compType;
    normMode_native = (cudnnLossNormalizationMode_t)normMode;
    gradMode_native = (cudnnNanPropagation_t)gradMode;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnSetCTCLossDescriptorEx(ctcLossDesc_native, compType_native, normMode_native, gradMode_native);

    // Write back native variable values
    // ctcLossDesc is read-only
    // compType is primitive
    // normMode is primitive
    // gradMode is primitive

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnSetCTCLossDescriptor_1v8Native(JNIEnv *env, jclass cls, jobject ctcLossDesc, jint compType, jint normMode, jint gradMode, jint maxLabelLength)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnSetCTCLossDescriptor_v8(ctcLossDesc=%p, compType=%d, normMode=%d, gradMode=%d, maxLabelLength=%d)\n",
        ctcLossDesc, compType, normMode, gradMode, maxLabelLength);

    // Native variable declarations
    cudnnCTCLossDescriptor_t ctcLossDesc_native;
    cudnnDataType_t compType_native;
    cudnnLossNormalizationMode_t normMode_native;
    cudnnNanPropagation_t gradMode_native;
    int maxLabelLength_native = 0;

    // Obtain native variable values
    ctcLossDesc_native = (cudnnCTCLossDescriptor_t)getNativePointerValue(env, ctcLossDesc);
    compType_native = (cudnnDataType_t)compType;
    normMode_native = (cudnnLossNormalizationMode_t)normMode;
    gradMode_native = (cudnnNanPropagation_t)gradMode;
    maxLabelLength_native = (int)maxLabelLength;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnSetCTCLossDescriptor_v8(ctcLossDesc_native, compType_native, normMode_native, gradMode_native, maxLabelLength_native);

    // Write back native variable values
    // ctcLossDesc is read-only
    // compType is primitive
    // normMode is primitive
    // gradMode is primitive
    // maxLabelLength is primitive

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetCTCLossDescriptorNative(JNIEnv *env, jclass cls, jobject ctcLossDesc, jintArray compType)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetCTCLossDescriptor(ctcLossDesc=%p, compType=%p)\n",
        ctcLossDesc, compType);

    // Native variable declarations
    cudnnCTCLossDescriptor_t ctcLossDesc_native;
    cudnnDataType_t compType_native;

    // Obtain native variable values
    ctcLossDesc_native = (cudnnCTCLossDescriptor_t)getNativePointerValue(env, ctcLossDesc);
    // compType is write-only

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetCTCLossDescriptor(ctcLossDesc_native, &compType_native);

    // Write back native variable values
    // ctcLossDesc is read-only
    if (!set(env, compType, 0, (jint)compType_native)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetCTCLossDescriptorExNative(JNIEnv *env, jclass cls, jobject ctcLossDesc, jintArray compType, jintArray normMode, jintArray gradMode)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetCTCLossDescriptorEx(ctcLossDesc=%p, compType=%p, normMode=%p, gradMode=%p)\n",
        ctcLossDesc, compType, normMode, gradMode);

    // Native variable declarations
    cudnnCTCLossDescriptor_t ctcLossDesc_native;
    cudnnDataType_t compType_native;
    cudnnLossNormalizationMode_t normMode_native;
    cudnnNanPropagation_t gradMode_native;

    // Obtain native variable values
    ctcLossDesc_native = (cudnnCTCLossDescriptor_t)getNativePointerValue(env, ctcLossDesc);
    // compType is write-only
    // normMode is write-only
    // gradMode is write-only

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetCTCLossDescriptorEx(ctcLossDesc_native, &compType_native, &normMode_native, &gradMode_native);

    // Write back native variable values
    // ctcLossDesc is read-only
    if (!set(env, compType, 0, (jint)compType_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, normMode, 0, (jint)normMode_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, gradMode, 0, (jint)gradMode_native)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetCTCLossDescriptor_1v8Native(JNIEnv *env, jclass cls, jobject ctcLossDesc, jintArray compType, jintArray normMode, jintArray gradMode, jintArray maxLabelLength)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetCTCLossDescriptor_v8(ctcLossDesc=%p, compType=%p, normMode=%p, gradMode=%p, maxLabelLength=%p)\n",
        ctcLossDesc, compType, normMode, gradMode, maxLabelLength);

    // Native variable declarations
    cudnnCTCLossDescriptor_t ctcLossDesc_native;
    cudnnDataType_t compType_native;
    cudnnLossNormalizationMode_t normMode_native;
    cudnnNanPropagation_t gradMode_native;
    int maxLabelLength_native;

    // Obtain native variable values
    ctcLossDesc_native = (cudnnCTCLossDescriptor_t)getNativePointerValue(env, ctcLossDesc);
    // compType is write-only
    // normMode is write-only
    // gradMode is write-only
    // maxLabelLength is write-only

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetCTCLossDescriptor_v8(ctcLossDesc_native, &compType_native, &normMode_native, &gradMode_native, &maxLabelLength_native);

    // Write back native variable values
    // ctcLossDesc is read-only
    if (!set(env, compType, 0, (jint)compType_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, normMode, 0, (jint)normMode_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, gradMode, 0, (jint)gradMode_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, maxLabelLength, 0, (jint)maxLabelLength_native)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnDestroyCTCLossDescriptorNative(JNIEnv *env, jclass cls, jobject ctcLossDesc)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnDestroyCTCLossDescriptor(ctcLossDesc=%p)\n",
        ctcLossDesc);

    // Native variable declarations
    cudnnCTCLossDescriptor_t ctcLossDesc_native;

    // Obtain native variable values
    ctcLossDesc_native = (cudnnCTCLossDescriptor_t)getNativePointerValue(env, ctcLossDesc);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnDestroyCTCLossDescriptor(ctcLossDesc_native);

    // Write back native variable values
    // ctcLossDesc is read-only

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

/** return the ctc costs and gradients, given the probabilities and labels */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnCTCLossNative(JNIEnv *env, jclass cls, jobject handle, jobject probsDesc, jobject probs, jintArray hostLabels, jintArray hostLabelLengths, jintArray hostInputLengths, jobject costs, jobject gradientsDesc, jobject gradients, jint algo, jobject ctcLossDesc, jobject workspace, jlong workSpaceSizeInBytes)
{
    // Null-checks for non-primitive arguments
    if (hostLabels == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hostLabels' is null for cudnnCTCLoss");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (hostLabelLengths == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hostLabelLengths' is null for cudnnCTCLoss");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (hostInputLengths == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hostInputLengths' is null for cudnnCTCLoss");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnCTCLoss(handle=%p, probsDesc=%p, probs=%p, hostLabels=%p, hostLabelLengths=%p, hostInputLengths=%p, costs=%p, gradientsDesc=%p, gradients=%p, algo=%d, ctcLossDesc=%p, workspace=%p, workSpaceSizeInBytes=%ld)\n",
        handle, probsDesc, probs, hostLabels, hostLabelLengths, hostInputLengths, costs, gradientsDesc, gradients, algo, ctcLossDesc, workspace, workSpaceSizeInBytes);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnTensorDescriptor_t probsDesc_native;
    void * probs_native = NULL;
    int * hostLabels_native = NULL;
    int * hostLabelLengths_native = NULL;
    int * hostInputLengths_native = NULL;
    void * costs_native = NULL;
    cudnnTensorDescriptor_t gradientsDesc_native;
    void * gradients_native = NULL;
    cudnnCTCLossAlgo_t algo_native;
    cudnnCTCLossDescriptor_t ctcLossDesc_native;
    void * workspace_native = NULL;
    size_t workSpaceSizeInBytes_native = 0;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    probsDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, probsDesc);
    probs_native = (void *)getPointer(env, probs);
    if (!initNative(env, hostLabels, hostLabels_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!initNative(env, hostLabelLengths, hostLabelLengths_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!initNative(env, hostInputLengths, hostInputLengths_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    costs_native = (void *)getPointer(env, costs);
    gradientsDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, gradientsDesc);
    gradients_native = (void *)getPointer(env, gradients);
    algo_native = (cudnnCTCLossAlgo_t)algo;
    ctcLossDesc_native = (cudnnCTCLossDescriptor_t)getNativePointerValue(env, ctcLossDesc);
    workspace_native = (void *)getPointer(env, workspace);
    workSpaceSizeInBytes_native = (size_t)workSpaceSizeInBytes;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnCTCLoss(handle_native, probsDesc_native, probs_native, hostLabels_native, hostLabelLengths_native, hostInputLengths_native, costs_native, gradientsDesc_native, gradients_native, algo_native, ctcLossDesc_native, workspace_native, workSpaceSizeInBytes_native);

    // Write back native variable values
    // handle is read-only
    // probsDesc is read-only
    // probs is a native pointer
    if (!releaseNative(env, hostLabels_native, hostLabels, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!releaseNative(env, hostLabelLengths_native, hostLabelLengths, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!releaseNative(env, hostInputLengths_native, hostInputLengths, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // costs is a native pointer
    // gradientsDesc is read-only
    // gradients is a native pointer
    // algo is primitive
    // ctcLossDesc is read-only
    // workspace is a native pointer
    // workSpaceSizeInBytes is primitive

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

/** return the ctc costs and gradients, given the probabilities and labels */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnCTCLoss_1v8Native(JNIEnv *env, jclass cls, jobject handle, jint algo, jobject ctcLossDesc, jobject probsDesc, jobject probs, jobject labels, jobject labelLengths, jobject inputLengths, jobject costs, jobject gradientsDesc, jobject gradients, jlong workSpaceSizeInBytes, jobject workspace)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnCTCLoss_v8(handle=%p, algo=%d, ctcLossDesc=%p, probsDesc=%p, probs=%p, labels=%p, labelLengths=%p, inputLengths=%p, costs=%p, gradientsDesc=%p, gradients=%p, workSpaceSizeInBytes=%ld, workspace=%p)\n",
        handle, algo, ctcLossDesc, probsDesc, probs, labels, labelLengths, inputLengths, costs, gradientsDesc, gradients, workSpaceSizeInBytes, workspace);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnCTCLossAlgo_t algo_native;
    cudnnCTCLossDescriptor_t ctcLossDesc_native;
    cudnnTensorDescriptor_t probsDesc_native;
    void * probs_native = NULL;
    int * labels_native = NULL;
    int * labelLengths_native = NULL;
    int * inputLengths_native = NULL;
    void * costs_native = NULL;
    cudnnTensorDescriptor_t gradientsDesc_native;
    void * gradients_native = NULL;
    size_t workSpaceSizeInBytes_native = 0;
    void * workspace_native = NULL;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    algo_native = (cudnnCTCLossAlgo_t)algo;
    ctcLossDesc_native = (cudnnCTCLossDescriptor_t)getNativePointerValue(env, ctcLossDesc);
    probsDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, probsDesc);
    probs_native = (void *)getPointer(env, probs);
    labels_native = (int *)getPointer(env, labels);
    labelLengths_native = (int *)getPointer(env, labelLengths);
    inputLengths_native = (int *)getPointer(env, inputLengths);
    costs_native = (void *)getPointer(env, costs);
    gradientsDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, gradientsDesc);
    gradients_native = (void *)getPointer(env, gradients);
    workSpaceSizeInBytes_native = (size_t)workSpaceSizeInBytes;
    workspace_native = (void *)getPointer(env, workspace);

    cudnnStatus_t jniResult_native = CUDNN_STATUS_NOT_INITIALIZED;
    // Native function call
    if(handle != nullptr)
        jniResult_native = cudnnCTCLoss_v8(handle_native, algo_native, ctcLossDesc_native, probsDesc_native, probs_native, labels_native, labelLengths_native, inputLengths_native, costs_native, gradientsDesc_native, gradients_native, workSpaceSizeInBytes_native, workspace_native);

    // Write back native variable values
    // handle is read-only
    // algo is primitive
    // ctcLossDesc is read-only
    // probsDesc is read-only
    // probs is a native pointer
    // labels is a native pointer
    // labelLengths is a native pointer
    // inputLengths is a native pointer
    // costs is a native pointer
    // gradientsDesc is read-only
    // gradients is a native pointer
    // workSpaceSizeInBytes is primitive
    // workspace is a native pointer

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

/** return the workspace size needed for ctc */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetCTCLossWorkspaceSizeNative(JNIEnv *env, jclass cls, jobject handle, jobject probsDesc, jobject gradientsDesc, jintArray labels, jintArray labelLengths, jintArray inputLengths, jint algo, jobject ctcLossDesc, jlongArray sizeInBytes)
{
    // Null-checks for non-primitive arguments
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cudnnGetCTCLossWorkspaceSize");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (probsDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'probsDesc' is null for cudnnGetCTCLossWorkspaceSize");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (gradientsDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'gradientsDesc' is null for cudnnGetCTCLossWorkspaceSize");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (labels == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'labels' is null for cudnnGetCTCLossWorkspaceSize");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (labelLengths == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'labelLengths' is null for cudnnGetCTCLossWorkspaceSize");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (inputLengths == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'inputLengths' is null for cudnnGetCTCLossWorkspaceSize");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    // algo is primitive
    if (ctcLossDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'ctcLossDesc' is null for cudnnGetCTCLossWorkspaceSize");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (sizeInBytes == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'sizeInBytes' is null for cudnnGetCTCLossWorkspaceSize");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetCTCLossWorkspaceSize(handle=%p, probsDesc=%p, gradientsDesc=%p, labels=%p, labelLengths=%p, inputLengths=%p, algo=%d, ctcLossDesc=%p, sizeInBytes=%p)\n",
        handle, probsDesc, gradientsDesc, labels, labelLengths, inputLengths, algo, ctcLossDesc, sizeInBytes);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnTensorDescriptor_t probsDesc_native;
    cudnnTensorDescriptor_t gradientsDesc_native;
    int * labels_native = NULL;
    int * labelLengths_native = NULL;
    int * inputLengths_native = NULL;
    cudnnCTCLossAlgo_t algo_native;
    cudnnCTCLossDescriptor_t ctcLossDesc_native;
    size_t sizeInBytes_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    probsDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, probsDesc);
    gradientsDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, gradientsDesc);
    labels_native = (int *)getPointer(env, labels);
    labelLengths_native = (int *)getPointer(env, labelLengths);
    inputLengths_native = (int *)getPointer(env, inputLengths);
    algo_native = (cudnnCTCLossAlgo_t)algo;
    ctcLossDesc_native = (cudnnCTCLossDescriptor_t)getNativePointerValue(env, ctcLossDesc);
    // sizeInBytes is write-only

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetCTCLossWorkspaceSize(handle_native, probsDesc_native, gradientsDesc_native, labels_native, labelLengths_native, inputLengths_native, algo_native, ctcLossDesc_native, &sizeInBytes_native);

    // Write back native variable values
    // handle is read-only
    // probsDesc is read-only
    // gradientsDesc is read-only
    // labels is a native pointer
    // labelLengths is a native pointer
    // inputLengths is a native pointer
    // algo is primitive
    // ctcLossDesc is read-only
    if (!set(env, sizeInBytes, 0, (jlong)sizeInBytes_native)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

/** return the workspace size needed for ctc */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetCTCLossWorkspaceSize_1v8Native(JNIEnv *env, jclass cls, jobject handle, jint algo, jobject ctcLossDesc, jobject probsDesc, jobject gradientsDesc, jlongArray sizeInBytes)
{
    // Null-checks for non-primitive arguments
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cudnnGetCTCLossWorkspaceSize_v8");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    // algo is primitive
    if (ctcLossDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'ctcLossDesc' is null for cudnnGetCTCLossWorkspaceSize_v8");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (probsDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'probsDesc' is null for cudnnGetCTCLossWorkspaceSize_v8");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (gradientsDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'gradientsDesc' is null for cudnnGetCTCLossWorkspaceSize_v8");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (sizeInBytes == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'sizeInBytes' is null for cudnnGetCTCLossWorkspaceSize_v8");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetCTCLossWorkspaceSize_v8(handle=%p, algo=%d, ctcLossDesc=%p, probsDesc=%p, gradientsDesc=%p, sizeInBytes=%p)\n",
        handle, algo, ctcLossDesc, probsDesc, gradientsDesc, sizeInBytes);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnCTCLossAlgo_t algo_native;
    cudnnCTCLossDescriptor_t ctcLossDesc_native;
    cudnnTensorDescriptor_t probsDesc_native;
    cudnnTensorDescriptor_t gradientsDesc_native;
    size_t sizeInBytes_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    algo_native = (cudnnCTCLossAlgo_t)algo;
    ctcLossDesc_native = (cudnnCTCLossDescriptor_t)getNativePointerValue(env, ctcLossDesc);
    probsDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, probsDesc);
    gradientsDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, gradientsDesc);
    // sizeInBytes is write-only

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetCTCLossWorkspaceSize_v8(handle_native, algo_native, ctcLossDesc_native, probsDesc_native, gradientsDesc_native, &sizeInBytes_native);

    // Write back native variable values
    // handle is read-only
    // algo is primitive
    // ctcLossDesc is read-only
    // probsDesc is read-only
    // gradientsDesc is read-only
    if (!set(env, sizeInBytes, 0, (jlong)sizeInBytes_native)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

/**
 * <pre>
 * Cross-library version checker..
 * This function is implemented differently in each sub-library. Each sublib
 * checks whether its own version matches that of its dependencies.
 * @return CUDNN_STATUS_SUCCESS if the version check passes,
 *          CUDNN_STATUS_VERSION_MISMATCH if the versions are inconsistent.
 * </pre>
 */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnAdvTrainVersionCheckNative(JNIEnv *env, jclass cls)
{
    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnAdvTrainVersionCheck()\n");

    // Native function call
    cudnnStatus_t jniResult_native = cudnnAdvTrainVersionCheck();

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

/** Create an instance of convolution descriptor */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnCreateConvolutionDescriptorNative(JNIEnv *env, jclass cls, jobject convDesc)
{
    // Null-checks for non-primitive arguments

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
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

/** Destroy an instance of convolution descriptor */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnDestroyConvolutionDescriptorNative(JNIEnv *env, jclass cls, jobject convDesc)
{
    // Null-checks for non-primitive arguments

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
    // convDesc is read-only

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnSetConvolutionMathTypeNative(JNIEnv *env, jclass cls, jobject convDesc, jint mathType)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnSetConvolutionMathType(convDesc=%p, mathType=%d)\n",
        convDesc, mathType);

    // Native variable declarations
    cudnnConvolutionDescriptor_t convDesc_native;
    cudnnMathType_t mathType_native;

    // Obtain native variable values
    convDesc_native = (cudnnConvolutionDescriptor_t)getNativePointerValue(env, convDesc);
    mathType_native = (cudnnMathType_t)mathType;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnSetConvolutionMathType(convDesc_native, mathType_native);

    // Write back native variable values
    // convDesc is read-only
    // mathType is primitive

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetConvolutionMathTypeNative(JNIEnv *env, jclass cls, jobject convDesc, jintArray mathType)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetConvolutionMathType(convDesc=%p, mathType=%p)\n",
        convDesc, mathType);

    // Native variable declarations
    cudnnConvolutionDescriptor_t convDesc_native;
    cudnnMathType_t mathType_native;

    // Obtain native variable values
    convDesc_native = (cudnnConvolutionDescriptor_t)getNativePointerValue(env, convDesc);
    // mathType is write-only

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetConvolutionMathType(convDesc_native, &mathType_native);

    // Write back native variable values
    // convDesc is read-only
    if (!set(env, mathType, 0, (jint)mathType_native)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnSetConvolutionGroupCountNative(JNIEnv *env, jclass cls, jobject convDesc, jint groupCount)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnSetConvolutionGroupCount(convDesc=%p, groupCount=%d)\n",
        convDesc, groupCount);

    // Native variable declarations
    cudnnConvolutionDescriptor_t convDesc_native;
    int groupCount_native = 0;

    // Obtain native variable values
    convDesc_native = (cudnnConvolutionDescriptor_t)getNativePointerValue(env, convDesc);
    groupCount_native = (int)groupCount;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnSetConvolutionGroupCount(convDesc_native, groupCount_native);

    // Write back native variable values
    // convDesc is read-only
    // groupCount is primitive

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetConvolutionGroupCountNative(JNIEnv *env, jclass cls, jobject convDesc, jintArray groupCount)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetConvolutionGroupCount(convDesc=%p, groupCount=%p)\n",
        convDesc, groupCount);

    // Native variable declarations
    cudnnConvolutionDescriptor_t convDesc_native;
    int groupCount_native;

    // Obtain native variable values
    convDesc_native = (cudnnConvolutionDescriptor_t)getNativePointerValue(env, convDesc);
    // groupCount is write-only

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetConvolutionGroupCount(convDesc_native, &groupCount_native);

    // Write back native variable values
    // convDesc is read-only
    if (!set(env, groupCount, 0, (jint)groupCount_native)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnSetConvolutionReorderTypeNative(JNIEnv *env, jclass cls, jobject convDesc, jint reorderType)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnSetConvolutionReorderType(convDesc=%p, reorderType=%d)\n",
        convDesc, reorderType);

    // Native variable declarations
    cudnnConvolutionDescriptor_t convDesc_native;
    cudnnReorderType_t reorderType_native;

    // Obtain native variable values
    convDesc_native = (cudnnConvolutionDescriptor_t)getNativePointerValue(env, convDesc);
    reorderType_native = (cudnnReorderType_t)reorderType;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnSetConvolutionReorderType(convDesc_native, reorderType_native);

    // Write back native variable values
    // convDesc is read-only
    // reorderType is primitive

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetConvolutionReorderTypeNative(JNIEnv *env, jclass cls, jobject convDesc, jintArray reorderType)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetConvolutionReorderType(convDesc=%p, reorderType=%p)\n",
        convDesc, reorderType);

    // Native variable declarations
    cudnnConvolutionDescriptor_t convDesc_native;
    cudnnReorderType_t reorderType_native;

    // Obtain native variable values
    convDesc_native = (cudnnConvolutionDescriptor_t)getNativePointerValue(env, convDesc);
    // reorderType is write-only

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetConvolutionReorderType(convDesc_native, &reorderType_native);

    // Write back native variable values
    // convDesc is read-only
    if (!set(env, reorderType, 0, (jint)reorderType_native)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnSetConvolution2dDescriptorNative(JNIEnv *env, jclass cls, jobject convDesc, jint pad_h, jint pad_w, jint u, jint v, jint dilation_h, jint dilation_w, jint mode, jint computeType)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnSetConvolution2dDescriptor(convDesc=%p, pad_h=%d, pad_w=%d, u=%d, v=%d, dilation_h=%d, dilation_w=%d, mode=%d, computeType=%d)\n",
        convDesc, pad_h, pad_w, u, v, dilation_h, dilation_w, mode, computeType);

    // Native variable declarations
    cudnnConvolutionDescriptor_t convDesc_native;
    int pad_h_native = 0;
    int pad_w_native = 0;
    int u_native = 0;
    int v_native = 0;
    int dilation_h_native = 0;
    int dilation_w_native = 0;
    cudnnConvolutionMode_t mode_native;
    cudnnDataType_t computeType_native;

    // Obtain native variable values
    convDesc_native = (cudnnConvolutionDescriptor_t)getNativePointerValue(env, convDesc);
    pad_h_native = (int)pad_h;
    pad_w_native = (int)pad_w;
    u_native = (int)u;
    v_native = (int)v;
    dilation_h_native = (int)dilation_h;
    dilation_w_native = (int)dilation_w;
    mode_native = (cudnnConvolutionMode_t)mode;
    computeType_native = (cudnnDataType_t)computeType;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnSetConvolution2dDescriptor(convDesc_native, pad_h_native, pad_w_native, u_native, v_native, dilation_h_native, dilation_w_native, mode_native, computeType_native);

    // Write back native variable values
    // convDesc is read-only
    // pad_h is primitive
    // pad_w is primitive
    // u is primitive
    // v is primitive
    // dilation_h is primitive
    // dilation_w is primitive
    // mode is primitive
    // computeType is primitive

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetConvolution2dDescriptorNative(JNIEnv *env, jclass cls, jobject convDesc, jintArray pad_h, jintArray pad_w, jintArray u, jintArray v, jintArray dilation_h, jintArray dilation_w, jintArray mode, jintArray computeType)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetConvolution2dDescriptor(convDesc=%p, pad_h=%p, pad_w=%p, u=%p, v=%p, dilation_h=%p, dilation_w=%p, mode=%p, computeType=%p)\n",
        convDesc, pad_h, pad_w, u, v, dilation_h, dilation_w, mode, computeType);

    // Native variable declarations
    cudnnConvolutionDescriptor_t convDesc_native;
    int pad_h_native;
    int pad_w_native;
    int u_native;
    int v_native;
    int dilation_h_native;
    int dilation_w_native;
    cudnnConvolutionMode_t mode_native;
    cudnnDataType_t computeType_native;

    // Obtain native variable values
    convDesc_native = (cudnnConvolutionDescriptor_t)getNativePointerValue(env, convDesc);
    // pad_h is write-only
    // pad_w is write-only
    // u is write-only
    // v is write-only
    // dilation_h is write-only
    // dilation_w is write-only
    // mode is write-only
    // computeType is write-only

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetConvolution2dDescriptor(convDesc_native, &pad_h_native, &pad_w_native, &u_native, &v_native, &dilation_h_native, &dilation_w_native, &mode_native, &computeType_native);

    // Write back native variable values
    // convDesc is read-only
    if (!set(env, pad_h, 0, (jint)pad_h_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, pad_w, 0, (jint)pad_w_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, u, 0, (jint)u_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, v, 0, (jint)v_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, dilation_h, 0, (jint)dilation_h_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, dilation_w, 0, (jint)dilation_w_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, mode, 0, (jint)mode_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, computeType, 0, (jint)computeType_native)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnSetConvolutionNdDescriptorNative(JNIEnv *env, jclass cls, jobject convDesc, jint arrayLength, jintArray padA, jintArray filterStrideA, jintArray dilationA, jint mode, jint computeType)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnSetConvolutionNdDescriptor(convDesc=%p, arrayLength=%d, padA=%p, filterStrideA=%p, dilationA=%p, mode=%d, computeType=%d)\n",
        convDesc, arrayLength, padA, filterStrideA, dilationA, mode, computeType);

    // Native variable declarations
    cudnnConvolutionDescriptor_t convDesc_native;
    int arrayLength_native = 0;
    int * padA_native = NULL;
    int * filterStrideA_native = NULL;
    int * dilationA_native = NULL;
    cudnnConvolutionMode_t mode_native;
    cudnnDataType_t computeType_native;

    // Obtain native variable values
    convDesc_native = (cudnnConvolutionDescriptor_t)getNativePointerValue(env, convDesc);
    arrayLength_native = (int)arrayLength;
    if (!initNative(env, padA, padA_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!initNative(env, filterStrideA, filterStrideA_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!initNative(env, dilationA, dilationA_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    mode_native = (cudnnConvolutionMode_t)mode;
    computeType_native = (cudnnDataType_t)computeType;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnSetConvolutionNdDescriptor(convDesc_native, arrayLength_native, padA_native, filterStrideA_native, dilationA_native, mode_native, computeType_native);

    // Write back native variable values
    // convDesc is read-only
    // arrayLength is primitive
    if (!releaseNative(env, padA_native, padA, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!releaseNative(env, filterStrideA_native, filterStrideA, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!releaseNative(env, dilationA_native, dilationA, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // mode is primitive
    // computeType is primitive

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

/** Helper function to return the dimensions of the output tensor given a convolution descriptor */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetConvolutionNdDescriptorNative(JNIEnv *env, jclass cls, jobject convDesc, jint arrayLengthRequested, jintArray arrayLength, jintArray padA, jintArray strideA, jintArray dilationA, jintArray mode, jintArray computeType)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetConvolutionNdDescriptor(convDesc=%p, arrayLengthRequested=%d, arrayLength=%p, padA=%p, strideA=%p, dilationA=%p, mode=%p, computeType=%p)\n",
        convDesc, arrayLengthRequested, arrayLength, padA, strideA, dilationA, mode, computeType);

    // Native variable declarations
    cudnnConvolutionDescriptor_t convDesc_native;
    int arrayLengthRequested_native = 0;
    int arrayLength_native;
    int * padA_native = NULL;
    int * strideA_native = NULL;
    int * dilationA_native = NULL;
    cudnnConvolutionMode_t mode_native;
    cudnnDataType_t computeType_native;

    // Obtain native variable values
    convDesc_native = (cudnnConvolutionDescriptor_t)getNativePointerValue(env, convDesc);
    arrayLengthRequested_native = (int)arrayLengthRequested;
    // arrayLength is write-only
    if (!initNative(env, padA, padA_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!initNative(env, strideA, strideA_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!initNative(env, dilationA, dilationA_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // mode is write-only
    // computeType is write-only

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetConvolutionNdDescriptor(convDesc_native, arrayLengthRequested_native, &arrayLength_native, padA_native, strideA_native, dilationA_native, &mode_native, &computeType_native);

    // Write back native variable values
    // convDesc is read-only
    // arrayLengthRequested is primitive
    if (!set(env, arrayLength, 0, (jint)arrayLength_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!releaseNative(env, padA_native, padA, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!releaseNative(env, strideA_native, strideA, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!releaseNative(env, dilationA_native, dilationA, true)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, mode, 0, (jint)mode_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, computeType, 0, (jint)computeType_native)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetConvolution2dForwardOutputDimNative(JNIEnv *env, jclass cls, jobject convDesc, jobject inputTensorDesc, jobject filterDesc, jintArray n, jintArray c, jintArray h, jintArray w)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetConvolution2dForwardOutputDim(convDesc=%p, inputTensorDesc=%p, filterDesc=%p, n=%p, c=%p, h=%p, w=%p)\n",
        convDesc, inputTensorDesc, filterDesc, n, c, h, w);

    // Native variable declarations
    cudnnConvolutionDescriptor_t convDesc_native;
    cudnnTensorDescriptor_t inputTensorDesc_native;
    cudnnFilterDescriptor_t filterDesc_native;
    int n_native;
    int c_native;
    int h_native;
    int w_native;

    // Obtain native variable values
    convDesc_native = (cudnnConvolutionDescriptor_t)getNativePointerValue(env, convDesc);
    inputTensorDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, inputTensorDesc);
    filterDesc_native = (cudnnFilterDescriptor_t)getNativePointerValue(env, filterDesc);
    // n is write-only
    // c is write-only
    // h is write-only
    // w is write-only

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetConvolution2dForwardOutputDim(convDesc_native, inputTensorDesc_native, filterDesc_native, &n_native, &c_native, &h_native, &w_native);

    // Write back native variable values
    // convDesc is read-only
    // inputTensorDesc is read-only
    // filterDesc is read-only
    if (!set(env, n, 0, (jint)n_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, c, 0, (jint)c_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, h, 0, (jint)h_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!set(env, w, 0, (jint)w_native)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

/** Helper function to return the dimensions of the output tensor given a convolution descriptor */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetConvolutionNdForwardOutputDimNative(JNIEnv *env, jclass cls, jobject convDesc, jobject inputTensorDesc, jobject filterDesc, jint nbDims, jintArray tensorOuputDimA)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetConvolutionNdForwardOutputDim(convDesc=%p, inputTensorDesc=%p, filterDesc=%p, nbDims=%d, tensorOuputDimA=%p)\n",
        convDesc, inputTensorDesc, filterDesc, nbDims, tensorOuputDimA);

    // Native variable declarations
    cudnnConvolutionDescriptor_t convDesc_native;
    cudnnTensorDescriptor_t inputTensorDesc_native;
    cudnnFilterDescriptor_t filterDesc_native;
    int nbDims_native = 0;
    int * tensorOuputDimA_native = NULL;

    // Obtain native variable values
    convDesc_native = (cudnnConvolutionDescriptor_t)getNativePointerValue(env, convDesc);
    inputTensorDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, inputTensorDesc);
    filterDesc_native = (cudnnFilterDescriptor_t)getNativePointerValue(env, filterDesc);
    nbDims_native = (int)nbDims;
    if (!initNative(env, tensorOuputDimA, tensorOuputDimA_native, true)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetConvolutionNdForwardOutputDim(convDesc_native, inputTensorDesc_native, filterDesc_native, nbDims_native, tensorOuputDimA_native);

    // Write back native variable values
    // convDesc is read-only
    // inputTensorDesc is read-only
    // filterDesc is read-only
    // nbDims is primitive
    if (!releaseNative(env, tensorOuputDimA_native, tensorOuputDimA, true)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

/** helper function to provide the convolution forward algo that fit best the requirement */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetConvolutionForwardAlgorithmMaxCountNative(JNIEnv *env, jclass cls, jobject handle, jintArray count)
{
    // Null-checks for non-primitive arguments
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cudnnGetConvolutionForwardAlgorithmMaxCount");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (count == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'count' is null for cudnnGetConvolutionForwardAlgorithmMaxCount");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetConvolutionForwardAlgorithmMaxCount(handle=%p, count=%p)\n",
        handle, count);

    // Native variable declarations
    cudnnHandle_t handle_native;
    int count_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    // count is write-only

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetConvolutionForwardAlgorithmMaxCount(handle_native, &count_native);

    // Write back native variable values
    // handle is read-only
    if (!set(env, count, 0, (jint)count_native)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetConvolutionForwardAlgorithm_1v7Native(JNIEnv *env, jclass cls, jobject handle, jobject srcDesc, jobject filterDesc, jobject convDesc, jobject destDesc, jint requestedAlgoCount, jintArray returnedAlgoCount, jobjectArray perfResults)
{
    // Null-checks for non-primitive arguments
    if (perfResults == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'perfResults' is null for cudnnGetConvolutionForwardAlgorithm_v7");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetConvolutionForwardAlgorithm_v7(handle=%p, srcDesc=%p, filterDesc=%p, convDesc=%p, destDesc=%p, requestedAlgoCount=%d, returnedAlgoCount=%p, perfResults=%p)\n",
        handle, srcDesc, filterDesc, convDesc, destDesc, requestedAlgoCount, returnedAlgoCount, perfResults);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnTensorDescriptor_t srcDesc_native;
    cudnnFilterDescriptor_t filterDesc_native;
    cudnnConvolutionDescriptor_t convDesc_native;
    cudnnTensorDescriptor_t destDesc_native;
    int requestedAlgoCount_native = 0;
    int returnedAlgoCount_native;
    cudnnConvolutionFwdAlgoPerf_t * perfResults_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    srcDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, srcDesc);
    filterDesc_native = (cudnnFilterDescriptor_t)getNativePointerValue(env, filterDesc);
    convDesc_native = (cudnnConvolutionDescriptor_t)getNativePointerValue(env, convDesc);
    destDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, destDesc);
    requestedAlgoCount_native = (int)requestedAlgoCount;
    // returnedAlgoCount is write-only
    if (!initNative(env, perfResults, perfResults_native, requestedAlgoCount)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetConvolutionForwardAlgorithm_v7(handle_native, srcDesc_native, filterDesc_native, convDesc_native, destDesc_native, requestedAlgoCount_native, &returnedAlgoCount_native, perfResults_native);

    // Write back native variable values
    // handle is read-only
    // srcDesc is read-only
    // filterDesc is read-only
    // convDesc is read-only
    // destDesc is read-only
    // requestedAlgoCount is primitive
    if (!set(env, returnedAlgoCount, 0, (jint)returnedAlgoCount_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!releaseNative(env, perfResults_native, perfResults, returnedAlgoCount_native)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnFindConvolutionForwardAlgorithmNative(JNIEnv *env, jclass cls, jobject handle, jobject xDesc, jobject wDesc, jobject convDesc, jobject yDesc, jint requestedAlgoCount, jintArray returnedAlgoCount, jobjectArray perfResults)
{
    // Null-checks for non-primitive arguments
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
    int returnedAlgoCount_native;
    cudnnConvolutionFwdAlgoPerf_t * perfResults_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    xDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, xDesc);
    wDesc_native = (cudnnFilterDescriptor_t)getNativePointerValue(env, wDesc);
    convDesc_native = (cudnnConvolutionDescriptor_t)getNativePointerValue(env, convDesc);
    yDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, yDesc);
    requestedAlgoCount_native = (int)requestedAlgoCount;
    // returnedAlgoCount is write-only
    if (!initNative(env, perfResults, perfResults_native, requestedAlgoCount)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnFindConvolutionForwardAlgorithm(handle_native, xDesc_native, wDesc_native, convDesc_native, yDesc_native, requestedAlgoCount_native, &returnedAlgoCount_native, perfResults_native);

    // Write back native variable values
    // handle is read-only
    // xDesc is read-only
    // wDesc is read-only
    // convDesc is read-only
    // yDesc is read-only
    // requestedAlgoCount is primitive
    if (!set(env, returnedAlgoCount, 0, (jint)returnedAlgoCount_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!releaseNative(env, perfResults_native, perfResults, returnedAlgoCount_native)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnFindConvolutionForwardAlgorithmExNative(JNIEnv *env, jclass cls, jobject handle, jobject xDesc, jobject x, jobject wDesc, jobject w, jobject convDesc, jobject yDesc, jobject y, jint requestedAlgoCount, jintArray returnedAlgoCount, jobjectArray perfResults, jobject workSpace, jlong workSpaceSizeInBytes)
{
    // Null-checks for non-primitive arguments
    if (perfResults == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'perfResults' is null for cudnnFindConvolutionForwardAlgorithmEx");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnFindConvolutionForwardAlgorithmEx(handle=%p, xDesc=%p, x=%p, wDesc=%p, w=%p, convDesc=%p, yDesc=%p, y=%p, requestedAlgoCount=%d, returnedAlgoCount=%p, perfResults=%p, workSpace=%p, workSpaceSizeInBytes=%ld)\n",
        handle, xDesc, x, wDesc, w, convDesc, yDesc, y, requestedAlgoCount, returnedAlgoCount, perfResults, workSpace, workSpaceSizeInBytes);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnTensorDescriptor_t xDesc_native;
    void * x_native = NULL;
    cudnnFilterDescriptor_t wDesc_native;
    void * w_native = NULL;
    cudnnConvolutionDescriptor_t convDesc_native;
    cudnnTensorDescriptor_t yDesc_native;
    void * y_native = NULL;
    int requestedAlgoCount_native = 0;
    int returnedAlgoCount_native;
    cudnnConvolutionFwdAlgoPerf_t * perfResults_native;
    void * workSpace_native = NULL;
    size_t workSpaceSizeInBytes_native = 0;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    xDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, xDesc);
    x_native = (void *)getPointer(env, x);
    wDesc_native = (cudnnFilterDescriptor_t)getNativePointerValue(env, wDesc);
    w_native = (void *)getPointer(env, w);
    convDesc_native = (cudnnConvolutionDescriptor_t)getNativePointerValue(env, convDesc);
    yDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, yDesc);
    y_native = (void *)getPointer(env, y);
    requestedAlgoCount_native = (int)requestedAlgoCount;
    // returnedAlgoCount is write-only
    if (!initNative(env, perfResults, perfResults_native, requestedAlgoCount)) return JCUDNN_STATUS_INTERNAL_ERROR;
    workSpace_native = (void *)getPointer(env, workSpace);
    workSpaceSizeInBytes_native = (size_t)workSpaceSizeInBytes;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnFindConvolutionForwardAlgorithmEx(handle_native, xDesc_native, x_native, wDesc_native, w_native, convDesc_native, yDesc_native, y_native, requestedAlgoCount_native, &returnedAlgoCount_native, perfResults_native, workSpace_native, workSpaceSizeInBytes_native);

    // Write back native variable values
    // handle is read-only
    // xDesc is read-only
    // x is a native pointer
    // wDesc is read-only
    // w is a native pointer
    // convDesc is read-only
    // yDesc is read-only
    // y is a native pointer
    // requestedAlgoCount is primitive
    if (!set(env, returnedAlgoCount, 0, (jint)returnedAlgoCount_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!releaseNative(env, perfResults_native, perfResults, returnedAlgoCount_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // workSpace is a native pointer
    // workSpaceSizeInBytes is primitive

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnIm2ColNative(JNIEnv *env, jclass cls, jobject handle, jobject xDesc, jobject x, jobject wDesc, jobject convDesc, jobject colBuffer)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnIm2Col(handle=%p, xDesc=%p, x=%p, wDesc=%p, convDesc=%p, colBuffer=%p)\n",
        handle, xDesc, x, wDesc, convDesc, colBuffer);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnTensorDescriptor_t xDesc_native;
    void * x_native = NULL;
    cudnnFilterDescriptor_t wDesc_native;
    cudnnConvolutionDescriptor_t convDesc_native;
    void * colBuffer_native = NULL;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    xDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, xDesc);
    x_native = (void *)getPointer(env, x);
    wDesc_native = (cudnnFilterDescriptor_t)getNativePointerValue(env, wDesc);
    convDesc_native = (cudnnConvolutionDescriptor_t)getNativePointerValue(env, convDesc);
    colBuffer_native = (void *)getPointer(env, colBuffer);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnIm2Col(handle_native, xDesc_native, x_native, wDesc_native, convDesc_native, colBuffer_native);

    // Write back native variable values
    // handle is read-only
    // xDesc is read-only
    // x is a native pointer
    // wDesc is read-only
    // convDesc is read-only
    // colBuffer is a native pointer

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnReorderFilterAndBiasNative(JNIEnv *env, jclass cls, jobject handle, jobject filterDesc, jint reorderType, jobject filterData, jobject reorderedFilterData, jint reorderBias, jobject biasData, jobject reorderedBiasData)
{
    // Null-checks for non-primitive arguments
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cudnnReorderFilterAndBias");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (filterDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'filterDesc' is null for cudnnReorderFilterAndBias");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    // reorderType is primitive
    if (filterData == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'filterData' is null for cudnnReorderFilterAndBias");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (reorderedFilterData == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'reorderedFilterData' is null for cudnnReorderFilterAndBias");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    // reorderBias is primitive
    if (biasData == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'biasData' is null for cudnnReorderFilterAndBias");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (reorderedBiasData == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'reorderedBiasData' is null for cudnnReorderFilterAndBias");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnReorderFilterAndBias(handle=%p, filterDesc=%p, reorderType=%d, filterData=%p, reorderedFilterData=%p, reorderBias=%d, biasData=%p, reorderedBiasData=%p)\n",
        handle, filterDesc, reorderType, filterData, reorderedFilterData, reorderBias, biasData, reorderedBiasData);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnFilterDescriptor_t filterDesc_native;
    cudnnReorderType_t reorderType_native;
    void * filterData_native = NULL;
    void * reorderedFilterData_native = NULL;
    int reorderBias_native = 0;
    void * biasData_native = NULL;
    void * reorderedBiasData_native = NULL;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    filterDesc_native = (cudnnFilterDescriptor_t)getNativePointerValue(env, filterDesc);
    reorderType_native = (cudnnReorderType_t)reorderType;
    filterData_native = (void *)getPointer(env, filterData);
    reorderedFilterData_native = (void *)getPointer(env, reorderedFilterData);
    reorderBias_native = (int)reorderBias;
    biasData_native = (void *)getPointer(env, biasData);
    reorderedBiasData_native = (void *)getPointer(env, reorderedBiasData);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnReorderFilterAndBias(handle_native, filterDesc_native, reorderType_native, filterData_native, reorderedFilterData_native, reorderBias_native, biasData_native, reorderedBiasData_native);

    // Write back native variable values
    // handle is read-only
    // filterDesc is read-only
    // reorderType is primitive
    // filterData is a native pointer
    // reorderedFilterData is a native pointer
    // reorderBias is primitive
    // biasData is a native pointer
    // reorderedBiasData is a native pointer

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

/** Helper function to return the minimum size of the workspace to be passed to the convolution given an algo*/
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetConvolutionForwardWorkspaceSizeNative(JNIEnv *env, jclass cls, jobject handle, jobject xDesc, jobject wDesc, jobject convDesc, jobject yDesc, jint algo, jlongArray sizeInBytes)
{
    // Null-checks for non-primitive arguments

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
    size_t sizeInBytes_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    xDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, xDesc);
    wDesc_native = (cudnnFilterDescriptor_t)getNativePointerValue(env, wDesc);
    convDesc_native = (cudnnConvolutionDescriptor_t)getNativePointerValue(env, convDesc);
    yDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, yDesc);
    algo_native = (cudnnConvolutionFwdAlgo_t)algo;
    // sizeInBytes is write-only

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetConvolutionForwardWorkspaceSize(handle_native, xDesc_native, wDesc_native, convDesc_native, yDesc_native, algo_native, &sizeInBytes_native);

    // Write back native variable values
    // handle is read-only
    // xDesc is read-only
    // wDesc is read-only
    // convDesc is read-only
    // yDesc is read-only
    // algo is primitive
    if (!set(env, sizeInBytes, 0, (jlong)sizeInBytes_native)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

/** Convolution functions: All of the form "output = alpha * Op(inputs) + beta * output" */
/** Function to perform the forward pass for batch convolution */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnConvolutionForwardNative(JNIEnv *env, jclass cls, jobject handle, jobject alpha, jobject xDesc, jobject x, jobject wDesc, jobject w, jobject convDesc, jint algo, jobject workSpace, jlong workSpaceSizeInBytes, jobject beta, jobject yDesc, jobject y)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnConvolutionForward(handle=%p, alpha=%p, xDesc=%p, x=%p, wDesc=%p, w=%p, convDesc=%p, algo=%d, workSpace=%p, workSpaceSizeInBytes=%ld, beta=%p, yDesc=%p, y=%p)\n",
        handle, alpha, xDesc, x, wDesc, w, convDesc, algo, workSpace, workSpaceSizeInBytes, beta, yDesc, y);

    // Native variable declarations
    cudnnHandle_t handle_native;
    void * alpha_native = NULL;
    cudnnTensorDescriptor_t xDesc_native;
    void * x_native = NULL;
    cudnnFilterDescriptor_t wDesc_native;
    void * w_native = NULL;
    cudnnConvolutionDescriptor_t convDesc_native;
    cudnnConvolutionFwdAlgo_t algo_native;
    void * workSpace_native = NULL;
    size_t workSpaceSizeInBytes_native = 0;
    void * beta_native = NULL;
    cudnnTensorDescriptor_t yDesc_native;
    void * y_native = NULL;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    PointerData *alpha_pointerData = initPointerData(env, alpha);
    if (alpha_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    alpha_native = (void *)alpha_pointerData->getPointer(env);
    xDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, xDesc);
    x_native = (void *)getPointer(env, x);
    wDesc_native = (cudnnFilterDescriptor_t)getNativePointerValue(env, wDesc);
    w_native = (void *)getPointer(env, w);
    convDesc_native = (cudnnConvolutionDescriptor_t)getNativePointerValue(env, convDesc);
    algo_native = (cudnnConvolutionFwdAlgo_t)algo;
    workSpace_native = (void *)getPointer(env, workSpace);
    workSpaceSizeInBytes_native = (size_t)workSpaceSizeInBytes;
    PointerData *beta_pointerData = initPointerData(env, beta);
    if (beta_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    beta_native = (void *)beta_pointerData->getPointer(env);
    yDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, yDesc);
    y_native = (void *)getPointer(env, y);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnConvolutionForward(handle_native, alpha_native, xDesc_native, x_native, wDesc_native, w_native, convDesc_native, algo_native, workSpace_native, workSpaceSizeInBytes_native, beta_native, yDesc_native, y_native);

    // Write back native variable values
    // handle is read-only
    if (!releasePointerData(env, alpha_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // xDesc is read-only
    // x is a native pointer
    // wDesc is read-only
    // w is a native pointer
    // convDesc is read-only
    // algo is primitive
    // workSpace is a native pointer
    // workSpaceSizeInBytes is primitive
    if (!releasePointerData(env, beta_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // yDesc is read-only
    // y is a native pointer

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

/** Fused conv/bias/activation operation : y = Act( alpha1 * conv(x) + alpha2 * z + bias ) */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnConvolutionBiasActivationForwardNative(JNIEnv *env, jclass cls, jobject handle, jobject alpha1, jobject xDesc, jobject x, jobject wDesc, jobject w, jobject convDesc, jint algo, jobject workSpace, jlong workSpaceSizeInBytes, jobject alpha2, jobject zDesc, jobject z, jobject biasDesc, jobject bias, jobject activationDesc, jobject yDesc, jobject y)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnConvolutionBiasActivationForward(handle=%p, alpha1=%p, xDesc=%p, x=%p, wDesc=%p, w=%p, convDesc=%p, algo=%d, workSpace=%p, workSpaceSizeInBytes=%ld, alpha2=%p, zDesc=%p, z=%p, biasDesc=%p, bias=%p, activationDesc=%p, yDesc=%p, y=%p)\n",
        handle, alpha1, xDesc, x, wDesc, w, convDesc, algo, workSpace, workSpaceSizeInBytes, alpha2, zDesc, z, biasDesc, bias, activationDesc, yDesc, y);

    // Native variable declarations
    cudnnHandle_t handle_native;
    void * alpha1_native = NULL;
    cudnnTensorDescriptor_t xDesc_native;
    void * x_native = NULL;
    cudnnFilterDescriptor_t wDesc_native;
    void * w_native = NULL;
    cudnnConvolutionDescriptor_t convDesc_native;
    cudnnConvolutionFwdAlgo_t algo_native;
    void * workSpace_native = NULL;
    size_t workSpaceSizeInBytes_native = 0;
    void * alpha2_native = NULL;
    cudnnTensorDescriptor_t zDesc_native;
    void * z_native = NULL;
    cudnnTensorDescriptor_t biasDesc_native;
    void * bias_native = NULL;
    cudnnActivationDescriptor_t activationDesc_native;
    cudnnTensorDescriptor_t yDesc_native;
    void * y_native = NULL;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    PointerData *alpha1_pointerData = initPointerData(env, alpha1);
    if (alpha1_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    alpha1_native = (void *)alpha1_pointerData->getPointer(env);
    xDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, xDesc);
    x_native = (void *)getPointer(env, x);
    wDesc_native = (cudnnFilterDescriptor_t)getNativePointerValue(env, wDesc);
    w_native = (void *)getPointer(env, w);
    convDesc_native = (cudnnConvolutionDescriptor_t)getNativePointerValue(env, convDesc);
    algo_native = (cudnnConvolutionFwdAlgo_t)algo;
    workSpace_native = (void *)getPointer(env, workSpace);
    workSpaceSizeInBytes_native = (size_t)workSpaceSizeInBytes;
    PointerData *alpha2_pointerData = initPointerData(env, alpha2);
    if (alpha2_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    alpha2_native = (void *)alpha2_pointerData->getPointer(env);
    zDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, zDesc);
    z_native = (void *)getPointer(env, z);
    biasDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, biasDesc);
    bias_native = (void *)getPointer(env, bias);
    activationDesc_native = (cudnnActivationDescriptor_t)getNativePointerValue(env, activationDesc);
    yDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, yDesc);
    y_native = (void *)getPointer(env, y);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnConvolutionBiasActivationForward(handle_native, alpha1_native, xDesc_native, x_native, wDesc_native, w_native, convDesc_native, algo_native, workSpace_native, workSpaceSizeInBytes_native, alpha2_native, zDesc_native, z_native, biasDesc_native, bias_native, activationDesc_native, yDesc_native, y_native);

    // Write back native variable values
    // handle is read-only
    if (!releasePointerData(env, alpha1_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // xDesc is read-only
    // x is a native pointer
    // wDesc is read-only
    // w is a native pointer
    // convDesc is read-only
    // algo is primitive
    // workSpace is a native pointer
    // workSpaceSizeInBytes is primitive
    if (!releasePointerData(env, alpha2_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // zDesc is read-only
    // z is a native pointer
    // biasDesc is read-only
    // bias is a native pointer
    // activationDesc is read-only
    // yDesc is read-only
    // y is a native pointer

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetConvolutionBackwardDataAlgorithmMaxCountNative(JNIEnv *env, jclass cls, jobject handle, jintArray count)
{
    // Null-checks for non-primitive arguments
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cudnnGetConvolutionBackwardDataAlgorithmMaxCount");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (count == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'count' is null for cudnnGetConvolutionBackwardDataAlgorithmMaxCount");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetConvolutionBackwardDataAlgorithmMaxCount(handle=%p, count=%p)\n",
        handle, count);

    // Native variable declarations
    cudnnHandle_t handle_native;
    int count_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    // count is write-only

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetConvolutionBackwardDataAlgorithmMaxCount(handle_native, &count_native);

    // Write back native variable values
    // handle is read-only
    if (!set(env, count, 0, (jint)count_native)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnFindConvolutionBackwardDataAlgorithmNative(JNIEnv *env, jclass cls, jobject handle, jobject wDesc, jobject dyDesc, jobject convDesc, jobject dxDesc, jint requestedAlgoCount, jintArray returnedAlgoCount, jobjectArray perfResults)
{
    // Null-checks for non-primitive arguments
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
    int returnedAlgoCount_native;
    cudnnConvolutionBwdDataAlgoPerf_t * perfResults_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    wDesc_native = (cudnnFilterDescriptor_t)getNativePointerValue(env, wDesc);
    dyDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, dyDesc);
    convDesc_native = (cudnnConvolutionDescriptor_t)getNativePointerValue(env, convDesc);
    dxDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, dxDesc);
    requestedAlgoCount_native = (int)requestedAlgoCount;
    // returnedAlgoCount is write-only
    if (!initNative(env, perfResults, perfResults_native, requestedAlgoCount)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnFindConvolutionBackwardDataAlgorithm(handle_native, wDesc_native, dyDesc_native, convDesc_native, dxDesc_native, requestedAlgoCount_native, &returnedAlgoCount_native, perfResults_native);

    // Write back native variable values
    // handle is read-only
    // wDesc is read-only
    // dyDesc is read-only
    // convDesc is read-only
    // dxDesc is read-only
    // requestedAlgoCount is primitive
    if (!set(env, returnedAlgoCount, 0, (jint)returnedAlgoCount_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!releaseNative(env, perfResults_native, perfResults, returnedAlgoCount_native)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnFindConvolutionBackwardDataAlgorithmExNative(JNIEnv *env, jclass cls, jobject handle, jobject wDesc, jobject w, jobject dyDesc, jobject dy, jobject convDesc, jobject dxDesc, jobject dx, jint requestedAlgoCount, jintArray returnedAlgoCount, jobjectArray perfResults, jobject workSpace, jlong workSpaceSizeInBytes)
{
    // Null-checks for non-primitive arguments
    if (perfResults == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'perfResults' is null for cudnnFindConvolutionBackwardDataAlgorithmEx");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnFindConvolutionBackwardDataAlgorithmEx(handle=%p, wDesc=%p, w=%p, dyDesc=%p, dy=%p, convDesc=%p, dxDesc=%p, dx=%p, requestedAlgoCount=%d, returnedAlgoCount=%p, perfResults=%p, workSpace=%p, workSpaceSizeInBytes=%ld)\n",
        handle, wDesc, w, dyDesc, dy, convDesc, dxDesc, dx, requestedAlgoCount, returnedAlgoCount, perfResults, workSpace, workSpaceSizeInBytes);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnFilterDescriptor_t wDesc_native;
    void * w_native = NULL;
    cudnnTensorDescriptor_t dyDesc_native;
    void * dy_native = NULL;
    cudnnConvolutionDescriptor_t convDesc_native;
    cudnnTensorDescriptor_t dxDesc_native;
    void * dx_native = NULL;
    int requestedAlgoCount_native = 0;
    int returnedAlgoCount_native;
    cudnnConvolutionBwdDataAlgoPerf_t * perfResults_native;
    void * workSpace_native = NULL;
    size_t workSpaceSizeInBytes_native = 0;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    wDesc_native = (cudnnFilterDescriptor_t)getNativePointerValue(env, wDesc);
    w_native = (void *)getPointer(env, w);
    dyDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, dyDesc);
    dy_native = (void *)getPointer(env, dy);
    convDesc_native = (cudnnConvolutionDescriptor_t)getNativePointerValue(env, convDesc);
    dxDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, dxDesc);
    dx_native = (void *)getPointer(env, dx);
    requestedAlgoCount_native = (int)requestedAlgoCount;
    // returnedAlgoCount is write-only
    if (!initNative(env, perfResults, perfResults_native, requestedAlgoCount)) return JCUDNN_STATUS_INTERNAL_ERROR;
    workSpace_native = (void *)getPointer(env, workSpace);
    workSpaceSizeInBytes_native = (size_t)workSpaceSizeInBytes;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnFindConvolutionBackwardDataAlgorithmEx(handle_native, wDesc_native, w_native, dyDesc_native, dy_native, convDesc_native, dxDesc_native, dx_native, requestedAlgoCount_native, &returnedAlgoCount_native, perfResults_native, workSpace_native, workSpaceSizeInBytes_native);

    // Write back native variable values
    // handle is read-only
    // wDesc is read-only
    // w is a native pointer
    // dyDesc is read-only
    // dy is a native pointer
    // convDesc is read-only
    // dxDesc is read-only
    // dx is a native pointer
    // requestedAlgoCount is primitive
    if (!set(env, returnedAlgoCount, 0, (jint)returnedAlgoCount_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!releaseNative(env, perfResults_native, perfResults, returnedAlgoCount_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // workSpace is a native pointer
    // workSpaceSizeInBytes is primitive

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetConvolutionBackwardDataAlgorithm_1v7Native(JNIEnv *env, jclass cls, jobject handle, jobject filterDesc, jobject diffDesc, jobject convDesc, jobject gradDesc, jint requestedAlgoCount, jintArray returnedAlgoCount, jobjectArray perfResults)
{
    // Null-checks for non-primitive arguments
    if (perfResults == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'perfResults' is null for cudnnGetConvolutionBackwardDataAlgorithm_v7");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetConvolutionBackwardDataAlgorithm_v7(handle=%p, filterDesc=%p, diffDesc=%p, convDesc=%p, gradDesc=%p, requestedAlgoCount=%d, returnedAlgoCount=%p, perfResults=%p)\n",
        handle, filterDesc, diffDesc, convDesc, gradDesc, requestedAlgoCount, returnedAlgoCount, perfResults);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnFilterDescriptor_t filterDesc_native;
    cudnnTensorDescriptor_t diffDesc_native;
    cudnnConvolutionDescriptor_t convDesc_native;
    cudnnTensorDescriptor_t gradDesc_native;
    int requestedAlgoCount_native = 0;
    int returnedAlgoCount_native;
    cudnnConvolutionBwdDataAlgoPerf_t * perfResults_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    filterDesc_native = (cudnnFilterDescriptor_t)getNativePointerValue(env, filterDesc);
    diffDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, diffDesc);
    convDesc_native = (cudnnConvolutionDescriptor_t)getNativePointerValue(env, convDesc);
    gradDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, gradDesc);
    requestedAlgoCount_native = (int)requestedAlgoCount;
    // returnedAlgoCount is write-only
    if (!initNative(env, perfResults, perfResults_native, requestedAlgoCount)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetConvolutionBackwardDataAlgorithm_v7(handle_native, filterDesc_native, diffDesc_native, convDesc_native, gradDesc_native, requestedAlgoCount_native, &returnedAlgoCount_native, perfResults_native);

    // Write back native variable values
    // handle is read-only
    // filterDesc is read-only
    // diffDesc is read-only
    // convDesc is read-only
    // gradDesc is read-only
    // requestedAlgoCount is primitive
    if (!set(env, returnedAlgoCount, 0, (jint)returnedAlgoCount_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!releaseNative(env, perfResults_native, perfResults, returnedAlgoCount_native)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

/**
 *  convolution algorithm (which requires potentially some workspace)
 */
/** Helper function to return the minimum size of the workspace to be passed to the convolution given an algo*/
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetConvolutionBackwardDataWorkspaceSizeNative(JNIEnv *env, jclass cls, jobject handle, jobject wDesc, jobject dyDesc, jobject convDesc, jobject dxDesc, jint algo, jlongArray sizeInBytes)
{
    // Null-checks for non-primitive arguments

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
    size_t sizeInBytes_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    wDesc_native = (cudnnFilterDescriptor_t)getNativePointerValue(env, wDesc);
    dyDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, dyDesc);
    convDesc_native = (cudnnConvolutionDescriptor_t)getNativePointerValue(env, convDesc);
    dxDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, dxDesc);
    algo_native = (cudnnConvolutionBwdDataAlgo_t)algo;
    // sizeInBytes is write-only

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetConvolutionBackwardDataWorkspaceSize(handle_native, wDesc_native, dyDesc_native, convDesc_native, dxDesc_native, algo_native, &sizeInBytes_native);

    // Write back native variable values
    // handle is read-only
    // wDesc is read-only
    // dyDesc is read-only
    // convDesc is read-only
    // dxDesc is read-only
    // algo is primitive
    if (!set(env, sizeInBytes, 0, (jlong)sizeInBytes_native)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnConvolutionBackwardDataNative(JNIEnv *env, jclass cls, jobject handle, jobject alpha, jobject wDesc, jobject w, jobject dyDesc, jobject dy, jobject convDesc, jint algo, jobject workSpace, jlong workSpaceSizeInBytes, jobject beta, jobject dxDesc, jobject dx)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnConvolutionBackwardData(handle=%p, alpha=%p, wDesc=%p, w=%p, dyDesc=%p, dy=%p, convDesc=%p, algo=%d, workSpace=%p, workSpaceSizeInBytes=%ld, beta=%p, dxDesc=%p, dx=%p)\n",
        handle, alpha, wDesc, w, dyDesc, dy, convDesc, algo, workSpace, workSpaceSizeInBytes, beta, dxDesc, dx);

    // Native variable declarations
    cudnnHandle_t handle_native;
    void * alpha_native = NULL;
    cudnnFilterDescriptor_t wDesc_native;
    void * w_native = NULL;
    cudnnTensorDescriptor_t dyDesc_native;
    void * dy_native = NULL;
    cudnnConvolutionDescriptor_t convDesc_native;
    cudnnConvolutionBwdDataAlgo_t algo_native;
    void * workSpace_native = NULL;
    size_t workSpaceSizeInBytes_native = 0;
    void * beta_native = NULL;
    cudnnTensorDescriptor_t dxDesc_native;
    void * dx_native = NULL;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    PointerData *alpha_pointerData = initPointerData(env, alpha);
    if (alpha_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    alpha_native = (void *)alpha_pointerData->getPointer(env);
    wDesc_native = (cudnnFilterDescriptor_t)getNativePointerValue(env, wDesc);
    w_native = (void *)getPointer(env, w);
    dyDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, dyDesc);
    dy_native = (void *)getPointer(env, dy);
    convDesc_native = (cudnnConvolutionDescriptor_t)getNativePointerValue(env, convDesc);
    algo_native = (cudnnConvolutionBwdDataAlgo_t)algo;
    workSpace_native = (void *)getPointer(env, workSpace);
    workSpaceSizeInBytes_native = (size_t)workSpaceSizeInBytes;
    PointerData *beta_pointerData = initPointerData(env, beta);
    if (beta_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    beta_native = (void *)beta_pointerData->getPointer(env);
    dxDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, dxDesc);
    dx_native = (void *)getPointer(env, dx);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnConvolutionBackwardData(handle_native, alpha_native, wDesc_native, w_native, dyDesc_native, dy_native, convDesc_native, algo_native, workSpace_native, workSpaceSizeInBytes_native, beta_native, dxDesc_native, dx_native);

    // Write back native variable values
    // handle is read-only
    if (!releasePointerData(env, alpha_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // wDesc is read-only
    // w is a native pointer
    // dyDesc is read-only
    // dy is a native pointer
    // convDesc is read-only
    // algo is primitive
    // workSpace is a native pointer
    // workSpaceSizeInBytes is primitive
    if (!releasePointerData(env, beta_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // dxDesc is read-only
    // dx is a native pointer

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

/** Helper function to calculate folding descriptors for dgrad */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetFoldedConvBackwardDataDescriptorsNative(JNIEnv *env, jclass cls, jobject handle, jobject filterDesc, jobject diffDesc, jobject convDesc, jobject gradDesc, jint transformFormat, jobject foldedFilterDesc, jobject paddedDiffDesc, jobject foldedConvDesc, jobject foldedGradDesc, jobject filterFoldTransDesc, jobject diffPadTransDesc, jobject gradFoldTransDesc, jobject gradUnfoldTransDesc)
{
    // Null-checks for non-primitive arguments
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cudnnGetFoldedConvBackwardDataDescriptors");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (filterDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'filterDesc' is null for cudnnGetFoldedConvBackwardDataDescriptors");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (diffDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'diffDesc' is null for cudnnGetFoldedConvBackwardDataDescriptors");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (convDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'convDesc' is null for cudnnGetFoldedConvBackwardDataDescriptors");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (gradDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'gradDesc' is null for cudnnGetFoldedConvBackwardDataDescriptors");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    // transformFormat is primitive
    if (foldedFilterDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'foldedFilterDesc' is null for cudnnGetFoldedConvBackwardDataDescriptors");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (paddedDiffDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'paddedDiffDesc' is null for cudnnGetFoldedConvBackwardDataDescriptors");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (foldedConvDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'foldedConvDesc' is null for cudnnGetFoldedConvBackwardDataDescriptors");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (foldedGradDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'foldedGradDesc' is null for cudnnGetFoldedConvBackwardDataDescriptors");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (filterFoldTransDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'filterFoldTransDesc' is null for cudnnGetFoldedConvBackwardDataDescriptors");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (diffPadTransDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'diffPadTransDesc' is null for cudnnGetFoldedConvBackwardDataDescriptors");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (gradFoldTransDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'gradFoldTransDesc' is null for cudnnGetFoldedConvBackwardDataDescriptors");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (gradUnfoldTransDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'gradUnfoldTransDesc' is null for cudnnGetFoldedConvBackwardDataDescriptors");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetFoldedConvBackwardDataDescriptors(handle=%p, filterDesc=%p, diffDesc=%p, convDesc=%p, gradDesc=%p, transformFormat=%d, foldedFilterDesc=%p, paddedDiffDesc=%p, foldedConvDesc=%p, foldedGradDesc=%p, filterFoldTransDesc=%p, diffPadTransDesc=%p, gradFoldTransDesc=%p, gradUnfoldTransDesc=%p)\n",
        handle, filterDesc, diffDesc, convDesc, gradDesc, transformFormat, foldedFilterDesc, paddedDiffDesc, foldedConvDesc, foldedGradDesc, filterFoldTransDesc, diffPadTransDesc, gradFoldTransDesc, gradUnfoldTransDesc);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnFilterDescriptor_t filterDesc_native;
    cudnnTensorDescriptor_t diffDesc_native;
    cudnnConvolutionDescriptor_t convDesc_native;
    cudnnTensorDescriptor_t gradDesc_native;
    cudnnTensorFormat_t transformFormat_native;
    cudnnFilterDescriptor_t foldedFilterDesc_native;
    cudnnTensorDescriptor_t paddedDiffDesc_native;
    cudnnConvolutionDescriptor_t foldedConvDesc_native;
    cudnnTensorDescriptor_t foldedGradDesc_native;
    cudnnTensorTransformDescriptor_t filterFoldTransDesc_native;
    cudnnTensorTransformDescriptor_t diffPadTransDesc_native;
    cudnnTensorTransformDescriptor_t gradFoldTransDesc_native;
    cudnnTensorTransformDescriptor_t gradUnfoldTransDesc_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    filterDesc_native = (cudnnFilterDescriptor_t)getNativePointerValue(env, filterDesc);
    diffDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, diffDesc);
    convDesc_native = (cudnnConvolutionDescriptor_t)getNativePointerValue(env, convDesc);
    gradDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, gradDesc);
    transformFormat_native = (cudnnTensorFormat_t)transformFormat;
    foldedFilterDesc_native = (cudnnFilterDescriptor_t)getNativePointerValue(env, foldedFilterDesc);
    paddedDiffDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, paddedDiffDesc);
    foldedConvDesc_native = (cudnnConvolutionDescriptor_t)getNativePointerValue(env, foldedConvDesc);
    foldedGradDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, foldedGradDesc);
    filterFoldTransDesc_native = (cudnnTensorTransformDescriptor_t)getNativePointerValue(env, filterFoldTransDesc);
    diffPadTransDesc_native = (cudnnTensorTransformDescriptor_t)getNativePointerValue(env, diffPadTransDesc);
    gradFoldTransDesc_native = (cudnnTensorTransformDescriptor_t)getNativePointerValue(env, gradFoldTransDesc);
    gradUnfoldTransDesc_native = (cudnnTensorTransformDescriptor_t)getNativePointerValue(env, gradUnfoldTransDesc);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetFoldedConvBackwardDataDescriptors(handle_native, filterDesc_native, diffDesc_native, convDesc_native, gradDesc_native, transformFormat_native, foldedFilterDesc_native, paddedDiffDesc_native, foldedConvDesc_native, foldedGradDesc_native, filterFoldTransDesc_native, diffPadTransDesc_native, gradFoldTransDesc_native, gradUnfoldTransDesc_native);

    // Write back native variable values
    // handle is read-only
    // filterDesc is read-only
    // diffDesc is read-only
    // convDesc is read-only
    // gradDesc is read-only
    // transformFormat is primitive
    // foldedFilterDesc is read-only
    // paddedDiffDesc is read-only
    // foldedConvDesc is read-only
    // foldedGradDesc is read-only
    // filterFoldTransDesc is read-only
    // diffPadTransDesc is read-only
    // gradFoldTransDesc is read-only
    // gradUnfoldTransDesc is read-only

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnCnnInferVersionCheckNative(JNIEnv *env, jclass cls)
{
    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnCnnInferVersionCheck()\n");

    // Native function call
    cudnnStatus_t jniResult_native = cudnnCnnInferVersionCheck();

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetConvolutionBackwardFilterAlgorithmMaxCountNative(JNIEnv *env, jclass cls, jobject handle, jintArray count)
{
    // Null-checks for non-primitive arguments
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cudnnGetConvolutionBackwardFilterAlgorithmMaxCount");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (count == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'count' is null for cudnnGetConvolutionBackwardFilterAlgorithmMaxCount");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(handle=%p, count=%p)\n",
        handle, count);

    // Native variable declarations
    cudnnHandle_t handle_native;
    int count_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    // count is write-only

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(handle_native, &count_native);

    // Write back native variable values
    // handle is read-only
    if (!set(env, count, 0, (jint)count_native)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnFindConvolutionBackwardFilterAlgorithmNative(JNIEnv *env, jclass cls, jobject handle, jobject xDesc, jobject dyDesc, jobject convDesc, jobject dwDesc, jint requestedAlgoCount, jintArray returnedAlgoCount, jobjectArray perfResults)
{
    // Null-checks for non-primitive arguments
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
    int returnedAlgoCount_native;
    cudnnConvolutionBwdFilterAlgoPerf_t * perfResults_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    xDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, xDesc);
    dyDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, dyDesc);
    convDesc_native = (cudnnConvolutionDescriptor_t)getNativePointerValue(env, convDesc);
    dwDesc_native = (cudnnFilterDescriptor_t)getNativePointerValue(env, dwDesc);
    requestedAlgoCount_native = (int)requestedAlgoCount;
    // returnedAlgoCount is write-only
    if (!initNative(env, perfResults, perfResults_native, requestedAlgoCount)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnFindConvolutionBackwardFilterAlgorithm(handle_native, xDesc_native, dyDesc_native, convDesc_native, dwDesc_native, requestedAlgoCount_native, &returnedAlgoCount_native, perfResults_native);

    // Write back native variable values
    // handle is read-only
    // xDesc is read-only
    // dyDesc is read-only
    // convDesc is read-only
    // dwDesc is read-only
    // requestedAlgoCount is primitive
    if (!set(env, returnedAlgoCount, 0, (jint)returnedAlgoCount_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!releaseNative(env, perfResults_native, perfResults, returnedAlgoCount_native)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnFindConvolutionBackwardFilterAlgorithmExNative(JNIEnv *env, jclass cls, jobject handle, jobject xDesc, jobject x, jobject dyDesc, jobject y, jobject convDesc, jobject dwDesc, jobject dw, jint requestedAlgoCount, jintArray returnedAlgoCount, jobjectArray perfResults, jobject workSpace, jlong workSpaceSizeInBytes)
{
    // Null-checks for non-primitive arguments
    if (perfResults == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'perfResults' is null for cudnnFindConvolutionBackwardFilterAlgorithmEx");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnFindConvolutionBackwardFilterAlgorithmEx(handle=%p, xDesc=%p, x=%p, dyDesc=%p, y=%p, convDesc=%p, dwDesc=%p, dw=%p, requestedAlgoCount=%d, returnedAlgoCount=%p, perfResults=%p, workSpace=%p, workSpaceSizeInBytes=%ld)\n",
        handle, xDesc, x, dyDesc, y, convDesc, dwDesc, dw, requestedAlgoCount, returnedAlgoCount, perfResults, workSpace, workSpaceSizeInBytes);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnTensorDescriptor_t xDesc_native;
    void * x_native = NULL;
    cudnnTensorDescriptor_t dyDesc_native;
    void * y_native = NULL;
    cudnnConvolutionDescriptor_t convDesc_native;
    cudnnFilterDescriptor_t dwDesc_native;
    void * dw_native = NULL;
    int requestedAlgoCount_native = 0;
    int returnedAlgoCount_native;
    cudnnConvolutionBwdFilterAlgoPerf_t * perfResults_native;
    void * workSpace_native = NULL;
    size_t workSpaceSizeInBytes_native = 0;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    xDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, xDesc);
    x_native = (void *)getPointer(env, x);
    dyDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, dyDesc);
    y_native = (void *)getPointer(env, y);
    convDesc_native = (cudnnConvolutionDescriptor_t)getNativePointerValue(env, convDesc);
    dwDesc_native = (cudnnFilterDescriptor_t)getNativePointerValue(env, dwDesc);
    dw_native = (void *)getPointer(env, dw);
    requestedAlgoCount_native = (int)requestedAlgoCount;
    // returnedAlgoCount is write-only
    if (!initNative(env, perfResults, perfResults_native, requestedAlgoCount)) return JCUDNN_STATUS_INTERNAL_ERROR;
    workSpace_native = (void *)getPointer(env, workSpace);
    workSpaceSizeInBytes_native = (size_t)workSpaceSizeInBytes;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnFindConvolutionBackwardFilterAlgorithmEx(handle_native, xDesc_native, x_native, dyDesc_native, y_native, convDesc_native, dwDesc_native, dw_native, requestedAlgoCount_native, &returnedAlgoCount_native, perfResults_native, workSpace_native, workSpaceSizeInBytes_native);

    // Write back native variable values
    // handle is read-only
    // xDesc is read-only
    // x is a native pointer
    // dyDesc is read-only
    // y is a native pointer
    // convDesc is read-only
    // dwDesc is read-only
    // dw is a native pointer
    // requestedAlgoCount is primitive
    if (!set(env, returnedAlgoCount, 0, (jint)returnedAlgoCount_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!releaseNative(env, perfResults_native, perfResults, returnedAlgoCount_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // workSpace is a native pointer
    // workSpaceSizeInBytes is primitive

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetConvolutionBackwardFilterAlgorithm_1v7Native(JNIEnv *env, jclass cls, jobject handle, jobject srcDesc, jobject diffDesc, jobject convDesc, jobject gradDesc, jint requestedAlgoCount, jintArray returnedAlgoCount, jobjectArray perfResults)
{
    // Null-checks for non-primitive arguments
    if (perfResults == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'perfResults' is null for cudnnGetConvolutionBackwardFilterAlgorithm_v7");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetConvolutionBackwardFilterAlgorithm_v7(handle=%p, srcDesc=%p, diffDesc=%p, convDesc=%p, gradDesc=%p, requestedAlgoCount=%d, returnedAlgoCount=%p, perfResults=%p)\n",
        handle, srcDesc, diffDesc, convDesc, gradDesc, requestedAlgoCount, returnedAlgoCount, perfResults);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnTensorDescriptor_t srcDesc_native;
    cudnnTensorDescriptor_t diffDesc_native;
    cudnnConvolutionDescriptor_t convDesc_native;
    cudnnFilterDescriptor_t gradDesc_native;
    int requestedAlgoCount_native = 0;
    int returnedAlgoCount_native;
    cudnnConvolutionBwdFilterAlgoPerf_t * perfResults_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    srcDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, srcDesc);
    diffDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, diffDesc);
    convDesc_native = (cudnnConvolutionDescriptor_t)getNativePointerValue(env, convDesc);
    gradDesc_native = (cudnnFilterDescriptor_t)getNativePointerValue(env, gradDesc);
    requestedAlgoCount_native = (int)requestedAlgoCount;
    // returnedAlgoCount is write-only
    if (!initNative(env, perfResults, perfResults_native, requestedAlgoCount)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetConvolutionBackwardFilterAlgorithm_v7(handle_native, srcDesc_native, diffDesc_native, convDesc_native, gradDesc_native, requestedAlgoCount_native, &returnedAlgoCount_native, perfResults_native);

    // Write back native variable values
    // handle is read-only
    // srcDesc is read-only
    // diffDesc is read-only
    // convDesc is read-only
    // gradDesc is read-only
    // requestedAlgoCount is primitive
    if (!set(env, returnedAlgoCount, 0, (jint)returnedAlgoCount_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    if (!releaseNative(env, perfResults_native, perfResults, returnedAlgoCount_native)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

/**
 *  convolution algorithm (which requires potentially some workspace)
 */
/** Helper function to return the minimum size of the workspace to be passed to the convolution given an algo*/
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetConvolutionBackwardFilterWorkspaceSizeNative(JNIEnv *env, jclass cls, jobject handle, jobject xDesc, jobject dyDesc, jobject convDesc, jobject gradDesc, jint algo, jlongArray sizeInBytes)
{
    // Null-checks for non-primitive arguments

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
    size_t sizeInBytes_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    xDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, xDesc);
    dyDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, dyDesc);
    convDesc_native = (cudnnConvolutionDescriptor_t)getNativePointerValue(env, convDesc);
    gradDesc_native = (cudnnFilterDescriptor_t)getNativePointerValue(env, gradDesc);
    algo_native = (cudnnConvolutionBwdFilterAlgo_t)algo;
    // sizeInBytes is write-only

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetConvolutionBackwardFilterWorkspaceSize(handle_native, xDesc_native, dyDesc_native, convDesc_native, gradDesc_native, algo_native, &sizeInBytes_native);

    // Write back native variable values
    // handle is read-only
    // xDesc is read-only
    // dyDesc is read-only
    // convDesc is read-only
    // gradDesc is read-only
    // algo is primitive
    if (!set(env, sizeInBytes, 0, (jlong)sizeInBytes_native)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnConvolutionBackwardFilterNative(JNIEnv *env, jclass cls, jobject handle, jobject alpha, jobject xDesc, jobject x, jobject dyDesc, jobject dy, jobject convDesc, jint algo, jobject workSpace, jlong workSpaceSizeInBytes, jobject beta, jobject dwDesc, jobject dw)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnConvolutionBackwardFilter(handle=%p, alpha=%p, xDesc=%p, x=%p, dyDesc=%p, dy=%p, convDesc=%p, algo=%d, workSpace=%p, workSpaceSizeInBytes=%ld, beta=%p, dwDesc=%p, dw=%p)\n",
        handle, alpha, xDesc, x, dyDesc, dy, convDesc, algo, workSpace, workSpaceSizeInBytes, beta, dwDesc, dw);

    // Native variable declarations
    cudnnHandle_t handle_native;
    void * alpha_native = NULL;
    cudnnTensorDescriptor_t xDesc_native;
    void * x_native = NULL;
    cudnnTensorDescriptor_t dyDesc_native;
    void * dy_native = NULL;
    cudnnConvolutionDescriptor_t convDesc_native;
    cudnnConvolutionBwdFilterAlgo_t algo_native;
    void * workSpace_native = NULL;
    size_t workSpaceSizeInBytes_native = 0;
    void * beta_native = NULL;
    cudnnFilterDescriptor_t dwDesc_native;
    void * dw_native = NULL;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    PointerData *alpha_pointerData = initPointerData(env, alpha);
    if (alpha_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    alpha_native = (void *)alpha_pointerData->getPointer(env);
    xDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, xDesc);
    x_native = (void *)getPointer(env, x);
    dyDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, dyDesc);
    dy_native = (void *)getPointer(env, dy);
    convDesc_native = (cudnnConvolutionDescriptor_t)getNativePointerValue(env, convDesc);
    algo_native = (cudnnConvolutionBwdFilterAlgo_t)algo;
    workSpace_native = (void *)getPointer(env, workSpace);
    workSpaceSizeInBytes_native = (size_t)workSpaceSizeInBytes;
    PointerData *beta_pointerData = initPointerData(env, beta);
    if (beta_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    beta_native = (void *)beta_pointerData->getPointer(env);
    dwDesc_native = (cudnnFilterDescriptor_t)getNativePointerValue(env, dwDesc);
    dw_native = (void *)getPointer(env, dw);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnConvolutionBackwardFilter(handle_native, alpha_native, xDesc_native, x_native, dyDesc_native, dy_native, convDesc_native, algo_native, workSpace_native, workSpaceSizeInBytes_native, beta_native, dwDesc_native, dw_native);

    // Write back native variable values
    // handle is read-only
    if (!releasePointerData(env, alpha_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // xDesc is read-only
    // x is a native pointer
    // dyDesc is read-only
    // dy is a native pointer
    // convDesc is read-only
    // algo is primitive
    // workSpace is a native pointer
    // workSpaceSizeInBytes is primitive
    if (!releasePointerData(env, beta_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // dwDesc is read-only
    // dw is a native pointer

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

/** Function to compute the bias gradient for batch convolution */
JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnConvolutionBackwardBiasNative(JNIEnv *env, jclass cls, jobject handle, jobject alpha, jobject dyDesc, jobject dy, jobject beta, jobject dbDesc, jobject db)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnConvolutionBackwardBias(handle=%p, alpha=%p, dyDesc=%p, dy=%p, beta=%p, dbDesc=%p, db=%p)\n",
        handle, alpha, dyDesc, dy, beta, dbDesc, db);

    // Native variable declarations
    cudnnHandle_t handle_native;
    void * alpha_native = NULL;
    cudnnTensorDescriptor_t dyDesc_native;
    void * dy_native = NULL;
    void * beta_native = NULL;
    cudnnTensorDescriptor_t dbDesc_native;
    void * db_native = NULL;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    PointerData *alpha_pointerData = initPointerData(env, alpha);
    if (alpha_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    alpha_native = (void *)alpha_pointerData->getPointer(env);
    dyDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, dyDesc);
    dy_native = (void *)getPointer(env, dy);
    PointerData *beta_pointerData = initPointerData(env, beta);
    if (beta_pointerData == NULL)
    {
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    beta_native = (void *)beta_pointerData->getPointer(env);
    dbDesc_native = (cudnnTensorDescriptor_t)getNativePointerValue(env, dbDesc);
    db_native = (void *)getPointer(env, db);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnConvolutionBackwardBias(handle_native, alpha_native, dyDesc_native, dy_native, beta_native, dbDesc_native, db_native);

    // Write back native variable values
    // handle is read-only
    if (!releasePointerData(env, alpha_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // dyDesc is read-only
    // dy is a native pointer
    if (!releasePointerData(env, beta_pointerData, JNI_ABORT)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // dbDesc is read-only
    // db is a native pointer

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnCreateFusedOpsConstParamPackNative(JNIEnv *env, jclass cls, jobject constPack, jint ops)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnCreateFusedOpsConstParamPack(constPack=%p, ops=%d)\n",
        constPack, ops);

    // Native variable declarations
    cudnnFusedOpsConstParamPack_t constPack_native;
    cudnnFusedOps_t ops_native;

    // Obtain native variable values
    // constPack is write-only
    ops_native = (cudnnFusedOps_t)ops;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnCreateFusedOpsConstParamPack(&constPack_native, ops_native);

    // Write back native variable values
    setNativePointerValue(env, constPack, (jlong)constPack_native);
    // ops is primitive

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnDestroyFusedOpsConstParamPackNative(JNIEnv *env, jclass cls, jobject constPack)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnDestroyFusedOpsConstParamPack(constPack=%p)\n",
        constPack);

    // Native variable declarations
    cudnnFusedOpsConstParamPack_t constPack_native;

    // Obtain native variable values
    constPack_native = (cudnnFusedOpsConstParamPack_t)getNativePointerValue(env, constPack);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnDestroyFusedOpsConstParamPack(constPack_native);

    // Write back native variable values
    // constPack is read-only

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnSetFusedOpsConstParamPackAttributeNative(JNIEnv *env, jclass cls, jobject constPack, jint paramLabel, jobject param)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnSetFusedOpsConstParamPackAttribute(constPack=%p, paramLabel=%d, param=%p)\n",
        constPack, paramLabel, param);

    // Native variable declarations
    cudnnFusedOpsConstParamPack_t constPack_native;
    cudnnFusedOpsConstParamLabel_t paramLabel_native;
    void * param_native = NULL;

    // Obtain native variable values
    constPack_native = (cudnnFusedOpsConstParamPack_t)getNativePointerValue(env, constPack);
    paramLabel_native = (cudnnFusedOpsConstParamLabel_t)paramLabel;
    param_native = (void *)getPointer(env, param);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnSetFusedOpsConstParamPackAttribute(constPack_native, paramLabel_native, param_native);

    // Write back native variable values
    // constPack is read-only
    // paramLabel is primitive
    // param is a native pointer

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetFusedOpsConstParamPackAttributeNative(JNIEnv *env, jclass cls, jobject constPack, jint paramLabel, jobject param, jintArray isNULL)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetFusedOpsConstParamPackAttribute(constPack=%p, paramLabel=%d, param=%p, isNULL=%p)\n",
        constPack, paramLabel, param, isNULL);

    // Native variable declarations
    cudnnFusedOpsConstParamPack_t constPack_native;
    cudnnFusedOpsConstParamLabel_t paramLabel_native;
    void * param_native = NULL;
    int isNULL_native;

    // Obtain native variable values
    constPack_native = (cudnnFusedOpsConstParamPack_t)getNativePointerValue(env, constPack);
    paramLabel_native = (cudnnFusedOpsConstParamLabel_t)paramLabel;
    param_native = (void *)getPointer(env, param);
    // isNULL is write-only

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetFusedOpsConstParamPackAttribute(constPack_native, paramLabel_native, param_native, &isNULL_native);

    // Write back native variable values
    // constPack is read-only
    // paramLabel is primitive
    // param is a native pointer
    if (!set(env, isNULL, 0, (jint)isNULL_native)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnCreateFusedOpsVariantParamPackNative(JNIEnv *env, jclass cls, jobject varPack, jint ops)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnCreateFusedOpsVariantParamPack(varPack=%p, ops=%d)\n",
        varPack, ops);

    // Native variable declarations
    cudnnFusedOpsVariantParamPack_t varPack_native;
    cudnnFusedOps_t ops_native;

    // Obtain native variable values
    // varPack is write-only
    ops_native = (cudnnFusedOps_t)ops;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnCreateFusedOpsVariantParamPack(&varPack_native, ops_native);

    // Write back native variable values
    setNativePointerValue(env, varPack, (jlong)varPack_native);
    // ops is primitive

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnDestroyFusedOpsVariantParamPackNative(JNIEnv *env, jclass cls, jobject varPack)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnDestroyFusedOpsVariantParamPack(varPack=%p)\n",
        varPack);

    // Native variable declarations
    cudnnFusedOpsVariantParamPack_t varPack_native;

    // Obtain native variable values
    varPack_native = (cudnnFusedOpsVariantParamPack_t)getNativePointerValue(env, varPack);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnDestroyFusedOpsVariantParamPack(varPack_native);

    // Write back native variable values
    // varPack is read-only

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnSetFusedOpsVariantParamPackAttributeNative(JNIEnv *env, jclass cls, jobject varPack, jint paramLabel, jobject ptr)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnSetFusedOpsVariantParamPackAttribute(varPack=%p, paramLabel=%d, ptr=%p)\n",
        varPack, paramLabel, ptr);

    // Native variable declarations
    cudnnFusedOpsVariantParamPack_t varPack_native;
    cudnnFusedOpsVariantParamLabel_t paramLabel_native;
    void * ptr_native = NULL;

    // Obtain native variable values
    varPack_native = (cudnnFusedOpsVariantParamPack_t)getNativePointerValue(env, varPack);
    paramLabel_native = (cudnnFusedOpsVariantParamLabel_t)paramLabel;
    ptr_native = (void *)getPointer(env, ptr);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnSetFusedOpsVariantParamPackAttribute(varPack_native, paramLabel_native, ptr_native);

    // Write back native variable values
    // varPack is read-only
    // paramLabel is primitive
    // ptr is a native pointer

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnGetFusedOpsVariantParamPackAttributeNative(JNIEnv *env, jclass cls, jobject varPack, jint paramLabel, jobject ptr)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnGetFusedOpsVariantParamPackAttribute(varPack=%p, paramLabel=%d, ptr=%p)\n",
        varPack, paramLabel, ptr);

    // Native variable declarations
    cudnnFusedOpsVariantParamPack_t varPack_native;
    cudnnFusedOpsVariantParamLabel_t paramLabel_native;
    void * ptr_native = NULL;

    // Obtain native variable values
    varPack_native = (cudnnFusedOpsVariantParamPack_t)getNativePointerValue(env, varPack);
    paramLabel_native = (cudnnFusedOpsVariantParamLabel_t)paramLabel;
    ptr_native = (void *)getPointer(env, ptr);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnGetFusedOpsVariantParamPackAttribute(varPack_native, paramLabel_native, ptr_native);

    // Write back native variable values
    // varPack is read-only
    // paramLabel is primitive
    // ptr is a native pointer

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnCreateFusedOpsPlanNative(JNIEnv *env, jclass cls, jobject plan, jint ops)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnCreateFusedOpsPlan(plan=%p, ops=%d)\n",
        plan, ops);

    // Native variable declarations
    cudnnFusedOpsPlan_t plan_native;
    cudnnFusedOps_t ops_native;

    // Obtain native variable values
    // plan is write-only
    ops_native = (cudnnFusedOps_t)ops;

    // Native function call
    cudnnStatus_t jniResult_native = cudnnCreateFusedOpsPlan(&plan_native, ops_native);

    // Write back native variable values
    setNativePointerValue(env, plan, (jlong)plan_native);
    // ops is primitive

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnDestroyFusedOpsPlanNative(JNIEnv *env, jclass cls, jobject plan)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnDestroyFusedOpsPlan(plan=%p)\n",
        plan);

    // Native variable declarations
    cudnnFusedOpsPlan_t plan_native;

    // Obtain native variable values
    plan_native = (cudnnFusedOpsPlan_t)getNativePointerValue(env, plan);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnDestroyFusedOpsPlan(plan_native);

    // Write back native variable values
    // plan is read-only

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnMakeFusedOpsPlanNative(JNIEnv *env, jclass cls, jobject handle, jobject plan, jobject constPack, jlongArray workspaceSizeInBytes)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnMakeFusedOpsPlan(handle=%p, plan=%p, constPack=%p, workspaceSizeInBytes=%p)\n",
        handle, plan, constPack, workspaceSizeInBytes);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnFusedOpsPlan_t plan_native;
    cudnnFusedOpsConstParamPack_t constPack_native;
    size_t workspaceSizeInBytes_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    plan_native = (cudnnFusedOpsPlan_t)getNativePointerValue(env, plan);
    constPack_native = (cudnnFusedOpsConstParamPack_t)getNativePointerValue(env, constPack);
    // workspaceSizeInBytes is write-only

    // Native function call
    cudnnStatus_t jniResult_native = cudnnMakeFusedOpsPlan(handle_native, plan_native, constPack_native, &workspaceSizeInBytes_native);

    // Write back native variable values
    // handle is read-only
    // plan is read-only
    // constPack is read-only
    if (!set(env, workspaceSizeInBytes, 0, (jlong)workspaceSizeInBytes_native)) return JCUDNN_STATUS_INTERNAL_ERROR;

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnFusedOpsExecuteNative(JNIEnv *env, jclass cls, jobject handle, jobject plan, jobject varPack)
{
    // Null-checks for non-primitive arguments
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cudnnFusedOpsExecute");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (plan == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'plan' is null for cudnnFusedOpsExecute");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }
    if (varPack == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'varPack' is null for cudnnFusedOpsExecute");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnFusedOpsExecute(handle=%p, plan=%p, varPack=%p)\n",
        handle, plan, varPack);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnFusedOpsPlan_t plan_native;
    cudnnFusedOpsVariantParamPack_t varPack_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    plan_native = (cudnnFusedOpsPlan_t)getNativePointerValue(env, plan);
    varPack_native = (cudnnFusedOpsVariantParamPack_t)getNativePointerValue(env, varPack);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnFusedOpsExecute(handle_native, plan_native, varPack_native);

    // Write back native variable values
    // handle is read-only
    // plan is read-only
    // varPack is read-only

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnCnnTrainVersionCheckNative(JNIEnv *env, jclass cls)
{
    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnCnnTrainVersionCheck()\n");

    // Native function call
    cudnnStatus_t jniResult_native = cudnnCnnTrainVersionCheck();

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnBackendCreateDescriptorNative(JNIEnv *env, jclass cls, jint descriptorType, jobject descriptor)
{
    // Null-checks for non-primitive arguments
    // descriptorType is primitive
    if (descriptor == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'descriptor' is null for cudnnBackendCreateDescriptor");
        return JCUDNN_STATUS_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnBackendCreateDescriptor(descriptorType=%d, descriptor=%p)\n",
        descriptorType, descriptor);

    // Native variable declarations
    cudnnBackendDescriptorType_t descriptorType_native;
    cudnnBackendDescriptor_t descriptor_native;

    // Obtain native variable values
    descriptorType_native = (cudnnBackendDescriptorType_t)descriptorType;
    // descriptor is write-only

    // Native function call
    cudnnStatus_t jniResult_native = cudnnBackendCreateDescriptor(descriptorType_native, &descriptor_native);

    // Write back native variable values
    // descriptorType is primitive
    setNativePointerValue(env, descriptor, (jlong)descriptor_native);

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnBackendDestroyDescriptorNative(JNIEnv *env, jclass cls, jobject descriptor)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnBackendDestroyDescriptor(descriptor=%p)\n",
        descriptor);

    // Native variable declarations
    cudnnBackendDescriptor_t descriptor_native;

    // Obtain native variable values
    descriptor_native = (cudnnBackendDescriptor_t)getNativePointerValue(env, descriptor);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnBackendDestroyDescriptor(descriptor_native);

    // Write back native variable values
    // descriptor is read-only

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnBackendInitializeNative(JNIEnv *env, jclass cls, jobject descriptor)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnBackendInitialize(descriptor=%p)\n",
        descriptor);

    // Native variable declarations
    cudnnBackendDescriptor_t descriptor_native;

    // Obtain native variable values
    descriptor_native = (cudnnBackendDescriptor_t)getNativePointerValue(env, descriptor);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnBackendInitialize(descriptor_native);

    // Write back native variable values
    // descriptor is read-only

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnBackendFinalizeNative(JNIEnv *env, jclass cls, jobject descriptor)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnBackendFinalize(descriptor=%p)\n",
        descriptor);

    // Native variable declarations
    cudnnBackendDescriptor_t descriptor_native;

    // Obtain native variable values
    descriptor_native = (cudnnBackendDescriptor_t)getNativePointerValue(env, descriptor);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnBackendFinalize(descriptor_native);

    // Write back native variable values
    // descriptor is read-only

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnBackendSetAttributeNative(JNIEnv *env, jclass cls, jobject descriptor, jint attributeName, jint attributeType, jlong elementCount, jobject arrayOfElements)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnBackendSetAttribute(descriptor=%p, attributeName=%d, attributeType=%d, elementCount=%ld, arrayOfElements=%p)\n",
        descriptor, attributeName, attributeType, elementCount, arrayOfElements);

    // Native variable declarations
    cudnnBackendDescriptor_t descriptor_native;
    cudnnBackendAttributeName_t attributeName_native;
    cudnnBackendAttributeType_t attributeType_native;
    int64_t elementCount_native = 0;
    void * arrayOfElements_native = NULL;

    // Obtain native variable values
    descriptor_native = (cudnnBackendDescriptor_t)getNativePointerValue(env, descriptor);
    attributeName_native = (cudnnBackendAttributeName_t)attributeName;
    attributeType_native = (cudnnBackendAttributeType_t)attributeType;
    elementCount_native = (int64_t)elementCount;
    arrayOfElements_native = (void *)getPointer(env, arrayOfElements);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnBackendSetAttribute(descriptor_native, attributeName_native, attributeType_native, elementCount_native, arrayOfElements_native);

    // Write back native variable values
    // descriptor is read-only
    // attributeName is primitive
    // attributeType is primitive
    // elementCount is primitive
    // arrayOfElements is a native pointer

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnBackendGetAttributeNative(JNIEnv *env, jclass cls, jobject descriptor, jint attributeName, jint attributeType, jlong requestedElementCount, jlongArray elementCount, jobject arrayOfElements)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnBackendGetAttribute(descriptor=%p, attributeName=%d, attributeType=%d, requestedElementCount=%ld, elementCount=%p, arrayOfElements=%p)\n",
        descriptor, attributeName, attributeType, requestedElementCount, elementCount, arrayOfElements);

    // Native variable declarations
    cudnnBackendDescriptor_t descriptor_native;
    cudnnBackendAttributeName_t attributeName_native;
    cudnnBackendAttributeType_t attributeType_native;
    int64_t requestedElementCount_native = 0;
    int64_t elementCount_native;
    void * arrayOfElements_native = NULL;

    // Obtain native variable values
    descriptor_native = (cudnnBackendDescriptor_t)getNativePointerValue(env, descriptor);
    attributeName_native = (cudnnBackendAttributeName_t)attributeName;
    attributeType_native = (cudnnBackendAttributeType_t)attributeType;
    requestedElementCount_native = (int64_t)requestedElementCount;
    // elementCount is write-only
    arrayOfElements_native = (void *)getPointer(env, arrayOfElements);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnBackendGetAttribute(descriptor_native, attributeName_native, attributeType_native, requestedElementCount_native, &elementCount_native, arrayOfElements_native);

    // Write back native variable values
    // descriptor is read-only
    // attributeName is primitive
    // attributeType is primitive
    // requestedElementCount is primitive
    if (!set(env, elementCount, 0, (jlong)elementCount_native)) return JCUDNN_STATUS_INTERNAL_ERROR;
    // arrayOfElements is a native pointer

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

JNIEXPORT jint JNICALL Java_jcuda_jcudnn_JCudnn_cudnnBackendExecuteNative(JNIEnv *env, jclass cls, jobject handle, jobject executionPlan, jobject variantPack)
{
    // Null-checks for non-primitive arguments

    // Log message
    Logger::log(LOG_TRACE, "Executing cudnnBackendExecute(handle=%p, executionPlan=%p, variantPack=%p)\n",
        handle, executionPlan, variantPack);

    // Native variable declarations
    cudnnHandle_t handle_native;
    cudnnBackendDescriptor_t executionPlan_native;
    cudnnBackendDescriptor_t variantPack_native;

    // Obtain native variable values
    handle_native = (cudnnHandle_t)getNativePointerValue(env, handle);
    executionPlan_native = (cudnnBackendDescriptor_t)getNativePointerValue(env, executionPlan);
    variantPack_native = (cudnnBackendDescriptor_t)getNativePointerValue(env, variantPack);

    // Native function call
    cudnnStatus_t jniResult_native = cudnnBackendExecute(handle_native, executionPlan_native, variantPack_native);

    // Write back native variable values
    // handle is read-only
    // executionPlan is read-only
    // variantPack is read-only

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

