# jcudnn
JCudnn - Java bindings for cuDNN

This is a first version of JCudnn. It is not part of the core JCuda
libraries, but depends on the following libraries:

- [jcuda](https://github.com/jcuda/jcuda)
- [jcuda-common](https://github.com/jcuda/jcuda-common)
- [jcublas](https://github.com/jcuda/jcublas)
 
Refer to [jcuda-main](https://github.com/jcuda/jcuda-main) for further
information and build instructions for these libraries.

## Building on Linux

If you have already built and installed jcuda-main, this is a short set of building instructions for linux (provided `$JAVA_HOME` is set). In the project root:

```
cmake JCudnnJNI/
make

mvn clean install
cd JCudnnJava/
mvn clean install
```
