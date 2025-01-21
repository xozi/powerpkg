## powerpkg

This is solver for distrubution models currently using CUDA cuBLAS to preform matrix operations.
There is plenty of work to made in terms of optimization, I'm currently not utilizing kernals and merely the matrix operations, so solving is not optimized yet. Optimizations will be made later. 

In [releases](https://github.com/xozi/powerpkg/releases/tag/homework) you can find the binaries for sets of homework problems. They require CUDA installed to operate, pure C++ implementation may be attempted, but only on request as complex value types from cuComplex are currently being used for all complex operations. I can transition these to C++ complex types from complex.h, but it will need to be seperated from the CUDA version entirely.
