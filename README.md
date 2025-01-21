## powerpkg

This is solver for distrubution models currently using CUDA cuBLAS to preform matrix operations.
There is plenty of work to made in terms of optimization, I'm currently not utilizing kernals and merely the matrix operations, so solving is not optimized yet. Optimizations will be made later. 

In [releases](https://github.com/xozi/powerpkg/releases/tag/homework) you can find the binaries for sets of homework problems. They require CUDA installed to operate, pure C++ implementation does not exist currently, but is possible. The only limitation to this is complex value types from cuComplex are currently being used for all complex operations and cuBLAS is handling all matrix operations. Both of these will have to be rewritten with native C++ complaint code, which C++ does have complex operations in `complex.h`.
