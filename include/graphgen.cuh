#ifndef GRAPHGEN_H
#define GRAPHGEN_H

#include <cuda_runtime.h>
#include <vector>
#include <functional>

// Function type for GPU kernels
using KernelFunction = std::function<void(cudaStream_t)>;

// Wrapper to generate and execute CUDA graph from kernels
cudaError_t executeKernelsAsGraph(const std::vector<KernelFunction>& kernels) {
    cudaStream_t stream;
    cudaGraph_t graph;
    cudaGraphExec_t graphExec = nullptr;
    
    // Create stream
    cudaError_t error = cudaStreamCreate(&stream);
    if (error != cudaSuccess) return error;

    // Begin capture
    error = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    if (error != cudaSuccess) {
        cudaStreamDestroy(stream);
        return error;
    }

    // Execute all kernels in capture mode
    for (const auto& kernel : kernels) {
        kernel(stream);
    }

    // End capture
    error = cudaStreamEndCapture(stream, &graph);
    if (error != cudaSuccess) {
        cudaStreamDestroy(stream);
        return error;
    }

    // Create executable graph
    error = cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);
    if (error != cudaSuccess) {
        cudaGraphDestroy(graph);
        cudaStreamDestroy(stream);
        return error;
    }

    // Launch graph
    error = cudaGraphLaunch(graphExec, stream);
    if (error != cudaSuccess) {
        cudaGraphExecDestroy(graphExec);
        cudaGraphDestroy(graph);
        cudaStreamDestroy(stream);
        return error;
    }

    // Synchronize
    error = cudaStreamSynchronize(stream);

    // Cleanup
    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);

    return error;
}

#endif
