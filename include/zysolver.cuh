#ifndef ZYSOLVER_CUH
#define ZYSOLVER_CUH

#include <cuda_runtime.h>
#include <cuComplex.h>
#include "power_abc.cuh"

template<typename T>
struct LineParam {  
    int conductors;
    T* res;      
    T* gmr;           
    T* x;   
    T* y;    
};

template<typename T>
struct GPULineParam {
    int size;
    T* d_res;
    T* d_gmr;
    T* d_x;
    T* d_y;
    cuFloatComplex* d_Z_prim;
    cuFloatComplex* d_Z_abc;
    cuFloatComplex* d_t_n;
    GPULineParam* d_this;

    // Constructor - allocates and copies from host LineParam
    GPULineParam(const LineParam<T>& params) {
        size = params.conductors;
        cudaDeviceSynchronize();
        
        // Allocate device pointer for this struct
        CUDA_CHECK(cudaMalloc(&d_this, sizeof(GPULineParam)));

        // Allocate and copy arrays
        if (size > 0) {
            CUDA_CHECK(cudaMalloc(&d_res, size * sizeof(T)));
            CUDA_CHECK(cudaMemcpy(d_res, params.res, size * sizeof(T), cudaMemcpyHostToDevice));
            
            CUDA_CHECK(cudaMalloc(&d_gmr, size * sizeof(T)));
            CUDA_CHECK(cudaMemcpy(d_gmr, params.gmr, size * sizeof(T), cudaMemcpyHostToDevice));
            
            CUDA_CHECK(cudaMalloc(&d_x, size * sizeof(T)));
            CUDA_CHECK(cudaMemcpy(d_x, params.x, size * sizeof(T), cudaMemcpyHostToDevice));
            
            CUDA_CHECK(cudaMalloc(&d_y, size * sizeof(T)));
            CUDA_CHECK(cudaMemcpy(d_y, params.y, size * sizeof(T), cudaMemcpyHostToDevice));
            
            CUDA_CHECK(cudaMalloc(&d_Z_prim, size * size * sizeof(cuFloatComplex)));
            CUDA_CHECK(cudaMalloc(&d_Z_abc, 3 * 3 * sizeof(cuFloatComplex)));
            CUDA_CHECK(cudaMalloc(&d_t_n, 3 * sizeof(cuFloatComplex)));
        }

        // Copy this struct to device
        CUDA_CHECK(cudaMemcpy(d_this, this, sizeof(GPULineParam), cudaMemcpyHostToDevice));
        cudaDeviceSynchronize();
    }

    // Copy results back to host
    void copyToHost(PowerLineMatrix<cuFloatComplex>& power) {
        CUDA_CHECK(cudaMemcpy(power.Z_abc, d_Z_abc, 9 * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(power.t_n, d_t_n, 3 * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost));
        cudaDeviceSynchronize();
    }

    // Destructor - handles cleanup
    ~GPULineParam() {
        cudaDeviceSynchronize();
        if (d_res) cudaFree(d_res);
        if (d_gmr) cudaFree(d_gmr);
        if (d_x) cudaFree(d_x);
        if (d_y) cudaFree(d_y);
        if (d_Z_prim) cudaFree(d_Z_prim);
        if (d_Z_abc) cudaFree(d_Z_abc);
        if (d_this) cudaFree(d_this);
        if (d_t_n) cudaFree(d_t_n);
        cudaDeviceSynchronize();
    }
};

__global__ void calc_zprim(GPULineParam<float>* params) {
    int i = threadIdx.x;  // Row index
    int j = threadIdx.y;  // Column index

    if (i < params->size && j < params->size) {
        if (i == j) {
            // Self-impedance
            float resistance = params->d_res[i];
            float gmr = params->d_gmr[i];
            params->d_Z_prim[i * params->size + j] = make_cuFloatComplex(
                resistance + 0.09530f, 
                0.12134f * (logf(1.0f / gmr) + 7.93402f)
            );
        } else {
            // Mutual impedance
            float distance = sqrtf((params->d_x[j] - params->d_x[i]) * (params->d_x[j] - params->d_x[i]) + 
                                   (params->d_y[j] - params->d_y[i]) * (params->d_y[j] - params->d_y[i]));
            params->d_Z_prim[i * params->size + j] = make_cuFloatComplex(
                0.09530f, 
                0.12134f * (logf(1.0f / distance) + 7.93402f)
            );
        }
    }
}

__device__ cuFloatComplex complex_div(cuFloatComplex a, cuFloatComplex b) {
    /*
        Complex Division (real(a), imag(a)) / (real(b), imag(b) = 
        (real(a) * real(b) + imag(a) * imag(b)) / (real(b)^2 + imag(b)^2),
        (imag(a) * real(b) - real(a) * imag(b)) / (real(b)^2 + imag(b)^2)
    */ 
    float denom = cuCrealf(b) * cuCrealf(b) + cuCimagf(b) * cuCimagf(b);
    return make_cuFloatComplex(
        (cuCrealf(a) * cuCrealf(b) + cuCimagf(a) * cuCimagf(b)) / denom,
        (cuCimagf(a) * cuCrealf(b) - cuCrealf(a) * cuCimagf(b)) / denom
    );
}

__device__ cuFloatComplex calc_tn(cuFloatComplex Z_ni, cuFloatComplex Z_nn) {
    // Calculate -[Znn]^(-1)[Zni]
    cuFloatComplex result = complex_div(Z_ni, Z_nn);
    return make_cuFloatComplex(-cuCrealf(result), -cuCimagf(result));
}

__global__ void kron_reduce(GPULineParam<float>* params) {
    int i = threadIdx.x;  // Row index
    int j = threadIdx.y;  // Column index

   if (i < 3 && j < 3) {
        // Start with Zij
        cuFloatComplex Z_reduced = params->d_Z_prim[i * params->size + j];
        
        // Get neutral conductor elements
        int n = params->size - 1;
        cuFloatComplex Z_nn = params->d_Z_prim[n * params->size + n];
        cuFloatComplex Z_in = params->d_Z_prim[i * params->size + n];
        cuFloatComplex Z_nj = params->d_Z_prim[n * params->size + j];

        // Kron reduction w/ complex division
        if (cuCrealf(Z_nn) != 0.0f || cuCimagf(Z_nn) != 0.0f) {
            cuFloatComplex temp = complex_div(cuCmulf(Z_in, Z_nj), Z_nn);
            Z_reduced = cuCsubf(Z_reduced, temp);
        }

        // Calculate neutral transformation matrix [tn] = -[Znn]^(-1)[Zni]
        // Only need to do this once per row (j == 0)
        if (j == 0 && i < 3) {
            cuFloatComplex Z_ni = params->d_Z_prim[n * params->size + i];
            if (cuCrealf(Z_nn) != 0.0f || cuCimagf(Z_nn) != 0.0f) {
                params->d_t_n[i] = calc_tn(Z_ni, Z_nn);
            }
        }

        params->d_Z_abc[i * 3 + j] = Z_reduced;
    }
}
#endif