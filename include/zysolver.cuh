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
    T* diam; //In inches
};

template<typename T>
struct GPULineParam {
    int size;
    T* d_res;
    T* d_gmr;
    T* d_x;
    T* d_y;
    T* d_rd; //radius in feet of conductor
    cuDoubleComplex* d_Z_prim;
    cuDoubleComplex* d_P_prim;
    cuDoubleComplex* d_P_abc;
    cuDoubleComplex* d_C_abc;
    cuDoubleComplex* d_Y_abc;
    cuDoubleComplex* d_Z_abc;
    cuDoubleComplex* d_t_n;
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
            
            CUDA_CHECK(cudaMalloc(&d_Z_prim, size * size * sizeof(cuDoubleComplex)));
            CUDA_CHECK(cudaMalloc(&d_Z_abc, 3 * 3 * sizeof(cuDoubleComplex)));
            CUDA_CHECK(cudaMalloc(&d_t_n, 3 * sizeof(cuDoubleComplex)));
            CUDA_CHECK(cudaMalloc(&d_P_prim, size * size * sizeof(cuDoubleComplex)));
            CUDA_CHECK(cudaMalloc(&d_C_abc, 3 * 3 * sizeof(cuDoubleComplex)));
            CUDA_CHECK(cudaMalloc(&d_Y_abc, 3 * 3 * sizeof(cuDoubleComplex)));
            CUDA_CHECK(cudaMalloc(&d_rd, size * sizeof(T)));
            T* rd = new T[size];
            for(int i = 0; i < size; i++) {
                rd[i] = (0.5 * params.diam[i]) / 12.0;
            }
            CUDA_CHECK(cudaMemcpy(d_rd, rd, size * sizeof(T), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMalloc(&d_P_abc, 3 * 3 * sizeof(cuDoubleComplex)));
            delete[] rd;
        }

        // Copy this struct to device
        CUDA_CHECK(cudaMemcpy(d_this, this, sizeof(GPULineParam), cudaMemcpyHostToDevice));
        cudaDeviceSynchronize();
    }

    // Copy results back to host
    void copyToHost(PowerLineMatrix<cuDoubleComplex>& power) {
        CUDA_CHECK(cudaMemcpy(power.Z_abc, d_Z_abc, 9 * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(power.t_n, d_t_n, 3 * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(power.Y_abc, d_Y_abc, 9 * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
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
        if (d_P_prim) cudaFree(d_P_prim);
        if (d_C_abc) cudaFree(d_C_abc);
        if (d_Y_abc) cudaFree(d_Y_abc);
        if (d_P_abc) cudaFree(d_P_abc);
        if (d_rd) cudaFree(d_rd);
        cudaDeviceSynchronize();
    }
};

struct tuple_return {
    cuDoubleComplex r1;
    cuDoubleComplex r2;
};

template<typename T>
__device__ tuple_return self_impedance(GPULineParam<T>* params, int i) {
    T resistance = params->d_res[i];
    T Sij = 2.0f * params->d_y[i]; 
    T Dij = params->d_gmr[i];
    T rd = params->d_rd[i];
    return tuple_return {
        .r1 = make_cuDoubleComplex(
            resistance + 0.09530, 
            0.12134 * (logf(1.0f / Dij) + 7.93402f)
        ),
        .r2 = make_cuDoubleComplex(
            11.17689e6 * logf(Sij/rd),
            0.0
        )
    };
}

template<typename T>
__device__ tuple_return mutual_impedance(GPULineParam<T>* params, int i, int j) {
    double Dij = sqrtf(
        (params->d_x[j] - params->d_x[i]) * (params->d_x[j] - params->d_x[i]) + 
        (params->d_y[j] - params->d_y[i]) * (params->d_y[j] - params->d_y[i]));
    float Sij = sqrtf(
        (params->d_x[j] - params->d_x[i]) * (params->d_x[j] - params->d_x[i]) + 
        (params->d_y[i] + params->d_y[j]) * (params->d_y[i] + params->d_y[j]));
    return tuple_return {
        .r1 = make_cuDoubleComplex(
            0.09530, 
            0.12134 * (logf(1.0f / Dij) + 7.93402f)
        ),
        .r2 = make_cuDoubleComplex(
            11.17689e6 * logf(Sij/Dij),
            0.0
        )  
    };
}

template<typename T>
__global__ void calc_prim(GPULineParam<T>* params) {
    int i = threadIdx.x;  // Row index
    int j = threadIdx.y;  // Column index
    __shared__ tuple_return result;
    if (i < params->size && j < params->size) {
        if (i == j) {
            // Self-impedance
            result = self_impedance(params, i);
            params->d_Z_prim[i * params->size + j] = result.r1;
            params->d_P_prim[i * params->size + j] = result.r2;
        } else {
            // Mutual impedance
            tuple_return mutual_imp = mutual_impedance(params, i, j);
            params->d_Z_prim[i * params->size + j] = mutual_imp.r1;
            params->d_P_prim[i * params->size + j] = mutual_imp.r2;
        }
    }
}

template<typename T>
__device__ T complex_div(T a, T b) {
    /*
        Complex Division (real(a), imag(a)) / (real(b), imag(b) = 
        (real(a) * real(b) + imag(a) * imag(b)) / (real(b)^2 + imag(b)^2),
        (imag(a) * real(b) - real(a) * imag(b)) / (real(b)^2 + imag(b)^2)
    */ 
    double denom = cuCreal(b) * cuCreal(b) + cuCimag(b) * cuCimag(b);
    return make_cuDoubleComplex(
        (cuCreal(a) * cuCreal(b) + cuCimag(a) * cuCimag(b)) / denom,
        (cuCimag(a) * cuCreal(b) - cuCreal(a) * cuCimag(b)) / denom
    );
}

template<typename T>
__device__ T calc_tn(T Z_ni, T Z_nn) {
    // Calculate -[Znn]^(-1)[Zni]
    T result = complex_div(Z_ni, Z_nn);
    return make_cuDoubleComplex(-cuCreal(result), -cuCimag(result));
}

template<typename T>
__device__ void run_kr_Z(T* Z_prim, T* Z_abc, T* t_n, int size, int i, int j) {
    // Start with element Zij
    T Z_reduced = Z_prim[i * size + j];
    
    // Get neutral conductor elements
    int n = size - 1;
    T Z_nn = Z_prim[n * size + n];
    T Z_in = Z_prim[i * size + n];
    T Z_nj = Z_prim[n * size + j];

    // Kron reduction formula: [Z_abc] = [Z_ij] - [Z_in][Z_nn]^(-1)[Z_nj]
    if (cuCreal(Z_nn) != 0.0f || cuCimag(Z_nn) != 0.0f) {
        T temp = complex_div(cuCmul(Z_in, Z_nj), Z_nn);
        Z_reduced = cuCsub(Z_reduced, temp);
    }

    // Store result
    Z_abc[i * 3 + j] = Z_reduced;

    // Calculate neutral transformation matrix [tn] = -[Znn]^(-1)[Zni]
    // Only need to do this once per row (j == 0)
    if (j == 0 && i < 3) {
        T Z_ni = Z_prim[n * size + i];
        t_n[i] = calc_tn(Z_ni, Z_nn);
    }    
}

__device__ void run_kr_P(cuDoubleComplex* P_prim, cuDoubleComplex* P_abc, int size, int i, int j) {
    // Get the base element Pij
    cuDoubleComplex P_ij = P_prim[i * size + j];
    
    // Get elements for Kron reduction
    int n = size - 1;  // index of neutral conductor (last row/column)
    cuDoubleComplex P_i4 = P_prim[i * size + n];  // Pi4 element
    cuDoubleComplex P_4j = P_prim[n * size + j];  // P4j element
    cuDoubleComplex P_44 = P_prim[n * size + n];  // P44 element (neutral-neutral term)

    // Perform Kron reduction: Pij = P̂ij - (P̂i4 * P̂4j) / P̂44
    cuDoubleComplex numerator = cuCmul(P_i4, P_4j);
    P_abc[i * 3 + j] = cuCsub(P_ij, cuCdiv(numerator, P_44));
}

template<typename T>
__global__ void kron_reduce(GPULineParam<T>* params) {
    int i = threadIdx.x;  // Row index
    int j = threadIdx.y;  // Column index
    if (i < 3 && j < 3) {
        run_kr_Z(params->d_Z_prim, params->d_Z_abc, params->d_t_n, params->size, i, j);
        run_kr_P(params->d_P_prim, params->d_P_abc, params->size, i, j);
    }
}

template<typename T>
void mvert(GPULineParam<T>* params) {
    // Setup arrays for batch inversion
    cuDoubleComplex** d_Parray;
    cuDoubleComplex** d_Carray;
    int* d_pivot;
    int* d_info;

    cublasHandle_t handle;
    cublasCreate(&handle);
    
    cudaMalloc(&d_Parray, sizeof(cuDoubleComplex*));
    cudaMalloc(&d_Carray, sizeof(cuDoubleComplex*));
    cudaMalloc(&d_pivot, 3 * sizeof(int));
    cudaMalloc(&d_info, sizeof(int));

    // Set up the arrays
    cudaMemcpy(d_Parray, &params->d_P_abc, sizeof(cuDoubleComplex*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Carray, &params->d_C_abc, sizeof(cuDoubleComplex*), cudaMemcpyHostToDevice);

    // Perform matrix inversion P_abc -> C_abc
    cublasZgetrfBatched(handle, 3, d_Parray, 3, d_pivot, d_info, 1);
    cublasZgetriBatched(handle, 3, d_Parray, 3, d_pivot, d_Carray, 3, d_info, 1);

    // Cleanup
    cudaFree(d_Parray);
    cudaFree(d_Carray);
    cudaFree(d_pivot);
    cudaFree(d_info);
    cublasDestroy(handle);
}

template<typename T>
__global__ void calc_Y(GPULineParam<T>* params) {
    int i = threadIdx.x;  // Row index
    int j = threadIdx.y;  // Column index
    if (i < 3 && j < 3) {
        params->d_Y_abc[i * 3 + j] = make_cuDoubleComplex(
            -cuCimag(params->d_C_abc[i * 3 + j]) * 376.9911,  
            cuCreal(params->d_C_abc[i * 3 + j]) * 376.9911    
        );
    }
}

#endif