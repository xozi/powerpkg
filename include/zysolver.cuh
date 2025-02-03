#ifndef ZYSOLVER_CUH
#define ZYSOLVER_CUH

#include <cuda_runtime.h>
#include <cuComplex.h>
#include "power_abc.cuh"

//T assumed double.
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
    int index;
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
    cuDoubleComplex** d_Parray;
    cuDoubleComplex** d_Carray;
    int* d_pivot;
    int* d_info;

    // Constructor - allocates and copies from host LineParam
    GPULineParam(const LineParam<T>& params) {
        size = params.conductors;
        index = size-1;
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
            
            //Prime arrays
            CUDA_CHECK(cudaMalloc(&d_Z_prim, size * size * sizeof(cuDoubleComplex)));
            CUDA_CHECK(cudaMalloc(&d_P_prim, size * size * sizeof(cuDoubleComplex)))

            //Kron reduction arrays
            CUDA_CHECK(cudaMalloc(&d_Z_abc, index * index * sizeof(cuDoubleComplex)));
            CUDA_CHECK(cudaMalloc(&d_t_n, index * sizeof(cuDoubleComplex)));
            CUDA_CHECK(cudaMalloc(&d_P_abc, index * index * sizeof(cuDoubleComplex)));
            CUDA_CHECK(cudaMalloc(&d_C_abc, index * index * sizeof(cuDoubleComplex)));
            CUDA_CHECK(cudaMalloc(&d_Y_abc, index * index * sizeof(cuDoubleComplex)));
            CUDA_CHECK(cudaMalloc(&d_rd, size * sizeof(T)));

            T* rd = new T[size];
            for(int i = 0; i < size; i++) {
                rd[i] = (0.5 * params.diam[i]) / 12.0;
            }
            CUDA_CHECK(cudaMemcpy(d_rd, rd, size * sizeof(T), cudaMemcpyHostToDevice));

            delete[] rd;

            // Allocate arrays for batch inversion
            CUDA_CHECK(cudaMalloc(&d_Parray, sizeof(cuDoubleComplex*)));
            CUDA_CHECK(cudaMalloc(&d_Carray, sizeof(cuDoubleComplex*)));
            CUDA_CHECK(cudaMalloc(&d_pivot, index * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));

            // Set up the arrays with proper host-to-device copy
            cuDoubleComplex* h_ptrs[1] = {d_P_abc};
            CUDA_CHECK(cudaMemcpy(d_Parray, &h_ptrs[0], sizeof(cuDoubleComplex*), cudaMemcpyHostToDevice));
            
            h_ptrs[0] = d_C_abc;
            CUDA_CHECK(cudaMemcpy(d_Carray, &h_ptrs[0], sizeof(cuDoubleComplex*), cudaMemcpyHostToDevice));
        }

        // Copy this struct to device
        CUDA_CHECK(cudaMemcpy(d_this, this, sizeof(GPULineParam), cudaMemcpyHostToDevice));
        cudaDeviceSynchronize();
    }

    // Copy results back to host
    void copyToHost(PowerLineMatrix<cuDoubleComplex>& power) {
        
        CUDA_CHECK(cudaMemcpy(power.Z_abc, d_Z_abc, index * index * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(power.t_n, d_t_n, index * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(power.Y_abc, d_Y_abc, index * index * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
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
        if (d_Parray) cudaFree(d_Parray);
        if (d_Carray) cudaFree(d_Carray);
        if (d_pivot) cudaFree(d_pivot);
        if (d_info) cudaFree(d_info);
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
    T Sij = 2.0 * params->d_y[i]; 
    T Dij = params->d_gmr[i];
    T rd = params->d_rd[i];
    return tuple_return {
        .r1 = make_cuDoubleComplex(
            resistance + 0.09530, 
            0.12134 * (log(1.0 / Dij) + 7.93402)
        ),
        .r2 = make_cuDoubleComplex(
            11.17689 * log(Sij/rd),
            0.0
        )
    };
}

template<typename T>
__device__ tuple_return mutual_impedance(GPULineParam<T>* params, int i, int j) {
    T Dij = sqrt(
        (params->d_x[j] - params->d_x[i]) * (params->d_x[j] - params->d_x[i]) + 
        (params->d_y[j] - params->d_y[i]) * (params->d_y[j] - params->d_y[i]));
    T Sij = sqrt(
        (params->d_x[j] - params->d_x[i]) * (params->d_x[j] - params->d_x[i]) + 
        (params->d_y[i] + params->d_y[j]) * (params->d_y[i] + params->d_y[j]));
    return tuple_return {
        .r1 = make_cuDoubleComplex(
            0.09530, 
            0.12134 * (log(1.0 / Dij) + 7.93402)
        ),
        .r2 = make_cuDoubleComplex(
            11.17689* log(Sij/Dij),
            0.0
        )  
    };
}

template<typename T>
__global__ void calc_prim(GPULineParam<T>* params) {
    int i = threadIdx.x; 
    int j = threadIdx.y;  
    if (i < params->size && j < params->size) {
        if (i == j) {
            tuple_return result = self_impedance(params, i);
            params->d_Z_prim[i * params->size + j] = result.r1;
            params->d_P_prim[i * params->size + j] = result.r2;
        } else {
            tuple_return mutual_imp = mutual_impedance(params, i, j);
            params->d_Z_prim[i * params->size + j] = mutual_imp.r1;
            params->d_P_prim[i * params->size + j] = mutual_imp.r2;
        }
    }
}

__device__ void run_kr(cuDoubleComplex* A_prim, cuDoubleComplex* A_abc, int size, int index, int i, int j) {
    // Start with element A
    cuDoubleComplex A_reduced = A_prim[i * size + j];
    
    // Get neutral conductor elements
    cuDoubleComplex A_nn = A_prim[index * size + index];
    cuDoubleComplex A_in = A_prim[i * size + index];
    cuDoubleComplex A_nj = A_prim[index * size + j];

    // Kron reduction formula: [A_abc] = [A_ij] - [A_in][A_nn]^(-1)[A_nj]
    if (cuCreal(A_nn) != 0.0 || cuCimag(A_nn) != 0.0) {
        cuDoubleComplex temp = cuCdiv(cuCmul(A_in, A_nj), A_nn);
        A_reduced = cuCsub(A_reduced, temp);
    }

    // Store result
    A_abc[i * index + j] = A_reduced;
}

template<typename T>
__global__ void kron_reduce(GPULineParam<T>* params) {
    int i = threadIdx.x;  
    int j = threadIdx.y; 
    if (i < params->size && j < params->size) {
        run_kr(params->d_Z_prim, params->d_Z_abc, params->size, params->index, i, j);
        run_kr(params->d_P_prim, params->d_P_abc, params->size, params->index, i, j);
        if (j == 0 && i < params->index) {
            cuDoubleComplex Z_ni = params->d_Z_prim[params->index * params->size + i];
            cuDoubleComplex Z_nn = params->d_Z_prim[params->index * params->size + params->index];
            params->d_t_n[i] = make_cuDoubleComplex(
                -cuCreal(cuCdiv(Z_ni, Z_nn)), 
                -cuCimag(cuCdiv(Z_ni, Z_nn))
            );
        }    
    }
}

template<typename T>
void mvert(GPULineParam<T>& params) {
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Perform matrix inversion P_abc -> C_abc
    cublasZgetrfBatched(handle, params.index,  params.d_Parray, params.index, params.d_pivot, params.d_info, 1);
    cublasZgetriBatched(handle, params.index,  params.d_Parray, params.index, params.d_pivot, params.d_Carray, params.index, params.d_info, 1);

    cublasDestroy(handle);
}

template<typename T>
__global__ void calc_Y(GPULineParam<T>* params) {
    int i = threadIdx.x;  
    int j = threadIdx.y;  
    int w = 2.0 * M_PI * 60.0;
    if (i < params->index && j < params->index) {
        params->d_Y_abc[i * params->index + j] = make_cuDoubleComplex(
            -cuCimag(params->d_C_abc[i * params->index + j]) * w,  
            cuCreal(params->d_C_abc[i * params->index + j]) * w    
        );
    }
}

#endif