#ifndef POWER_ABC_CUH
#define POWER_ABC_CUH

#include <cuda_runtime.h>
#include <cuda/std/cmath>
#include <cuda/std/cassert>
#include <cublas_v2.h>
#include <cuComplex.h>
#include <cstdio> 
#include <iostream>

// Check for CUDA errors
#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

// Simple RAII wrapper for CUDA memory
template<typename T>
class CudaTempMemory {
public:
    CudaTempMemory(size_t size) {
        CUDA_CHECK(cudaMalloc(&ptr_, size));
    }
    ~CudaTempMemory() {
        if (ptr_) cudaFree(ptr_);
    }
    T* get() { return ptr_; }
    
private:
    T* ptr_ = nullptr;
    // Prevent copying
    CudaTempMemory(const CudaTempMemory&) = delete;
    CudaTempMemory& operator=(const CudaTempMemory&) = delete;
};


//Assuming 3 Phase System, but code has been written to handle any number of phases if PowerMatrix is changed
template<typename T>
struct MatrixMemory {
    const int PHASE = 3;
    const float PI_F = 3.14159265358979323846f;
    T* Zabc;
    T* Yabc;
    T* I_R;
    T* V_R_LN;
    T* V_R_LL;
    T* V_S_LN;
    T* V_S_LL;
    T* I_S;
    T* a;
    T* A;
    T* B;
};


template<typename T>
struct PowerMatrix {
    const int PHASE = 3;
    const float PI_F = 3.14159265358979323846f;
    T Z_abc[3 * 3];
    T Y_abc[3 * 3];
    //Resulting Matrices (c d b are generated)
    T a_R[3 * 3];
    T A_R[3 * 3];
    T B_R[3 * 3];
    //Currents for (Sent) and (Received)
    T I_R[3];
    T I_S[3];
    //Voltages for L-N and L-L
    T V_S_LN[3];
    T V_R_LN[3];
    T V_R_LL[3];
    T V_S_LL[3];
    //Voltage Unbalance Percentage
    float V_unb_perc;
    //Voltage Drop Percentage per Phase
    float phase_vdrop_perc[3];
    //Complex Power per Phase at Source
    T S_source[3];
    //Complex Power per Phase at Load (S = V * I*)
    T S_load[3];
    //Complex Power per Phase at Loss (S_loss = S_S - S_R)
    T S_loss[3];
};

// Assumes 3 Phase System
template<typename T>
PowerMatrix<T> vizy(float rated_voltage, 
                   float VA, 
                   float PF, 
                   T* Z_abc, 
                   T* Y_abc) {
    PowerMatrix<T> power;
    //Load V_LN calculation and Load Current
    const float voltage = (rated_voltage)/cuda::std::sqrtf(3.0f);
    const float current = VA / (cuda::std::sqrtf(3.0f) * rated_voltage);

    //Load PF angle and convert to radians
    const float PF_angle = -acosf(PF);
    const float deg_to_rad = power.PI_F / 180.0f;

    //Load V_LN angles
    const float v_angles[power.PHASE] = {0.0f, -120.0f, 120.0f};

    //Load V_LL angles
    const float LL_angles[power.PHASE] = {30.0f, -90.0f, 150.0f}; 

    //Load I_R angles
    const float i_angles[power.PHASE] = {
        0.0f + (PF_angle * 180.0f/power.PI_F),  
        -120.0f + (PF_angle * 180.0f/power.PI_F), 
        120.0f + (PF_angle * 180.0f/power.PI_F)
    };

    //Combined Magnitude and Angle for V_R_LN, V_R_LL, and I_R
    for(int i = 0; i < power.PHASE; i++) {
        power.V_R_LN[i] = make_cuFloatComplex(
            voltage * cosf(v_angles[i] * deg_to_rad),
            voltage * sinf(v_angles[i] * deg_to_rad)
        );

        power.V_R_LL[i] = make_cuFloatComplex(
            rated_voltage * cosf(LL_angles[i] * deg_to_rad),
            rated_voltage * sinf(LL_angles[i] * deg_to_rad)
        );

        power.I_R[i] = make_cuFloatComplex(
            current * cosf(i_angles[i] * deg_to_rad),
            current * sinf(i_angles[i] * deg_to_rad)
        );
    }

    //Copy Impedance and Admittance Matrices to PowerMatrix
    memcpy(power.Y_abc, Y_abc, power.PHASE * power.PHASE * sizeof(cuFloatComplex));
    memcpy(power.Z_abc, Z_abc, power.PHASE * power.PHASE * sizeof(cuFloatComplex));

    return power;
}

// Allocate memory function
template<typename T>
MatrixMemory<T> gpu_alloc(PowerMatrix<T> power) {
    //Memory structure and Matrix function handler (cuBLAS)
    MatrixMemory<T> gpu_m;
    //Allocate memory for matrices (Impedance and Admittance Matrices)
    CUDA_CHECK(cudaMalloc(&gpu_m.Zabc, power.PHASE * power.PHASE * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&gpu_m.Yabc, power.PHASE * power.PHASE * sizeof(T)));
    //Allocate memory for matrices (Current and Voltage Recieving)
    CUDA_CHECK(cudaMalloc(&gpu_m.I_R, power.PHASE * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&gpu_m.V_R_LN, power.PHASE * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&gpu_m.V_R_LL, power.PHASE * sizeof(T)));
    //Allocate memory for matrices (Current and Voltage Sending)
    CUDA_CHECK(cudaMalloc(&gpu_m.I_S, power.PHASE * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&gpu_m.V_S_LN, power.PHASE * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&gpu_m.V_S_LL, power.PHASE * sizeof(T)));
    //Allocate memory for matrices (Resulting Matrices)
    CUDA_CHECK(cudaMalloc(&gpu_m.a, power.PHASE * power.PHASE * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&gpu_m.A, power.PHASE * power.PHASE * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&gpu_m.B, power.PHASE * power.PHASE * sizeof(T)));
    //Copy implicitly known values to GPU memory
    CUDA_CHECK(cudaMemcpy(gpu_m.Zabc, power.Z_abc, power.PHASE * power.PHASE * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu_m.Yabc, power.Y_abc, power.PHASE * power.PHASE * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu_m.I_R, power.I_R, power.PHASE * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu_m.V_R_LN, power.V_R_LN, power.PHASE * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu_m.V_R_LL, power.V_R_LL, power.PHASE * sizeof(T), cudaMemcpyHostToDevice));
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError());
    return gpu_m;
}

// Deallocate memory function
template<typename T>
void dealloc(MatrixMemory<T> gpu_m) {
    CUDA_CHECK(cudaFree(gpu_m.Zabc));
    CUDA_CHECK(cudaFree(gpu_m.Yabc));
    CUDA_CHECK(cudaFree(gpu_m.I_R));
    CUDA_CHECK(cudaFree(gpu_m.V_R_LN));
    CUDA_CHECK(cudaFree(gpu_m.V_S_LN));
    CUDA_CHECK(cudaFree(gpu_m.a));
    CUDA_CHECK(cudaFree(gpu_m.A));
    CUDA_CHECK(cudaFree(gpu_m.B));
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError());
}

template<typename T>
T* mumuadd(cublasHandle_t handle, 
           T* var1,        // First input (receiving)
           T* var2,        // Second input (receiving)
           T* mult1,       // First matrix multiplier
           T* mult2,       // Second matrix multiplier
           T* result,      // Output (sending)
           const int PHASE) {
    // Allocate temporary arrays
    CudaTempMemory<T> temp1(PHASE * sizeof(T));
    CudaTempMemory<T> temp2(PHASE * sizeof(T));

    // Scalars for matrix operations
    const T alpha = make_cuFloatComplex(1.0f, 0.0f);
    const T beta_zero = make_cuFloatComplex(0.0f, 0.0f); 
    const T beta_one = make_cuFloatComplex(1.0f, 0.0f);  

    // First multiplication
    cublasCgemm3m(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
        PHASE, 1, PHASE, 
        &alpha,          
        mult1, 
        PHASE, 
        var1, 
        PHASE, 
        &beta_zero,      
        temp1.get(), 
        PHASE);

    // Second multiplication
    cublasCgemm3m(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
        PHASE, 1, PHASE, 
        &alpha,          
        mult2, 
        PHASE, 
        var2, 
        PHASE, 
        &beta_zero,     
        temp2.get(), 
        PHASE);

    // Add results
    cublasCgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
        PHASE, 1, 
        &beta_one,      
        temp1.get(), 
        PHASE, 
        &beta_one,      
        temp2.get(), 
        PHASE,
        result, 
        PHASE);

    return result;
}

template<typename T>
T* m_invert(cublasHandle_t handle, T* var1, T* var2, const int PHASE) {
    // Create temporary memory with automatic cleanup
    CudaTempMemory<T*> d_Aarray(sizeof(T*));
    CudaTempMemory<T*> d_Carray(sizeof(T*));
    CudaTempMemory<int> PivotArray(PHASE * sizeof(int));
    CudaTempMemory<int> infoArray(sizeof(int));

   // Copy the pointers to device
    CUDA_CHECK(cudaMemcpy(d_Aarray.get(), &var1, sizeof(T*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Carray.get(), &var2, sizeof(T*), cudaMemcpyHostToDevice));

    // LU decomposition of [a] -> [A]
    cublasCgetrfBatched(handle, 
        PHASE, 
        d_Aarray.get(),
        PHASE, 
        PivotArray.get(), 
        infoArray.get(), 
        1);
    
    // Inversion of [a] -> [A]
    cublasCgetriBatched(handle, 
        PHASE, 
        d_Aarray.get(), 
        PHASE, 
        PivotArray.get(), 
        d_Carray.get(), 
        PHASE, 
        infoArray.get(), 
        1);

    return var2;
}

// Matrix solver function, Backward Sweep
template<typename T>
PowerMatrix<T> matrix_solver(PowerMatrix<T> power) {

    /*
    Documentation regarding how some of the cuBLAS functions work:
    https://docs.nvidia.com/cuda/cublas/#cublas-t-gemm3m
    https://docs.nvidia.com/cuda/cublas/#cublas-t-getrfbatched
    https://docs.nvidia.com/cuda/cublas/#cublas-t-getribatched
    https://docs.nvidia.com/cuda/cublas/#cublas-t-geam
    */

    //Allocate memory for GPU
    MatrixMemory<cuFloatComplex> gpu_m = gpu_alloc(power);

    // Create a handle for cuBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Scalars for matrix operations
    const T alpha = make_cuFloatComplex(0.5f, 0.0f);
    const T beta = make_cuFloatComplex(1.0f, 0.0f);
    
    // Identity Matrix
    CudaTempMemory<cuFloatComplex> u(power.PHASE * power.PHASE * sizeof(cuFloatComplex));
    cuFloatComplex h_u[power.PHASE * power.PHASE] = {
        make_cuFloatComplex(1, 0),  make_cuFloatComplex(0, 0),  make_cuFloatComplex(0, 0),
        make_cuFloatComplex(0, 0),  make_cuFloatComplex(1, 0),  make_cuFloatComplex(0, 0),
        make_cuFloatComplex(0, 0),  make_cuFloatComplex(0, 0),  make_cuFloatComplex(1, 0)
    };
    CUDA_CHECK(cudaMemcpy(u.get(), h_u, power.PHASE * power.PHASE * sizeof(cuFloatComplex), cudaMemcpyHostToDevice));

    // Calculate (1/2) * Z * Y => [a]
    cublasCgemm3m(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
        gpu_m.PHASE, gpu_m.PHASE, gpu_m.PHASE, 
        &alpha, 
        gpu_m.Zabc, 
        power.PHASE, 
        gpu_m.Yabc, 
        power.PHASE, 
        &beta, 
        gpu_m.a,  
        power.PHASE);
    
    // Adding [u] to [a]
    cublasCgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
        gpu_m.PHASE, gpu_m.PHASE, 
        &beta, 
        u, 
        gpu_m.PHASE, 
        &beta, 
        gpu_m.a, 
        gpu_m.PHASE, 
        gpu_m.a,    
        power.PHASE);

    // Retrieve [a] from GPU memory
    CUDA_CHECK(cudaMemcpy(power.a_R, gpu_m.a, gpu_m.PHASE * gpu_m.PHASE * sizeof(T), cudaMemcpyDeviceToHost));
   
    // Invert [a] -> [A]
    gpu_m.A = m_invert(handle, gpu_m.a, gpu_m.A, gpu_m.PHASE);

    // Retrieve [A] from GPU memory
    CUDA_CHECK(cudaMemcpy(power.A_R, gpu_m.A, gpu_m.PHASE * gpu_m.PHASE * sizeof(T), cudaMemcpyDeviceToHost));

    // Calculate [B]
    cublasCgemm3m(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
        gpu_m.PHASE, gpu_m.PHASE, gpu_m.PHASE, 
        &beta, 
        gpu_m.A, 
        power.PHASE, 
        gpu_m.Zabc, 
        power.PHASE, 
        &beta, 
        gpu_m.B,  
        power.PHASE);
    
    // Retrieve [B] from GPU memory
    CUDA_CHECK(cudaMemcpy(power.B_R, gpu_m.B, gpu_m.PHASE * gpu_m.PHASE * sizeof(T), cudaMemcpyDeviceToHost));

    // Calculate [V_S_LN]
    gpu_m.V_S_LN = mumuadd(handle, gpu_m.V_R_LN, gpu_m.I_R, gpu_m.a, gpu_m.Zabc, gpu_m.V_S_LN, gpu_m.PHASE);
    
    // Retrieve [V_S_LN] from GPU memory
    CUDA_CHECK(cudaMemcpy(power.V_S_LN, gpu_m.V_S_LN, gpu_m.PHASE * sizeof(T), cudaMemcpyDeviceToHost));

    // Calculate [I_S]
    gpu_m.I_S = mumuadd(handle, gpu_m.V_R_LN, gpu_m.I_R, gpu_m.Yabc, gpu_m.a, gpu_m.I_S, gpu_m.PHASE);

    // Retrieve [I_S] from GPU memory
    CUDA_CHECK(cudaMemcpy(power.I_S, gpu_m.I_S, gpu_m.PHASE * sizeof(T), cudaMemcpyDeviceToHost));

    //Calculate [V_S_LL]
    gpu_m.V_S_LL = mumuadd(handle, gpu_m.V_R_LL, gpu_m.I_R, gpu_m.B, gpu_m.Zabc, gpu_m.V_S_LL, gpu_m.PHASE);    

    // Retrieve [V_S_LL] from GPU memory
    CUDA_CHECK(cudaMemcpy(power.V_S_LL, gpu_m.V_S_LL, gpu_m.PHASE * sizeof(T), cudaMemcpyDeviceToHost));

    // CPU Power Calculations
    // Calculate the average voltage
    float voltage_average = 0;
    for (int i = 0; i < gpu_m.PHASE; i++) {
        voltage_average += cuCabsf(power.V_S_LN[i]);
    }
    voltage_average /= gpu_m.PHASE;
    
    // Percentage of voltage unbalance
    power.V_unb_perc = (((cuCabsf(power.V_S_LN[0])-voltage_average)/voltage_average))*100;

    // Calculate the voltage drop percentage per phase
    for (int i = 0; i < gpu_m.PHASE; i++) {
        power.phase_vdrop_perc[i] = (((cuCabsf(power.V_S_LN[i])/cuCabsf(power.V_R_LN[i]))-1)*100);
    }

    // Calculate complex power per phase at source (S = V * I*)
    for(int i = 0; i < power.PHASE; i++) {
        // Complex conjugate of I_S
        T I_S_conj = make_cuFloatComplex(
            cuCrealf(power.I_S[i]), 
            -cuCimagf(power.I_S[i])
        );
        
        // Complex power = V_S_LN * conj(I_S)
        power.S_source[i] = cuCmulf(power.V_S_LN[i], I_S_conj);
    }

    // Calculate complex power per phase at receiving end (load)
    for(int i = 0; i < power.PHASE; i++) {
        // Complex conjugate of I_R
        T I_R_conj = make_cuFloatComplex(
            cuCrealf(power.I_R[i]), 
            -cuCimagf(power.I_R[i])
        );
        
        // Complex power = V_R_LN * conj(I_R)
        power.S_load[i] = cuCmulf(power.V_R_LN[i], I_R_conj);
    }

    // Calculate power loss per phase (S_loss = S_S - S_R) else to inspect (or some further explanations
    for(int i = 0; i < power.PHASE; i++) {
        power.S_loss[i] = cuCsubf(power.S_source[i], power.S_load[i]);
    }

    //Sync Errors and Destroy cuBLAS
    cublasDestroy(handle);

    // Free device memory/synchronize
    dealloc(gpu_m);
    return power;
}

#endif