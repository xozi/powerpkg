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

template<typename T>
struct PowerLineMatrix {
    int phase;
    T* u;
    T* Z_abc;
    T* Y_abc;
    //Resulting Matrices (c d b are generated)
    T* a;
    T* A;
    T* B;
    //Currents for (Sent) and (Received)
    T* I_R;
    T* I_S;
    //Voltages for L-N and L-L
    T* V_S_LN;
    T* V_R_LN;
    T* V_R_LL;
    T* V_S_LL;
    //Voltage Unbalance Percentage
    float V_unb_perc;
    //Voltage Drop Percentage per Phase
    float* phase_vdrop_perc;
    //Complex Power per Phase at Source
    T* S_source;
    //Complex Power per Phase at Load (S = V * I*)
    T* S_load;
    //Complex Power per Phase at Loss (S_loss = S_S - S_R)
    T* S_loss;
    //Neutral Transformation Matrix
    T* t_n;

    // Constructor to initialize everything to zero
    PowerLineMatrix() : phase(0) {
        allocateArrays();
    }

    // Constructor with phase parameter
    explicit PowerLineMatrix(int p) : phase(p) {
        allocateArrays();
    }

    private:
    void allocateArrays() {
        // Allocate memory for arrays
        u = new T[phase * phase];
        Z_abc = new T[phase * phase];
        Y_abc = new T[phase * phase];
        a = new T[phase * phase];
        A = new T[phase * phase];
        B = new T[phase * phase];
        I_R = new T[phase];
        I_S = new T[phase];
        V_S_LN = new T[phase];
        V_R_LN = new T[phase];
        V_R_LL = new T[phase];
        V_S_LL = new T[phase];
        phase_vdrop_perc = new float[phase];
        S_source = new T[phase];
        S_load = new T[phase];
        S_loss = new T[phase];
        t_n = new T[phase];

        // Initialize complex arrays to zero
        cuDoubleComplex zero_complex = make_cuDoubleComplex(0.0f, 0.0f);
        
        // Initialize phase x phase matrices
        for(int i = 0; i < phase * phase; i++) {
            u[i] = zero_complex;
            Z_abc[i] = zero_complex;
            Y_abc[i] = zero_complex;
            a[i] = zero_complex;
            A[i] = zero_complex;
            B[i] = zero_complex;
        }
        
        // Initialize phase-element arrays
        for(int i = 0; i < phase; i++) {
            I_R[i] = zero_complex;
            I_S[i] = zero_complex;
            V_S_LN[i] = zero_complex;
            V_R_LN[i] = zero_complex;
            V_R_LL[i] = zero_complex;
            V_S_LL[i] = zero_complex;
            S_source[i] = zero_complex;
            S_load[i] = zero_complex;
            S_loss[i] = zero_complex;
            t_n[i] = zero_complex;
            phase_vdrop_perc[i] = 0.0f;
        }
        
        // Initialize float values
        V_unb_perc = 0.0f;
    }
};

template<typename T>
struct GPUPowerLineMatrix {
    int phase;
    const T alpha_one = make_cuDoubleComplex(1.0f, 0.0f);
    const T alpha_half = make_cuDoubleComplex(0.5f, 0.0f);
    const T beta_zero = make_cuDoubleComplex(0.0f, 0.0f);
    const T beta_one = make_cuDoubleComplex(1.0f, 0.0f);
    //Pointers to GPU Memory
    T* d_u;
    T* d_Z_abc;
    T* d_Y_abc;
    T* d_a;
    T* d_A;
    T* d_B;
    T* d_I_R;
    T* d_I_S;
    T* d_V_S_LN;
    T* d_V_R_LN;
    T* d_V_R_LL;
    T* d_V_S_LL;
    float* d_voltage_average;
    float* d_V_unb_perc;
    float* d_phase_vdrop_perc;
    T* d_S_source;
    T* d_S_load;
    T* d_S_loss;
    GPUPowerLineMatrix* d_this;
    T** d_Aarray;
    T** d_Carray;
    int* d_pivot;
    int* d_info;
    T* d_temp1;
    T* d_temp2;

    // Constructor - allocates and copies from host PowerLineMatrix
    GPUPowerLineMatrix(const PowerLineMatrix<T>& power) {
        // Allocate device pointer for this struct
        CUDA_CHECK(cudaMalloc(&d_this, sizeof(GPUPowerLineMatrix)));

        // Allocate and copy 3x3 matrices
        CUDA_CHECK(cudaMalloc(&d_u, power.phase * power.phase * sizeof(T)));
        CUDA_CHECK(cudaMemcpy(d_u, power.u, power.phase * power.phase * sizeof(T), cudaMemcpyHostToDevice));
        
        CUDA_CHECK(cudaMalloc(&d_Z_abc, power.phase * power.phase * sizeof(T)));
        CUDA_CHECK(cudaMemcpy(d_Z_abc, power.Z_abc, power.phase * power.phase * sizeof(T), cudaMemcpyHostToDevice));
        
        CUDA_CHECK(cudaMalloc(&d_Y_abc, power.phase * power.phase * sizeof(T)));
        CUDA_CHECK(cudaMemcpy(d_Y_abc, power.Y_abc, power.phase * power.phase * sizeof(T), cudaMemcpyHostToDevice));
        
        CUDA_CHECK(cudaMalloc(&d_a, power.phase * power.phase * sizeof(T)));
        CUDA_CHECK(cudaMalloc(&d_A, power.phase * power.phase * sizeof(T)));
        CUDA_CHECK(cudaMalloc(&d_B, power.phase * power.phase * sizeof(T)));

        // Allocate and copy 3-element arrays
        CUDA_CHECK(cudaMalloc(&d_I_R, power.phase * sizeof(T)));
        CUDA_CHECK(cudaMemcpy(d_I_R, power.I_R, power.phase * sizeof(T), cudaMemcpyHostToDevice));
        
        CUDA_CHECK(cudaMalloc(&d_I_S, power.phase * sizeof(T)));
        CUDA_CHECK(cudaMalloc(&d_V_S_LN, power.phase * sizeof(T)));
        CUDA_CHECK(cudaMalloc(&d_V_R_LN, power.phase * sizeof(T)));
        CUDA_CHECK(cudaMemcpy(d_V_R_LN, power.V_R_LN, power.phase * sizeof(T), cudaMemcpyHostToDevice));
        
        CUDA_CHECK(cudaMalloc(&d_V_R_LL, power.phase * sizeof(T)));
        CUDA_CHECK(cudaMemcpy(d_V_R_LL, power.V_R_LL, power.phase * sizeof(T), cudaMemcpyHostToDevice));
        
        CUDA_CHECK(cudaMalloc(&d_V_S_LL, power.phase * sizeof(T)));
        
        // Allocate float arrays
        CUDA_CHECK(cudaMalloc(&d_voltage_average, sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_V_unb_perc, sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_phase_vdrop_perc, power.phase * sizeof(float)));
        
        // Allocate complex power arrays
        CUDA_CHECK(cudaMalloc(&d_S_source, power.phase * sizeof(T)));
        CUDA_CHECK(cudaMalloc(&d_S_load, power.phase * sizeof(T)));
        CUDA_CHECK(cudaMalloc(&d_S_loss, power.phase * sizeof(T)));

        // Allocate temporary storage
        CUDA_CHECK(cudaMalloc(&d_Aarray, sizeof(T*)));
        CUDA_CHECK(cudaMalloc(&d_Carray, sizeof(T*)));
        CUDA_CHECK(cudaMalloc(&d_pivot, phase * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_temp1, phase * sizeof(T)));
        CUDA_CHECK(cudaMalloc(&d_temp2, phase * sizeof(T)));

        // Set up the arrays with proper host-to-device copy
        T* h_ptrs[1] = {d_a};
        CUDA_CHECK(cudaMemcpy(d_Aarray, &h_ptrs[0], sizeof(T*), cudaMemcpyHostToDevice));
        
        h_ptrs[0] = d_A;
        CUDA_CHECK(cudaMemcpy(d_Carray, &h_ptrs[0], sizeof(T*), cudaMemcpyHostToDevice));

        // Copy this struct to device
        CUDA_CHECK(cudaMemcpy(d_this, this, sizeof(GPUPowerLineMatrix), cudaMemcpyHostToDevice));
        cudaDeviceSynchronize();
    }

    // Copy results back to host
    void copyToHost(PowerLineMatrix<T>& power) {
        CUDA_CHECK(cudaMemcpy(power.a, d_a, power.phase * power.phase * sizeof(T), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(power.A, d_A, power.phase * power.phase * sizeof(T), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(power.B, d_B, power.phase * power.phase * sizeof(T), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(power.I_S, d_I_S, power.phase * sizeof(T), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(power.V_S_LN, d_V_S_LN, power.phase * sizeof(T), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(power.V_S_LL, d_V_S_LL, power.phase * sizeof(T), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&power.V_unb_perc, d_V_unb_perc, sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(power.phase_vdrop_perc, d_phase_vdrop_perc, power.phase * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(power.S_source, d_S_source, power.phase * sizeof(T), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(power.S_load, d_S_load, power.phase * sizeof(T), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(power.S_loss, d_S_loss, power.phase * sizeof(T), cudaMemcpyDeviceToHost));
        cudaDeviceSynchronize();
    }

    // Destructor - handles cleanup
    ~GPUPowerLineMatrix() {
        if (d_u) cudaFree(d_u);
        if (d_Z_abc) cudaFree(d_Z_abc);
        if (d_Y_abc) cudaFree(d_Y_abc);
        if (d_a) cudaFree(d_a);
        if (d_A) cudaFree(d_A);
        if (d_B) cudaFree(d_B);
        if (d_I_R) cudaFree(d_I_R);
        if (d_I_S) cudaFree(d_I_S);
        if (d_V_S_LN) cudaFree(d_V_S_LN);
        if (d_V_R_LN) cudaFree(d_V_R_LN);
        if (d_V_R_LL) cudaFree(d_V_R_LL);
        if (d_V_S_LL) cudaFree(d_V_S_LL);
        if (d_voltage_average) cudaFree(d_voltage_average);
        if (d_V_unb_perc) cudaFree(d_V_unb_perc);
        if (d_phase_vdrop_perc) cudaFree(d_phase_vdrop_perc);
        if (d_S_source) cudaFree(d_S_source);
        if (d_S_load) cudaFree(d_S_load);
        if (d_S_loss) cudaFree(d_S_loss);
        if (d_Aarray) cudaFree(d_Aarray);
        if (d_Carray) cudaFree(d_Carray);
        if (d_pivot) cudaFree(d_pivot);
        if (d_info) cudaFree(d_info);
        if (d_temp1) cudaFree(d_temp1);
        if (d_temp2) cudaFree(d_temp2);
        if (d_this) cudaFree(d_this);
        cudaDeviceSynchronize();
    }
};

// Assumes 3 Phase System
template<typename T>
PowerLineMatrix<T> vizy(float rated_voltage, 
                   double VA, 
                   double PF, 
                   T* Z_abc, 
                   T* Y_abc,
                   PowerLineMatrix<T>& power) {
    //Load V_LN calculation and Load Current
    const double voltage = (rated_voltage)/cuda::std::sqrtf(3.0);
    const double current = VA / (cuda::std::sqrtf(3.0) * rated_voltage);

    //Load PF angle and convert to radians
    const double PF_angle = -acos(PF);
    const double deg_to_rad = M_PI / 180.0;

    //Load V_LN angles
    const double v_angles[power.phase] = {
        0.0,                    // Phase A always present
        (power.phase >= 2) ? -120.0 : 0.0,  // Phase B if 2 or 3 phase
        (power.phase == 3) ? 120.0 : 0.0    // Phase C only if 3 phase
    };

    //Load V_LL angles
    const double LL_angles[power.phase] = {
        30.0,                   // Phase A always present
        (power.phase >= 2) ? -90.0 : 0.0,   // Phase B if 2 or 3 phase
        (power.phase == 3) ? 150.0 : 0.0    // Phase C only if 3 phase
    };

    //Load I_R angles
    const double i_angles[power.phase] = {
        0.0 + (PF_angle * 180.0/M_PI),    // Phase A always present
        (power.phase >= 2) ? -120.0 + (PF_angle * 180.0/M_PI) : 0.0,  // Phase B if 2 or 3 phase
        (power.phase == 3) ? 120.0 + (PF_angle * 180.0/M_PI) : 0.0    // Phase C only if 3 phase
    };

    //Combined Magnitude and Angle for V_R_LN, V_R_LL, and I_R
    for(int i = 0; i < power.phase; i++) {
        power.V_R_LN[i] = make_cuDoubleComplex(
            voltage * cos(v_angles[i] * deg_to_rad),
            voltage * sin(v_angles[i] * deg_to_rad)
        );

        power.V_R_LL[i] = make_cuDoubleComplex(
            rated_voltage * cos(LL_angles[i] * deg_to_rad),
            rated_voltage * sin(LL_angles[i] * deg_to_rad)
        );

        power.I_R[i] = make_cuDoubleComplex(
            current * cos(i_angles[i] * deg_to_rad),
            current * sin(i_angles[i] * deg_to_rad)
        );
    }

    // Create dynamic unity matrix for any phase size
    for(int i = 0; i < power.phase; i++) {
        for(int j = 0; j < power.phase; j++) {
            // Set diagonal elements to 1, others to 0
            if(i == j) {
                power.u[i * power.phase + j] = make_cuDoubleComplex(1.0, 0.0);
            } else {
                power.u[i * power.phase + j] = make_cuDoubleComplex(0.0, 0.0);
            }
        }
    }
    //Copy Impedance and Admittance Matrices to PowerLineMatrix
    memcpy(power.Y_abc, Y_abc, power.phase * power.phase * sizeof(cuDoubleComplex));
    memcpy(power.Z_abc, Z_abc, power.phase * power.phase * sizeof(cuDoubleComplex));

    return power;
}

__global__ void voltage_metrics(GPUPowerLineMatrix<cuDoubleComplex>* d_m) {
    __threadfence();
    // Calculate voltage average
    if (threadIdx.x == 0) {
        float sum = 0.0f;
        for (int i = 0; i < d_m->phase; i++) {
            sum += cuCabs(d_m->d_V_S_LN[i]);
        }
        *d_m->d_voltage_average = sum / d_m->phase;
        *d_m->d_V_unb_perc = ((cuCabs(d_m->d_V_S_LN[0]) - *d_m->d_voltage_average) / *d_m->d_voltage_average) * 100.0f;
    }

    // Parallel phase calculations
    if (threadIdx.x > 0 && threadIdx.x < (d_m->phase + 1)) {
        d_m->d_phase_vdrop_perc[(threadIdx.x-1)] = 
            ((cuCabs(d_m->d_V_S_LN[(threadIdx.x-1)]) / cuCabs(d_m->d_V_R_LN[(threadIdx.x-1)])) - 1.0f) * 100.0f;
    }
    __threadfence();
}

__global__ void power_loss(GPUPowerLineMatrix<cuDoubleComplex>* d_m) {
    __threadfence();
    // Parallel phase calculations
    if (threadIdx.x < d_m->phase) {
        // Calculate source power
        cuDoubleComplex I_S_conj = make_cuDoubleComplex(
            cuCreal(d_m->d_I_S[threadIdx.x]),
            -cuCimag(d_m->d_I_S[threadIdx.x])
        );
        d_m->d_S_source[threadIdx.x] = cuCmul(d_m->d_V_S_LN[threadIdx.x], I_S_conj);
        
        // Calculate load power
        cuDoubleComplex I_R_conj = make_cuDoubleComplex(
            cuCreal(d_m->d_I_R[threadIdx.x]),
            -cuCimag(d_m->d_I_R[threadIdx.x])
        );
        d_m->d_S_load[threadIdx.x] = cuCmul(d_m->d_V_R_LN[threadIdx.x], I_R_conj);
        
        // Calculate power loss
        d_m->d_S_loss[threadIdx.x] = cuCsub(d_m->d_S_source[threadIdx.x], d_m->d_S_load[threadIdx.x]);
    }
    __threadfence();
}

struct BlockInit {
    dim3 grid;
    dim3 block;
    
    BlockInit(int num_tasks, bool is_matrix = false) {
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, 0);
        
        // Get max resources and divide by number of concurrent tasks
        int max_threads = props.maxThreadsPerBlock / num_tasks;
        int warp_size = props.warpSize;
        
        // Ensure we're aligned to warp size
        max_threads = (max_threads / warp_size) * warp_size;
        
        if (is_matrix) {
            // 2D configuration for matrix operations
            int block_dim = static_cast<int>(sqrt(max_threads));
            block = dim3(block_dim, block_dim);
            grid = dim3(block_dim, block_dim);
        } else {
            // 1D configuration for vector operations
            block = dim3(max_threads);
            grid = dim3(max_threads);
        }
    }
};

void line_matrix_op(GPUPowerLineMatrix<cuDoubleComplex>& m){
    cublasHandle_t handle;
    cublasCreate(&handle);

   // Cycle 1: Calculate [a] matrix
    // Operation: a = (1/2)ZY + u
    cublasZgemm3m(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        m.phase, m.phase, m.phase,
        &m.alpha_half,
        m.d_Z_abc,
        m.phase,
        m.d_Y_abc,
        m.phase,
        &m.beta_one,
        m.d_a,
        m.phase);

    cublasZgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        m.phase, m.phase,
        &m.beta_one,
        m.d_u,
        m.phase,
        &m.beta_one,
        m.d_a,
        m.phase,
        m.d_a,
        m.phase);

    // Cycle 2: Parallel computations dependent on [a]
    // Operation: Matrix inversion A = a^(-1)
  
    // Regular matrix inversion instead of batch
    cublasZgetrfBatched(handle, m.phase, m.d_Aarray, m.phase, 
                        m.d_pivot, m.d_info, 1);
    cublasZgetriBatched(handle, m.phase, m.d_Aarray, m.phase,
                        m.d_pivot, m.d_Carray, m.phase, 
                        m.d_info, 1);

    // Operation: V_S_LN = a*V_R_LN + b*I_R

    cublasZgemm3m(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        m.phase, 1, m.phase,
        &m.alpha_one, m.d_a, m.phase,
        m.d_V_R_LN, m.phase,
        &m.beta_zero, m.d_temp1, m.phase);

    cublasZgemm3m(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        m.phase, 1, m.phase,
        &m.alpha_one, m.d_Z_abc, m.phase,
        m.d_I_R, m.phase,
        &m.beta_zero, m.d_temp2, m.phase);

    cublasZgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        m.phase, 1,
        &m.beta_one, m.d_temp1, m.phase,
        &m.beta_one, m.d_temp2, m.phase,
        m.d_V_S_LN, m.phase);

    // Operation: I_S = c*V_R_LN + d*I_R
    cublasZgemm3m(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        m.phase, 1, m.phase,
        &m.alpha_one, m.d_Y_abc, m.phase,
        m.d_V_R_LN, m.phase,
        &m.beta_zero, m.d_temp1, m.phase);

    cublasZgemm3m(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        m.phase, 1, m.phase,
        &m.alpha_one, m.d_a, m.phase,
        m.d_I_R, m.phase,
        &m.beta_zero, m.d_temp2, m.phase);

    cublasZgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        m.phase, 1,
        &m.beta_one, m.d_temp1, m.phase,
        &m.beta_one, m.d_temp2, m.phase,
        m.d_I_S, m.phase);

    // Operation: V_S_LL = a*V_R_LL + b*I_R
    cublasZgemm3m(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        m.phase, 1, m.phase,
        &m.alpha_one, m.d_a, m.phase,
        m.d_V_R_LL, m.phase,
        &m.beta_zero, m.d_temp1, m.phase);

    cublasZgemm3m(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        m.phase, 1, m.phase,
        &m.alpha_one, m.d_Z_abc, m.phase,
        m.d_I_R, m.phase,
        &m.beta_zero, m.d_temp2, m.phase);

    cublasZgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        m.phase, 1,
        &m.beta_one, m.d_temp1, m.phase,
        &m.beta_one, m.d_temp2, m.phase,
        m.d_V_S_LL, m.phase);

    // Cycle 3: Parallel computations dependent on [A]
    // Operation: B = Ab

    cublasZgemm3m(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        m.phase, m.phase, m.phase,
        &m.alpha_one,
        m.d_A,
        m.phase,
        m.d_Z_abc,
        m.phase,
        &m.beta_one,
        m.d_B,
        m.phase);
    cublasDestroy(handle);
}

#endif