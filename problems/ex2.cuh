#ifndef EX2_H
#define EX2_H

#include "power_abc.cuh"
#include "zysolver.cuh"

PowerLineMatrix<cuFloatComplex> ex2_unit_test() {
    PowerLineMatrix<cuFloatComplex> power;
    float res[4] = {
        0.3060f,  // Phase A
        0.3060f,  // Phase B
        0.3060f,  // Phase C
        0.5920f   // Neutral
    };

    float gmr[4] = {
        0.0244f,  // Phase A GMR
        0.0244f,  // Phase B GMR
        0.0244f,  // Phase C GMR
        0.0081f   // Neutral GMR
    };

    float x[4] = {
        0.0f,  // Phase A
        2.5f,  // Phase B
        7.0f,  // Phase C
        4.0f  // Neutral
    };

    float y[4] = {
        29.0f,  // Phase A
        29.0f,  // Phase B
        29.0f,  // Phase C
        25.0f  // Neutral
    };

    LineParam<float> line_params = {
        4, //amount of conductors
        res,
        gmr,
        x,
        y
    };

    GPULineParam<float> gpu_line_params(line_params);
    cudaDeviceSynchronize();

    // Simple nxn grid for Z_prim calculation
    dim3 block_zprim(gpu_line_params.size, gpu_line_params.size); 
    dim3 grid_zprim(1, 1);  
    calc_zprim<<<grid_zprim, block_zprim>>>(gpu_line_params.d_this);
    cudaDeviceSynchronize();

   
    // Simple 3x3 grid for Kron reduction
    dim3 block_kron(3, 3);  
    dim3 grid_kron(1, 1);  
    kron_reduce<<<grid_kron, block_kron>>>(gpu_line_params.d_this);
    cudaDeviceSynchronize();

    gpu_line_params.copyToHost(power);   
    return power;
}


#endif 