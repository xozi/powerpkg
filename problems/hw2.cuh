#ifndef HW2_H
#define HW2_H

#include "power_abc.cuh"
#include "zysolver.cuh"

PowerLineMatrix<cuDoubleComplex> hw2_unit_test_p1() {
    PowerLineMatrix<cuDoubleComplex> power(3);
    //Phase Conductor 556,500 26/7 ACSR
    //Neutral Conductor 4/0 ACSR

    double res[4] = {
        0.1859,  // Phase A
        0.1859,  // Phase B
        0.1859,  // Phase C
        0.5920   // Neutral
    };

    double gmr[4] = {
        0.0313,  // Phase A GMR
        0.0313,  // Phase B GMR
        0.0313,  // Phase C GMR
        0.00814   // Neutral GMR
    };

    double diam[4] = {
        0.927,  // Phase A
        0.927,  // Phase B
        0.927,  // Phase C
        0.563    // Neutral
    };

    double x[4] = {
        0.0,  // Phase A
        2.5,  // Phase B
        7.0,  // Phase C
        4.0   // Neutral
    };

    double y[4] = {
        29.0,  // Phase A
        29.0,  // Phase B
        29.0,  // Phase C
        25.0   // Neutral
    };

    LineParam<double> line_params = {
        4, //amount of conductors
        res,
        gmr,
        x,
        y,
        diam
    };

    GPULineParam<double> gpu_line_params(line_params);
    cudaDeviceSynchronize();

    // Simple nxn grid for Z_prim calculation
    dim3 block_zprim(gpu_line_params.size, gpu_line_params.size); 
    dim3 grid_zprim(1, 1);  
    calc_prim<<<grid_zprim, block_zprim>>>(gpu_line_params.d_this);
    cudaDeviceSynchronize();

    // Simple 3x3 grid for Kron reduction
    dim3 block_kron(gpu_line_params.index, gpu_line_params.index);  
    dim3 grid_kron(1, 1);  
    kron_reduce<<<grid_kron, block_kron>>>(gpu_line_params.d_this);
    cudaDeviceSynchronize();

    mvert(gpu_line_params);
    // Simple 3x3 grid for Y calculation
    dim3 block_y(gpu_line_params.index, gpu_line_params.index);  
    dim3 grid_y(1, 1);  
    calc_Y<<<grid_y, block_y>>>(gpu_line_params.d_this);
    cudaDeviceSynchronize();

    gpu_line_params.copyToHost(power);   
    return power;
}

PowerLineMatrix<cuDoubleComplex> hw2_unit_test_p2() {
    PowerLineMatrix<cuDoubleComplex> power(2);
    //Phase Conductor 336,400 26/7 ACSR
    //Neutral Conductor 4/0 ACSR

    double res[3] = {
        0.306,  // Phase A
        0.306,  // Phase B
        0.5920   // Neutral
    };

    double gmr[3] = {
        0.0244,  // Phase A GMR
        0.0244,  // Phase B GMR
        0.00814   // Neutral GMR
    };

    double diam[3] = {
        0.721,  // Phase A
        0.721,  // Phase B
        0.563    // Neutral
    };

    double x[3] = {
        0.0,  // Phase A
        7.0,  // Phase B
        4.0   // Neutral
    };

    double y[3] = {
        29.0,  // Phase A
        29.0,  // Phase B
        25.0   // Neutral
    };

    LineParam<double> line_params = {
        3, //amount of conductors
        res,
        gmr,
        x,
        y,
        diam
    };

    GPULineParam<double> gpu_line_params(line_params);
    cudaDeviceSynchronize();

    // Simple nxn grid for Z_prim calculation
    dim3 block_zprim(gpu_line_params.size, gpu_line_params.size); 
    dim3 grid_zprim(1, 1);  
    calc_prim<<<grid_zprim, block_zprim>>>(gpu_line_params.d_this);
    cudaDeviceSynchronize();

    // Simple 3x3 grid for Kron reduction
    dim3 block_kron(gpu_line_params.index, gpu_line_params.index);  
    dim3 grid_kron(1, 1);  
    kron_reduce<<<grid_kron, block_kron>>>(gpu_line_params.d_this);
    cudaDeviceSynchronize();

    mvert(gpu_line_params);
    // Simple 3x3 grid for Y calculation
    dim3 block_y(gpu_line_params.index, gpu_line_params.index);  
    dim3 grid_y(1, 1);  
    calc_Y<<<grid_y, block_y>>>(gpu_line_params.d_this);
    cudaDeviceSynchronize();

    gpu_line_params.copyToHost(power);   
    return power;
}
#endif 