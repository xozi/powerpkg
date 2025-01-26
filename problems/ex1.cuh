#ifndef EX1_H
#define EX1_H

#include "power_abc.cuh"

PowerLineMatrix<cuFloatComplex> ex1_unit_test() {
    PowerLineMatrix<cuFloatComplex> power;
    // Scaling factor (line length in miles)
    const float scale = 10000.0f/5280.0f;

    // Impedance matrix [Z]
    cuFloatComplex Z_abc[3 * 3] = {
        make_cuFloatComplex(scale * 0.4576, scale * 1.0780), make_cuFloatComplex(scale * 0.1560, scale * 0.5017), make_cuFloatComplex(scale * 0.1535, scale * 0.3849),
        make_cuFloatComplex(scale * 0.1560, scale * 0.5017), make_cuFloatComplex(scale * 0.4666, scale * 1.0482), make_cuFloatComplex(scale * 0.1580, scale * 0.4236),
        make_cuFloatComplex(scale * 0.1535, scale * 0.3849), make_cuFloatComplex(scale * 0.1580, scale * 0.4236), make_cuFloatComplex(scale * 0.4615, scale * 1.0651)
    };

    // Admittance matrix [Y]
    cuFloatComplex Y_abc[3 * 3] = {
        make_cuFloatComplex(0.0, scale * 5.6711e-6), make_cuFloatComplex(0.0, scale * -1.8362e-6), make_cuFloatComplex(0.0, scale * -0.7033e-6),
        make_cuFloatComplex(0.0, scale * -1.8362e-6), make_cuFloatComplex(0.0, scale * 5.9774e-6), make_cuFloatComplex(0.0, scale * -1.1690e-6),
        make_cuFloatComplex(0.0, scale * -0.7033e-6), make_cuFloatComplex(0.0, scale * -1.1690e-6), make_cuFloatComplex(0.0, scale * 5.3911e-6)
    };

    // Inputs (VA, Rated Voltage, Power Factor)
    const float VA = 6000.0f*1000.0f;
    const float RATED_VOLTAGE = 12.47f*1000.0f;
    const float PF = 0.9f;  

   //Voltage and Current Setup
    vizy(RATED_VOLTAGE, VA, PF, Z_abc, Y_abc, power);

    //Allocate to GPU 
    GPUPowerLineMatrix<cuFloatComplex> line1(power);
    BlockInit init(2, false);
    {
        line_matrix_op(line1);
        voltage_metrics<<<init.grid, init.block>>>(line1.d_this);
        power_loss<<<init.grid, init.block>>>(line1.d_this); 
    }
    line1.copyToHost(power);
    return power;
}

#endif 