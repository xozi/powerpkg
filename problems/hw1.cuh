#ifndef HW1_H
#define HW1_H

#include "power_abc.cuh"


PowerLineMatrix<cuDoubleComplex> hw1_unit_test() {
    PowerLineMatrix<cuDoubleComplex> power;
    //Scale (line length in miles)
    const double scale = 2.0;

    // Impedance matrix [Z]
    cuDoubleComplex Z_abc[3 * 3] = {
        make_cuDoubleComplex(scale * 0.3375, scale * 1.0478), make_cuDoubleComplex(scale * 0.1535, scale * 0.3849), make_cuDoubleComplex(scale * 0.1559, scale * 0.5017),
        make_cuDoubleComplex(scale * 0.1535, scale * 0.3849), make_cuDoubleComplex(scale * 0.3414, scale * 1.0348), make_cuDoubleComplex(scale * 0.1580, scale * 0.4236),
        make_cuDoubleComplex(scale * 0.1559, scale * 0.5017), make_cuDoubleComplex(scale * 0.1580, scale * 0.4236), make_cuDoubleComplex(scale * 0.3465, scale * 1.0179)
    };

    // Admittance matrix [Y]
    cuDoubleComplex Y_abc[3 * 3] = {
        make_cuDoubleComplex(0.0, scale * 5.9540e-6), make_cuDoubleComplex(0.0, scale * -0.7471e-6), make_cuDoubleComplex(0.0, scale * -2.0030e-6),
        make_cuDoubleComplex(0.0, scale * -0.7471e-6), make_cuDoubleComplex(0.0, scale * 5.6322e-6), make_cuDoubleComplex(0.0, scale * -1.2641e-6),
        make_cuDoubleComplex(0.0, scale * -2.0030e-6), make_cuDoubleComplex(0.0, scale * -1.2641e-6), make_cuDoubleComplex(0.0, scale * 6.3962e-6)
    };

    // Inputs (VA, Rated Voltage, Power Factor)
    const double VA = 10000.0*1000.0;
    const double RATED_VOLTAGE = 13.2*1000.0;
    const double PF = 0.85;

   //Voltage and Current Setup
   vizy(RATED_VOLTAGE, VA, PF, Z_abc, Y_abc, power);

    //Allocate to GPU
    GPUPowerLineMatrix<cuDoubleComplex> line1(power);
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