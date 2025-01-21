#ifndef EX1_H
#define EX1_H

#include "power_abc.cuh"

PowerMatrix<cuFloatComplex> ex1_unit_test() {
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
    PowerMatrix<cuFloatComplex> init = vizy(RATED_VOLTAGE, VA, PF, Z_abc, Y_abc);

    // Solve for values
    PowerMatrix<cuFloatComplex> result = matrix_solver(init);
    
    return result;
}

#endif 