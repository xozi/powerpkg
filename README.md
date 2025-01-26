## powerpkg

This is solver for power distrubution models using CUDA kernels for parralize calculations.
There isn't any proper optimizations as I'm just getting functions to work with CUDA for powerflow, with some written kernels to speed up the calculations (they're not hard to begin with). The ideal utilization of this is parallelize the configuration of line/xfer/load at the same time, then preform quick forward/back flow calculations. 

I'm merely writing examples of solving singular problems until I can do that.This obviously requires CUDA installed to operate, pure C++ implementation does not exist currently.

I have a notebook that I generated to run the tests, feel free to use it as a demo.