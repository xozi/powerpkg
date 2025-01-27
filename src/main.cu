#include "logW.cuh"
#include "power_abc.cuh"
#include "ex1.cuh"
#include "hw1.cuh"
#include "ex2.cuh"
#include <cstring>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: %s <test_type>\n", argv[0]);
        return 1; 
    }

    PowerLineMatrix<cuDoubleComplex> result;

    // Determine which test to run based on the argument
    if (strcmp(argv[1], "ex1") == 0) {
        // Example 1 is calculation of backwardsweep of power flow
        result = ex1_unit_test();
    } else if (strcmp(argv[1], "hw1") == 0) {
        // Homework 1 is calculation of backward sweep, similar to Example 1
        result = hw1_unit_test(); 
    } else if (strcmp(argv[1], "ex2") == 0) {
        // Example 2 is calculation of Z_abc, Y_abc, t_n from line parameters
        result = ex2_unit_test(); 
    } else {
        printf("Error: Unknown test type '%s'...", argv[1]);
        return 1; 
    }

    int filewrite = write_log(result);
    if (filewrite != 0) {
        printf("Error: write_log failed with code %d\n", filewrite);
        return filewrite; 
    }
    return 0;
}