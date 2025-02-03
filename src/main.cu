#include "logW.cuh"
#include "power_abc.cuh"
#include "ex1.cuh"
#include "hw1.cuh"
#include "ex2.cuh"
#include "hw2.cuh"
#include <cstring>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: %s <test_type>\n", argv[0]);
        return 1; 
    }
    PowerLineMatrix<cuDoubleComplex> result;
    // Determine which test to run based on the argument
    FILE* fp = clear_log();
    if (strcmp(argv[1], "ex1") == 0) {
        // Example 1 is calculation of backwardsweep of power flow
        result = ex1_unit_test();
        write_log(result, fp, "Example 1");
    } else if (strcmp(argv[1], "hw1") == 0) {
        // Homework 1 is calculation of backward sweep, similar to Example 1
        result = hw1_unit_test(); 
        write_log(result, fp, "Homework 1");
    } else if (strcmp(argv[1], "ex2") == 0) {
        // Example 2 is calculation of Z_abc, Y_abc, t_n from line parameters
        result = ex2_unit_test(); 
        write_log(result, fp, "Example 2");
    } else if (strcmp(argv[1], "hw2") == 0) {
        // Homework 2 is calculation of Z_abc, Y_abc, t_n from line parameters
        result = hw2_unit_test_p1(); 
        write_log(result, fp, "Homework 2 Part 1");
        result = hw2_unit_test_p2(); 
        write_log(result, fp, "Homework 2 Part 2");
    } else {
        printf("Error: Unknown test type '%s'...", argv[1]);
    }
    fclose(fp);
    return 0;
}