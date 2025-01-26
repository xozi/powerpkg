#include "logW.cuh"
#include "power_abc.cuh"
#include "ex1.cuh"
#include "hw1.cuh"
#include <cstring>

int main(int argc, char* argv[]) {
    // Check for specific command line arguments
    if (argc < 2) {
        printf("Usage: %s <test_type>\n", argv[0]);
        return 1; // Exit with error if no arguments are provided
    }

    PowerLineMatrix<cuFloatComplex> result;

    // Determine which test to run based on the argument
    if (strcmp(argv[1], "ex1") == 0) {
        // Test Example 1 (from ex1.cuh)
        result = ex1_unit_test();
    } else if (strcmp(argv[1], "hw1") == 0) {
        // Test Homework 1 (from hw1.cuh)
        result = hw1_unit_test(); 
    } else {
        printf("Error: Unknown test type '%s'. Use 'ex1' or 'hw1'.\n", argv[1]);
        return 1; 
    }

    int filewrite = write_log(result);
    if (filewrite != 0) {
        printf("Error: write_log failed with code %d\n", filewrite);
        return filewrite; 
    }
    return 0;
}