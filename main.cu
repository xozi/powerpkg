#include "logW.cuh"
#include "power_abc.cuh"
#include "ex1.cuh"
#include "hw1.cuh"
int main() {
    //Test Example 1 (from ex1.cuh)
    //PowerMatrix<cuFloatComplex> result = ex1_unit_test();

    //Test Homework 1 (from hw1.cuh)
    PowerMatrix<cuFloatComplex> result = hw1_unit_test();

    int filewrite = write_log(result);
    if (filewrite != 0) {
        printf("Error: write_log failed with code %d\n", filewrite);
        return filewrite; // Exit with the error code
    }
    return 0;
}