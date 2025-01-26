#ifndef LOGW_H
#define LOGW_H
#include <cstdio>
#include <cuComplex.h>
#include "power_abc.cuh"

// Return phasor string
std::string get_phasor_string(const char* prefix, int index, cuFloatComplex z, float PI_F) {
    float magnitude = sqrtf(cuCrealf(z) * cuCrealf(z) + cuCimagf(z) * cuCimagf(z));
    float angle = atan2f(cuCimagf(z), cuCrealf(z)) * 180.0f / PI_F;
    
    char buffer[100];
    snprintf(buffer, sizeof(buffer), "%s[%d] = %.2f ∠ %.2f°\n", prefix, (index+1), magnitude, angle);
    return std::string(buffer);
}

int write_log(PowerLineMatrix<cuFloatComplex> result) {
    // Open file for writing results, truncate any existing content
    FILE* fp = NULL;
    #ifdef _WIN32
        errno_t err = fopen_s(&fp, "result.txt", "w");
        if (err != 0 || fp == NULL) {
            printf("Error creating/opening file! Error code: %d\n", err);
            return 1;
        }
    #else
        fp = fopen("result.txt", "w");
        if (fp == NULL) {
            printf("Error creating/opening file!\n");
            return 1;
        }
    #endif
    // Print Returned Values
    fprintf(fp, "\nComplex Power per Phase at Source (VA):\n");
    for (int i = 0; i < result.phase; i++) {
        fprintf(fp, "%s", get_phasor_string("S_S", i, result.S_source[i], result.PI_F).c_str());
    }

    fprintf(fp, "\nComplex Power per Phase at Load (VA):\n");
    for (int i = 0; i < result.phase; i++) {
        fprintf(fp, "%s", get_phasor_string("S_R", i, result.S_load[i], result.PI_F).c_str());
    }

    fprintf(fp, "\nPower Loss per Phase (VA):\n");
    for (int i = 0; i < result.phase; i++) {
        fprintf(fp, "%s", get_phasor_string("S_loss", i, result.S_loss[i], result.PI_F).c_str());
    }

    fprintf(fp, "\nCurrent-R (RMS, A):\n");
    for (int i = 0; i < result.phase; i++) {
        fprintf(fp, "%s", get_phasor_string("I_R", i, result.I_R[i], result.PI_F).c_str());
    }

    fprintf(fp, "\nVoltage-R (L-N) (RMS, V):\n");
    for (int i = 0; i < result.phase; i++) {
        fprintf(fp, "%s", get_phasor_string("V_R_LN", i, result.V_R_LN[i], result.PI_F).c_str());
    }

    fprintf(fp, "\nVoltage-R (L-L) (RMS, V):\n");
    for (int i = 0; i < result.phase; i++) {
        fprintf(fp, "%s", get_phasor_string("V_R_LL", i, result.V_R_LL[i], result.PI_F).c_str());
    }

    fprintf(fp, "\nCurrent-S (RMS, A):\n");
    for (int i = 0; i < result.phase; i++) {
        fprintf(fp, "%s", get_phasor_string("I_S", i, result.I_S[i], result.PI_F).c_str());
    }

    fprintf(fp, "\nVoltage-S (L-N) (RMS, V):\n");
    for (int i = 0; i < result.phase; i++) {
        fprintf(fp, "%s", get_phasor_string("V_S_LN", i, result.V_S_LN[i], result.PI_F).c_str());
    }

    fprintf(fp, "\nVoltage-S (L-L) (RMS, V):\n");
    for (int i = 0; i < result.phase; i++) {
        fprintf(fp, "%s", get_phasor_string("V_S_LL", i, result.V_S_LL[i], result.PI_F).c_str());
    }

    fprintf(fp, "\nVoltage Unbalance Percentage:\n %f%%\n", result.V_unb_perc);

    fprintf(fp, "\nVoltage Drop Percentage per Phase:\n %f%%\n %f%%\n %f%%\n", 
           result.phase_vdrop_perc[0], 
           result.phase_vdrop_perc[1], 
           result.phase_vdrop_perc[2]);

    fprintf(fp, "\n[t_n]:\n");
    for (int i = 0; i < result.phase; i++) {
        fprintf(fp, "%f + %fi\t\n", cuCrealf(result.t_n[i]), cuCimagf(result.t_n[i]));
    }

    fprintf(fp, "\n[Z_abc]:\n");
    for (int i = 0; i < result.phase; i++) {
        for (int j = 0; j < result.phase; j++) {
            fprintf(fp, "%f + %fi\t", cuCrealf(result.Z_abc[i * result.phase + j]), 
                                    cuCimagf(result.Z_abc[i * result.phase + j]));
        }
        fprintf(fp, "\n");
    }

    fprintf(fp, "\n[a]:\n");
    for (int i = 0; i < result.phase; i++) {
        for (int j = 0; j < result.phase; j++) {
            fprintf(fp, "%f + %fi\t", cuCrealf(result.a[i * result.phase + j]), 
                                    cuCimagf(result.a[i * result.phase + j]));
        }
        fprintf(fp, "\n");
    }

    fprintf(fp, "\n[b]:\n");
    for (int i = 0; i < result.phase; i++) {
        for (int j = 0; j < result.phase; j++) {
            fprintf(fp, "%f + %fi\t", cuCrealf(result.Z_abc[i * result.phase + j]), 
                                    cuCimagf(result.Z_abc[i * result.phase + j]));
        }
        fprintf(fp, "\n");
    }

    fprintf(fp, "\n[c]:\n");
    for (int i = 0; i < result.phase; i++) {
        for (int j = 0; j < result.phase; j++) {
            fprintf(fp, "%f + %fi\t", cuCrealf(result.Y_abc[i * result.phase + j]), 
                                    cuCimagf(result.Y_abc[i * result.phase + j]));
        }
        fprintf(fp, "\n");
    }

    fprintf(fp, "\n[d]:\n");
    for (int i = 0; i < result.phase; i++) {
        for (int j = 0; j < result.phase; j++) {
            fprintf(fp, "%f + %fi\t", cuCrealf(result.a[i * result.phase + j]), 
                                    cuCimagf(result.a[i * result.phase + j]));
        }
        fprintf(fp, "\n");
    }

    fprintf(fp, "\n[A]:\n");
    for (int i = 0; i < result.phase; i++) {
        for (int j = 0; j < result.phase; j++) {
            fprintf(fp, "%f + %fi\t", cuCrealf(result.A[i * result.phase + j]), 
                                    cuCimagf(result.A[i * result.phase + j]));
        }
        fprintf(fp, "\n");
    }

    fprintf(fp, "\n[B]:\n");
    for (int i = 0; i < result.phase; i++) {
        for (int j = 0; j < result.phase; j++) {
            fprintf(fp, "%f + %fi\t", cuCrealf(result.B[i * result.phase + j]), 
                                    cuCimagf(result.B[i * result.phase + j]));
        }
        fprintf(fp, "\n");
    }

    fclose(fp);
    return 0;
}

#endif 