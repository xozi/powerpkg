#ifndef LOGW_H
#define LOGW_H
#include <cstdio>
#include <cuComplex.h>
#include "power_abc.cuh"

// Return phasor string
std::string get_phasor_string(const char* prefix, int index, cuDoubleComplex z) {
    double magnitude = sqrt(cuCreal(z) * cuCreal(z) + cuCimag(z) * cuCimag(z));
    double angle = atan2(cuCimag(z), cuCreal(z)) * 180.0 / M_PI;
    
    char buffer[100];
    snprintf(buffer, sizeof(buffer), "%s[%d] = %.8f ∠ %.8f°\n", prefix, (index+1), magnitude, angle);
    return std::string(buffer);
}

// Clear the log file and return file pointer
FILE* clear_log() {
    FILE* fp = NULL;
    #ifdef _WIN32
        errno_t err = fopen_s(&fp, "result.txt", "w");
        if (err != 0 || fp == NULL) {
            printf("Error creating/opening file! Error code: %d\n", err);
            return NULL;
        }
    #else
        fp = fopen("result.txt", "w");
        if (fp == NULL) {
            printf("Error creating/opening file!\n");
            return NULL;
        }
    #endif
    return fp;
}

void write_log(PowerLineMatrix<cuDoubleComplex> result, FILE* fp, std::string title) {
    // Open file for writing results, truncate any existing content
    if (fp == NULL) {
        return;
    }
    // Print Returned Values
    fprintf(fp, "\n%s:\n", title.c_str());
    fprintf(fp, "\nComplex Power per Phase at Source (VA):\n");
    for (int i = 0; i < result.phase; i++) {
        fprintf(fp, "%s", get_phasor_string("S_S", i, result.S_source[i]).c_str());
    }

    fprintf(fp, "\nComplex Power per Phase at Load (VA):\n");
    for (int i = 0; i < result.phase; i++) {
        fprintf(fp, "%s", get_phasor_string("S_R", i, result.S_load[i]).c_str());
    }

    fprintf(fp, "\nPower Loss per Phase (VA):\n");
    for (int i = 0; i < result.phase; i++) {
        fprintf(fp, "%s", get_phasor_string("S_loss", i, result.S_loss[i]).c_str());
    }

    fprintf(fp, "\nCurrent-R (RMS, A):\n");
    for (int i = 0; i < result.phase; i++) {
        fprintf(fp, "%s", get_phasor_string("I_R", i, result.I_R[i]).c_str());
    }

    fprintf(fp, "\nVoltage-R (L-N) (RMS, V):\n");
    for (int i = 0; i < result.phase; i++) {
        fprintf(fp, "%s", get_phasor_string("V_R_LN", i, result.V_R_LN[i]).c_str());
    }

    fprintf(fp, "\nVoltage-R (L-L) (RMS, V):\n");
    for (int i = 0; i < result.phase; i++) {
        fprintf(fp, "%s", get_phasor_string("V_R_LL", i, result.V_R_LL[i]).c_str());
    }

    fprintf(fp, "\nCurrent-S (RMS, A):\n");
    for (int i = 0; i < result.phase; i++) {
        fprintf(fp, "%s", get_phasor_string("I_S", i, result.I_S[i]).c_str());
    }

    fprintf(fp, "\nVoltage-S (L-N) (RMS, V):\n");
    for (int i = 0; i < result.phase; i++) {
        fprintf(fp, "%s", get_phasor_string("V_S_LN", i, result.V_S_LN[i]).c_str());
    }

    fprintf(fp, "\nVoltage-S (L-L) (RMS, V):\n");
    for (int i = 0; i < result.phase; i++) {
        fprintf(fp, "%s", get_phasor_string("V_S_LL", i, result.V_S_LL[i]).c_str());
    }

    fprintf(fp, "\nVoltage Unbalance Percentage (V):\n %.8f%%\n", result.V_unb_perc);


    fprintf(fp, "\nVoltage Drop Percentage per Phase (V):\n");  
    for (int i = 0; i < result.phase; i++) {
        fprintf(fp, "%.8f%%\n", result.phase_vdrop_perc[i]);
    }

    fprintf(fp, "\n[t_n]:\n");
    for (int i = 0; i < result.phase; i++) {
        fprintf(fp, "%.8f + %.8fi\t\n", cuCreal(result.t_n[i]), cuCimag(result.t_n[i]));
    }

    fprintf(fp, "\n[Z_abc]: (Ω per mile)\n");
    for (int i = 0; i < result.phase; i++) {
        for (int j = 0; j < result.phase; j++) {
            fprintf(fp, "%.8f + %.8fi\t", cuCreal(result.Z_abc[i * result.phase + j]), 
                                    cuCimag(result.Z_abc[i * result.phase + j]));
        }
        fprintf(fp, "\n");
    }

    fprintf(fp, "\n[Y_abc]: (µS per mile)\n");
    for (int i = 0; i < result.phase; i++) {
        for (int j = 0; j < result.phase; j++) {
            fprintf(fp, "%.8f + %.8fi\t", cuCreal(result.Y_abc[i * result.phase + j]), 
                                    cuCimag(result.Y_abc[i * result.phase + j]));
        }
        fprintf(fp, "\n");
    }

    fprintf(fp, "\n[a]:\n");
    for (int i = 0; i < result.phase; i++) {
        for (int j = 0; j < result.phase; j++) {
            fprintf(fp, "%.8f + %.8fi\t", cuCreal(result.a[i * result.phase + j]), 
                                    cuCimag(result.a[i * result.phase + j]));
        }
        fprintf(fp, "\n");
    }

    fprintf(fp, "\n[b]:\n");
    for (int i = 0; i < result.phase; i++) {
        for (int j = 0; j < result.phase; j++) {
            fprintf(fp, "%.8f + %.8fi\t", cuCreal(result.Z_abc[i * result.phase + j]), 
                                    cuCimag(result.Z_abc[i * result.phase + j]));
        }
        fprintf(fp, "\n");
    }

    fprintf(fp, "\n[c]:\n");
    for (int i = 0; i < result.phase; i++) {
        for (int j = 0; j < result.phase; j++) {
            fprintf(fp, "%.8f + %.8fi\t", cuCreal(result.Y_abc[i * result.phase + j]), 
                                    cuCimag(result.Y_abc[i * result.phase + j]));
        }
        fprintf(fp, "\n");
    }

    fprintf(fp, "\n[d]:\n");
    for (int i = 0; i < result.phase; i++) {
        for (int j = 0; j < result.phase; j++) {
            fprintf(fp, "%.8f + %.8fi\t", cuCreal(result.a[i * result.phase + j]), 
                                    cuCimag(result.a[i * result.phase + j]));
        }
        fprintf(fp, "\n");
    }

    fprintf(fp, "\n[A]:\n");
    for (int i = 0; i < result.phase; i++) {
        for (int j = 0; j < result.phase; j++) {
            fprintf(fp, "%.8f + %.8fi\t", cuCreal(result.A[i * result.phase + j]), 
                                    cuCimag(result.A[i * result.phase + j]));
        }
        fprintf(fp, "\n");
    }

    fprintf(fp, "\n[B]:\n");
    for (int i = 0; i < result.phase; i++) {
        for (int j = 0; j < result.phase; j++) {
            fprintf(fp, "%.8f + %.8fi\t", cuCreal(result.B[i * result.phase + j]), 
                                    cuCimag(result.B[i * result.phase + j]));
        }
        fprintf(fp, "\n");
    }
}

#endif 