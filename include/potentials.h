// --- include/potentials.h ---
#ifndef POTENTIALS_H
#define POTENTIALS_H

#include "grid.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Pre-calcula V_nuc */
double* potentials_init_Vnuc_gpu(const GridInfo* grid);


void potentials_sum_total_gpu(double* d_V_tot,
                              const double* d_V_nuc,
                              const double* d_V_H,
                              int n_total);

#ifdef __cplusplus
}
#endif

#endif // POTENTIALS_H
