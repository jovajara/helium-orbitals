// --- include/density.h ---
#ifndef DENSITY_H
#define DENSITY_H

#include "grid.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * @brief Inicializa el orbital phi (d_phi) en la GPU con un orbital 1s de Slater.
 *
 * Esta es nuestra "conjetura" (guess) inicial para el bucle SCF.
 *
 * @param d_phi_out Puntero de GPU (double*) al arreglo donde se escribirá el orbital.
 * @param grid Puntero a la GridInfo (para la geometría de la malla).
 */
void density_init_phi_gpu(double* d_phi_out, const GridInfo* grid);

/* ¡NUEVA! Inicializa orbital 2p (Estado Excitado) */
void density_init_phi_2p_gpu(double* d_phi_out, const GridInfo* grid);

/* Inicializa orbital 3d (Trébol) */
void density_init_phi_3d_gpu(double* d_phi_out, const GridInfo* grid);

/* Inicializa orbital 4f (Octupolo / 8 lóbulos) */
void density_init_phi_4f_gpu(double* d_phi_out, const GridInfo* grid);

/*
 * @brief Calcula la densidad electrónica (rho = 2 * |phi|^2) en la GPU.
 *
 * @param d_rho_out Puntero de GPU (double*) al arreglo donde se escribirá la densidad.
 * @param d_phi_in Puntero de GPU (double*) con el orbital de entrada.
 * @param grid Puntero a la GridInfo (solo para N_total).
 */
void density_compute_gpu(double* d_rho_out,
                         const double* d_phi_in,
                         const GridInfo* grid);

#ifdef __cplusplus
}
#endif

#endif // DENSITY_H
