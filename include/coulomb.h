// --- include/coulomb.h ---
#ifndef COULOMB_H
#define COULOMB_H

#include "grid.h"
#include "fft_utils.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * @brief Pre-calcula el kernel de Coulomb (4*pi / k^2) en la GPU.
 *
 * Esto crea un arreglo en la GPU del tamaño del espacio-k (complejo)
 * que contiene el valor 4*pi/k^2 para cada vector k.
 *
 * @param grid Puntero a la GridInfo (para obtener N y el espaciado).
 * @return Un puntero de GPU (double*) al kernel k-space pre-calculado.
 */
double* coulomb_kernel_init_gpu(const GridInfo* grid);

/*
 * @brief Resuelve la ecuación de Poisson (V_H) para una densidad (rho).
 *
 * Realiza la operación completa:
 * 1. FFT(rho) -> rho_k
 * 2. V_k = rho_k * (4*pi / k^2_kernel)
 * 3. iFFT(V_k) -> V_H
 *
 * @param d_V_H_out Arreglo de GPU (real) donde se escribirá el potencial.
 * @param d_rho_in Arreglo de GPU (real) con la densidad de entrada.
 * @param d_k_kernel Arreglo de GPU (real) del kernel pre-calculado.
 * @param plan El plan de FFT (para ejecutar las transformadas).
 */
void coulomb_solve_gpu(double* d_V_H_out,
                       const double* d_rho_in,
                       const double* d_k_kernel,
                       FFTPlan* plan);

#ifdef __cplusplus
}
#endif

#endif // COULOMB_H
