// --- include/itp_solver.h ---
#ifndef ITP_SOLVER_H
#define ITP_SOLVER_H

#include "grid.h"
#include "fft_utils.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * @brief Pre-calcula el operador de energía cinética (T = p^2/2) en k-space.
 *
 * Crea un arreglo en la GPU del tamaño del espacio-k (complejo)
 * que contiene el valor p^2/2 para cada vector p (o k).
 *
 * @param grid Puntero a la GridInfo (para obtener N y el espaciado).
 * @return Un puntero de GPU (cufftDoubleComplex*) al kernel T(k) pre-calculado.
 */
cufftDoubleComplex* itp_init_T_kernel_gpu(const GridInfo* grid);

/*
 * @brief Ejecuta UN paso de la propagación en tiempo imaginario (ITP).
 *
 * Realiza la operación "Split-Step":
 * phi_new = exp(-V*dt/2) * iFFT[ exp(-T*dt) * FFT[ exp(-V*dt/2) * phi_old ] ]
 *
 * @param d_phi Puntero al orbital en GPU. Se actualiza "in-place".
 * @param d_V_total Puntero al potencial total (V_nuc + V_H) en la GPU.
 * @param d_T_kernel Puntero al kernel de energía cinética (p^2/2) en k-space.
 * @param plan El plan de FFT (para ejecutar las transformadas).
 * @param delta_tau El paso de tiempo imaginario (dt).
 */
void itp_step_gpu(double* d_phi,
                  const double* d_V_total,
                  const cufftDoubleComplex* d_T_kernel,
                  FFTPlan* plan,
                  double delta_tau);

/*
 * @brief Normaliza el orbital phi en la GPU.
 *
 * Calcula la norma ||phi|| y luego actualiza phi = phi / ||phi||.
 * Retorna la norma (útil para monitorear la convergencia).
 *
 * @param d_phi Puntero al orbital en GPU (se actualiza "in-place").
 * @param grid Puntero a la GridInfo (para N_total y dv).
 * @return El valor de la norma <phi|phi> antes de normalizar.
 */
double itp_normalize_phi_gpu(double* d_phi, const GridInfo* grid);

double itp_calculate_repulsion_energy_gpu(const double* d_rho, 
                                          const double* d_V_H, 
                                          const GridInfo* grid);

#ifdef __cplusplus
}
#endif

#endif // ITP_SOLVER_H
