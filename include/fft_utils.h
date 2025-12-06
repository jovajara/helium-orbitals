// --- include/fft_utils.h ---
#ifndef FFT_UTILS_H
#define FFT_UTILS_H

#include <cufft.h>
#include "grid.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * @brief Estructura que envuelve un plan de cuFFT y sus buffers en GPU.
 */
typedef struct {
    cufftHandle plan_r2c; // Plan para Real -> Complejo (Adelante)
    cufftHandle plan_c2r; // Plan para Complejo -> Real (Reversa)

    int N;                // Dimensión (N x N x N)
    int N_total_real;     // N*N*N
    int N_total_complex;  // N*N*(N/2 + 1)

    // Punteros a la memoria EN LA GPU (device)
    double* d_real;           // Buffer para datos reales
    cufftDoubleComplex* d_complex; // Buffer para datos complejos

} FFTPlan;

/*
 * @brief Crea un plan de FFT 3D y asigna los buffers en la GPU.
 * @param grid Puntero a la GridInfo que describe la malla.
 * @return Un puntero a la estructura FFTPlan inicializada.
 */
FFTPlan* fft_plan_create(const GridInfo* grid);

/*
 * @brief Libera la memoria de GPU y destruye los planes de cuFFT.
 * @param plan Puntero al plan a destruir.
 */
void fft_plan_destroy(FFTPlan* plan);

/*
 * @brief Ejecuta la FFT (Adelante): Real -> Complejo.
 * @param plan Puntero al plan.
 * @param d_real_in Puntero GPU a los datos reales de entrada.
 */
void fft_execute_r2c(FFTPlan* plan, double* d_real_in);

/*
 * @brief Ejecuta la FFT (Reversa): Complejo -> Real.
 * @param plan Puntero al plan.
 * @param d_real_out Puntero GPU donde se guardará el resultado real.
 */
void fft_execute_c2r(FFTPlan* plan, double* d_real_out);

#ifdef __cplusplus
}
#endif

#endif // FFT_UTILS_H
