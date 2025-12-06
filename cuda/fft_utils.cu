// --- cuda/fft_utils.cu ---
#include "../include/fft_utils.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <stdio.h>
#include <stdlib.h>

// Macro de utilidad para chequear errores de CUDA
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error en %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Macro de utilidad para chequear errores de cuFFT
#define CUFFT_CHECK(call) \
    do { \
        cufftResult err = call; \
        if (err != CUFFT_SUCCESS) { \
            fprintf(stderr, "cuFFT Error en %s:%d, Código: %d\n", __FILE__, __LINE__, (int)err); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)


/*
 * @brief Kernel simple para escalar un grid por un factor.
 * (Necesario porque la iFFT de cuFFT no está normalizada)
 */
__global__ void kernel_scale_grid(double* d_grid, double scale_factor, int n_total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_total) {
        d_grid[idx] *= scale_factor;
    }
}


/* * Implementación de fft_plan_create
 */
FFTPlan* fft_plan_create(const GridInfo* grid) {
    printf("[FFT] Creando plan para malla %dx%dx%d...\n", grid->N, grid->N, grid->N);

    FFTPlan* plan = (FFTPlan*)malloc(sizeof(FFTPlan));
    if (plan == NULL) {
        fprintf(stderr, "Error al asignar memoria para FFTPlan (CPU)\n");
        exit(1);
    }

    plan->N = grid->N;
    plan->N_total_real = grid->N_total;
    plan->N_total_complex = grid->N * grid->N * (grid->N / 2 + 1);

    // Asignar buffers en GPU
    CUDA_CHECK(cudaMalloc((void**)&plan->d_real,
                           plan->N_total_real * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&plan->d_complex,
                           plan->N_total_complex * sizeof(cufftDoubleComplex)));

    // Crear planes de cuFFT
    CUFFT_CHECK(cufftPlan3d(&plan->plan_r2c,
                            grid->N, grid->N, grid->N, CUFFT_D2Z));
    CUFFT_CHECK(cufftPlan3d(&plan->plan_c2r,
                            grid->N, grid->N, grid->N, CUFFT_Z2D));

    printf("[FFT] Plan creado. Buffers de GPU asignados.\n");
    return plan;
}


/* * Implementación de fft_plan_destroy
 */
void fft_plan_destroy(FFTPlan* plan) {
    printf("[FFT] Destruyendo plan y liberando buffers...\n");

    CUFFT_CHECK(cufftDestroy(plan->plan_r2c));
    CUFFT_CHECK(cufftDestroy(plan->plan_c2r));

    CUDA_CHECK(cudaFree(plan->d_real));
    CUDA_CHECK(cudaFree(plan->d_complex));

    free(plan);
}


/* * Implementación de fft_execute_r2c (Real -> Complejo)
 */
void fft_execute_r2c(FFTPlan* plan, double* d_real_in) {
    // Copiar datos de entrada al buffer interno del plan
    CUDA_CHECK(cudaMemcpy(plan->d_real, d_real_in,
                          plan->N_total_real * sizeof(double),
                          cudaMemcpyDeviceToDevice));

    // Ejecutar FFT
    CUFFT_CHECK(cufftExecD2Z(plan->plan_r2c, plan->d_real, plan->d_complex));
}


/* * Implementación de fft_execute_c2r (Complejo -> Real)
 */
void fft_execute_c2r(FFTPlan* plan, double* d_real_out) {
    // Ejecutar FFT inversa
    CUFFT_CHECK(cufftExecZ2D(plan->plan_c2r, plan->d_complex, plan->d_real));

    // Normalizar el resultado (dividir por N_total)
    double scale_factor = 1.0 / (double)plan->N_total_real;

    int threads_per_block = 256;
    int blocks = (plan->N_total_real + threads_per_block - 1) / threads_per_block;

    kernel_scale_grid<<<blocks, threads_per_block>>>(plan->d_real,
                                                     scale_factor,
                                                     plan->N_total_real);
    CUDA_CHECK(cudaGetLastError());

    // Copiar resultado al buffer de salida del usuario
    CUDA_CHECK(cudaMemcpy(d_real_out, plan->d_real,
                          plan->N_total_real * sizeof(double),
                          cudaMemcpyDeviceToDevice));
}
