// --- cuda/coulomb.cu ---
#include "../include/coulomb.h"
#include "../include/params.h" // Para PI
#include <cuda_runtime.h>
#include <cufft.h> // Para cufftDoubleComplex
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Macro de chequeo de errores de CUDA
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error en %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)


/*
 * @brief KERNEL: Construye el kernel 4*pi / k^2 en el espacio-k.
 *
 * NOTA: El kernel de Coulomb se almacena como un arreglo de 'double',
 * aunque se aplicará a un arreglo 'cufftDoubleComplex'.
 * Esto ahorra memoria y funciona bien para la multiplicación.
 */
__global__ void kernel_build_k_kernel(double* d_k_kernel, int N, double spacing) {
    // k-space es N x N x (N/2 + 1)
    int N_k_z = N / 2 + 1;

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= N || iy >= N || iz >= N_k_z) {
        return;
    }

    // Calcular los componentes del vector k (kx, ky, kz)
    double dk = 2.0 * PI / (N * spacing); // Frecuencia fundamental

    // kx y ky "envuelven" (wrap-around) para frecuencias negativas
    // ej. N=64: 0, 1, ..., 31, -32, -31, ..., -1
    double kx = (ix < N / 2) ? (double)ix : (double)(ix - N);
    double ky = (iy < N / 2) ? (double)iy : (double)(iy - N);

    // kz no envuelve, solo frecuencias positivas
    double kz = (double)iz;

    kx *= dk;
    ky *= dk;
    kz *= dk;

    double k_sq = kx*kx + ky*ky + kz*kz; // k^2

    int flat_idx = (ix * N + iy) * N_k_z + iz;

    // --- ¡Manejo de la singularidad k=0! ---
    if (k_sq < 1e-10) { // (si k_sq es cero)
        d_k_kernel[flat_idx] = 0.0;
    } else {
        d_k_kernel[flat_idx] = 4.0 * PI / k_sq;
    }
}

/*
 * @brief KERNEL: Multiplica el rho_k por el kernel k-space
 *
 * Realiza: V_k = rho_k * k_kernel
 */
__global__ void kernel_apply_coulomb_kspace(cufftDoubleComplex* d_V_k,
                                            const cufftDoubleComplex* d_rho_k,
                                            const double* d_k_kernel,
                                            int n_total_complex) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_total_complex) {
        return;
    }

    // Cargar rho_k (complejo)
    cufftDoubleComplex rho_val = d_rho_k[idx];

    // Cargar k_kernel (real)
    double k_val = d_k_kernel[idx];

    // Multiplicar (real * complejo)
    d_V_k[idx].x = rho_val.x * k_val; // Parte real
    d_V_k[idx].y = rho_val.y * k_val; // Parte imaginaria
}

/*
 * * Implementación de coulomb_kernel_init_gpu
 */
double* coulomb_kernel_init_gpu(const GridInfo* grid) {
    printf("[Coulomb] Pre-calculando kernel 4*pi/k^2 en GPU...\n");

    int N = grid->N;
    int N_k_z = N / 2 + 1;
    int n_total_complex = N * N * N_k_z;

    double* d_k_kernel;
    CUDA_CHECK(cudaMalloc((void**)&d_k_kernel,
                           n_total_complex * sizeof(double)));

    // Configuración de la grilla 3D para k-space
    dim3 threads(8, 8, 8);
    dim3 blocks(
        (N + threads.x - 1) / threads.x,
        (N + threads.y - 1) / threads.y,
        (N_k_z + threads.z - 1) / threads.z
    );

    kernel_build_k_kernel<<<blocks, threads>>>(d_k_kernel, N, grid->spacing);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize()); // Asegurarse que el kernel esté listo

    return d_k_kernel;
}

/*
 * * Implementación de coulomb_solve_gpu
 */
void coulomb_solve_gpu(double* d_V_H_out,
                       const double* d_rho_in,
                       const double* d_k_kernel,
                       FFTPlan* plan) {

    // 1. FFT(rho) -> rho_k
    //    (d_rho_in va a plan->d_real, resultado queda en plan->d_complex)
    fft_execute_r2c(plan, (double*)d_rho_in);

    // 2. V_k = rho_k * k_kernel
    //    (Resultado se guarda de vuelta en plan->d_complex)
    int n_total_complex = plan->N_total_complex;
    int threads_per_block = 256;
    int blocks = (n_total_complex + threads_per_block - 1) / threads_per_block;

    kernel_apply_coulomb_kspace<<<blocks, threads_per_block>>>(
        plan->d_complex,     // Escribir en V_k (buffer interno)
        plan->d_complex,     // Leer desde rho_k (buffer interno)
        d_k_kernel,
        n_total_complex
    );
    CUDA_CHECK(cudaGetLastError());

    // 3. iFFT(V_k) -> V_H
    //    (Lee desde plan->d_complex, resultado va a d_V_H_out)
    fft_execute_c2r(plan, d_V_H_out);
}
