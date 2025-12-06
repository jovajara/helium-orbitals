// --- cuda/itp_solver.cu ---
#include "../include/itp_solver.h"
#include "../include/params.h" // Para PI
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// ¡Importante! Incluimos la librería CUB para la reducción en paralelo
#include <cub/cub.cuh>

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
 * @brief KERNEL: Construye el kernel T(k) = p^2/2 en el espacio-k.
 */
__global__ void kernel_build_T_kernel(cufftDoubleComplex* d_T_kernel, int N, double spacing) {
    int N_k_z = N / 2 + 1;

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= N || iy >= N || iz >= N_k_z) {
        return;
    }

    double dk = 2.0 * PI / (N * spacing);
    double kx = (ix < N / 2) ? (double)ix : (double)(ix - N);
    double ky = (iy < N / 2) ? (double)iy : (double)(iy - N);
    double kz = (double)iz;
    kx *= dk;
    ky *= dk;
    kz *= dk;

    double k_sq = kx*kx + ky*ky + kz*kz; // p^2 (ya que h_bar = 1)

    int flat_idx = (ix * N + iy) * N_k_z + iz;

    // T = p^2 / 2m (m=1 en a.u.)
    // Guardamos T(k) solo en la parte real.
    d_T_kernel[flat_idx].x = 0.5 * k_sq;
    d_T_kernel[flat_idx].y = 0.0;
}

/*
 * @brief KERNEL: Aplica el operador de potencial (Split-Step)
 *
 * phi = phi * exp(-V * dt)
 */
__global__ void kernel_apply_potential(double* d_phi,
                                     const double* d_V_total,
                                     double dt,
                                     int n_total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_total) {
        return;
    }

    double V = d_V_total[idx];
    d_phi[idx] *= exp(-V * dt);
}

/*
 * @brief KERNEL: Aplica el operador cinético (Split-Step)
 *
 * phi_k = phi_k * exp(-T * dt)
 */
__global__ void kernel_apply_kinetic(cufftDoubleComplex* d_phi_k,
                                   const cufftDoubleComplex* d_T_kernel,
                                   double dt,
                                   int n_total_complex) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_total_complex) {
        return;
    }

    double T = d_T_kernel[idx].x; // T(k) es real
    double factor = exp(-T * dt);

    d_phi_k[idx].x *= factor;
    d_phi_k[idx].y *= factor;
}

/*
 * @brief KERNEL: Calcula el cuadrado de la norma |phi|^2 * dv
 */
__global__ void kernel_calc_norm_sq(double* d_norm_sq_array,
                                    const double* d_phi,
                                    double dv,
                                    int n_total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_total) {
        return;
    }

    double phi_val = d_phi[idx];
    d_norm_sq_array[idx] = phi_val * phi_val * dv;
}

/*
 * @brief KERNEL: Normaliza el orbital phi = phi / sqrt(norma)
 */
__global__ void kernel_normalize(double* d_phi, double norm_factor, int n_total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_total) {
        return;
    }

    d_phi[idx] *= norm_factor;
}

// --- KERNEL PARA INTEGRAL DE REPULSIÓN ---
// Calcula rho[i] * V_H[i] * dv para cada punto
__global__ void kernel_repulsion_term(double* d_out, 
                                      const double* d_rho, 
                                      const double* d_V_H, 
                                      double dv, 
                                      int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // La integral es rho * V_H.
        // El factor 0.5 del doble conteo lo aplicaremos al final.
        d_out[idx] = d_rho[idx] * d_V_H[idx] * dv;
    }
}

/*
 * * Implementación de itp_init_T_kernel_gpu
 */
cufftDoubleComplex* itp_init_T_kernel_gpu(const GridInfo* grid) {
    printf("[ITP] Pre-calculando kernel cinético T(k) = p^2/2 en GPU...\n");

    int N = grid->N;
    int N_k_z = N / 2 + 1;
    int n_total_complex = N * N * N_k_z;

    cufftDoubleComplex* d_T_kernel;
    CUDA_CHECK(cudaMalloc((void**)&d_T_kernel,
                           n_total_complex * sizeof(cufftDoubleComplex)));

    dim3 threads(8, 8, 8);
    dim3 blocks(
        (N + threads.x - 1) / threads.x,
        (N + threads.y - 1) / threads.y,
        (N_k_z + threads.z - 1) / threads.z
    );

    kernel_build_T_kernel<<<blocks, threads>>>(d_T_kernel, N, grid->spacing);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    return d_T_kernel;
}

/*
 * * Implementación de itp_step_gpu
 */
void itp_step_gpu(double* d_phi,
                  const double* d_V_total,
                  const cufftDoubleComplex* d_T_kernel,
                  FFTPlan* plan,
                  double delta_tau) {

    int n_total = plan->N_total_real;
    int n_total_complex = plan->N_total_complex;
    double dt_half = delta_tau / 2.0;

    int threads_per_block = 256;
    int blocks = (n_total + threads_per_block - 1) / threads_per_block;

    // 1. Aplicar exp(-V*dt/2)
    kernel_apply_potential<<<blocks, threads_per_block>>>(d_phi, d_V_total, dt_half, n_total);
    CUDA_CHECK(cudaGetLastError());

    // 2. FFT(phi) -> phi_k
    //    (d_phi va a plan->d_real, resultado queda en plan->d_complex)
    fft_execute_r2c(plan, d_phi);

    // 3. Aplicar exp(-T*dt)
    int blocks_complex = (n_total_complex + threads_per_block - 1) / threads_per_block;
    kernel_apply_kinetic<<<blocks_complex, threads_per_block>>>(plan->d_complex,
                                                               d_T_kernel,
                                                               delta_tau,
                                                               n_total_complex);
    CUDA_CHECK(cudaGetLastError());

    // 4. iFFT(phi_k) -> phi
    //    (Lee de plan->d_complex, resultado va a d_phi)
    fft_execute_c2r(plan, d_phi);

    // 5. Aplicar exp(-V*dt/2)
    kernel_apply_potential<<<blocks, threads_per_block>>>(d_phi, d_V_total, dt_half, n_total);
    CUDA_CHECK(cudaGetLastError());
}

/*
 * * Implementación de itp_normalize_phi_gpu
 */
double itp_normalize_phi_gpu(double* d_phi, const GridInfo* grid) {
    int n_total = grid->N_total;

    // --- 1. Calcular |phi|^2 * dv en un arreglo temporal ---
    double* d_norm_sq_array;
    CUDA_CHECK(cudaMalloc((void**)&d_norm_sq_array, n_total * sizeof(double)));

    int threads_per_block = 256;
    int blocks = (n_total + threads_per_block - 1) / threads_per_block;

    kernel_calc_norm_sq<<<blocks, threads_per_block>>>(d_norm_sq_array, d_phi, grid->dv, n_total);
    CUDA_CHECK(cudaGetLastError());

    // --- 2. Reducción en paralelo (Suma) usando CUB ---
    // CUB necesita memoria temporal
    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    double* d_norm_total; // Puntero al resultado en GPU
    CUDA_CHECK(cudaMalloc((void**)&d_norm_total, sizeof(double)));

    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes,
                           d_norm_sq_array, d_norm_total, n_total);
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes,
                           d_norm_sq_array, d_norm_total, n_total);

    // --- 3. Copiar el resultado (norma_total) de GPU a CPU ---
    double h_norm_total;
    CUDA_CHECK(cudaMemcpy(&h_norm_total, d_norm_total,
                          sizeof(double), cudaMemcpyDeviceToHost));

    // --- 4. Calcular el factor de normalización (1 / sqrt(norma)) ---
    double norm_factor = 1.0 / sqrt(h_norm_total);

    // --- 5. Aplicar el factor de normalización en la GPU ---
    kernel_normalize<<<blocks, threads_per_block>>>(d_phi, norm_factor, n_total);
    CUDA_CHECK(cudaGetLastError());

    // --- 6. Limpieza ---
    CUDA_CHECK(cudaFree(d_norm_sq_array));
    CUDA_CHECK(cudaFree(d_temp_storage));
    CUDA_CHECK(cudaFree(d_norm_total));

    return h_norm_total; // Retornamos la norma <phi|phi>
}

double itp_calculate_repulsion_energy_gpu(const double* d_rho, 
                                                     const double* d_V_H, 
                                                     const GridInfo* grid) {
    int n = grid->N_total;
    
    // 1. Memoria temporal para el producto punto
    double* d_temp;
    CUDA_CHECK(cudaMalloc((void**)&d_temp, n * sizeof(double)));
    
    int threads = 256;
    int blocks = (n + 255) / 256;
    
    kernel_repulsion_term<<<blocks, threads>>>(d_temp, d_rho, d_V_H, grid->dv, n);
    
    // 2. Reducción (Suma total) usando CUB
    void* d_storage = NULL;
    size_t bytes = 0;
    double* d_sum;
    CUDA_CHECK(cudaMalloc((void**)&d_sum, sizeof(double)));
    
    // Query
    cub::DeviceReduce::Sum(d_storage, bytes, d_temp, d_sum, n);
    CUDA_CHECK(cudaMalloc(&d_storage, bytes));
    
    // Ejecutar
    cub::DeviceReduce::Sum(d_storage, bytes, d_temp, d_sum, n);
    
    // 3. Traer resultado
    double integral_val;
    CUDA_CHECK(cudaMemcpy(&integral_val, d_sum, sizeof(double), cudaMemcpyDeviceToHost));
    
    // Limpieza
    cudaFree(d_temp); cudaFree(d_storage); cudaFree(d_sum);
    
    // La energía de repulsión (Corrección de doble conteo) es 0.5 * Integral
    return 0.5 * integral_val;
}