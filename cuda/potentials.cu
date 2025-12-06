#include "../include/potentials.h"
#include "../include/params.h" // Para ZNUC
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error en %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// --- KERNEL 1: Construir V_nuc ---
__global__ void kernel_build_Vnuc(double* d_V_nuc_out, GridInfo grid, double Z) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= grid.N || iy >= grid.N || iz >= grid.N) return;

    double x = -grid.box_size + ix * grid.spacing;
    double y = -grid.box_size + iy * grid.spacing;
    double z = -grid.box_size + iz * grid.spacing;

    double r_sq = x*x + y*y + z*z;
    double a_sq = grid.spacing * grid.spacing; // Suavizado para evitar -inf en r=0
    double r = sqrt(r_sq + a_sq);

    int flat_idx = (ix * grid.N + iy) * grid.N + iz;
    d_V_nuc_out[flat_idx] = -Z / r;
}

// --- KERNEL 2: Sumar Potenciales ---
__global__ void kernel_sum_potentials(double* d_V_tot,
                                      const double* d_V_nuc,
                                      const double* d_V_H,
                                      int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_V_tot[idx] = d_V_nuc[idx] + 0.5 * d_V_H[idx];
    }
}

// --- Funciones Host (Wrappers) ---

double* potentials_init_Vnuc_gpu(const GridInfo* grid) {
    printf("[Potentials] Pre-calculando potencial nuclear V_nuc = -Z/r en GPU...\n");
    double* d_V_nuc;
    CUDA_CHECK(cudaMalloc((void**)&d_V_nuc, grid->N_total * sizeof(double)));

    dim3 threads(8, 8, 8);
    dim3 blocks((grid->N + threads.x - 1)/threads.x,
                (grid->N + threads.y - 1)/threads.y,
                (grid->N + threads.z - 1)/threads.z);

    // Pasamos *grid por valor
    kernel_build_Vnuc<<<blocks, threads>>>(d_V_nuc, *grid, ZNUC);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    return d_V_nuc;
}

void potentials_sum_total_gpu(double* d_V_tot,
                              const double* d_V_nuc,
                              const double* d_V_H,
                              int n_total) {
    int threads = 256;
    int blocks = (n_total + threads - 1) / threads;

    kernel_sum_potentials<<<blocks, threads>>>(d_V_tot, d_V_nuc, d_V_H, n_total);
    CUDA_CHECK(cudaGetLastError());
}
