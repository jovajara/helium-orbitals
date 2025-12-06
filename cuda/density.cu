// --- cuda/density.cu ---
#include "../include/density.h"
#include "../include/params.h" // Para TEST_SLATER_ZETA y PI
#include <cuda_runtime.h>
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
 * @brief KERNEL: Inicializa el orbital 1s de Slater normalizado
 *
 * phi(r) = N * exp(-zeta * r)
 * N = sqrt(zeta^3 / pi)
 */
__global__ void kernel_init_slater_phi(double* d_phi_out,
                                       const GridInfo grid,
                                       const double zeta) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= grid.N || iy >= grid.N || iz >= grid.N) {
        return;
    }

    // Calcular la coordenada (x,y,z) real
    double x = -grid.box_size + ix * grid.spacing;
    double y = -grid.box_size + iy * grid.spacing;
    double z = -grid.box_size + iz * grid.spacing;

    double r = sqrt(x*x + y*y + z*z);

    // Calcular el factor de normalización N
    double norm = sqrt( (zeta * zeta * zeta) / PI );

    // Calcular el valor del orbital
    double phi_val = norm * exp(-zeta * r);

    // Escribir el resultado
    int flat_idx = (ix * grid.N + iy) * grid.N + iz;
    d_phi_out[flat_idx] = phi_val;
}

/*
 * @brief KERNEL: Calcula la densidad rho = 2 * |phi|^2
 */
__global__ void kernel_compute_density(double* d_rho_out,
                                       const double* d_phi_in,
                                       int n_total) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_total) {
        return;
    }

    double phi_val = d_phi_in[idx];
    d_rho_out[idx] = 2.0 * phi_val * phi_val;
}

/*
 * Orbital 2px de Slater
 * Forma: x * exp(-zeta * r / 2)
 * Nota: Tiene un nodo en z=0 (plano yz).
 */
__global__ void kernel_init_slater_2pz(double* d_phi_out,
                                       GridInfo grid,
                                       double zeta) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= grid.N || iy >= grid.N || iz >= grid.N) return;

    double x = -grid.box_size + ix * grid.spacing;
    double y = -grid.box_size + iy * grid.spacing;
    double z = -grid.box_size + iz * grid.spacing; // ¡Importante para 2pz!

    double r = sqrt(x*x + y*y + z*z);

    // El factor x hace que sea antisimétrico 
    // Usamos zeta/2 porque los orbitales n=2 son más difusos.
    double phi_val = x * exp(-zeta * r * 0.5);

    int flat_idx = (ix * grid.N + iy) * grid.N + iz;
    d_phi_out[flat_idx] = phi_val;
}

/*
 * KERNEL 3D: Orbital Trébol (x^2 - y^2)
 * n=3, l=2. Forma: (x^2 - y^2) * exp(-zeta * r / 3)
 */
__global__ void kernel_init_slater_3d_clover(double* d_phi_out, GridInfo grid, double zeta) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= grid.N || iy >= grid.N || iz >= grid.N) return;

    double x = -grid.box_size + ix * grid.spacing;
    double y = -grid.box_size + iy * grid.spacing;
    double z = -grid.box_size + iz * grid.spacing;
    double r = sqrt(x*x + y*y + z*z);

    // Parte angular d_(x^2-y^2)
    double angular = (x * x - y * y);

    d_phi_out[ix * grid.N * grid.N + iy * grid.N + iz] = angular * exp(-zeta * r * 0.3333);
}

/*
 * * Implementación de density_init_phi_gpu
 */
void density_init_phi_gpu(double* d_phi_out, const GridInfo* grid) {

    dim3 threads(8, 8, 8);
    dim3 blocks(
        (grid->N + threads.x - 1) / threads.x,
        (grid->N + threads.y - 1) / threads.y,
        (grid->N + threads.z - 1) / threads.z
    );

    kernel_init_slater_phi<<<blocks, threads>>>(d_phi_out, *grid, TEST_SLATER_ZETA);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void density_init_phi_2p_gpu(double* d_phi_out, const GridInfo* grid) {

    dim3 threads(8, 8, 8);
    dim3 blocks(
        (grid->N + threads.x - 1) / threads.x,
        (grid->N + threads.y - 1) / threads.y,
        (grid->N + threads.z - 1) / threads.z
    );

    // Usamos el mismo zeta de prueba, pero el kernel aplica el factor 0.5
    kernel_init_slater_2pz<<<blocks, threads>>>(d_phi_out, *grid, TEST_SLATER_ZETA);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void density_init_phi_3d_gpu(double* d_phi_out, const GridInfo* grid) {
    dim3 threads(8, 8, 8);
    dim3 blocks((grid->N + 7)/8, (grid->N + 7)/8, (grid->N + 7)/8);
    kernel_init_slater_3d_clover<<<blocks, threads>>>(d_phi_out, *grid, TEST_SLATER_ZETA);
    CUDA_CHECK(cudaGetLastError());
}

/*
 * * Implementación de density_compute_gpu
 */
void density_compute_gpu(double* d_rho_out,
                         const double* d_phi_in,
                         const GridInfo* grid) {

    int n_total = grid->N_total;
    int threads_per_block = 256;
    int blocks = (n_total + threads_per_block - 1) / threads_per_block;

    kernel_compute_density<<<blocks, threads_per_block>>>(d_rho_out, d_phi_in, n_total);
    CUDA_CHECK(cudaGetLastError());
}
