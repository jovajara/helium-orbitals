#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#include "../include/params.h"
#include "../include/grid.h"
#include "../include/fft_utils.h"
#include "../include/coulomb.h"
#include "../include/potentials.h"
#include "../include/density.h"
#include "../include/itp_solver.h"

#define CUDA_CHECK(call) do { cudaError_t err = call; if (err != cudaSuccess) { fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err)); exit(1); } } while (0)

GridInfo grid;
FFTPlan* fft_plan = NULL;
double *d_phi, *d_rho, *d_V_nuc, *d_V_H, *d_V_tot;
double *d_Coulomb_kernel = NULL;
cufftDoubleComplex *d_T_kernel = NULL;

// --- GUARDAR BINARIO ---
void save_binary(const char* label) {
    size_t bytes = grid.N_total * sizeof(double);
    double* h_rho = (double*)malloc(bytes);
    density_compute_gpu(d_rho, d_phi, &grid);
    CUDA_CHECK(cudaMemcpy(h_rho, d_rho, bytes, cudaMemcpyDeviceToHost));
    char filename[64]; sprintf(filename, "helium_%s.bin", label);
    FILE* fp = fopen(filename, "wb");
    if(fp) {
        fwrite(&grid.N, sizeof(int), 1, fp);
        fwrite(&grid.box_size, sizeof(double), 1, fp);
        fwrite(h_rho, sizeof(double), grid.N_total, fp);
        fclose(fp);
        printf("   -> Guardado: %s\n", filename);
    }
    free(h_rho);
}

// --- SETUP FÍSICO DINÁMICO ---
void setup_physics(double box_size) {
    if (d_V_nuc) cudaFree(d_V_nuc);
    if (d_Coulomb_kernel) cudaFree(d_Coulomb_kernel);
    if (d_T_kernel) cudaFree(d_T_kernel);
    if (fft_plan) fft_plan_destroy(fft_plan);
    grid = create_grid(GRID_RESOLUTION, box_size);
    fft_plan = fft_plan_create(&grid);
    d_V_nuc = potentials_init_Vnuc_gpu(&grid);
    d_Coulomb_kernel = coulomb_kernel_init_gpu(&grid);
    d_T_kernel = itp_init_T_kernel_gpu(&grid);
}

// --- SOLVER  ---
void solve_orbital(const char* name, int init_type) {
    printf("\n========================================\n");
    printf(" CALCULANDO ORBITAL: %s\n", name);
    printf("========================================\n");

    // Inicialización según simetría:
    // Tipo 1: Esférico (1s)
    if (init_type == 1) density_init_phi_gpu(d_phi, &grid);
    // Tipo 2: Mancuerna (2p)
    if (init_type == 2) density_init_phi_2p_gpu(d_phi, &grid);
    // Tipo 3: Trébol (3d)
    if (init_type == 3) density_init_phi_3d_gpu(d_phi, &grid);

    double E_orb = 0.0;
    for (int iter = 0; iter < MAX_SCF_ITER; iter++) {
        density_compute_gpu(d_rho, d_phi, &grid);
        coulomb_solve_gpu(d_V_H, d_rho, d_Coulomb_kernel, fft_plan);
        potentials_sum_total_gpu(d_V_tot, d_V_nuc, d_V_H, grid.N_total);
        itp_step_gpu(d_phi, d_V_tot, d_T_kernel, fft_plan, IMAGINARY_TIME_STEP);

        double norm_sq = itp_normalize_phi_gpu(d_phi, &grid);
        E_orb = -log(norm_sq) / (2.0 * IMAGINARY_TIME_STEP);

        if (iter % 20 == 0) printf("Iter %3d | E_orb: %.5f Ha\n", iter, E_orb);
    }

    // --- RESULTADOS FINALES Y ENERGÍA TOTAL ---

    // Calculamos la integral de repulsión cruda
    double E_rep_raw = itp_calculate_repulsion_energy_gpu(d_rho, d_V_H, &grid);

    // CORRECCIÓN:
    // E_rep_raw contiene la interacción de 2 electrones con el potencial de 2 electrones (4 interacciones / 2 = 2J).
    // La energía de repulsión física real es J (1 interacción).
    // Por lo tanto, usamos 0.5 * E_rep_raw.
    double E_rep_real = 0.5 * E_rep_raw;

    // Fórmula: E_tot = 2*epsilon - J
    double E_total = 2.0 * E_orb - E_rep_real;

    double Ha_to_eV = 27.211386;

    printf("============================================\n");
    printf(" > Energía Orbital (e):   %.6f Ha\n", E_orb);
    printf(" > Repulsión (J):         %.6f Ha\n", E_rep_real);
    printf(" -------------------------------------------\n");
    printf(" > ENERGÍA TOTAL:         %.6f Ha\n", E_total);
    printf(" > ENERGÍA TOTAL (eV):    %.3f eV\n", E_total * Ha_to_eV);
    printf("============================================\n");

    save_binary(name);
}

int main() {
    printf("=== HF-Helium Completo (1s -> 4f) ===\n");

    // Memoria inicial
    grid = create_grid(GRID_RESOLUTION, 10.0);
    size_t bytes = grid.N_total * sizeof(double);
    CUDA_CHECK(cudaMalloc((void**)&d_phi, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_rho, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_V_H, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_V_tot, bytes));

    // ==================================================
    // GRUPO 1: SERIE S (Caja 15.0 para buena definición)
    // ==================================================
    printf("\n>>> GRUPO 1: Orbitales S (Caja 15.0) <<<\n");
    setup_physics(15.0);

    // 1s
    solve_orbital("1s", 1);

    // ==================================================
    // GRUPO 2: SERIE P y D (Caja 25.0)
    // ==================================================
    printf("\n>>> GRUPO 2: Orbitales P y D (Caja 25.0) <<<\n");
    setup_physics(25.0);

    // 2p
    solve_orbital("2p", 2);

    // 3d
    solve_orbital("3d", 3);

    // Limpieza
    cudaFree(d_phi); cudaFree(d_rho); cudaFree(d_V_H); cudaFree(d_V_tot);
    cudaFree(d_V_nuc); cudaFree(d_Coulomb_kernel); cudaFree(d_T_kernel);
    fft_plan_destroy(fft_plan);
    printf("\n=== Cálculo Terminado ===\n");
    return 0;
}