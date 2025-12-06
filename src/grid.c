// --- src/grid.c ---
#include "../include/grid.h"
#include <stdio.h>  // Para printf
#include <math.h>   // Para pow

GridInfo create_grid(int N, double box_size) {
    GridInfo grid;
    grid.N = N;
    grid.box_size = box_size;
    grid.N_total = N * N * N;

    // Hay (N-1) intervalos entre N puntos
    grid.spacing = (2.0 * box_size) / (double)(N - 1);
    grid.dv = pow(grid.spacing, 3.0);

    printf("[Grid] Malla 3D creada: %dx%dx%d puntos.\n", N, N, N);
    printf("[Grid] Espaciado (dx): %e\n", grid.spacing);

    return grid;
}
