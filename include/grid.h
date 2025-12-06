// --- include/grid.h ---
#ifndef GRID_H
#define GRID_H

typedef struct {
    int N;              // Resolución (ej. 64)
    int N_total;        // N * N * N
    double box_size;    // Tamaño físico (ej. 12.0)
    double spacing;     // Distancia entre puntos (Delta_x)
    double dv;          // Elemento de volumen (Delta_x^3)
} GridInfo;

/*
 * @brief Inicializa y retorna una estructura GridInfo.
 * @param N Resolución (puntos por lado).
 * @param box_size Tamaño físico de la caja (se centrará en 0).
 * @return Una estructura GridInfo llenada.
 */
GridInfo create_grid(int N, double box_size);

#endif // GRID_H
