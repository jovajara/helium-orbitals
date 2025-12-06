// --- include/params.h ---
#ifndef PARAMS_H
#define PARAMS_H

// --- 1. Constantes de la Malla ---
#define GRID_RESOLUTION 256    // Puntos por lado (N)

// --- 2. Constantes de Física ---
#define ZNUC 2.0              // Carga nuclear del Helio

// --- 3. Constantes del SCF / ITP ---
#define MAX_SCF_ITER 100
#define SCF_TOLERANCE 1e-7
#define IMAGINARY_TIME_STEP 1e-4 // (Delta_tau) Paso de tiempo para ITP

// --- 4. Constantes de PRUEBA ---
#define TEST_SLATER_ZETA 1.6875

// --- 5. Constantes Matemáticas ---
#define PI 3.14159265358979323846

#endif // PARAMS_H
