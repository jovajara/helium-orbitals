# --- Makefile ---

CC = gcc
NVCC = nvcc
TARGET = hf_helium

# Directorios del proyecto
INC_DIR = include
SRC_DIR = src
CUDA_DIR = cuda
OBJ_DIR = obj

# Directorio donde vive CUDA en Colab
CUDA_PATH = /usr/local/cuda

# --- FLAGS ---

# 1. CFLAGS (Para GCC):
#    -I$(INC_DIR): Busca tus headers (.h)
#    -I$(CUDA_PATH)/include: ¡CRUCIAL! Busca cufft.h, cuda_runtime.h, etc.
CFLAGS = -I$(INC_DIR) -I$(CUDA_PATH)/include -Wall -O3

# 2. NVFLAGS (Para NVCC):
#    -arch=sm_75: Optimización para Tesla T4
NVFLAGS = -I$(INC_DIR) -O3 -arch=sm_75

# 3. LDFLAGS (Enlazador):
#    -L$(CUDA_PATH)/lib64: Busca las librerías compiladas (.so)
LDFLAGS = -L$(CUDA_PATH)/lib64 -lm -lcudart -lcufft

# --- ARCHIVOS ---
C_SRCS := $(wildcard $(SRC_DIR)/*.c)
CUDA_SRCS := $(wildcard $(CUDA_DIR)/*.cu)

C_OBJS := $(C_SRCS:$(SRC_DIR)/%.c=$(OBJ_DIR)/%.o)
CUDA_OBJS := $(CUDA_SRCS:$(CUDA_DIR)/%.cu=$(OBJ_DIR)/%.o)

OBJS := $(C_OBJS) $(CUDA_OBJS)

# --- REGLAS ---

all: $(TARGET)

$(TARGET): $(OBJS)
	@echo "Enlazando..."
	$(NVCC) $(OBJS) -o $@ $(LDFLAGS)

# Regla para compilar C (main.c, grid.c)
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	@echo "Compilando C: $<"
	$(CC) $(CFLAGS) -c $< -o $@

# Regla para compilar CUDA (kernels)
$(OBJ_DIR)/%.o: $(CUDA_DIR)/%.cu | $(OBJ_DIR)
	@echo "Compilando CUDA: $<"
	$(NVCC) $(NVFLAGS) -c $< -o $@

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

clean:
	rm -rf $(OBJ_DIR) $(TARGET)

.PHONY: all clean
