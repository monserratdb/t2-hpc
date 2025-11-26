#!/bin/bash
#SBATCH --job-name=kmeans_config2
#SBATCH --partition=hpc-iic3533
#SBATCH --output=salida_config2_%j.out  # Renombrado para organizar las 5 configs
#SBATCH --error=error_config2_%j.err
#SBATCH --time=00:15:00
#SBATCH --mem=4G

# --- CONFIGURACIÓN DE PARALELISMO HÍBRIDO (Configuración 2: 16 CPUs TOTALES) ---
# Requiere 2 nodos (8 CPUs/nodo)
#SBATCH --nodes=2             # <-- CORRECCIÓN: Pide 2 nodos para 16 CPUs
#SBATCH --ntasks=4            # 4 Procesos MPI (2 procesos en cada nodo)
#SBATCH --cpus-per-task=4     # 4 Hilos Numba por proceso (usa los 4 hilos por proceso MPI)
#SBATCH --ntasks-per-node=2   # Indicamos 2 tareas MPI por nodo para distribuirlas bien

# Activar entorno
source ~/miniconda3/bin/activate kmeans_env

# Configurar Numba y OpenMP
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Ejecutar (mpirun correrá $SLURM_NTASKS = 4 procesos)
mpirun -n $SLURM_NTASKS python kmeans_distributed.py