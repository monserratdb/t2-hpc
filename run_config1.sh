#!/bin/bash
#SBATCH --job-name=kmeans_config1
#SBATCH --partition=hpc-iic3533
#SBATCH --output=salida_config1_%j.out
#SBATCH --error=error_config1_%j.err
#SBATCH --time=00:15:00
#SBATCH --mem=4G

# --- CONFIGURACIÓN DE PARALELISMO HÍBRIDO (Configuración 1: 16 CPUs TOTALES) ---
#SBATCH --nodes=2             # NECESARIO: Pide 2 nodos
#SBATCH --ntasks=2            # 2 Procesos MPI (1 proceso en cada nodo)
#SBATCH --cpus-per-task=8     # 8 Hilos Numba por proceso
#SBATCH --ntasks-per-node=1   # 1 tarea MPI por nodo (para asegurar que cada tarea use todos los 8 CPUs)

# Activar entorno
source ~/miniconda3/bin/activate kmeans_env

# Configurar Numba y OpenMP
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Ejecutar
mpirun -n $SLURM_NTASKS python kmeans_distributed.py