#!/bin/bash
#SBATCH --job-name=kmeans_config0
#SBATCH --partition=hpc-iic3533
#SBATCH --output=salida_config0_%j.out
#SBATCH --error=error_config0_%j.err
#SBATCH --time=00:15:00
#SBATCH --mem=4G

# --- CONFIGURACIÓN DE PARALELISMO (Configuración 0: 1 CPU TOTAL) ---
# Necesita 1 nodo. (1 MPI x 1 Numba)
#SBATCH --nodes=1             
#SBATCH --ntasks=1            # 1 Proceso MPI
#SBATCH --cpus-per-task=1     # 1 Hilo Numba por proceso
#SBATCH --ntasks-per-node=1   # 1 tarea MPI por nodo

# Activar entorno
source ~/miniconda3/bin/activate kmeans_env

# Configurar Numba y OpenMP
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Ejecutar
mpirun -n $SLURM_NTASKS python kmeans_distributed.py