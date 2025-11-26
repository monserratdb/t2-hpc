#!/bin/bash
#SBATCH --job-name=kmeans_config3
#SBATCH --partition=hpc-iic3533
#SBATCH --output=salida_config3_%j.out
#SBATCH --error=error_config3_%j.err
#SBATCH --time=00:15:00
#SBATCH --mem=4G

# --- CONFIGURACIÓN 3: 8 MPI tasks x 2 Numba threads = 16 CPUs TOTALES ---
#SBATCH --nodes=2             # Pide 2 nodos
#SBATCH --ntasks=8            # 8 Procesos MPI (4 en cada nodo)
#SBATCH --cpus-per-task=2     # 2 Hilos Numba por proceso
#SBATCH --ntasks-per-node=4   # 4 tareas MPI por nodo (4 * 2 = 8 CPUs/nodo)

# Activar entorno
source ~/miniconda3/bin/activate kmeans_env

# Configurar Numba y OpenMP
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Ejecutar
mpirun -n $SLURM_NTASKS python kmeans_distributed.p