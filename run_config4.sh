#!/bin/bash
#SBATCH --job-name=kmeans_config4
#SBATCH --partition=hpc-iic3533
#SBATCH --output=salida_config4_%j.out
#SBATCH --error=error_config4_%j.err
#SBATCH --time=00:15:00
#SBATCH --mem=4G

# --- CONFIGURACIÓN 4: 16 MPI tasks x 1 Numba thread = 16 CPUs TOTALES ---
#SBATCH --nodes=2             # Pide 2 nodos
#SBATCH --ntasks=16           # 16 Procesos MPI
#SBATCH --cpus-per-task=1     # 1 Hilo Numba por proceso
#SBATCH --ntasks-per-node=8   # 8 tareas MPI por nodo (8 * 1 = 8 CPUs/nodo)

# Activar entorno
source ~/miniconda3/bin/activate kmeans_env

# Configurar Numba y OpenMP
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Ejecutar
mpirun -n $SLURM_NTASKS python kmeans_distributed.py