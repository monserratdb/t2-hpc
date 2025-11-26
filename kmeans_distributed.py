import numpy as np
from mpi4py import MPI
from numba import njit, prange
import time
import os # Necesario para obtener el número de hilos Numba desde SLURM

# --- Funciones Numba (compute_distances_and_assign es paralela, compute_local_sums es secuencial segura) ---
@njit(parallel=True, fastmath=True)
def compute_distances_and_assign(data, centroids):
    # Lógica de asignación de labels (es el cuello de botella, debe ser paralela)
    n_samples = data.shape[0]
    labels = np.zeros(n_samples, dtype=np.int32)
    for i in prange(n_samples):
        min_dist = 1e30
        label = -1
        for j in range(centroids.shape[0]):
            dist = 0.0
            for d in range(data.shape[1]):
                diff = data[i, d] - centroids[j, d]
                dist += diff * diff
            if dist < min_dist:
                min_dist = dist
                label = j
        labels[i] = label
    return labels

@njit(fastmath=True) # SEGURO: Eliminamos parallel=True para evitar el Race Condition
def compute_local_sums(data, labels, k, d):
    # Lógica de acumulación (se ejecuta secuencialmente dentro de cada tarea MPI)
    local_sums = np.zeros((k, d), dtype=np.float64)
    local_counts = np.zeros(k, dtype=np.int32)
    for i in range(data.shape[0]):
        l = labels[i]
        for feat in range(d):
            local_sums[l, feat] += data[i, feat] 
        local_counts[l] += 1
    return local_sums, local_counts

# --- PROGRAMA PRINCIPAL MPI ---
def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Parámetros (4M x 20 features)
    N_TOTAL = 4000000 
    D = 20
    K = 10
    SEED = 1234
    
    # 1. Generar datos y centroides (inicialización MPI)
    n_local = N_TOTAL // size
    np.random.seed(SEED + rank) 
    local_data = np.random.randn(n_local, D).astype(np.float64)
    if rank == 0:
        centroids = np.random.randn(K, D).astype(np.float64)
    else:
        centroids = np.empty((K, D), dtype=np.float64)
    comm.Bcast(centroids, root=0) 

    # === INICIO DE MEDICIÓN DE TIEMPO CRÍTICA ===
    comm.Barrier() 
    t_start = MPI.Wtime() 

    # Bucle K-Means (10 iteraciones)
    MAX_ITER = 10
    for iteration in range(MAX_ITER):
        # 1. Asignar labels (Numba Paralelo)
        labels = compute_distances_and_assign(local_data, centroids)
        
        # 2. Sumas locales (Numba Secuencial/Seguro)
        local_sums, local_counts = compute_local_sums(local_data, labels, K, D)
        
        # 3. Reducción global (MPI Allreduce)
        global_sums = np.zeros_like(local_sums)
        global_counts = np.zeros_like(local_counts)
        comm.Allreduce(local_sums, global_sums, op=MPI.SUM)
        comm.Allreduce(local_counts, global_counts, op=MPI.SUM)
        
        # 4. Actualizar centroides
        new_centroids = np.zeros_like(centroids)
        for i in range(K):
            if global_counts[i] > 0:
                new_centroids[i] = global_sums[i] / global_counts[i]
            else:
                new_centroids[i] = centroids[i] 
        centroids = new_centroids
            
    # === FIN DE MEDICIÓN DE TIEMPO CRÍTICA ===
    comm.Barrier() 
    t_end = MPI.Wtime()

    # Impresión del resultado FINAL (Solo el Rank 0)
    if rank == 0:
        elapsed_time = t_end - t_start
        numba_threads = os.environ.get('SLURM_CPUS_PER_TASK') 
        print("-" * 50)
        print(f"Configuración: {size} MPI x {numba_threads} Numba")
        print(f"Tiempo total: {elapsed_time:.4f} segundos")
        print("-" * 50)

if __name__ == "__main__":
    main()