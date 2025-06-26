import numpy as np
import dimod
import matplotlib.pyplot as plt
import time

start_time = time.time()
# Grid size and number of clusters
N = 9
K = 5

# Generate a 9x9 grid of integers from 0 to 4
np.random.seed(3)
grid = np.random.randint(0, 5, size=(N, N))

# Parameters
lambda_onehot = 1.5
lambda_similarity = 4.0
lambda_balance = 0.05

# Helper function for variable naming
def var(i, j, k):
    return f"x_{i}_{j}_{k}"

# Initialize QUBO dictionary
Q = {}

# 1. One-hot encoding constraint: Each cell must belong to exactly one cluster
for i in range(N):
    for j in range(N):
        for k1 in range(K):
            Q[(var(i, j, k1), var(i, j, k1))] = Q.get((var(i, j, k1), var(i, j, k1)), 0) + lambda_onehot
            for k2 in range(k1 + 1, K):
                Q[(var(i, j, k1), var(i, j, k2))] = Q.get((var(i, j, k1), var(i, j, k2)), 0) + 2 * lambda_onehot

# 2. Similarity reward between adjacent cells with similar values (exponential decay)
for i in range(N):
    for j in range(N):
        for ni, nj in [(i+1, j), (i, j+1)]:  # only right and down neighbors
            if 0 <= ni < N and 0 <= nj < N:
                diff = grid[i, j] - grid[ni, nj]
                similarity = np.exp(-(diff ** 2) / 2.0)
                for k in range(K):
                    key = (var(i, j, k), var(ni, nj, k))
                    Q[key] = Q.get(key, 0) - lambda_similarity * similarity

# 3. Cluster size balancing: penalize unevenly large clusters
ideal_size = (N * N) / K
for k in range(K):
    for i1 in range(N):
        for j1 in range(N):
            for i2 in range(N):
                for j2 in range(N):
                    key = (var(i1, j1, k), var(i2, j2, k))
                    Q[key] = Q.get(key, 0) + lambda_balance

    for i in range(N):
        for j in range(N):
            Q[(var(i, j, k), var(i, j, k))] -= 2 * lambda_balance * ideal_size

    # Optional constant term (not used by dimod)
    Q[(f"const_{k}", f"const_{k}")] = Q.get((f"const_{k}", f"const_{k}"), 0) + lambda_balance * (ideal_size ** 2)

# 4. Add small random bias to break symmetry
for i in range(N):
    for j in range(N):
        for k in range(K):
            Q[(var(i, j, k), var(i, j, k))] += np.random.uniform(-0.3, 0.3)

# Build and solve the QUBO using Simulated Annealing
bqm = dimod.BinaryQuadraticModel.from_qubo(Q)
sampler = dimod.SimulatedAnnealingSampler()
sampleset = sampler.sample(bqm, num_reads=100)
best_sample = sampleset.first.sample

# Build cluster assignment grid
cluster_grid = np.zeros((N, N), dtype=int)
for i in range(N):
    for j in range(N):
        for k in range(K):
            if best_sample.get(var(i, j, k), 0) == 1:
                cluster_grid[i, j] = k

# --- Visualization ---
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Original values
axes[0].imshow(grid, cmap='gray', vmin=0, vmax=4)
axes[0].set_title("Original Grid Values")
for i in range(N):
    for j in range(N):
        axes[0].text(j, i, str(grid[i, j]), ha='center', va='center', color='red', fontsize=8)

# Cluster assignments
axes[1].imshow(cluster_grid, cmap='tab10', vmin=0, vmax=K-1)
axes[1].set_title("Cluster Assignment")
for i in range(N):
    for j in range(N):
        axes[1].text(j, i, str(cluster_grid[i, j]), ha='center', va='center', color='black', fontsize=8)

end_time = time.time()
print(f"total time: {(end_time - start_time):.4f}")

plt.tight_layout()
plt.show()

# Print final cluster assignment array
print("Cluster assignments:")
print(cluster_grid)

