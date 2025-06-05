import numpy as np
import dimod
import time

def create_qubo(n):
    # Define the distance matrix ( symmetric and diagonal=0 )
    distance_matrix = np.array([[0, 1, 2],
                               [1, 0, 3],
                               [2, 3, 0]])

    # Initialize Q and c matrices
    Q = np.zeros((n**2, n**2))
    c = np.zeros(n**2)

    # Objective function: minimize the total distance traveled
    for i in range(n):
        for j in range(n):
            if i != j:
                for k in range(n):
                    if k != i and k != j:
                        Q[i*n + j, k*n + i] = distance_matrix[i, j]
                        Q[j*n + i, i*n + k] = distance_matrix[j, i]

    # Constraints: each city is visited exactly once
    for i in range(n):
        for j in range(n):
            if i != j:
                Q[i*n + j, i*n + j] = 2

    for i in range(n):
        c[i*n + i] = -1

    # Constraints: return to the starting city
    Q[0*n + 1, 0*n + 1] += 2
    Q[1*n + 0, 1*n + 0] += 2
    Q[0*n + 2, 2*n + 2] += 2
    Q[2*n + 0, 0*n + 0] += 2

    return Q, c

# Start 1st timer
start_time = time.time()

n = 3
Q, c = create_qubo(n)
print("QUBO Matrix Q:")
print(Q)
print("\nQUBO Vector c:")
print(c)

# End 1st timer
end_time = time.time()

print(f"QUBO creation elapsed time: {(end_time - start_time):.4f} seconds")

# Start 2nd timer
start_time = time.time()

# Create the QUBO model
dict_bqm = dimod.BQM.from_qubo(Q)

# Create an exact solver
sampler_exact = dimod.ExactSolver()

# Solve the QUBO problem
sampleset = sampler_exact.sample(dict_bqm)

# Print the solution
print(sampleset)

# End 2nd timer
end_time = time.time()

print(f"QUBO solver elapsed time: {(end_time - start_time):.4f} seconds")