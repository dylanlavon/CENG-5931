import numpy as np
import dimod
import time

def create_qubo(sudoku_grid):
    num_rows, num_cols = sudoku_grid.shape
    num_numbers = num_rows
    num_variables = num_rows * num_cols * num_numbers

    Q = np.zeros((num_variables, num_variables))
    c = 0.0

    # Each cell contains exactly one number
    for i in range(num_rows):
        for j in range(num_cols):
            for k in range(num_numbers):
                Q[i * num_cols * num_numbers + j * num_numbers + k, i * num_cols * num_numbers + j * num_numbers + k] = 1

    # Each number appears exactly once in each row and column
    for i in range(num_rows):
        for k in range(num_numbers):
            for j1 in range(num_cols):
                for j2 in range(j1 + 1, num_cols):
                    Q[i * num_cols * num_numbers + j1 * num_numbers + k, i * num_cols * num_numbers + j2 * num_numbers + k] = 2

    for j in range(num_cols):
        for k in range(num_numbers):
            for i1 in range(num_rows):
                for i2 in range(i1 + 1, num_rows):
                    Q[i1 * num_cols * num_numbers + j * num_numbers + k, i2 * num_cols * num_numbers + j * num_numbers + k] = 2

    # Each 2x2 sub-grid contains each number exactly once
    for s in range(num_rows // 2):
        for t in range(num_cols // 2):
            for k in range(num_numbers):
                for i1 in range(2 * s, 2 * s + 2):
                    for j1 in range(2 * t, 2 * t + 2):
                        for i2 in range(2 * s, 2 * s + 2):
                            for j2 in range(2 * t, 2 * t + 2):
                                if (i1, j1) != (i2, j2):
                                    Q[i1 * num_cols * num_numbers + j1 * num_numbers + k, i2 * num_cols * num_numbers + j2 * num_numbers + k] = 2

    # Fix known values
    for i in range(num_rows):
        for j in range(num_cols):
            if sudoku_grid[i, j] != 0:
                for k in range(num_numbers):
                    if k + 1 != sudoku_grid[i, j]:
                        Q[i * num_cols * num_numbers + j * num_numbers + k, i * num_cols * num_numbers + j * num_numbers + k] = 1000

    return Q

def solve_qubo(Q):
    bqm = dimod.BQM.from_qubo(Q)
    sampler = dimod.SimulatedAnnealingSampler()
    response = sampler.sample(bqm, num_reads=1000)

    # Convert sampleset to sudoku grid
    best_sample = response.record[0]
    print(best_sample)
    return sudoku_grid

start_time = time.time()

# Initial grid
sudoku_grid = np.array([[1, 0, 3, 0],
                        [2, 0, 0, 0],
                        [0, 0, 0, 4],
                        [0, 1, 0, 0]])

Q = create_qubo(sudoku_grid)
solution = solve_qubo(Q)

end_time = time.time()
print(f"QUBO solver elapsed time: {(end_time - start_time):.4f} seconds")