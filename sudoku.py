import numpy as np
import dimod
from dwave.system import DWaveSampler

def create_qubo(sudoku_grid):
    num_rows, num_cols = sudoku_grid.shape
    num_numbers = num_rows
    num_variables = num_rows * num_cols * num_numbers

    Q = np.zeros((num_variables, num_variables))
    c = np.zeros(num_variables)

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
                        c[i * num_cols * num_numbers + j * num_numbers + k] = 1000
                    else:
                        c[i * num_cols * num_numbers + j * num_numbers + k] = -2000

    return Q, c

def solve_qubo(Q):
    bqm = dimod.BQM.from_qubo(Q)
    sampler = DWaveSampler()
    response = sampler.sample(bqm)
    sampleset = response.sampleset

    # Convert sampleset to sudoku grid
    sudoku_grid = np.zeros((4, 4), dtype=int)
    for sample in sampleset.record:
        grid = np.zeros((4, 4), dtype=int)
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    if sample[i * 4 * 4 + j * 4 + k] == 1:
                        grid[i, j] = k + 1
        sudoku_grid = grid

    return sudoku_grid

# Initial grid
sudoku_grid = np.array([[1, 0, 3, 0],
                        [2, 0, 0, 0],
                        [0, 0, 0, 4],
                        [0, 1, 0, 0]])

Q, c = create_qubo(sudoku_grid)
solution = solve_qubo(Q)

print("Solution:")
print(solution)