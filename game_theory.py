import dimod

# Terrain and line-of-sight costs
terrain = [4, 2, 5,
           3, 1, 3,
           6, 2, 4]
line_of_sight = [3, 1, 3,
                 4, 2, 2,
                 5, 1, 3]
costs = [t + l for t, l in zip(terrain, line_of_sight)]

num_cells = 9
desired_path_len = 5
lagrange = 10.0  # Penalty multiplier

# --- Step 1: Construct the QUBO dictionary ---
Q = {}

# Objective term: Minimize sum of (terrain + line-of-sight)
for i in range(num_cells):
    Q[(f'x{i}', f'x{i}')] = costs[i]

# Constraint: (sum x_i - desired_len)^2 = sum x_i + 2*sum x_ixj - 2*desired_len*sum x_i + const
# Linear part: (-2 * desired_len + 1) * x_i
for i in range(num_cells):
    var = f'x{i}'
    Q[(var, var)] = Q.get((var, var), 0.0) + lagrange * (-2 * desired_path_len + 1)

# Quadratic part: 2 * x_i * x_j
for i in range(num_cells):
    for j in range(i + 1, num_cells):
        Q[(f'x{i}', f'x{j}')] = Q.get((f'x{i}', f'x{j}'), 0.0) + lagrange * 2

# Start and end cell constraints: (1 - x)^2 = 1 - 2x + x^2 => linear coeff += -2 + 1 = -1
for endpoint in [0, 8]:
    var = f'x{endpoint}'
    Q[(var, var)] = Q.get((var, var), 0.0) + lagrange * (-2 + 1)

# --- Step 2: Build the BQM from QUBO ---
bqm = dimod.BinaryQuadraticModel.from_qubo(Q)

# --- Step 3: Solve the QUBO ---
sampler = dimod.ExactSolver()
sampleset = sampler.sample(bqm)

# --- Step 4: Output ---
best = sampleset.first
selected_path = [i for i in range(num_cells) if best.sample[f'x{i}'] == 1]

# Show raw QUBO energy and true cost
true_cost = sum(costs[i] for i in selected_path)

print("Best path indices:", selected_path)
print("QUBO total energy (includes penalties):", best.energy)
print("Actual path cost (terrain + line-of-sight only):", true_cost)