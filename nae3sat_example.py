# Copyright 2022 D-Wave Systems Inc.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import numpy as np
import matplotlib.pyplot as plt
import requests

from dimod.generators import random_nae3sat
from qdeepsdk import QDeepHybridSolver

# Define problems
num_variables = 75
rho_list = [2.1, 3.0]  # the clause-to-variable ratio

# Create directory for plots
plots_dir = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(plots_dir, exist_ok=True)

# Initialize QDeep Hybrid Solver
solver = QDeepHybridSolver()
solver.token = "mtagdfsplb"  # <-- Set your authentication token here
# Optionally configure quantum optimization parameters:
solver.m_budget = 50000   # Measurement budget (adjust if needed)
solver.num_reads = 10000  # Number of reads (adjust if needed)

def bqm_to_qubo_matrix(bqm):
    """Convert a Binary Quadratic Model (BQM) to a full symmetric QUBO matrix.
    
    Returns a tuple (Q, variables) where:
      - Q is an n x n NumPy array
      - variables is the list of variable names in order corresponding to Q
    """
    variables = list(bqm.variables)
    n = len(variables)
    Q = np.zeros((n, n))
    # Set linear coefficients on the diagonal.
    for i, v in enumerate(variables):
        Q[i, i] = bqm.linear[v]
    # Set quadratic coefficients (ensuring symmetry).
    for (u, v), coeff in bqm.quadratic.items():
        i = variables.index(u)
        j = variables.index(v)
        Q[i, j] = coeff
        Q[j, i] = coeff
    return Q, variables

# Generate two NAE3SAT problems with clause-to-variable ratios rho 2.1 and 3.0
for rho in rho_list:
    print(f"\nCreating an NAE3SAT problem with rho={rho} and N={num_variables}")
    num_clauses = round(num_variables * rho)
    bqm = random_nae3sat(num_variables, num_clauses, seed=42)
    
    # Convert the BQM into a QUBO matrix suitable for the hybrid solver.
    qubo_matrix, variables = bqm_to_qubo_matrix(bqm)
    
    # Solve the QUBO problem using QDeep Hybrid Solver.
    print("Sending problem to QDeep Hybrid Solver...")
    try:
        response = solver.solve(qubo_matrix)
        # Expected response structure:
        # {
        #     "QdeepHybridSolver": {
        #         "configuration": [0.0, 1.0, ...],
        #         "energy": -1.0,
        #         "time": 3.6245157718658447
        #     }
        # }
        result = response.get("QdeepHybridSolver", {})
        energy = result.get("energy")
        configuration = result.get("configuration")
        time_taken = result.get("time")
        print("Hybrid Solver Results:", result)
    except ValueError as e:
        print(f"Error: {e}")
        continue
    except requests.RequestException as e:
        print(f"API Error: {e}")
        continue
