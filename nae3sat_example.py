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
import dimod

from dimod.generators import random_nae3sat
from qdeepsdk import QDeepHybridSolver

# Define problems
num_variables = 75
rho_list = [2.1, 3.0]  # the clause-to-variable ratio

# Create directory for plots
os.makedirs(os.path.join(os.path.dirname(__file__), "plots"), exist_ok=True)

# Initialize the qdeepsdk solver and set authentication token
solver = QDeepHybridSolver()
solver.token = "your-auth-token"  # set your authentication token here


# For each clause-to-variable ratio, generate and solve the problem repeatedly
for rho in rho_list:
    print(f"\nCreating an NAE3SAT problem with rho={rho} and N={num_variables}")
    num_clauses = round(num_variables * rho)
    bqm = random_nae3sat(num_variables, num_clauses, seed=42)

    # Convert the binary quadratic model (BQM) to a QUBO dictionary and offset.
    Q_dict, offset = bqm.to_qubo()
    # Create a dense QUBO matrix
    Q_matrix = np.zeros((num_variables, num_variables))
    for (i, j), value in Q_dict.items():
        Q_matrix[int(i), int(j)] = value

    print("Solving QUBO problem using QDeepHybridSolver...")

    # Simulate multiple reads by solving repeatedly and collecting energies
    energies = []
    num_reads = 5  # number of independent solution attempts
    for _ in range(num_reads):
        print(_)
        try:
            resp = solver.solve(Q_matrix)
            # The response is expected to be a dictionary with a "QdeepHybridSolver" key
            result = resp.get("QdeepHybridSolver", {})
            energies.append(result.get("energy", np.nan))
        except Exception as e:
            print(f"Solver error: {e}")
            energies.append(np.nan)

    energies = np.array(energies)

    # Plot energy distributions
    plt.figure(rho * 100 + 1, figsize=(9, 4))
    counts, bins, patches = plt.hist(
        energies,
        bins=20,
        alpha=0.7,
        label="QDeepHybridSolver",
    )
    mean_energy = np.average(energies[np.isfinite(energies)])
    plt.axvline(
        mean_energy,
        linestyle="dashed",
        linewidth=1,
        color=patches[0].get_facecolor() if patches else "black",
        label=f"Mean: {mean_energy:.2f}",
    )
    plt.xlabel("Energy")
    plt.ylabel("Count")
    plt.title(f"$\\rho={rho}$, $N={num_variables}$")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig("./plots/rho_{}_energies.png".format(int(rho * 100)))

print("\nResults saved under the plots folder.\n")
