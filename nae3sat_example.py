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

import dimod
import matplotlib.pyplot as plt
import minorminer
import numpy as np

from dimod.generators import random_nae3sat
from dwave.system import DWaveSampler, FixedEmbeddingComposite
from dwave.samplers import SimulatedAnnealingSampler
from dwave.preprocessing import ScaleComposite


# Define problems
num_variables = 75
rho_list = [2.1, 3.0]  # the clause-to-variable ratio

# Create directory for plots
os.makedirs(os.path.join(os.path.dirname(__file__), "plots"), exist_ok=True)

# Get an Advantage sampler
# adv_sampler = DWaveSampler(solver=dict(topology__type="pegasus"))

# Get an Advantage2 prototype sampler
# adv2p_sampler = DWaveSampler(solver=dict(topology__type="zephyr"))

sampler = SimulatedAnnealingSampler()

# Generate two NAE3SAT problems with clause-to-variable ratio rho 2.1 and 3.0
for rho in rho_list:
    print(f"\nCreating an NAE3SAT problem with rho={rho} and N={num_variables}")

    num_clauses = round(num_variables * rho)

    bqm = random_nae3sat(num_variables, num_clauses, seed=42)

    # Solve problem
    print(f"Sending problem to Simulated Annealing...")
    sampleset = sampler.sample(
        bqm,
        chain_strength=3,
        num_reads=100,
        auto_scale=False,
        label="Example - NAE3SAT",
    )

    # Plot energy distributions
    plt.figure(rho * 100 + 1, figsize=(9, 4))
    _, _, bar = plt.hist(
        sampleset.record.energy,
        weights=sampleset.record.num_occurrences,
        label="Simulated Annealing",
        alpha=0.7,
    )
    plt.axvline(
        np.average(
            sampleset.record.energy, weights=sampleset.record.num_occurrences
        ),
        linestyle="dashed",
        linewidth=1,
        color=bar[0].get_facecolor(),
        label=f"Mean, Simulated Annealing",
    )
    plt.xlabel("Energy")
    plt.ylabel("Count")
    plt.title(f"$\\rho={rho}$, $N={num_variables}$")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig("./plots/rho_{}_energies.png".format(int(rho * 100)))

print("\nResults saved under the plots folder.\n")
