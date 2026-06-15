"""
evolution/population.py

Population management for the evolutionary outer loop.
Handles tournament selection, elitism, and offspring generation.
"""

from __future__ import annotations

import random
from typing import List, Optional, Tuple

from creature.morphology import Genome
from evolution.genome_ops import (
    add_random_bone,
    crossover,
    mutate_bone_size,
    mutate_density,
    mutate_joint_params,
    remove_random_leaf_bone,
    toggle_motor,
)

# (genome, fitness) pair type
Individual = Tuple[Genome, Optional[float]]


class Population:
    """Manages a list of (Genome, fitness) pairs.

    Args:
        size:         Target population size.
        seed_genomes: Initial genomes to populate generation 0.
    """

    def __init__(self, size: int, seed_genomes: List[Genome]) -> None:
        self.size = size
        self.individuals: List[Individual] = []

        # Fill population by cycling through seeds and copying
        for i in range(size):
            seed = seed_genomes[i % len(seed_genomes)]
            g = seed.copy()
            g.name = f"{seed.name}_gen0_{i}"
            self.individuals.append((g, None))

    def set_fitness(self, index: int, fitness: float) -> None:
        genome, _ = self.individuals[index]
        self.individuals[index] = (genome, fitness)

    def get_genome(self, index: int) -> Genome:
        return self.individuals[index][0]

    def get_fitness(self, index: int) -> Optional[float]:
        return self.individuals[index][1]

    def sorted_by_fitness(self) -> List[Individual]:
        """Return individuals sorted by fitness (highest first).
        Individuals with None fitness are placed last.
        """
        def _key(ind: Individual) -> float:
            return ind[1] if ind[1] is not None else -1e9
        return sorted(self.individuals, key=_key, reverse=True)

    def best(self) -> Individual:
        return self.sorted_by_fitness()[0]

    def mean_fitness(self) -> float:
        fitnesses = [f for _, f in self.individuals if f is not None]
        return sum(fitnesses) / len(fitnesses) if fitnesses else 0.0

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def select_parent(self, k: int = 3) -> Genome:
        """Tournament selection: pick k individuals, return the best."""
        candidates = random.sample(self.individuals, min(k, len(self.individuals)))
        best = max(candidates, key=lambda ind: ind[1] if ind[1] is not None else -1e9)
        return best[0].copy()

    # ------------------------------------------------------------------
    # Next generation
    # ------------------------------------------------------------------

    def next_generation(
        self,
        generation: int,
        elitism: int = 2,
        mutation_rate: float = 0.4,
        crossover_rate: float = 0.3,
        tournament_k: int = 3,
    ) -> "Population":
        """Produce the next generation via elitism + offspring.

        Args:
            generation:     Current generation number (used for naming).
            elitism:        Number of top individuals kept unchanged.
            mutation_rate:  Probability of applying each mutation operator.
            crossover_rate: Probability of crossover vs. clonal mutation.
            tournament_k:   Tournament selection pool size.

        Returns:
            A new Population with size individuals and fitness=None.
        """
        sorted_inds = self.sorted_by_fitness()
        new_individuals: List[Individual] = []

        # Elitism: carry top N forward unchanged
        for i in range(min(elitism, len(sorted_inds))):
            g = sorted_inds[i][0].copy()
            g.name = f"{g.name.split('_gen')[0]}_gen{generation}_elite{i}"
            new_individuals.append((g, None))

        # Fill the rest with offspring
        while len(new_individuals) < self.size:
            parent_a = self.select_parent(tournament_k)

            if random.random() < crossover_rate:
                # Crossover with a second parent
                parent_b = self.select_parent(tournament_k)
                child = crossover(parent_a, parent_b)
                if child is None:
                    child = parent_a.copy()
            else:
                child = parent_a.copy()

            # Apply mutations stochastically
            child = _apply_mutations(child, mutation_rate)
            child.name = f"{child.name.split('_gen')[0]}_gen{generation}_{len(new_individuals)}"
            child.metadata["generation"] = generation
            child.metadata["fitness"] = None
            new_individuals.append((child, None))

        new_pop = Population.__new__(Population)
        new_pop.size = self.size
        new_pop.individuals = new_individuals[: self.size]
        return new_pop


# ---------------------------------------------------------------------------
# Mutation application
# ---------------------------------------------------------------------------

_MUTATION_OPS = [
    mutate_bone_size,
    mutate_joint_params,
    add_random_bone,
    remove_random_leaf_bone,
    toggle_motor,
    mutate_density,
]


def _apply_mutations(genome: Genome, mutation_rate: float) -> Genome:
    """Apply each mutation operator with probability mutation_rate."""
    g = genome
    for op in _MUTATION_OPS:
        if random.random() < mutation_rate:
            result = op(g)
            if result is not None:
                g = result
    return g
