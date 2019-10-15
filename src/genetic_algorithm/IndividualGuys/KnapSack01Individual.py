from src.genetic_algorithm.Individual import Individual
import numpy as np


class KnapSack01Individual(Individual):

    def __init__(self, length_gen: int):
        super().__init__(length_gen=length_gen)
        self.length_gen = length_gen
        self.selected_elements = np.random.randint(0, 2, size=length_gen)

    def mutate(self, gen_mutation_rate):

        mask = np.random.randint(0, self.length_gen, size=int(gen_mutation_rate * self.length_gen))
        for idx in mask:
            self.selected_elements[idx] = 0 if self.selected_elements[idx] == 1 else 0

    def cross_over(self, other: Individual) -> Individual:

        # Selects a random point
        separator = np.random.randint(0, self.length_gen)
        other_gen = other.get_gen()
        left_gen_half = self.selected_elements[:separator]
        right_gen_half = other_gen[separator:]
        new_bits = np.concatenate([left_gen_half, right_gen_half])
        child = KnapSack01Individual(len(new_bits))
        child.fit_gen(new_bits)
        return child

    def get_gen(self):
        return self.selected_elements

    def fit_gen(self, gen):
        assert len(gen) == self.length_gen, "The gen must be of same size that the empty individual"
        self.selected_elements = gen

    def __repr__(self):
        return str(self.selected_elements)