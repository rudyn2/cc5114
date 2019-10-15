from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from src.genetic_algorithm.Individual import Individual
from src.genetic_algorithm.PopManager import PopManager


class GeneticEngine:

    def __init__(self, population_size: int,
                 gen_size: int,
                 mutation_rate: float,
                 gen_mutation_rate: float,
                 elitism_rate: float,
                 fitness_function,
                 individual_generator,
                 tournament_selection_proportion: float = 0.2,
                 max_iter: int = 100):

        self.population_size = population_size
        self.gen_size = gen_size
        self.mutation_rate = mutation_rate
        self.gen_mutation_rate = gen_mutation_rate
        self.elitism_rate = elitism_rate
        self.fitness = fitness_function
        self.individual_generator = individual_generator
        self.tournament_selection_proportion = tournament_selection_proportion
        self.max_iter = max_iter

        self.summary = None
        self.population = None

    def tournament_selection(self) -> Individual:
        """
        Performs tournament selection over the population.
        :return:
        """
        random_choices = self.population.select_random(self.tournament_selection_proportion)
        temp_manager = PopManager(self.fitness)
        temp_manager.fit(random_choices)
        return temp_manager.select_best()

    def run(self):
        """
        Runs the Genetic Algorithm.
        :return:
        """

        # Create the initial population
        self.population = PopManager(self.fitness)
        self.population.create(individual_class=self.individual_generator,
                               population_size=100,
                               length_gen=self.gen_size)

        # Calculate the scores
        scores = self.population.calculate_scores()
        generation = 1

        # Summary dict
        summary = defaultdict(list)
        for _ in range(self.max_iter):

            # Elitism
            new_generation = self.population.select_rate_best(self.elitism_rate)

            for _ in range(int(self.population_size * self.mutation_rate)):
                parent_a = self.tournament_selection()
                parent_b = self.tournament_selection()

                # Applying cross over
                new_individual = parent_a.cross_over(parent_b)

                # Applying mutation
                r = np.random.rand()
                if r <= self.mutation_rate:
                    new_individual.mutate(self.gen_mutation_rate)

                new_generation.append(new_individual)
            self.population.fit(new_generation)
            # ------------------------------------------------------

            scores = self.population.calculate_scores()

            # Summary of the progress
            summary['best_scores'].append(np.max(scores))
            summary['worst_scores'].append(np.min(scores))
            summary['mean_scores'].append((np.mean(scores)))
            print(f"Generation {generation}, best score: {summary['best_scores'][-1]}, "
                  f"worst score: {summary['worst_scores'][-1]}, mean score: {summary['mean_scores'][-1]}")
            generation += 1

        print(f"Best individual of last generation: {self.population.select_best()}")
        self.summary = summary

    def plot_evolution(self):
        """
        Generates a plot of the generation's evolution.
        """

        generations = list(range(1, len(self.summary['mean_scores'])+1))
        plt.plot(generations, self.summary['best_scores'], label='best')
        plt.plot(generations, self.summary['worst_scores'], label='worst')
        plt.plot(generations, self.summary['mean_scores'], label='mean')
        plt.xlabel('# Generation')
        plt.ylabel('Score')
        plt.legend()
        plt.show()



