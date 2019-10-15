import numpy as np

from src.genetic_algorithm.Fitness import Fitness


class PopManager:

    def __init__(self, fitness: Fitness):
        self.fitness = fitness
        self.population = None

    def fit(self, individuals: list):
        self.population = individuals

    def select_random(self, rate: float):
        """
        Selects some proportion of the population randomly.
        :param rate:                    A number between 0 and 1 with the proportion wanted.
        :return:                        A list of individuals
        """
        assert 0 <= rate <= 1
        return list(np.random.choice(self.population, size=int(rate * len(self.population))))

    def select_rate_best(self, rate: float):
        """
        Selects the "rate" percent of the best individuals by a fitness function criterion.
        :param rate:                    A number between 0 and 1
        :return:                        A list of individuals
        """
        assert 0 <= rate <= 1
        sorted_individuals = sorted(self.population, key=lambda x: self.fitness.eval(x), reverse=True)
        return sorted_individuals[:int(rate * len(self.population))]

    def select_best(self):
        """
        Selects the best individual of the given population
        :return:
        """
        return max(self.population, key=lambda x: self.fitness.eval(x))

    def calculate_scores(self):
        """
        Calculates the score per individual given some fitness function.
        :return:                        A list of scores.
        """
        return [self.fitness.eval(individual) for individual in self.population]

    def create(self, individual_class, population_size: int, **kwargs):
        """
        Creates a population. The parameter individual_class must be a child class of Individual class. The key word
        arguments must provide all the arguments needed to initialize a individual_class instance; also, must provide
        the number argument to indicate the size of the population.
        :param individual_class:        An individual class generator.
        :param population_size:         The size of the population.
        :param kwargs:                  Keywords argument to initialize and
        :return:
        """
        try:
            individuals = [individual_class(**kwargs) for _ in range(population_size)]
            self.population = individuals
        except KeyError:
            raise ValueError('You must provide a gen length for the population')


