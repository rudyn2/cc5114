from src.genetic_algorithm.Utils import plot_hotmap
from AstIndividualGenerator import AstIndividualGenerator
from fitness.FindNumberFitness_v3 import FindNumberFitness
from arboles import *
import numpy as np
from Utils import heatmap
import matplotlib.pyplot as plt
from GeneticEngine import GeneticEngine

# setting up the problem parameters
target_number = 65346
allowed_functions = [AddNode, SubNode, MultNode]
allowed_terminals = [25, 7, 8, 100, 4, 2]
max_depth = 4

# Initialization of genetic engine
find_number_fitness = FindNumberFitness(target_number=target_number)
ast_gen = AstIndividualGenerator(allowed_functions=allowed_functions, allowed_terminals=allowed_terminals)

data = plot_hotmap(gen_size=5,
                fitness_function=find_number_fitness,
                mode='max_iter',
                individual_generator=ast_gen,
                elitism_rate=0.5,
                gen_mutation_rate=0.5,
                n_iter=15,
                y_log_scale=True)

data = np.log10(-data)
pop_sizes = np.linspace(50, 500, 11)
mutation_rates = np.linspace(0, 1, 11)
pop_sizes_labels = [f'{int(size):d}' for size in pop_sizes]
mut_rates_labels = [f'{rate:.2f}' for rate in mutation_rates]
fig, ax = plt.subplots()
heatmap(data, row_labels=pop_sizes_labels, col_labels=mut_rates_labels, xlabel='Mutation rates',
        ylabel='Population sizes', ax=ax, cmap="YlGn", cbarlabel="Mean fitness at 10th generation")
plt.show()