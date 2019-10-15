from src.genetic_algorithm.GeneticEngine import GeneticEngine
import numpy as np
import matplotlib.pyplot as plt


# Print iterations progress
def printProgressBar (iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar.
    Source: https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def plot_hotmap(gen_size,
                fitness_function,
                individual_generator, *,
                elitism_rate: float = 0.1,
                gen_mutation_rate: float = 0.2,
                n_iter: int = 10):
    """
    Creates a hotmap of the mean fitness after n iterations.
    :return:
    """

    pop_sizes = np.linspace(50, 1000, 10)
    mutation_rates = np.linspace(0, 1, 10)
    mean_scores = []

    total_iteration = 10*10
    iteration = 1
    for pop_size in pop_sizes:
        sub_mean_scores = []
        for mutation_rate in mutation_rates:
            new_ga = GeneticEngine(population_size=int(pop_size),
                                   gen_size=gen_size,
                                   mutation_rate=mutation_rate,
                                   gen_mutation_rate=gen_mutation_rate,
                                   elitism_rate=elitism_rate,
                                   fitness_function=fitness_function,
                                   individual_generator=individual_generator,
                                   max_iter=n_iter)
            new_ga.run(verbose=False)
            summary = new_ga.get_summary()
            sub_mean_scores.append(summary['mean_scores'][-1])
            printProgressBar(iteration, total_iteration, prefix='Progress', suffix='Complete')
            iteration += 1
        mean_scores.append(sub_mean_scores)

    mean_scores = np.array(mean_scores)
    plt.imshow(mean_scores)
    plt.colorbar()
    plt.show()
