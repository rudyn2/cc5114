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


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.
    Source: https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/image_annotated_heatmap.html.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


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
    fig, ax = plt.subplots()
    heatmap(mean_scores, row_labels=pop_sizes, col_labels=mutation_rates, ax=ax,
            cmap="YlGn", cbarlabel="Mean fitness at 10th generation")
    fig.tight_layout()
    plt.show()
