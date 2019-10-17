# Genetic Algorithms

The code hosted here contains the implementation of a genetic algorithm engine, some utilities needed
to the engine, and also some examples in order to illustrate its use.

### DISCLAIMER

I am not a native english speaker so maybe you might find some spelling mistakes (sorry). I am trying to improve 
my english abilities, thanks for your patience.

Table of Contents
================

  * [Cloning the repository](#cloning)
  * [Usage](#usage)
    * [Example execution](#example-execution)
  * [Analysis](#analysis)
    * [Implementation](#implementation)
    * [The rocks in the way](#the-rocks-in-the-way)
    * [Results](#results)
    
Cloning the repository
===

To clone the repository please follow the next steps (you must have the git software in your system).

1) Choose a folder where the repository will be stored and go to it.
2) Run ``git clone https://github.com/rudyn2/cc5114.git``
3) Ready.

Usage
====

In order to use the genetic algorithm engine you must provide the basics of GA implementations: a fitness function and a 
individual specification. The example showed above implements the minimal code needed to execute a genetic algorithm
that found the solution for the sequence of bits problem (see the description in the sections below).


````
word_to_find = 'ElBromaS'
w_fitness = WordFitness.WordFitness(word_to_find)
ga = GeneticEngine(population_size=100,
                   gen_size=len(word_to_find),
                   mutation_rate=0.8,
                   gen_mutation_rate=0.2,
                   elitism_rate=0.1,
                   fitness_function=w_fitness,
                   individual_generator=WordIndividual.WordIndividual,
                   max_iter=20)

ga.run()
ga.plot_evolution()
````

Example execution
---

In order to give to the first user a fast understanding of the code he can review the TestGA.py code available
in the examples folder. To execute it:

1) Open a terminal.
2) Change your directory to this folder.
3) Execute in the console: ``python Example.py``.
4) Ready.

The example will load the Iris Dataset and train a neural network using some of the tools that this
repository provides.

UML Class
---

As usual, the logic semantic segmentation of the classes for a genetic algorithm are: individuals (that can mutate, 
they can reproduce them and are the building blocks of some population), fitness functions (takes and individual
and can evaluate them accordingly to some user-defined metric), and finally, the genetic 
algorithm engine (executes the logical steps of a genetic algorithm).

![UML](https://raw.githubusercontent.com/rudyn2/cc5114/master/src/genetic_algorithm/resources/UML_class.png)


Implementation
===

* Individuals: Each new individual must be created as a instance of a child class from the Individual class. The
individual class is an pseudo-abstract class that provides the specification for the actions (methods) that an individual
must be able to do, like mutation, cross-over and others. In this class the method eval(Fitness) is implemented and
are therefore it is available for all child classes. 

* Fitness: Each new fitness function definition must be created as a instance of a child class from the Fitness class. 
The Fitness class is also a pseudo-abstract class that provides the specification for the actions that a 
fitness functions must be able to do, these are basically the eval(Individual) method, that takes an individual and 
returns some score using the user-defined procedures.

* PopManager: This is an helper class. Basically, it is a manager of the population. To create a PopManager a fitness
function must be specified. The manager handles the individuals with a simple python array and can perform things like
select the best individual, select some random individuals or calculate theirs scores, all this using the provided 
fitness function. An extra ability of this manager is create a new population of individuals from scratch, 
the method "create" implements this functionality but needs an individual generator as input.

* GeneticEngine: The most important code is here. A genetic engine is a class that execute a classic genetic algorithm
given some parameters, functions and classes. The algorithm is executed as follows. 
    1) Create some random population with N individuals.
    2) Evaluate them using a fitness function. If some condition is reached then it ends.
    3) Selects some proportion of the best individuals.
    4) Selects 2N parents using the Tournament strategy and from them create N children with cross-over.
    5) For each children created, performs a mutation with some probability.
    6) Go to step 2.
    
    At the moment, the only supported criterion is a maximum number of iterations. All the parameters like the 
    probability of mutation, the size of the population, the fitness function, the individual generator, the size of 
    each chromosome, the elitism rate, and others must be provided in the constructor of the engine. Remember that the **fitness function**
    and the **individual generator** must be provided and these are the elements that will define your problem. Once
    that a GA engine is created just call the run() method, to see the evolution of the generation executes the 
    plot_evolution() method. To see in detail the meaning of each parameter please visit the class documentation.
    
* Utils: It has the plot hotmap method that given some gen size, fitness function and a individual generator can plot
a hotmap that shows the performance of the genetic engine for a combination of population sizes and mutation rates.

Exercises
====

This genetic algorithm was tested for three kind of problems.

Sequence of bits
---

This problem is about finding a sequence of bits. Each individual corresponds to a list of bits of the same size as the
specified gen size. Then, the fitness function is the total of coincidences bit by bit (e.g. 010 and 011 has 2 
coincidences). The genetic evolution for this experimented is shown in the next figure, the sequence of bits that wants
to be found is: 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0.

Word
---

This problem is a little different from the sequence of bits but here a word wants to be found. So, for each position
the amount of possible choices is bigger. This problems considers that each letter of word can be uppercase or 
lower case. As the intuition says, the individual is modeled as a string, so, a mutation will mean that some letter
will change for other letter. The fitness function corresponds to the amount of letters that are equals between the
target word and the individual. For the experiment shown below the target word is "ElBromas".

0-1 Knapsack problem
========

In the context of the cc5114 homework #2, the selected problem to solve is 0-1 Knapsack. To introduce a little bit,
The 0-1 knapsack (01KP) is an combinatorial optimization problem classified as a NP complete problem.
 
 In simple words,
the situation described is that someones has to choose between several items where each of them has an 
value and weight, and just one item can be choose per class. But, in order to find the good selection, the sum of all
 the weights selected cant be greater than some value (it is a restriction). The optimization is trying to select an 
 combination of items that maximizes the total value.
 
 For this particular problem, 5 different 
 elements can be selected:

- A box of weight 12 and value 4.
- A box of weight 2 and value 2.
- A box of weight 1 and value 2.
- A box of weight 1 and value 1.
- A box of weight 4 and value 10.

The total weight restriction is fixed in 15. 

A solution of this problem can be achieved using GA algorithms but a good model is needed. The proposed 
model is the following:

- *Individual*: Each individual has genes that can be represented as a sequence of bits. The length of this
genes are equals to the total of different items that can be choose (5). The element i of this sequence correspond to a
bit where 1 means that the item i was selected and 0 if not. So, one individual represents one selection and the 
gene describe it.

- *Fitness*: The weights and values of the items will be represented in an array of equal length than the total of 
different items, then, they will have the same length as the gen of the individuals. To compute the fitness of each 
individual first the total weights is computed calculating the dot product between the gen and the weights, so, given
the binary representation, just the selected items will be included in the sum. If the total weight is greater than the
maximum capacity (15) then the fitness is 0. Otherwise, the fitness will be the dot product between the values and the 
genes (this means the sum of values of the items selected).

In terms of code, no modifications was introduced in the engine. The classes KnapSackFitness_01 and KnapSack01Individual
represents the fitness and individual, respectively. Using this approach the results are good, and a solution is found.
A plot of the evolution per generation is shown below.

![UML](https://raw.githubusercontent.com/rudyn2/cc5114/master/src/genetic_algorithm/resources/gen_evolution.png)

Also, the hotmap for the population sizes and mutation rates is the following, the variable being analyzed is the
mean fitness score at the 10th generation.

![UML](https://raw.githubusercontent.com/rudyn2/cc5114/master/src/genetic_algorithm/resources/hotmap.png)

From this results we can extract several conclusions. In first place, the number of iterations needed to found a solution
to the problem are just a few (in the mean scenario it is approximately 5 generations). The solution found is [0 1 1 1 1]
that means that all the lighter items were selected summing a total of 15 and weight of 8, we would tend to think that
there is space that is not being taken advantage of, but, given the restrictions that doesn't matters because the total
values is maximum; as we can see, the heavier item is not being selected because it uses all the capacity and dont sum a
great value. From the hotmap we can see an interesting fact: when we increase the population size the mutation began
to don't matter. That happens because when we have a great population, the probability of having diversity is greater 
even with a slow mutation rate, but in the opposite case, when the population size is little, we need a good mutation rate
in order to achieve gen diversity over the generations. g




