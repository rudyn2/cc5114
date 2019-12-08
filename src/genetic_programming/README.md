# Genetic Algorithms

The code hosted here contains the implementation of genetic programming algorithms, some solved problems using this 
technique and also some examples in order to illustrate its use. This project use the genetic engine implemented
in the last homework with the aim to keep the same genetic algorithm. The innovation here is the creation of a new
type of individual, a tree individual. These trees, also called Abstract Syntax Tree, are a representation of the 
abstract syntactic structure of source code written in a programming language. To reduce the problem, this project
uses this approach but for just mathematical operations as the addition, subtraction, division and multiplication.
So now, the algorithm is the same but the individuals are semantically different. This leads the problem to do a good
implementation of the cross over and mutation operations.

### DISCLAIMER

I am not a native english speaker so maybe you might find some spelling mistakes (sorry). I am trying to improve 
my english abilities, thanks for your patience.

Table of Contents
================

* [Cloning the repository](#cloning)
* [Usage](#usage)
* [Example execution](#example-execution)
* [Implementation](#implementation)
    * [Crossover](#cross-over)
    * [Mutation](#mutation)
* [The little hack](#the-little-hack)
* [Exercises](#exercises)
    * [Finding a number with repetition](#find-a-number-with-repetition)
    * [Finding a number with repetition and constrains](#find-a-number-with-repetition-and-constrains)
    * [Finding a number without repetition and constrains](#find-a-number-without-repetition-and-constrains)
    * [Symbolic Regression](#symbolic-regression)
    * [Division](#division)
* [Results and Analysis](#results-and-analysis)

# Cloning the repository


To clone the repository please follow the next steps (you must have the git software in your system).

1) Choose a folder where the repository will be stored and go to it.
2) Run ``git clone https://github.com/rudyn2/cc5114.git``
3) Ready.

# Usage

In order to use the genetic algorithm engine you must provide the basics of GA implementations: a fitness function and a 
individual specification. The example showed above implements the minimal code needed to execute a genetic programming
algorithm to find a mathematical expression which results is some target number(see the description in the 
sections below).


````
from arboles import *
from AstIndividualGenerator import AstIndividualGenerator
from GeneticEngine import GeneticEngine
from fitness.FindNumberFitness_v3 import FindNumberFitness

max_depth = 5
allowed_functions = [AddNode, SubNode, MultNode]
allowed_terminals = [-2, -1, 0, 1, 2]

ast_gen = AstIndividualGenerator(allowed_functions=allowed_functions, allowed_terminals=allowed_terminals)
find_number_fitness = FindNumberFitness(target_number=65346)
ga = GeneticEngine(population_size=100,
                   gen_size=max_depth,
                   mutation_rate=0.6,
                   gen_mutation_rate=0.3,
                   elitism_rate=0.6,
                   fitness_function=regression_fitness,
                   individual_generator=ast_gen,
                   max_iter=5)
ga.run(mode='max_iter')
ga.plot_evolution()
````

Maybe it is now a little bit confusing but keep reading and it'll be much clear.

## Example execution

In order to give to the first user a fast understanding of the code, he can review the examples/find_number.py code
available in the examples folder. To execute it:

1) Open a terminal.
2) Change your directory to the src/genetic_algorithm/examples folder.
3) Execute in the console: ``python find_number.py``.
4) Ready.

The example will execute a Genetic Programming Algorithm to solve the same problem of the usage example.

# UML Class

As usual, the logic semantic segmentation of the classes for a genetic algorithm are: individuals (that can mutate, 
they can reproduce them and are the building blocks of some population), fitness functions (takes and individual
and can evaluate them accordingly to some user-defined metric), and finally, the genetic 
algorithm engine (executes the logical steps of a genetic algorithm).

![UML](https://raw.githubusercontent.com/rudyn2/cc5114/master/src/genetic_algorithm/resources/UML_class.png)


# Implementation


To understand better the logic behind the genetic engine used in this project please go to 
[Genetic Algorithms](https://github.com/rudyn2/cc5114/tree/master/src/genetic_algorithm "Genetic Algorithms") folder, 
there you can find a much nicer explanation of how this works. 

As it was said before, the innovation is the creation of a new type of individuals: the Abstract Syntax Trees. The
semantic representation suggest that this new kind of individual must be a child class of the Individual class defined
before, this class will be called *AstIndividual*. The individual defined in the [Genetic Algorithms](https://github.com/rudyn2/cc5114/tree/master/src/genetic_algorithm "Genetic Algorithms")
section demands to provide a implementation for the cross over, mutation, get gen and fit gen operations. We will see 
how each of them were solved but before that, it is important to say that the AstIndividual class is based in the 
**arboles** and **abstract_syntax_tree** libraries that have been provided by the teaching staff, these classes
have functionality to operate and generate trees performing random creation, replacing nodes, evaluation, copy, among
others.

First of all, each AstIndividual contains 5 attributes. The first three are the **allowed_functions**, **allowed_terminals** and
**max_depth**, this attributes must be specified in the constructor. Both, allowed functions and allowed terminals
must be child classes of the *Node* class defined in the **arboles** library. This is the way that the user has to 
specify which functions can the tree use (e.g. addition is specified as a SumNode) and which will be the terminals
(e.g. a set of numbers or variables). The remained 2 attributes are the tree and tree_gen. The first one is an
instance of the Node class that has to be generated using the **abstract_syntax_tree** class (this library provides
an amazing tree generator), and the second one is is the generator used to generate the tree attribute, it is saved
because will be need to perform the mutation operation. 

## Cross over
As we know, the cross over operation has to be perform to a tree. To do it, several steps are needed. These steps
are shown below.

````
# First: a copy of the first parent is created
copy_parent_1 = parent_1.copy()

# Second: a node from this copy is randomly selected
cross_over_node = random.choice(copy_parent_1.serialize())
print(f"Selected node: {cross_over_node}")

# Third: a node from the second parent is randomly selected
second_parent_sub_tree = random.choice(parent_2.serialize()).copy()
print(f"Selected subtree : {second_parent_sub_tree}")

# Fourth: the second sub tree is replaced in the selected node from the first parent
copy_parent_1.replace(second_parent_sub_tree)
````

This implementation assumes that we have a parent_1 and parent_2 instances (Node's class instances) and ensures
that the cross-over between them will produce a new individual totally independent of their parents.

## Mutation

The mutation is a little different. The steps are shown below.

````
# Step 1: Choose some node from the copy, this node will be mutated.
node_to_mutate = random.choice(self.tree.serialize())

# Step 2: The mutation is generated with max depth equal to the node to mutate depth
mutation = self.tree_gen(max_depth=node_to_mutate.get_depth())
assert node_to_mutate.get_depth() >= mutation.get_depth()

# Step 3: The mutation is performed in the selected node
node_to_mutate.replace(mutation)
````
It is important to mention that this operation is performed into the tree and makes effect to this same tree.

# The little hack

From the implementation of the genetic algorithm we know the most important things to specify are the 
fitness function and the individual generator. In the case of the individual generator, the requirements says
that this must be a "class" whose instances are some individual that can be evaluated using the specified fitness
function. The problem is that this class must be such as the only parameter that can receive is the gen size or
length gen so the PopManager can create new individuals without issues. The thing is that the AstIndividual 
also needs the allowed functions and allowed terminals. To address this problem the AstIndividualGenerator class is
created using a "proxy" approach. To create an instance of this class we specify the same that we need for each new
random AstIndividual but this instances have the __call__ method which imitates the __init__ method that is used in the
engine, returning at this way new AstIndividuals, with the difference that to create them just the max_depth is needed 
and the other parameters were specified in the AstIndividualGenerator constructor. So, instead of using classes to 
generate new instances using the init method, we use a instance of a class to generate new instances of other classes
but using the call method.


# Exercises

This genetic algorithm was tested for three kind of problems.

## Find a number with repetition


This problem is about finding a mathematical expression which its evaluation result in some target number.
For example, if the target number is 10, a optimal solution may be 5 + 5 or 2 * 5. No constrains are needed.

Using the implementation for the AST's we just need to specify a fitness function to evaluate them. So,
given some tree, it is desired to minimize its error that will be defined as the absolute value of the difference
of the result of the tree and the expected result. But as we know, the GA searches individuals with the highest scores 
(tries to maximize), so instead of minimize the error, we maximize the inverse of the error that is mathematically 
equal. The key line to implement it is shown below.

````
fitness = -(abs(self.target_number - tree_to_eval.eval(feed_dict={'values':[]})))
````

The full implementation can be revisited in: 
[Find a nuber with repetition: Fitness function implementation](https://github.com/rudyn2/cc5114/blob/master/src/genetic_programming/fitness/FindNumberFitness_v0.py "Find a number with repetition")

## Find a number with repetition and constrains


This problem is the same as the before but adds a constrain. Now, it is also desired trees more smalls. So, the problem
is turned into a multiple objective optimization problem. To solved problem a new component is introduced to the 
fitness function: the depth of the tree. As before, instead of minimize the depth of the tree we maximize the
inverse of the depth. To calculate the depth a new method was extended to the Node class of the **arboles**
library. This method finds the depth efficiently using recursion.

Finally, the implementation in python is something like this,

````
fitness = -(abs(self.target_number - tree_to_eval.eval(feed_dict={'values':[]})) + tree_to_eval.get_depth())
````

The full implementation can be revisited in: 
[Find a nuber with repetition and constrains: Fitness function implementation](https://github.com/rudyn2/cc5114/blob/master/src/genetic_programming/fitness/FindNumberFitness_v1.py "Find a number with repetition")

## Find a number without repetition and constrains

The thing is getting harder. This problem is the same as before BUT also needs that the solution trees don't have
repetitions. To address this new difficult the fitness function needs to be adapted. The key idea is to punish 
the trees that have repetitions. So, we'll call **pureness** the property of being a tree that don't have repeated
terminals. Therefore, the adaption is clear. Each tree not pure will be punished, such as each not pure 
tree score will be decreased some factor. To implement this, we need something that can tell us if a particular tree
is pure. Then, the ***is_pure*** method is implemented in the node class of the *arboles* library, this method
takes a node, serialize it, extracts just the terminal nodes, and checks if there is repeated terminal nodes. Using
this method, the key line in the adapted fitness function is shown below.

````
punishment = 1 if tree_to_eval.is_pure(feed_dict={'values': []}) else 100
fitness = -(abs(self.target_number - tree_to_eval.eval(feed_dict={'values': []})) + tree_to_eval.get_depth())*punishment
````

The full implementation can be revisited in: 
[Find a nuber without repetition and constrains: Fitness function implementation](https://github.com/rudyn2/cc5114/blob/master/src/genetic_programming/fitness/FindNumberFitness_v2.py "Find a number without repetition")

## Symbolic regression

This problem is a little more complex but like always finding a good fitness function is enough. What is desired here
is finding an equation that fits to some set of data. For example if we have the function $$ x^2 + x + 6 $$ we could
generate some dataset of 100 pairs of (x, f(x)) points, then, if we passed this to our GP algorithm this will return
the ***symbolic*** function $$ x^2 + x + 6 $$. In order to do this we will need an implementation for *variables*, 
such as we can be able to have trees that we can evaluate over some data. This variables will be the "x" of our 
objective function. Then, the implementation of the variables will be seen before. 

### Variable implementation

The approach to have trees with variables is simple. As we know that a variable is a kind of terminal node, we will
modify the TerminalNode class defined in the *arboles* library so it can have string values. All we need is a way
to differentiate when a terminal node value is string or float so we can evaluate them different. The key code for the
eval method of the terminal node is here:

````
# If the value is a string we return the value stored in the feed_dict, otherwise we return the value (it is assumed
# it is a number)
try:
    return feed_dict[self.value] if isinstance(self.value, str) else self.value
except KeyError:
    raise KeyError(f"El diccionario no posee un valor para la variable {self.value}.")
````

As we can see, a feed_dict is needed. This feed_dict must be (obviously) a dictionary with some structure constrains.
This constrains are simple. If we have a terminal node with a value of "x" (it means is a variable), then at the moment
of evaluate it the feed_dict *must* have the "x" key stored, otherwise that terminal node can't be evaluated.

### Fitness implementation

Using this variable implementation we can define a fitness function. This fitness function just iterates over some
set of data and calculates the absolute difference between the target values and the generated values, accumulates it 
and finally returns it. The meaning of this is like finding the discrete integral of the absolute error. As before, the
GP can maximize and we want to minimize the error, then we maximize the inverse of the error (final score). 

````
difference = 0
    for idx in range(len(self.target_data['values'])):
        # build the feed dict with all the variables
        feed_dict = {}
        for var in self.target_data.keys():
            if var == 'values':
                continue
            feed_dict[var] = self.target_data[var][idx]
        difference += abs(tree_to_eval.eval(feed_dict)-self.target_data['values'][idx])
````            
As we can see, some constrains needs to be taken in count to execute this fitness function, in particular, to the
target data (specified in the fitness constructor). This constrains are the following: It *must* have a key called
"values" at which the objective function values are stored as a list. All the other keys, *must* 
be the arguments (like x, y, z or other thing, the name doesn't matter). Also, the values of each key *must*
be an iterable of same length, so each point can be generated (otherwise will be more than arguments than values
representing them). This way the feed_dict can represent each (x, y, z, ..., f(x, y, z, ...)) point. 

This implementation allows the searching of arbitrary multi-variable real functions.



## Division

The *arboles* library provides a lot of internal nodes (functions) like addition, subtraction, multiplication and 
maximum but we would like to have the division operation. The problem with this is that if we have '0' terminals
a ZeroDivisionError can occur. To address this problem a simple approach is used. We implement the division as the other
operations, letting it raise errors and as we know which error will happen we can catch it later.

### Fitness implementation

To illustrate the use of this approach we will use the new "DivNode" fo the find this number problem. The fitness
function is shown below.

Initialization of the fitness function and individual generator:

````
allowed_functions = [AddNode, SubNode, MultNode, DivNode]
allowed_terminals = [25, 7, 8, 100, 4, 2]

find_number_fitness = FindNumberFitness(target_number=65346)
ast_gen = AstIndividualGenerator(allowed_functions=allowed_functions, allowed_terminals=allowed_terminals)
````

Fragment of the definition of the eval() method in the FindNumberFitness implementation:

````
try:
    return -(abs(self.target_number - tree_to_eval.eval(feed_dict={'values': []})) + tree_to_eval.get_depth())
except ZeroDivisionError:
    return -10**12
````

As we can see, if the tree that is being evaluated has a division by zero the ZeroDivisionError wil be caught and 
a really bad fitness is returned (so that malicious trees can be detected and ignored).

# Results and Analysis

This section shows the results of the execution of the solution for the problems mentioned before. It is important to 
say that given the volatility of the GP algorithm sometimes there are really bad results and to see better the 
fitness evolution the y-scale was set up to a logarithm scale to improve the understanding of the evolution.

## Finding a number with repetition

The target number was set up to 65346. The allowed functions are addition, subtraction, multiplication and maximum. The 
max depth for the tree generator was set up to 5. The allowed terminals are {25, 7, 8, 100, 4, 2}.
The fitness function that provides the restrictions for this problem is the defined in FindNumberFitness_v0.py. 
The parameters of the genetic algorithm are the following:

- Population size: 200
- Mutation rate: 0.8
- Elitism rate: 0.7
- Maximum number of iterations: 10

After a couple of executions, the following results gives a 0 error. It means that its evaluation
gives exactly the expected number: 65346 with a depth of 12 at the generation number 32.

Result:

(max({((7 + 25) * (25 * 25)), max({(max({(7 * (25 * 100)), max({(((7 + 25) * (2 * 8)) + 25), (max({((7 + 25) * (25 * 100)), max({(((7 + 25) * (25 * 100)) + 25), (2 - 4)})}) - 4)})}) + 25), (2 * (100 - 7))})}) + ((max({2, 2}) * (max({2, 2}) - 100)) * (25 + (25 + 25))))

The fitness evolution is shown below.

![Fitness evolution](https://raw.githubusercontent.com/rudyn2/cc5114/master/src/genetic_programming/experiments_results/_v0/4.png)

## Finding a number with repetition and constrains

As it was said before, we now introduce the depth constrain defined in the FindNumberFitness_v1.py file. We also
modify a little bit the parameters reducing the diversity setting the mutation rate to 0.4 and the max depth to 4, so 
there is more probability of having little trees.The other parameters remains as the previous problem.

At the third execution of the algorithm we found after 4 iterations the following expression:

((((4 * 4) - (25 + 100)) * (25 - 8)) - (((100 * 4) * (4 - 25)) * (4 - (4 - 8))))

This expression represents a tree with depth of 4 and its evaluation gives an amazing result of 65347. This is a better
tree with even less depth than the previous approach. 

The fitness evolution is shown below.

![Fitness evolution](https://raw.githubusercontent.com/rudyn2/cc5114/master/src/genetic_programming/experiments_results/_v1/3.png)

You can see this and another results in folder _v0 located at the experimental results section: [Find a nuber without repetition and constrains: Fitness function implementation](https://github.com/rudyn2/cc5114/blob/master/src/genetic_programming/fitness/FindNumberFitness_v2.py "Find a number without repetition")

## Finding a number without repetition and constrains

Now the difficult increases. The problem is the same as the previous one but now we want trees with unique terminal
nodes. The implementation of this problem was described in previous section. After 16 executions of the 
genetic algorithm this tree was found.

((7 * ((2 + 100) - 8)) * (4 * 25))

The evaluation gives 65800, a really close result. The depth is 4, not too big and most important it has now
the property that we were searching: the uniques terminal nodes. 6 of the 6 terminals allowed were used on this tree
to give an amazing result. The main problem with this exercise was that the GP algorithm was executed a lot of times
in order to achieve this result. It is important to say that in the process of testing the most repeated result
was this tree: ((25 * 100) * (4 * 7)) that gives a number of 70000 with a depth of 2 (not really bad). 

The fitness evolution is shown below.

![Fitness evolution](https://raw.githubusercontent.com/rudyn2/cc5114/master/src/genetic_programming/experiments_results/_v0/4.png)

## Symbolic regression

The allowed functions used in this problem were the addition, subtraction and multiplication. The allowed terminals
are the integer numbers from -10 to 10 and 21 times the "x" symbol (to represent a expression). The target function 
is x^2 + x + 6, from which the data to feed the algorithm was generated. The parameters used for the GP algorithm are 
the following.

- Population size: 100
- Mutation rate: 0.6
- Elitism rate: 0.6
- Maximum depth of generated trees: 4
- Max number of iterations: ? (Fitness criterion used)

The best result found was the following which gives this expression:

((x - 6) + (x * x))

With a depth of 2 this expression matches exactly to the expected one. The criterion to stop the algorithm
was set up when the best tree has a fitness of 0 (remember that the scores are in the
negative domain). Then, the algorithm took 98 generations to find a solution (a couple of minutes
of processing). This was executed several times and this results shows the tree with minimal depth. During this
experiment other solutions were found but they had greater depth. 

The fitness evolution is shown below.

![Fitness evolution](https://raw.githubusercontent.com/rudyn2/cc5114/master/src/genetic_programming/experiments_results/_v0/4.png)

As we can see, an elitism rate greater than zero makes that the best individual fitness never goes down. Also
the difference between the worst and the best individuals suggest that there is enough diversity at each 
generation. This genetic diversity is beneficial to the algorithm exploration.

## Division

To show an example of the division node working the *find this number* problem is solved using this kind of 
operation. So, the allowed functions will be the same as before: addition, subtraction, multiplication and this time 
also the division. The target number is 65346 and the parameters can be reviewed in the 
***Finding a number with repetition*** section. 
