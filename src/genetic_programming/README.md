# Genetic Algorithms

The code hosted here contains the implementation of a genetic programming algorithms, some solved problems using this 
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

**Table of Contents**

[TOCM]

[TOC]

Cloning the repository
===

To clone the repository please follow the next steps (you must have the git software in your system).

1) Choose a folder where the repository will be stored and go to it.
2) Run ``git clone https://github.com/rudyn2/cc5114.git``
3) Ready.

Usage
====

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

Example execution
---

In order to give to the first user a fast understanding of the code, he can review the examples/find_number.py code
available in the examples folder. To execute it:

1) Open a terminal.
2) Change your directory to the src/genetic_algorithm/examples folder.
3) Execute in the console: ``python find_number.py``.
4) Ready.

The example will execute a Genetic Programming Algorithm to solve the same problem of the usage example.

UML Class
---

As usual, the logic semantic segmentation of the classes for a genetic algorithm are: individuals (that can mutate, 
they can reproduce them and are the building blocks of some population), fitness functions (takes and individual
and can evaluate them accordingly to some user-defined metric), and finally, the genetic 
algorithm engine (executes the logical steps of a genetic algorithm).

![UML](https://raw.githubusercontent.com/rudyn2/cc5114/master/src/genetic_algorithm/resources/UML_class.png)


Implementation
===

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

### Cross over
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

### Mutation

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

The little hack
===

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


Exercises
====

This genetic algorithm was tested for three kind of problems.

Find a number with repetition
---

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

Find a number with repetition and constrains
---

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

Find a number without repetition and constrains
---

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

Symbolic regression
---



Division
--





