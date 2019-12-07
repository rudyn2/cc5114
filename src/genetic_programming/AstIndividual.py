from Individual import Individual
from abstract_syntax_tree import AST
import random


class AstIndividual(Individual):

    def __init__(self, *, allowed_functions, allowed_terminals, **kwargs):

        try:
            self.max_depth = kwargs['max_depth'] if 'max_depth' in kwargs.keys() else kwargs['length_gen']
        except KeyError:
            raise KeyError("You must provide a max depth parameter for the Individual")

        super().__init__(self.max_depth)
        self.allowed_functions = allowed_functions
        self.allowed_terminals = allowed_terminals
        self.tree_gen = AST(allowed_functions=allowed_functions, allowed_terminals=allowed_terminals)
        self.tree = self.tree_gen(max_depth=self.max_depth)

    def cross_over(self, other):
        parent_2 = other.get_gen()

        # Simulating the process of cross over
        # First: a copy of the first parent is created
        copy_parent_1 = self.tree.copy()

        # Second: a node from this copy is randomly selected
        cross_over_node = random.choice(copy_parent_1.serialize())

        # Third: a node from the second parent is randomly selected
        second_parent_sub_tree = random.choice(parent_2.serialize()).copy()

        # Fourth: the second sub tree is replaced in the selected node from the first parent
        cross_over_node.replace(second_parent_sub_tree)

        # Extra step: Create a AstIndividual instance from this new child
        child = AstIndividual(max_depth=self.length_gen, allowed_functions=self.allowed_functions,
                              allowed_terminals=self.allowed_terminals)
        child.fit_gen(copy_parent_1)
        return child

    def mutate(self, gen_mutation_rate):

        # Step 1: Choose some node from the copy, this node will be mutated.
        node_to_mutate = random.choice(self.tree.serialize())

        # Step 2: The mutation is generated with max depth equal to the node to mutate depth
        mutation = self.tree_gen(max_depth=node_to_mutate.get_depth())
        assert node_to_mutate.get_depth() >= mutation.get_depth()

        # Step 3: The mutation is performed in the selected node
        node_to_mutate.replace(mutation)

    def get_gen(self):
        return self.tree

    def fit_gen(self, gen):
        self.tree = gen

    def __repr__(self):
        return str(self.tree)


