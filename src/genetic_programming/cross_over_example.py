import abstract_syntax_tree
from arboles import *
random.seed(42)

ast_generator = abstract_syntax_tree.AST(allowed_functions=[AddNode, SubNode], allowed_terminals=[1, 2, 3])

parent_1 = ast_generator(max_depth=3)
parent_2 = ast_generator(max_depth=3)
print(f"Parent 1: {parent_1}")
print(f"Parent 2: {parent_2}")

# Simulating the process of cross over
# First: a copy of the first parent is created
copy_parent_1 = parent_1.copy()

# Second: a node from this copy is randomly selected
cross_over_node = random.choice(copy_parent_1.serialize())
print(f"Selected node: {cross_over_node}")

# Third: a node from the second parent is randomly selected
second_parent_sub_tree = random.choice(parent_2.serialize())
print(f"Selected subtree : {second_parent_sub_tree}")

# Fourth: the second sub tree is replaced in the selected node from the first parent
copy_parent_1.replace(second_parent_sub_tree)
print(f"Child: {copy_parent_1}")
