from AstIndividual import AstIndividual


class AstIndividualGenerator:

    def __init__(self, allowed_functions: list, allowed_terminals: list):
        self.allowed_functions = allowed_functions
        self.allowed_terminals = allowed_terminals

    def __call__(self, *args, **kwargs):
        return AstIndividual(max_depth=kwargs['max_depth'] if 'max_depth' in kwargs.keys() else kwargs['length_gen'],
                             allowed_functions=self.allowed_functions,
                             allowed_terminals=self.allowed_terminals)
