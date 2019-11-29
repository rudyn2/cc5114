import random
from arboles import *

# un AST es un arbol que representa un programa, la idea aqui es tener
# un generador, que pueda generar ASTs aleatorios
class AST:
    def __init__(self, allowed_functions, allowed_terminals, prob_terminal=0.3):
        # las funciones (nodos en nuestro caso) que nuestro programa puede tener
        self.functions = allowed_functions
        # los terminales admitidos en nuestro programa. Numeros por ejemplo
        self.terminals = allowed_terminals
        # para no tener un arbol infinitamente profundo, existe una posibilidad
        # de que, a pesar de que ahora toque hacer otro sub arbol, que se ignore
        # eso y se ponga un terminal en su lugar.
        self.prob = prob_terminal

    # esta funcion ya la hemos visto, nos permite llamar al AST como si fuera
    # una funcion. max_depth es la produndidad que queremos tenga el arbol
    def __call__(self, max_depth=10):
        # aqui tenemos una funcion auxiliar. Nos permitira hacer esto recursivo
        def create_rec_tree(depth):
            # si `depth` es mayor a 0, nos toca crear un sub-arbol
            if depth > 0:
                # elegimos una funcion aleatoriamente
                node_cls = random.choice(self.functions)
                # aqui iremos dejando los argumentos que necesita la funcion
                arguments = []
                # para cada argumento que la funcion necesite...
                for _ in range(node_cls.num_args):
                    # existe un `prob` probabilidad de que no sigamos creando
                    # sub-arboles y lleguemos y cortemos aqui para hacer
                    # un nodo terminal
                    if random.random() < self.prob:
                        arguments.append(create_rec_tree(0))
                    else:
                        # la otra es seguir creando sub-arboles recursivamente
                        arguments.append(create_rec_tree(depth - 1))
                
                # `arguments` es una lista y los nodos necesitan argumentos
                # asi que hacemos "unpacking" de la lista
                return node_cls(*arguments)
            else:
                # si `depth` es 0 entonces creamos un nodo terminal con
                # alguno de los terminales permitidos que definimos inicialmente
                return TerminalNode(random.choice(self.terminals))

        # llamamos a la funcion auxiliar para crear un arbol de profundidad `max_depth`
        return create_rec_tree(max_depth)