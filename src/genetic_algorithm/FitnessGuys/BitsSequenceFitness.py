import numpy as np

from src.genetic_algorithm.Fitness import Fitness
from src.genetic_algorithm.IndividualGuys.BitsSequenceIndividual import BitsSequenceIndividual


class BitsSequenceFitness(Fitness):

    def __init__(self, target_sequence: list):
        super().__init__()
        self.target_sequence = target_sequence

    def eval(self, individual: BitsSequenceIndividual):
        """
        Evaluates a BitsSequenceIndividual.
        :param individual:                  A BitsSequenceIndividual
        :return:                            A Score.
        """

        other_gen = individual.get_gen()
        assert len(other_gen) == len(self.target_sequence), f"Can't evaluate a sequence of bits of different length." \
                                                            f"Expected {len(self.target_sequence)} and got {len(other_gen)}"
        return np.sum(np.array(self.target_sequence) == np.array(other_gen))

