from src.genetic_algorithm.Fitness import Fitness
from src.genetic_algorithm.IndividualGuys import WordIndividual


class WordFitness(Fitness):

    def __init__(self, target_word: str):
        super().__init__()
        self.target_word = target_word

    def eval(self, word_individual: WordIndividual):
        other_word = word_individual.get_word()
        assert len(other_word) == len(self.target_word), "Can't eval words of different length"

        count = 0
        for idx, char in enumerate(other_word):
            if char == self.target_word[idx]:
                count += 1
        return count
