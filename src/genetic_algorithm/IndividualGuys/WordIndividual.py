import numpy as np

from src.genetic_algorithm.Individual import Individual


class WordIndividual(Individual):
    def __init__(self, length_gen: int):
        super().__init__(length_gen)
        self.length_gen = length_gen
        self.dictionary = self.create_dictionary()
        self.word = self.create_random_word(length_gen, self.dictionary)

    def cross_over(self, other) -> Individual:
        """
        Performs single point cross over with another WordIndividual. The first part of the child is from this
        chromosome and the second part from the other WordIndividual.
        :param other:                           WordIndividual to mate.
        :return:                                Child WordIndividual instance.
        """
        other_word = other.get_word()
        r = np.random.randint(0, len(other_word))
        child_chromosome = self.word[:r] + other_word[r:]
        child = WordIndividual(len(child_chromosome))
        child.fit_gen(child_chromosome)
        return child

    def mutate(self, gen_mutation_rate: float):
        """
        Performs the mutation of a word. The proportion of gen that will change depends on the gen_mutation_rate
        parameter.
        :param gen_mutation_rate:               Proportion of genes to mutate.
        """
        assert 0 <= gen_mutation_rate <= 1, "The gen mutation rate must be between 0 and 1."
        original_word = list(self.word)
        to_mutate_index = np.random.randint(0, len(original_word), size=int(gen_mutation_rate*len(original_word)))
        for idx in to_mutate_index:
            original_word[idx] = self.dictionary[np.random.randint(0, len(self.dictionary))]
        self.word = "".join(original_word)

    def get_word(self):
        return self.word

    def fit_gen(self, word: list):
        self.word = "".join(word)

    @staticmethod
    def create_dictionary():
        lower_letters = [chr(letter_code) for letter_code in range(65, 91)]
        upper_letters = [chr(letter_code) for letter_code in range(97, 123)]
        lower_letters.extend(upper_letters)
        return lower_letters

    @staticmethod
    def create_random_word(size: int, dictionary: list):
        random_word_list = np.random.choice(dictionary, size=size)
        return "".join(random_word_list)

    def __repr__(self):
        return self.word
