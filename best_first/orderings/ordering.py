"""Author: Brandon Trabucco, Copyright 2019"""


from best_first.insertion import Insertion
from abc import ABC, abstractmethod
import numpy as np


class Ordering(ABC):

    def __init__(
            self,
            max_violations=0
    ):
        self.max_violations = max_violations

    def expand(
            self,
            words,
            tags,
            left_words=np.zeros([], dtype=np.int32),
            left_tags=np.zeros([], dtype=np.int32),
            slots=np.zeros([], dtype=np.int32),
            violations=0
    ):
        if not isinstance(words, np.ndarray):
            words = np.array(words)
        if not isinstance(tags, np.ndarray):
            tags = np.array(tags)
        possible_orderings = []
        if words.size == 0:
            return possible_orderings

        removal_index = np.argmin(self.score(words, tags))
        for position, word, tag in enumerate(zip(words, tags)):
            order_violations = (
                violations if position == removal_index else violations + 1)

            if order_violations <= self.max_violations:
                split_words = np.append(words[:position], words[position + 1:])
                split_tags = np.append(tags[:position], tags[position + 1:])
                candidate_words = np.append([word], left_words)
                candidate_tags = np.append([tag], left_tags)
                candidate_slots = np.append(
                    np.where(position < slots, slots - 1, slots), [position])

                insertion_index = np.argmax(self.score(candidate_words, candidate_tags))
                insertion_word = candidate_words[insertion_index]
                insertion_tag = candidate_tags[insertion_index]
                insertion_slot = candidate_slots[insertion_index]
                insertion = Insertion(
                    words=np.concatenate([[2], split_words, [3]]),
                    tags=np.concatenate([[1], split_tags, [1]]),
                    next_word=insertion_word,
                    next_tag=insertion_tag,
                    slot=insertion_slot,
                    violations=order_violations)

                possible_orderings.append(insertion)
                possible_orderings.extend(
                    self.expand(
                        split_words,
                        split_tags,
                        left_words=candidate_words,
                        left_tags=candidate_tags,
                        slots=candidate_slots,
                        violations=order_violations))

        return possible_orderings

    @abstractmethod
    def score(
            self,
            words,
            tags,
    ):
        return NotImplemented
