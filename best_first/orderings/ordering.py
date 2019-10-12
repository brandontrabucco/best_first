"""Author: Brandon Trabucco, Copyright 2019"""


from best_first.caption_utils import Insertion
from abc import ABC, abstractmethod
import numpy as np


class Ordering(ABC):

    def __init__(
            self,
            max_violations=0
    ):
        self.max_violations = max_violations

    def get_orderings(
            self,
            split_words,
            split_tags,
            candidate_words,
            candidate_tags,
            candidate_slots,
            order_violations,
            closed
    ):
        orderings = self.expand(
            split_words,
            split_tags,
            left_words=candidate_words,
            left_tags=candidate_tags,
            slots=candidate_slots,
            violations=order_violations,
            closed=closed) if split_words.size > 0 else []

        insertion_index = np.argmax(self.score(candidate_words, candidate_tags))
        return orderings + [Insertion(
            words=np.concatenate([[2], split_words, [3]]),
            tags=np.concatenate([[1], split_tags, [1]]),
            next_word=candidate_words[insertion_index],
            next_tag=candidate_tags[insertion_index],
            slot=candidate_slots[insertion_index],
            violations=order_violations)]

    def expand(
            self,
            words,
            tags,
            left_words=np.zeros([0], dtype=np.int32),
            left_tags=np.zeros([0], dtype=np.int32),
            slots=np.zeros([0], dtype=np.int32),
            violations=0,
            closed=None
    ):
        orderings = []
        closed = closed if closed is not None else set()
        removal_index = np.argmin(self.score(words, tags))
        for i, (word, tag) in enumerate(zip(words, tags)):
            order_violations = violations if i == removal_index else violations + 1
            if order_violations <= self.max_violations:

                split_words = np.append(words[:i], words[i + 1:])
                if split_words.tostring() not in closed:
                    closed.add(split_words.tostring())

                    split = 0
                    while split < slots.size and slots[split] <= i:
                        split += 1

                    orderings.extend(self.get_orderings(
                        split_words,
                        np.append(tags[:i], tags[i + 1:]),
                        np.concatenate([left_words[:split], [word], left_words[split:]]),
                        np.concatenate([left_tags[:split], [tag], left_tags[split:]]),
                        np.concatenate([slots[:split], [i], slots[split:] + 1]),
                        order_violations,
                        closed))

        return orderings

    @abstractmethod
    def score(
            self,
            words,
            tags,
    ):
        return NotImplemented
