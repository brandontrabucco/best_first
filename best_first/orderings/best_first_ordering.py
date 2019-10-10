"""Author: Brandon Trabucco, Copyright 2019"""


from best_first.orderings.ordering import Ordering
import numpy as np


class BestFirstOrdering(Ordering):

    def __init__(
            self,
            word_weight=0,
            tag_weight=0,
            max_violations=0
    ):
        self.word_weight = word_weight
        self.tag_weight = tag_weight
        Ordering.__init__(self, max_violations=max_violations)

    def score(
            self,
            words,
            tags,
    ):
        return np.log(words.astype(np.float32) * self.word_weight + np.exp(
            tags.astype(np.float32) * self.tag_weight))
