"""Author: Brandon Trabucco, Copyright 2019"""


from best_first.orderings.ordering import Ordering
import numpy as np


class ForwardOrdering(Ordering):

    def score(
            self,
            words,
            tags,
    ):
        return -np.arange(words.size)
