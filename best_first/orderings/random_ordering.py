"""Author: Brandon Trabucco, Copyright 2019"""


from best_first.orderings.ordering import Ordering
from collections import defaultdict
import numpy as np


class RandomOrdering(Ordering):

    def __init__(
            self,
            max_violations=0
    ):
        self.random_state = defaultdict(lambda: np.random.uniform())
        Ordering.__init__(self, max_violations=max_violations)

    def score(
            self,
            words,
            tags,
    ):
        return [self.random_state[w] for w in words]
