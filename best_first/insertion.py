"""Author: Brandon Trabucco, Copyright 2019"""


from collections import namedtuple


Insertion = namedtuple("Insertion", [
    "words",
    "tags",
    "next_word",
    "next_tag",
    "slot",
    "violations"])
