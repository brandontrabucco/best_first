"""Author: Brandon Trabucco, Copyright 2019"""


import pickle as pkl
import tensorflow as tf
from collections import namedtuple


Insertion = namedtuple("Insertion", [
    "words", "tags",
    "next_word", "next_tag", "slot", "violations"])


def load_vocabulary(
        vocab_file="vocab.txt"
):
    with tf.io.gfile.GFile(vocab_file, "r") as f:
        return Vocabulary(f.read().strip().lower().split("\n"),
                          unknown_word="<unk>", unknown_id=1)


parts_of_speech = [
    "<pad>", "<unk>", ".", "CONJ", "DET", "ADP", "PRT", "PRON",
    "ADV", "NUM", "ADJ", "VERB", "NOUN"]


def load_parts_of_speech():
    return Vocabulary(
        parts_of_speech, unknown_word="<unk>", unknown_id=1)


def load_tagger(
        tagger_file="tagger.pkl"
):
    with tf.io.gfile.GFile(tagger_file, 'rb') as f:
        return pkl.loads(f.read())


def create_hash(
        reverse_vocab,
        unknown_word="<unk>",
        unknown_id=1
):
    words, ids = tf.constant(reverse_vocab), tf.range(len(reverse_vocab))
    words_to_ids_hash = tf.lookup.StaticHashTable(
        initializer=tf.lookup.KeyValueTensorInitializer(
            keys=words, values=ids), default_value=tf.constant(unknown_id))
    ids_to_words_hash = tf.lookup.StaticHashTable(
        initializer=tf.lookup.KeyValueTensorInitializer(
            keys=ids, values=words), default_value=tf.constant(unknown_word))
    return words_to_ids_hash, ids_to_words_hash


class Vocabulary(object):

    def __init__(
            self,
            *args,
            **kwargs
    ):
        words_to_ids_hash, ids_to_words_hash = create_hash(*args, **kwargs)
        self.words_to_ids_hash = words_to_ids_hash
        self.ids_to_words_hash = ids_to_words_hash

    def size(
            self
    ):
        return self.words_to_ids_hash.size()

    def words_to_ids(
            self,
            keys
    ):
        return self.words_to_ids_hash.lookup(keys)

    def ids_to_words(
            self,
            keys
    ):
        return self.ids_to_words_hash.lookup(keys)
