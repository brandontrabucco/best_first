"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf


def create_hash(
        reverse_vocab,
        unknown_word="<unk>",
        unknown_id=1
):
    words = tf.constant(reverse_vocab)
    ids = tf.range(len(reverse_vocab))
    words_to_ids_hash = tf.lookup.StaticHashTable(
        initializer=tf.lookup.KeyValueTensorInitializer(
            keys=words,
            values=ids
        ),
        default_value=tf.constant(unknown_id)
    )
    ids_to_words_hash = tf.lookup.StaticHashTable(
        initializer=tf.lookup.KeyValueTensorInitializer(
            keys=ids,
            values=words
        ),
        default_value=tf.constant(unknown_word)
    )
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

    def size(self):
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
