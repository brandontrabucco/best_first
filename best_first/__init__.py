"""Author: Brandon Trabucco, Copyright 2019"""


import best_first.config as args
import tensorflow as tf
import pickle as pkl
from best_first.vocabulary import Vocabulary
from best_first.best_first_decoder import BestFirstDecoder


def load_vocabulary():
    if not tf.io.gfile.exists(args.vocab_file):
        from best_first.backend.process_captions import process_captions
        process_captions()
    with tf.io.gfile.GFile(args.vocab_file, "r") as f:
        reverse_vocab = f.read().strip().lower().split("\n")
        return Vocabulary(
            reverse_vocab,
            unknown_word="<unk>",
            unknown_id=1)


def load_parts_of_speech():
    return Vocabulary(
        args.parts_of_speech,
        unknown_word="<unk>",
        unknown_id=1)


def load_tagger():
    if not tf.io.gfile.exists(args.tagger_file):
        from best_first.backend.create_tagger import create_tagger
        create_tagger()
    with tf.io.gfile.GFile(args.tagger_file, 'rb') as f:
        return pkl.loads(f.read())


def load_model():
    vocab = load_vocabulary()
    parts_of_speech = load_parts_of_speech()
    return vocab, parts_of_speech, BestFirstDecoder(
        vocab.size().numpy(),
        args.word_embedding_size,
        parts_of_speech.size().numpy(),
        args.tag_embedding_size,
        args.num_heads,
        args.attention_hidden_size,
        args.dense_hidden_size,
        args.num_layers,
        args.hidden_size,
        args.output_size)
