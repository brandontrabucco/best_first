"""Author: Brandon Trabucco, Copyright 2019"""


import best_first.config as args
import tensorflow as tf
import numpy as np
import pickle as pkl
import nltk
import os
from collections import defaultdict
from best_first import load_tagger, load_parts_of_speech
from best_first.vocabulary import Vocabulary
from best_first.orderings.best_first_ordering import BestFirstOrdering
from best_first.orderings.random_ordering import RandomOrdering
from best_first.orderings.forward_ordering import ForwardOrdering
from best_first.orderings.backward_ordering import BackwardOrdering
from best_first.insertion import Insertion


def load_captions(
        caption_files
):
    tagger = load_tagger()
    inner_words, inner_tags, inner_frequencies = [], [], defaultdict(int)
    for inner_path in caption_files:
        this_file_words, this_file_tags = [], []
        with tf.io.gfile.GFile(inner_path) as this_file:
            for line in this_file.readlines():
                tokens = nltk.word_tokenize(line.strip().lower())[:(args.max_caption_length - 1)]
                this_file_words.append(tokens)
                this_file_tags.append(list(zip(*tagger.tag(tokens)))[1])
        inner_words.append(this_file_words)
        inner_tags.append(this_file_tags)
        for this_file_line in this_file_words:
            for this_file_word in this_file_line:
                inner_frequencies[this_file_word] += 1
    return inner_words, inner_tags, inner_frequencies


def create_vocabulary(
    inner_frequencies
):
    sorted_words, sorted_frequencies = list(zip(*sorted(
        inner_frequencies.items(), key=(lambda x: x[1]), reverse=True)))
    split = 0
    for split, frequency in enumerate(sorted_frequencies):
        if frequency < args.min_word_frequency:
            break
    reverse_vocab = ("<pad>", "<unk>", "<start>", "<end>") + sorted_words[:(split + 1)]
    with tf.io.gfile.GFile(args.vocab_file, "w") as this_file:
        this_file.write("\n".join(reverse_vocab))
    return Vocabulary(reverse_vocab, unknown_word="<unk>", unknown_id=1)


if __name__ == "__main__":

    tf.io.gfile.makedirs(args.caption_feature_folder)
    all_caption_files = tf.io.gfile.glob(os.path.join(args.caption_folder, "*.txt"))
    all_words, all_tags, word_frequencies = load_captions(all_caption_files)
    vocab = create_vocabulary(word_frequencies)

    if args.ordering_type == "best_first":
        ordering = BestFirstOrdering(
            word_weight=1.0 / vocab.size().numpy(),
            tag_weight=1.0 / len(args.parts_of_speech),
            max_violations=args.max_violations)
    elif args.ordering_type == "forward":
        ordering = ForwardOrdering(max_violations=args.max_violations)
    elif args.ordering_type == "backward":
        ordering = BackwardOrdering(max_violations=args.max_violations)
    else:
        ordering = RandomOrdering(max_violations=args.max_violations)

    parts_of_speech = load_parts_of_speech()
    for caption_path, word_examples, tag_examples in zip(
            all_caption_files, all_words, all_tags):
        samples = []
        for words, tags in zip(word_examples, tag_examples):
            word_ids = vocab.words_to_ids(tf.constant(words)).numpy()
            tag_ids = parts_of_speech.words_to_ids(tf.constant(tags)).numpy()
            samples.extend(ordering.expand(word_ids, tag_ids))
            samples.append(Insertion(
                words=np.concatenate([[2], word_ids, [3]]),
                tags=np.concatenate([[1], tag_ids, [1]]),
                next_word=0,
                next_tag=0,
                slot=1 + word_ids.size,
                violations=0))

        sample_path = os.path.join(
            args.caption_feature_folder, os.path.basename(caption_path) + ".pkl")
        with tf.io.gfile.GFile(sample_path, "wb") as f:
            f.write(pkl.dumps(samples))
