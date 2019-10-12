"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
import numpy as np
import pickle as pkl
import nltk
import os
import argparse
from collections import defaultdict
from best_first import load_tagger, load_parts_of_speech
from best_first.vocabulary import Vocabulary
from best_first.insertion import Insertion
from best_first.orderings import BestFirstOrdering
from best_first.orderings import RandomWordOrdering
from best_first.orderings import RandomPositionOrdering
from best_first.orderings import ForwardSequentialOrdering
from best_first.orderings import BackwardSequentialOrdering


def load_captions(
        caption_files,
        max_caption_length
):
    tagger = load_tagger()
    inner_words, inner_tags, inner_frequencies = [], [], defaultdict(int)
    for inner_path in caption_files:
        this_file_words, this_file_tags = [], []
        with tf.io.gfile.GFile(inner_path) as this_file:
            for line in this_file.readlines():
                tokens = nltk.word_tokenize(line.strip().lower())[:(max_caption_length - 1)]
                this_file_words.append(tokens)
                this_file_tags.append(list(zip(*tagger.tag(tokens)))[1])
        inner_words.append(this_file_words)
        inner_tags.append(this_file_tags)
        for this_file_line in this_file_words:
            for this_file_word in this_file_line:
                inner_frequencies[this_file_word] += 1
    return inner_words, inner_tags, inner_frequencies


def create_vocabulary(
        inner_frequencies,
        min_word_frequency,
        vocab_file
):
    sorted_words, sorted_frequencies = list(zip(*sorted(
        inner_frequencies.items(), key=(lambda x: x[1]), reverse=True)))
    split = 0
    for split, frequency in enumerate(sorted_frequencies):
        if frequency < min_word_frequency:
            break
    reverse_vocab = ("<pad>", "<unk>", "<start>", "<end>") + sorted_words[:(split + 1)]
    with tf.io.gfile.GFile(vocab_file, "w") as this_file:
        this_file.write("\n".join(reverse_vocab))
    return Vocabulary(reverse_vocab, unknown_word="<unk>", unknown_id=1)


def get_ordering(
        inner_vocab,
        inner_parts_of_speech,
        ordering_type,
        max_violations
):
    if ordering_type == "best_first":
        return BestFirstOrdering(
            word_weight=1.0 / inner_vocab.size().numpy(),
            tag_weight=1.0 / inner_parts_of_speech.size().numpy(),
            max_violations=max_violations)
    elif ordering_type == "forward_sequential":
        return ForwardSequentialOrdering(max_violations=max_violations)
    elif ordering_type == "backward_sequential":
        return BackwardSequentialOrdering(max_violations=max_violations)
    elif ordering_type == "random_word":
        return RandomWordOrdering(max_violations=max_violations)
    elif ordering_type == "random_position":
        return RandomPositionOrdering(max_violations=max_violations)
    else:
        return RandomWordOrdering(max_violations=max_violations)


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Create a part of speech tagger")
    parser.add_argument("--caption_feature_folder", type=str, default="./caption_features")
    parser.add_argument("--caption_folder", type=str, default="./captions")
    parser.add_argument("--max_caption_length", type=int, default=13)
    parser.add_argument("--ordering_type", type=str, default="best_first")
    parser.add_argument("--max_violations", type=int, default=0)
    parser.add_argument("--min_word_frequency", type=int, default=1)
    parser.add_argument("--vocab_file", type=str, default="vocab.txt")
    args = parser.parse_args()

    tf.io.gfile.makedirs(args.caption_feature_folder)
    all_caption_files = tf.io.gfile.glob(os.path.join(args.caption_folder, "*.txt"))
    all_words, all_tags, word_frequencies = load_captions(all_caption_files, args.max_caption_length)
    vocab = create_vocabulary(word_frequencies, args.min_word_frequency, args.vocab_file)
    parts_of_speech = load_parts_of_speech()
    ordering = get_ordering(vocab, parts_of_speech, args.ordering_type, args.max_violations)

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
