"""Author: Brandon Trabucco, Copyright 2019"""


import best_first.config as args
import tensorflow as tf
import pickle as pkl
import nltk
import os
from collections import defaultdict
from best_first import load_tagger, load_parts_of_speech
from best_first.vocabulary import Vocabulary
from best_first.recovery_samples import recovery_samples


def process_captions():
    print("Processing captions")
    tf.io.gfile.makedirs(args.caption_feature_folder)

    all_caption_files = tf.io.gfile.glob(os.path.join(args.caption_folder, "*.txt"))
    tagger = load_tagger()

    all_words = []
    all_tags = []
    word_frequencies = defaultdict(int)
    for caption_path in all_caption_files:
        words = []
        tags = []
        with tf.io.gfile.GFile(caption_path) as f:
            for line in f.readlines():
                tokens = nltk.word_tokenize(line.strip().lower())[:(args.max_caption_length - 1)]
                words.append(tokens)
                tags.append(list(zip(*tagger.tag(tokens)))[1])
        all_words.append(words)
        all_tags.append(tags)
        for line in words:
            for word in line:
                word_frequencies[word] += 1

    sorted_words, sorted_frequencies = list(zip(*sorted(
        word_frequencies.items(),
        key=(lambda x: x[1]), reverse=True)))
    split = 0
    for split, frequency in enumerate(sorted_frequencies):
        if frequency < args.min_word_frequency:
            break
    reverse_vocab = ("<pad>", "<unk>", "<start>", "<end>") + sorted_words[:split]
    with tf.io.gfile.GFile(args.vocab_file, "w") as f:
        f.write("\n".join(reverse_vocab))

    vocab = Vocabulary(reverse_vocab, unknown_word="<unk>", unknown_id=1)
    parts_of_speech = load_parts_of_speech()
    for caption_path, words, tags in zip(
            all_caption_files,
            all_words,
            all_tags):

        samples = []
        for i in range(len(words)):
            word_ids = vocab.words_to_ids(tf.constant(words[i])).numpy()
            tag_ids = parts_of_speech.words_to_ids(tf.constant(tags[i])).numpy()
            samples.extend(
                recovery_samples(
                    word_ids,
                    tag_ids,
                    (1.0 / len(reverse_vocab)),
                    (1.0 / len(args.parts_of_speech))))

        sample_path = os.path.join(
            args.caption_feature_folder,
            os.path.basename(caption_path) + ".pkl")
        with tf.io.gfile.GFile(sample_path, "wb") as f:
            f.write(pkl.dumps(samples))
