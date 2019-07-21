"""Author: Brandon Trabucco, Copyright 2019"""


import best_first.config as args
import tensorflow as tf
import pickle as pkl
import os
import nltk


def create_tagger():
    print("Creating tagger")
    tf.io.gfile.makedirs(os.path.dirname(args.tagger_file))

    brown_tagged_sentences = nltk.corpus.brown.tagged_sents(tagset='universal')

    training_split = int(len(brown_tagged_sentences) * 0.9)
    train_sentences = brown_tagged_sentences[:training_split]
    test_sentences = brown_tagged_sentences[training_split:]

    t0 = nltk.DefaultTagger('<unk>')
    t1 = nltk.UnigramTagger(train_sentences, backoff=t0)
    t2 = nltk.BigramTagger(train_sentences, backoff=t1)
    t3 = nltk.TrigramTagger(train_sentences, backoff=t2)

    scores = [[t0.evaluate(test_sentences), t0],
        [t1.evaluate(test_sentences), t1],
        [t2.evaluate(test_sentences), t2],
        [t3.evaluate(test_sentences), t3]]

    best_score, best_tagger = max(scores, key=lambda x: x[0])
    print("Finished building tagger {0:.2f}%".format(best_score * 100))
    with tf.io.gfile.GFile(args.tagger_file, 'wb') as f:
        pkl.dump(best_tagger, f)
