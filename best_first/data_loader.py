"""Author: Brandon Trabucco, Copyright 2019"""


import best_first.config as args
import tensorflow as tf
import numpy as np
import os


def decode_image_backend(image_path):
    return np.load(image_path.decode('utf-8'))


def process_sequence_example(sequence_example):
    context, sequence = tf.io.parse_single_sequence_example(
        sequence_example,
        context_features = {
            "image": tf.io.FixedLenFeature([], dtype=tf.string),
            "new_word": tf.io.FixedLenFeature([], dtype=tf.int64),
            "new_tag": tf.io.FixedLenFeature([], dtype=tf.int64),
            "slot": tf.io.FixedLenFeature([], dtype=tf.int64)
        },
        sequence_features = {
            "words": tf.io.FixedLenSequenceFeature([], dtype=tf.int64),
            "tags": tf.io.FixedLenSequenceFeature([], dtype=tf.int64)
        }
    )

    image_path = context["image"]
    image = tf.numpy_function(decode_image_backend, [image_path], tf.float32)
    new_word = context["new_word"]
    new_tag = context["new_tag"]
    slot = context["slot"]
    words = sequence["words"]
    tags = sequence["tags"]

    return {
        "image_path": image_path,
        "image": image,
        "new_word": tf.cast(new_word, tf.int32),
        "new_tag": tf.cast(new_tag, tf.int32),
        "slot": tf.cast(slot, tf.int32),
        "words": tf.cast(words, tf.int32),
        "tags": tf.cast(tags, tf.int32),
        "indicators": tf.ones(tf.shape(words), dtype=tf.float32)
    }


def data_loader():
    if not tf.io.gfile.exists(
            os.path.join(args.tfrecord_folder,"0000000000000.tfrecord")):
        from best_first.backend.create_tfrecords import create_tfrecords
        create_tfrecords()

    record_files = tf.data.Dataset.list_files(
        os.path.join(args.tfrecord_folder, "*.tfrecord"))
    dataset = record_files.interleave(
        (lambda filename: tf.data.TFRecordDataset(filename).map(
            process_sequence_example,
            num_parallel_calls=tf.data.experimental.AUTOTUNE)),
        4,
        16,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.shuffle(args.queue_size)
    dataset = dataset.repeat(args.num_epochs)
    dataset = dataset.padded_batch(args.batch_size, padded_shapes={
        "image_path": [],
        "image": [64, 2048],
        "new_word": [],
        "new_tag": [],
        "slot": [],
        "words": [None],
        "tags": [None],
        "indicators": [None]
    })

    return dataset.apply(tf.data.experimental.prefetch_to_device(
        "/gpu:0", buffer_size=tf.data.experimental.AUTOTUNE))
