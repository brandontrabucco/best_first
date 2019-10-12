"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
import numpy as np
import os


def load_image_features(
        image_path
):
    return np.load(image_path.decode('utf-8'))


def parse_sequence_example(
        sequence_example
):
    context, sequence = tf.io.parse_single_sequence_example(
        sequence_example,
        context_features={
            "image_path": tf.io.FixedLenFeature([], dtype=tf.string),
            "next_word": tf.io.FixedLenFeature([], dtype=tf.int64),
            "next_tag": tf.io.FixedLenFeature([], dtype=tf.int64),
            "slot": tf.io.FixedLenFeature([], dtype=tf.int64),
            "violations": tf.io.FixedLenFeature([], dtype=tf.int64)},
        sequence_features={
            "words": tf.io.FixedLenSequenceFeature([], dtype=tf.int64),
            "tags": tf.io.FixedLenSequenceFeature([], dtype=tf.int64)})

    indicators = tf.ones(tf.shape(sequence["words"]), dtype=tf.float32)
    image = tf.numpy_function(
        load_image_features, [context["image_path"]], tf.float32)

    return dict(
        image=image, image_path=context["image_path"],
        next_word=tf.cast(context["next_word"], tf.int32),
        next_tag=tf.cast(context["next_tag"], tf.int32),
        slot=tf.cast(context["slot"], tf.int32),
        violations=tf.cast(context["violations"], tf.int32),
        words=tf.cast(sequence["words"], tf.int32),
        tags=tf.cast(sequence["tags"], tf.int32),
        indicators=indicators)


def parse_tf_records(
        record_files
):
    return tf.data.TFRecordDataset(record_files).map(
        parse_sequence_example,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)


def create_dataset(
        tfrecord_folder="./data/tfrecords",
        shuffle_queue_size=10000,
        num_epochs=-1,
        batch_size=32,
):
    record_files = tf.data.Dataset.list_files(os.path.join(
        tfrecord_folder, "*.tfrecord"))
    dataset = record_files.interleave(
        parse_tf_records,
        cycle_length=tf.data.experimental.AUTOTUNE,
        block_length=2,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,)
    dataset = dataset.shuffle(shuffle_queue_size)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.padded_batch(batch_size, padded_shapes={
        "image_path": [],
        "image": [64, 2048],
        "next_word": [],
        "next_tag": [],
        "slot": [],
        "violations": [],
        "words": [None],
        "tags": [None],
        "indicators": [None]})
    return dataset.prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE)
