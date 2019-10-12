"""Author: Brandon Trabucco, Copyright 2019"""


import best_first.config as args
import tensorflow as tf
import pickle as pkl
import os
import sys


def create_sequence_example(
        inner_image_bytes,
        inner_sample
):
    image_feature = tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[inner_image_bytes]))
    next_word_feature = tf.train.Feature(
        int64_list=tf.train.Int64List(value=[inner_sample.next_word]))
    next_tag_feature = tf.train.Feature(
        int64_list=tf.train.Int64List(value=[inner_sample.next_tag]))
    slot_feature = tf.train.Feature(
        int64_list=tf.train.Int64List(value=[inner_sample.slot]))
    violations_feature = tf.train.Feature(
        int64_list=tf.train.Int64List(value=[inner_sample.violations]))
    words_feature_list = tf.train.FeatureList(feature=[tf.train.Feature(
        int64_list=tf.train.Int64List(value=[word])) for word in inner_sample.words])
    tags_feature_list = tf.train.FeatureList(feature=[tf.train.Feature(
        int64_list=tf.train.Int64List(value=[tag])) for tag in inner_sample.tags])
    return tf.train.SequenceExample(
        context=tf.train.Features(
            feature={"image": image_feature,
                     "next_word": next_word_feature, "next_tag": next_tag_feature,
                     "slot": slot_feature, "violations": violations_feature}),
        feature_lists=tf.train.FeatureLists(
            feature_list={"words": words_feature_list, "tags": tags_feature_list}))


if __name__ == "__main__":

    tf.io.gfile.makedirs(os.path.dirname(args.tfrecord_folder))
    all_caption_files = sorted(tf.io.gfile.glob(os.path.join(
        args.caption_feature_folder, "*.txt.pkl")))
    all_image_files = sorted(tf.io.gfile.glob(os.path.join(
        args.image_feature_folder, "*.jpg.npy")))

    shard = 0
    num_samples_so_far = 0
    writer = tf.io.TFRecordWriter(os.path.join(
        args.tfrecord_folder, "{:013d}.tfrecord".format(shard)))

    for caption_file, image_file in zip(all_caption_files, all_image_files):
        with tf.io.gfile.GFile(image_file, "rb") as f:
            image_bytes = f.read()

        with tf.io.gfile.GFile(caption_file, "rb") as f:
            samples = pkl.loads(f.read())

        for sample in samples:
            if num_samples_so_far >= args.num_samples_per_shard:
                sys.stdout.flush()
                writer.close()
                shard += 1
                num_samples_so_far = 0
                writer = tf.io.TFRecordWriter(os.path.join(
                    args.tfrecord_folder, "{:013d}.tfrecord".format(shard)))

            sequence_sample = create_sequence_example(image_bytes, sample)
            writer.write(sequence_sample.SerializeToString())
            num_samples_so_far += 1

    sys.stdout.flush()
    writer.close()
