"""Author: Brandon Trabucco, Copyright 2019"""


import best_first.config as args
import tensorflow as tf
import pickle as pkl
import os
import sys


def create_tfrecords():
    print("Creating tfrecords")
    tf.io.gfile.makedirs(os.path.dirname(args.tfrecord_folder))
    if not tf.io.gfile.exists(args.image_feature_folder):
        from best_first.backend.process_images import process_images
        process_images()

    all_caption_files = tf.io.gfile.glob(os.path.join(args.caption_feature_folder, "*.txt.pkl"))
    all_image_files = tf.io.gfile.glob(os.path.join(args.image_feature_folder, "*.jpg.npy"))

    all_caption_files = sorted(all_caption_files)
    all_image_files = sorted(all_image_files)

    shard = 0
    num_samples_so_far = 0
    writer = tf.io.TFRecordWriter(
        os.path.join(
            args.tfrecord_folder,
            "{:013d}.tfrecord".format(shard)))

    for caption_file, image_file in zip(
            all_caption_files, all_image_files):

        with tf.io.gfile.GFile(caption_file, "rb") as f:
            for sample in pkl.loads(f.read()):

                if num_samples_so_far >= args.num_samples_per_shard:
                    sys.stdout.flush()
                    writer.close()
                    shard += 1
                    num_samples_so_far = 0
                    writer = tf.io.TFRecordWriter(
                        os.path.join(
                            args.tfrecord_folder,
                            "{:013d}.tfrecord".format(shard)))

                sequence_example = tf.train.SequenceExample(
                    context=tf.train.Features(
                        feature={
                            "image": tf.train.Feature(
                                bytes_list=tf.train.BytesList(value=[bytes(image_file, "utf-8")])
                            ),
                            "new_word": tf.train.Feature(
                                int64_list=tf.train.Int64List(value=[sample.new_word])
                            ),
                            "new_tag": tf.train.Feature(
                                int64_list=tf.train.Int64List(value=[sample.new_tag])
                            ),
                            "slot": tf.train.Feature(
                                int64_list=tf.train.Int64List(value=[sample.slot])
                            )
                        }
                    ),
                    feature_lists=tf.train.FeatureLists(
                        feature_list={
                            "words": tf.train.FeatureList(
                                feature=[
                                    tf.train.Feature(
                                        int64_list=tf.train.Int64List(value=[w])
                                    )
                                    for w in sample.words
                                ]
                            ),
                            "tags": tf.train.FeatureList(
                                feature=[
                                    tf.train.Feature(
                                        int64_list=tf.train.Int64List(value=[t])
                                    )
                                    for t in sample.tags
                                ]
                            ),
                        }
                    )
                )

                writer.write(sequence_example.SerializeToString())
                num_samples_so_far += 1

    sys.stdout.flush()
    writer.close()
