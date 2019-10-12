"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
import numpy as np
import os
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Process images into InceptionV3 features")
    parser.add_argument("--image_feature_folder", type=str, default="./image_features")
    parser.add_argument("--image_height", type=int, default=299)
    parser.add_argument("--image_width", type=int, default=299)
    parser.add_argument("--image_folder", type=str, default="./images")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    tf.io.gfile.makedirs(args.image_feature_folder)
    image_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
    image_features_extract_model = tf.keras.Model(
        image_model.input, image_model.layers[-1].output)

    def process_image(data_path):
        return data_path, tf.keras.applications.inception_v3.preprocess_input(
            tf.image.resize(
                tf.image.decode_jpeg(
                    tf.io.read_file(data_path),
                    channels=3), [args.image_height, args.image_width]))

    image_dataset = tf.data.Dataset.from_tensor_slices(
        tf.io.gfile.glob(os.path.join(args.image_folder, "*.jpg")))
    image_dataset = image_dataset.map(
        process_image,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    image_dataset = image_dataset.batch(args.batch_size)
    image_dataset = image_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    for image_path, image in image_dataset:
        batch_features = image_features_extract_model(image)
        batch_features = tf.reshape(
            batch_features, [batch_features.shape[0], -1, batch_features.shape[3]])

        for data, path in zip(batch_features, image_path):
            np.save(os.path.join(
                args.image_feature_folder,
                os.path.basename(path.numpy().decode("utf-8"))), data.numpy())
