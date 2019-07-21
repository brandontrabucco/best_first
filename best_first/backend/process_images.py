"""Author: Brandon Trabucco, Copyright 2019"""


import best_first.config as args
import tensorflow as tf
import numpy as np
import os


def process_images():
    print("Processing images")
    tf.io.gfile.makedirs(args.image_feature_folder)

    def process_image(data_path):
        image = tf.keras.applications.inception_v3.preprocess_input(
            tf.image.resize(
                tf.image.decode_jpeg(
                    tf.io.read_file(data_path),
                    channels=3
                ),
                [args.image_height, args.image_width]
            )
        )
        return image, data_path

    image_model = tf.keras.applications.InceptionV3(
        include_top=False,
        weights='imagenet'
    )

    image_features_extract_model = tf.keras.Model(
        image_model.input,
        image_model.layers[-1].output
    )

    all_image_files = tf.io.gfile.glob(
        os.path.join(args.image_folder, "*.jpg")
    )

    image_dataset = tf.data.Dataset.from_tensor_slices(all_image_files)
    image_dataset = image_dataset.map(
        process_image,
        num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(args.batch_size)

    for image, image_path in image_dataset:
        batch_features = image_features_extract_model(image)
        batch_features = tf.reshape(
            batch_features,
            [batch_features.shape[0], -1, batch_features.shape[3]]
        )
        for data, path in zip(batch_features, image_path):
            np.save(
                os.path.join(
                    args.image_feature_folder,
                    os.path.basename(path.numpy().decode("utf-8"))
                ),
                data.numpy()
            )
