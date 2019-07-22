"""Author: Brandon Trabucco, Copyright 2019"""


import argparse
import best_first.config as args
import tensorflow as tf
from best_first import load_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="image.jpg")
    cmd_args = parser.parse_args()

    vocab, parts_of_speech, decoder = load_model()
    optimizer = tf.keras.optimizers.Adam(lr=args.learning_rate)

    ckpt = tf.train.Checkpoint(decoder=decoder, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(
        ckpt, args.checkpoint_dir, max_to_keep=2)
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print("Restoring model from {}".format(ckpt_manager.latest_checkpoint))

    image = tf.keras.applications.inception_v3.preprocess_input(
        tf.image.resize(
            tf.image.decode_jpeg(
                tf.io.read_file(cmd_args.image),
                channels=3
            ),
            [args.image_height, args.image_width]
        )
    )

    image_model = tf.keras.applications.InceptionV3(
        include_top=False,
        weights='imagenet'
    )

    image_features_extract_model = tf.keras.Model(
        image_model.input,
        image_model.layers[-1].output
    )

    batch_features = image_features_extract_model(tf.expand_dims(image, 0))
    batch_features = tf.reshape(
        batch_features,
        [1, -1, batch_features.shape[3]]
    )

    words = tf.constant([[2, 3]])
    slot = None
    for i in range(args.max_caption_length):

        pointer_logits, tag_logits, word_logits = decoder([batch_features, words])
        slot = tf.argmax(pointer_logits[0], output_type=tf.int32).numpy()
        new_word = tf.argmax(word_logits[0], output_type=tf.int32).numpy()

        print(vocab.ids_to_words(tf.constant(words)))
        print(vocab.ids_to_words(tf.constant(new_word)))
        print(slot)

        if slot == tf.size(words[0]).numpy() - 1:
            break
        base_list = words.numpy().tolist()[0]
        words = tf.constant([base_list[:(slot + 1)] + [new_word] + base_list[(slot + 1):]])

    print("{}".format(tf.strings.reduce_join(
        vocab.ids_to_words(words[0]), separator=" ").numpy()))
