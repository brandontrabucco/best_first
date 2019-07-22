"""Author: Brandon Trabucco, Copyright 2019"""


import best_first.config as args
import tensorflow as tf
import time
from tensorboard import program
from best_first.data_loader import data_loader
from best_first import load_model


if __name__ == "__main__":
    vocab, parts_of_speech, decoder = load_model()
    optimizer = tf.keras.optimizers.Adam(lr=args.learning_rate)

    writer = tf.summary.create_file_writer(args.logging_dir)
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', args.logging_dir])
    print("Launching tensorboard at {}".format(tb.launch()))

    for iteration, batch in enumerate(data_loader()):
        start_time = time.time()

        image_path = batch["image_path"]
        image = batch["image"]
        new_word = batch["new_word"]
        new_tag = batch["new_tag"]
        slot = batch["slot"]
        words = batch["words"]
        tags = batch["tags"]
        indicators = batch["indicators"]

        with writer.as_default():
            tf.summary.experimental.set_step(iteration)
            tf.summary.text("image_path", image_path[0])
            tf.summary.text("new_word", vocab.ids_to_words(new_word[0]))
            tf.summary.text("new_tag", parts_of_speech.ids_to_words(new_tag[0]))
            tf.summary.text("slot", tf.as_string(slot[0]))
            tf.summary.text("words", tf.strings.reduce_join(vocab.ids_to_words(
                words[0, :]), separator=" "))
            tf.summary.text("tags", tf.strings.reduce_join(parts_of_speech.ids_to_words(
                tags[0, :]), separator=" "))

        def loss_function():
            pointer_logits, tag_logits, word_logits = decoder([
                image, words, tf.ones(tf.shape(image)[:2]), indicators, slot, new_tag])

            pointer_loss = tf.reduce_mean(
                tf.losses.sparse_categorical_crossentropy(
                    slot,
                    pointer_logits,
                    from_logits=True))
            tag_loss = tf.reduce_mean(
                tf.losses.sparse_categorical_crossentropy(
                    new_tag,
                    tag_logits,
                    from_logits=True))
            word_loss = tf.reduce_mean(
                tf.losses.sparse_categorical_crossentropy(
                    new_word,
                    word_logits,
                    from_logits=True))
            total_loss = (
                    args.pointer_loss_weight * pointer_loss +
                    args.tag_loss_weight * tag_loss +
                    args.word_loss_weight * word_loss)

            with writer.as_default():
                tf.summary.scalar("Pointer Loss", pointer_loss)
                tf.summary.scalar("Tag Loss", tag_loss)
                tf.summary.scalar("Word Loss", word_loss)
                tf.summary.scalar("Total Loss", total_loss)

            return total_loss

        optimizer.minimize(loss_function, decoder.trainable_variables)
        with writer.as_default():
            tf.summary.scalar("Images Per Second", args.batch_size / (
                time.time() - start_time))
