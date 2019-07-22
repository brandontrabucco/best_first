"""Author: Brandon Trabucco, Copyright 2019"""


import best_first.config as args
import tensorflow as tf
from tensorboard import program
import time
from best_first.data_loader import data_loader
from best_first import load_vocabulary
from best_first import load_parts_of_speech
from best_first.best_first_decoder import BestFirstDecoder


if __name__ == "__main__":

    vocab = load_vocabulary()
    parts_of_speech = load_parts_of_speech()

    decoder = BestFirstDecoder(
        vocab.size().numpy(),
        args.word_embedding_size,
        parts_of_speech.size().numpy(),
        args.tag_embedding_size,
        args.num_heads,
        args.attention_hidden_size,
        args.dense_hidden_size,
        args.num_layers,
        args.hidden_size,
        args.output_size)

    optimizer = tf.keras.optimizers.Adam()

    writer = tf.summary.create_file_writer(args.logging_dir)
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', args.logging_dir])
    print("Launching tensorboard at {}".format(tb.launch()))

    for iteration, batch in enumerate(data_loader()):
        break

    for iteration in range(1000):

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
            tf.summary.text("words", tf.strings.reduce_join(
                vocab.ids_to_words(words[0, :]), separator=" "))
            tf.summary.text("tags", tf.strings.reduce_join(
                parts_of_speech.ids_to_words(tags[0, :]), separator=" "))

        def loss_function():
            start_time = time.time()
            pointer_logits, tag_logits, word_logits = decoder([
                image, words, indicators, slot, new_tag])

            pointer_loss = tf.reduce_mean(
                tf.losses.sparse_categorical_crossentropy(
                    slot,
                    pointer_logits))
            tag_loss = tf.reduce_mean(
                tf.losses.sparse_categorical_crossentropy(
                    new_tag,
                    tag_logits))
            word_loss = tf.reduce_mean(
                tf.losses.sparse_categorical_crossentropy(
                    new_word,
                    word_logits))
            total_loss = pointer_loss + tag_loss + word_loss

            with writer.as_default():
                tf.summary.scalar("Pointer Loss", pointer_loss)
                tf.summary.scalar("Tag Loss", tag_loss)
                tf.summary.scalar("Word Loss", word_loss)
                tf.summary.scalar("Total Loss", total_loss)
                tf.summary.scalar(
                    "Images Per Second", args.batch_size / (time.time() - start_time))

            return total_loss

        optimizer.minimize(loss_function, decoder.trainable_variables)
