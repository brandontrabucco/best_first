"""Author: Brandon Trabucco, Copyright 2019"""


import best_first.config as args
import tensorflow as tf
from tensorboard import program
import time
from best_first.data_loader import data_loader
from best_first import load_vocabulary
from best_first import load_parts_of_speech
from best_first.nets.transformer import Transformer


if __name__ == "__main__":

    vocab = load_vocabulary()
    parts_of_speech = load_parts_of_speech()

    word_embedding_layer = tf.keras.layers.Embedding(vocab.size().numpy(), 1024)
    tag_embedding_layer = tf.keras.layers.Embedding(parts_of_speech.size().numpy(), 1024)

    model = Transformer(8, 32, 512, 3, 256, 1024)

    pointer_layer = tf.keras.layers.Dense(1)
    tag_layer = tf.keras.layers.Dense(parts_of_speech.size().numpy())
    word_layer = tf.keras.layers.Dense(vocab.size().numpy())

    optimizer = tf.keras.optimizers.Adam()
    trainable_variables = (
        model.trainable_variables +
        word_embedding_layer.trainable_variables +
        tag_embedding_layer.trainable_variables +
        pointer_layer.trainable_variables +
        tag_layer.trainable_variables +
        word_layer.trainable_variables)

    writer = tf.summary.create_file_writer(args.logging_dir)
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', args.logging_dir])
    tb.launch()

    start_time = time.time()
    for iteration, batch in enumerate(data_loader()):

        image_path = batch["image_path"]
        image = batch["image"]
        new_word = batch["new_word"]
        new_tag = batch["new_tag"]
        slot = batch["slot"]
        words = batch["words"]
        tags = batch["tags"]
        indicators = batch["indicators"]

        def loss_function():

            word_embeddings = word_embedding_layer(words)
            hidden_activations = model([image, word_embeddings, indicators])
            pointer_logits = tf.squeeze(pointer_layer(hidden_activations), 2)

            selected_hidden_activations = tf.squeeze(tf.gather(
                hidden_activations, tf.expand_dims(slot, 1), batch_dims=1), 1)
            tag_logits = tag_layer(selected_hidden_activations)

            tag_embeddings = tag_embedding_layer(new_tag)
            context_vector = tf.concat([selected_hidden_activations, tag_embeddings], 1)
            word_logits = word_layer(context_vector)

            pointer_loss = tf.losses.sparse_categorical_crossentropy(
                slot,
                pointer_logits)
            tag_loss = tf.losses.sparse_categorical_crossentropy(
                new_tag,
                tag_logits)
            word_loss = tf.losses.sparse_categorical_crossentropy(
                new_word,
                word_logits)

            return tf.reduce_mean(pointer_loss + tag_loss + word_loss)

        optimizer.minimize(loss_function, trainable_variables)

        if (iteration + 1) % args.logging_delay == 0:
            end_time = time.time()
            duration = end_time - start_time
            start_time = end_time
            with writer.as_default():
                tf.summary.experimental.set_step(iteration)
                tf.summary.scalar("Images Per Second", args.logging_delay * args.batch_size / duration)
                tf.summary.scalar("Loss", loss_function())

