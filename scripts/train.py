"""Author: Brandon Trabucco, Copyright 2019"""


import best_first.config as args
import tensorflow as tf
import time
from tensorboard import program
from best_first.data_loader import data_loader
from best_first import load_model


if __name__ == "__main__":
    tf.io.gfile.makedirs(args.checkpoint_dir)

    vocab, parts_of_speech, decoder = load_model()
    optimizer = tf.keras.optimizers.Adam(lr=args.learning_rate)

    ckpt = tf.train.Checkpoint(decoder=decoder, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(
        ckpt, args.checkpoint_dir, max_to_keep=2)
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print("Restoring model from {}".format(ckpt_manager.latest_checkpoint))

    writer = tf.summary.create_file_writer(args.logging_dir)
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', args.logging_dir])
    print("Launching tensorboard at {}".format(tb.launch()))

    for iteration, batch in enumerate(data_loader()):
        tf.summary.experimental.set_step(iteration)
        start_time = time.time()

        image_path = batch["image_path"]
        image = batch["image"]
        new_word = batch["new_word"]
        new_tag = batch["new_tag"]
        slot = batch["slot"]
        words = batch["words"]
        tags = batch["tags"]
        indicators = batch["indicators"]

        def loss_function():
            pointer_logits, tag_logits, word_logits = decoder([
                image, words, tf.ones(tf.shape(image)[:2]), indicators, slot, new_tag])

            predicted_new_word = tf.argmax(word_logits, axis=(-1), output_type=tf.int32)
            predicted_new_tag = tf.argmax(tag_logits, axis=(-1), output_type=tf.int32)
            predicted_slot = tf.argmax(pointer_logits, axis=(-1), output_type=tf.int32)

            pointer_accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted_slot, slot), tf.float32))
            tag_accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted_new_tag, new_tag), tf.float32))
            word_accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted_new_word, new_word), tf.float32))

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
                tf.summary.scalar("Train Pointer Accuracy", pointer_accuracy)
                tf.summary.scalar("Train Tag Accuracy", tag_accuracy)
                tf.summary.scalar("Train Word Accuracy", word_accuracy)
                tf.summary.scalar("Train Pointer Loss", pointer_loss)
                tf.summary.scalar("Train Tag Loss", tag_loss)
                tf.summary.scalar("Train Word Loss", word_loss)
                tf.summary.scalar("Train Total Loss", total_loss)

            return total_loss

        optimizer.minimize(loss_function, decoder.trainable_variables)
        with writer.as_default():
            tf.summary.scalar("Train Images Per Second", args.batch_size / (
                time.time() - start_time))

        if (iteration + 1) % args.checkpoint_delay == 0:
            ckpt_manager.save()
