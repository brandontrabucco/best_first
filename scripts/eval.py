"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from best_first import load_vocabulary
from best_first import load_parts_of_speech
from best_first import create_dataset
from best_first import OrderedDecoder
from best_first import decoder_params
import os
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Train an Ordered Decoder module")
    parser.add_argument("--logging_dir", type=str, default="../data")
    parser.add_argument("--tfrecord_folder", type=str, default="../data/tfrecords")
    parser.add_argument("--vocab_file", type=str, default="../data/vocab.txt")
    parser.add_argument("--word_weight", type=float, default=1.0)
    parser.add_argument("--tag_weight", type=float, default=1.0)
    parser.add_argument("--pointer_weight", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--ckpt", type=str, default="../data/model.ckpt")
    args = parser.parse_args()
    
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    writer = tf.summary.create_file_writer(args.logging_dir)
    tf.io.gfile.makedirs(os.path.dirname(args.ckpt))
    tf.io.gfile.makedirs(args.logging_dir)
    vocab = load_vocabulary(vocab_file=args.vocab_file)
    parts_of_speech = load_parts_of_speech()
    dataset = create_dataset(tfrecord_folder=args.tfrecord_folder, batch_size=args.batch_size)

    decoder_params["vocab_size"] = vocab.size()
    decoder_params["parts_of_speech_size"] = parts_of_speech.size()
    decoder = OrderedDecoder(decoder_params)
    decoder.load_weights(args.ckpt)

    for iteration, batch in enumerate(dataset):
        tf.summary.experimental.set_step(iteration)

        pointer_logits, tag_logits, word_logits = decoder(
            batch["image"],
            batch["words"],
            batch["tags"],
            word_paddings=batch["indicators"],
            training=True)

        pointer_loss = tf.reduce_mean(
            tf.losses.sparse_categorical_crossentropy(
                batch["slot"],
                pointer_logits,
                from_logits=True))

        tag_loss = tf.reduce_mean(
            tf.losses.sparse_categorical_crossentropy(
                batch["next_tag"],
                tag_logits,
                from_logits=True))

        word_loss = tf.reduce_mean(
            tf.losses.sparse_categorical_crossentropy(
                batch["next_word"],
                word_logits,
                from_logits=True))

        loss = (
            args.pointer_weight * pointer_loss +
            args.tag_weight * tag_loss +
            args.word_weight * word_loss)

        with writer.as_default():
            tf.summary.text(
                "Eval/Words",
                tf.strings.reduce_join(vocab.ids_to_words(batch["words"]), separator=" ", axis=-1)[:3])
            tf.summary.text(
                "Eval/Next Word",
                vocab.ids_to_words(batch["next_word"])[:3])
            tf.summary.text(
                "Eval/Predicted Next Word",
                vocab.ids_to_words(tf.argmax(word_logits, axis=-1, output_type=tf.int32))[:3])

            tf.summary.scalar("Eval/Loss Pointer", pointer_loss)
            tf.summary.scalar("Eval/Loss Tag", tag_loss)
            tf.summary.scalar("Eval/Loss Word", word_loss)
            tf.summary.scalar("Eval/Loss Total", loss)
