"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from best_first import load_vocabulary
from best_first import load_parts_of_speech
from best_first import create_dataset
from best_first import OrderedDecoder
from best_first import decoder_params
from best_first.beam_search import beam_search
from nlgeval import NLGEval
import os
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Train an Ordered Decoder module")
    parser.add_argument("--logging_dir", type=str, default="../data")
    parser.add_argument("--tfrecord_folder", type=str, default="../data/tfrecords")
    parser.add_argument("--caption_folder", type=str, default="../data/captions")
    parser.add_argument("--vocab_file", type=str, default="../data/vocab.txt")
    parser.add_argument("--num_images_per_eval", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--beam_size", type=int, default=3)
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

    nlgeval = NLGEval()

    reference_captions = {}
    hypothesis_captions = {}

    for iteration, batch in enumerate(dataset):
        tf.summary.experimental.set_step(iteration)

        paths = [x.decode("utf-8") for x in batch["image_path"].numpy()]
        paths = [os.path.join(args.caption_folder, os.path.basename(x)[:-7] + "txt") for x in paths]
        for file_path in paths:
            with tf.io.gfile.GFile(file_path, "r") as f:
                reference_captions[file_path] = [x for x in f.read().strip().lower().split("\n") if len(x) > 0]

        words, tags, slots, log_probs = beam_search(
            batch["image"],
            decoder,
            beam_size=args.beam_size,
            training=False)

        for i in range(words.shape[0]):
            hypothesis_captions[paths[i]] = tf.strings.reduce_join(
                vocab.ids_to_words(words[i, 0, :]), separator=" ").numpy().decode("utf-8").replace(
                    "<pad>", "").replace("<start>", "").replace("<end>", "").strip()

        if len(reference_captions.keys()) >= args.num_images_per_eval:

            for key in hypothesis_captions.keys():
                x, y = hypothesis_captions[key], reference_captions[key]
                print("\nhypothesis: {}".format(x))
                for z in y:
                    print("    reference: {}".format(z))
            metrics_dict = nlgeval.compute_metrics([*zip(*reference_captions)], hypothesis_captions)
            reference_captions, hypothesis_captions = [], []

            with writer.as_default():
                print("")
                for key in metrics_dict.keys():
                    print(
                        "Eval/{}".format(key), metrics_dict[key])
                    tf.summary.scalar(
                        "Eval/{}".format(key), metrics_dict[key])

