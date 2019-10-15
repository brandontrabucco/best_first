"""Author: Brandon Trabucco, Copyright 2019"""


import argparse
import tensorflow as tf
from best_first.caption_utils import load_tagger
from best_first import load_vocabulary
from best_first import load_parts_of_speech
from best_first import OrderedDecoder
from best_first import decoder_params
from best_first.beam_search import beam_search


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="image.jpg")
    parser.add_argument("--ckpt", type=str, default="./data/model.ckpt")
    parser.add_argument("--tagger_file", type=str, default="./data/tagger.pkl")
    parser.add_argument("--vocab_file", type=str, default="./data/vocab.txt")
    parser.add_argument("--image_height", type=int, default=299)
    parser.add_argument("--image_width", type=int, default=299)
    parser.add_argument("--beam_size", type=int, default=100)
    parser.add_argument("--max_caption_length", type=int, default=20)
    args = parser.parse_args()
    
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    tagger = load_tagger(tagger_file=args.tagger_file)
    vocab = load_vocabulary(vocab_file=args.vocab_file)
    parts_of_speech = load_parts_of_speech()

    decoder_params["vocab_size"] = vocab.size()
    decoder_params["parts_of_speech_size"] = parts_of_speech.size()
    decoder = OrderedDecoder(decoder_params)
    decoder.load_weights(args.ckpt)

    image_model = tf.keras.applications.InceptionV3(
        include_top=False, weights='imagenet')
    image_features_extract_model = tf.keras.Model(
        image_model.input, image_model.layers[-1].output)

    image = tf.keras.applications.inception_v3.preprocess_input(
        tf.image.resize(
            tf.image.decode_jpeg(
                tf.io.read_file(args.image),
                channels=3), [args.image_height, args.image_width]))

    images = image_features_extract_model(tf.expand_dims(image, axis=0))
    images = tf.reshape(images, [tf.shape(images)[0], -1, images.shape[3]])

    words, tags, slots, log_probs = beam_search(
        images,
        decoder,
        beam_size=args.beam_size,
        training=False)

    captions = tf.strings.reduce_join(
        vocab.ids_to_words(words), separator=" ", axis=(-1))

    caption = captions[0, 0].numpy().decode("utf-8").replace(
        "<pad>", "").replace("<start>", "").replace("<end>", "").strip().split(" ")

    editted_caption = []
    for slot in reversed(slots[0, 0, :-1].numpy().tolist()):
        editted_caption.append(list(caption))
        caption.pop(slot)

    editted_caption.reverse()

    for slot, partial_caption in zip(slots[0, 0, :-1].numpy().tolist(), editted_caption):
        partial_caption.insert(slot, "[")
        partial_caption.insert(slot  + 2, "]")
        print(" ".join(partial_caption))
