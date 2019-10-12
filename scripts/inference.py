"""Author: Brandon Trabucco, Copyright 2019"""


import argparse
import tensorflow as tf
from best_first.caption_utils import load_tagger
from best_first import load_vocabulary
from best_first import load_parts_of_speech
from best_first import OrderedDecoder
from best_first import decoder_params


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="image.jpg")
    parser.add_argument("--ckpt", type=str, default="./data/model.ckpt")
    parser.add_argument("--tagger_file", type=str, default="./data/tagger.pkl")
    parser.add_argument("--vocab_file", type=str, default="./data/vocab.txt")
    parser.add_argument("--image_height", type=int, default=299)
    parser.add_argument("--image_width", type=int, default=299)
    parser.add_argument("--max_caption_length", type=int, default=20)
    args = parser.parse_args()

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

    images = image_features_extract_model(tf.expand_dims(image, 0))
    images = tf.reshape(images, [1, -1, images.shape[3]])

    words, tags = tf.constant([[2, 3]]), tf.constant([[1, 1]])
    for i in range(args.max_caption_length):
        pointer_logits, tag_logits, word_logits = decoder(
            images, words, tags,  training=False)

        slot = tf.argmax(pointer_logits[0], output_type=tf.int32).numpy()
        if slot == tf.size(words[0]).numpy() - 1:
            break

        next_word = tf.argmax(word_logits[0], output_type=tf.int32).numpy()
        next_word_string = vocab.ids_to_words(tf.constant(next_word)).numpy().decode("utf-8")
        next_tag_string = tagger.tag([next_word_string])[0][1]
        next_tag = parts_of_speech.words_to_ids(tf.constant(next_tag_string)).numpy()

        words = words.numpy().tolist()[0]
        words = tf.constant([words[:(slot + 1)] + [next_word] + words[(slot + 1):]])
        tags = tags.numpy().tolist()[0]
        tags = tf.constant([tags[:(slot + 1)] + [next_tag] + tags[(slot + 1):]])

        word_list = vocab.ids_to_words(words[0]).numpy().tolist()
        word_list.insert(slot + 2, "]")
        word_list.insert(slot + 1, "[")
        caption = tf.strings.reduce_join(word_list, separator=" ").numpy().decode("utf-8")
        print("Caption: {}".format(caption))
