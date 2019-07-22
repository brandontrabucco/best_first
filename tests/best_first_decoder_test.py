"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from best_first.best_first_decoder import BestFirstDecoder


if __name__ == "__main__":

    layer = BestFirstDecoder(
        1024,
        1024,
        1024,
        1024,
        8,
        32,
        512,
        3,
        256,
        4)

    image = tf.random.normal([1, 7, 4])
    words = tf.random.uniform([1, 5], maxval=1000, dtype=tf.int32)
    indicators_image = tf.concat([tf.ones([1, 5]), tf.zeros([1, 2])], 1)
    indicators_words = tf.concat([tf.ones([1, 3]), tf.zeros([1, 2])], 1)

    with tf.GradientTape(persistent=True) as tape:

        tape.watch(image)

        pointer_logits, tag_logits, word_logits = layer([
            image, words, indicators_image, indicators_words])

    grad = tape.gradient(pointer_logits, image)
    print(grad)

    grad = tape.gradient(tag_logits, image)
    print(grad)

    grad = tape.gradient(word_logits, image)
    print(grad)
