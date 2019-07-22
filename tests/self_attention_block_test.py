"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from best_first.layers.self_attention_block import SelfAttentionBlock


if __name__ == "__main__":

    layer = SelfAttentionBlock(
        8,
        32,
        512,
        3,
        4)

    sequence_one = tf.random.normal([1, 7, 4])
    indicators_one = tf.concat([tf.ones([1, 5]), tf.zeros([1, 2])], 1)

    with tf.GradientTape() as tape:

        tape.watch(sequence_one)

        outputs = layer([sequence_one, indicators_one])
        grad = tape.gradient(outputs[0, 0, :], sequence_one)
        print(grad)
