"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from best_first.layers.dual_attention_block import DualAttentionBlock


if __name__ == "__main__":

    layer = DualAttentionBlock(
        8,
        32,
        512,
        3,
        4)

    sequence_one = tf.random.normal([1, 7, 4])
    sequence_two = tf.random.normal([1, 7, 4])
    indicators_one = tf.concat([tf.ones([1, 5]), tf.zeros([1, 2])], 1)
    indicators_two = tf.concat([tf.ones([1, 5]), tf.zeros([1, 2])], 1)

    with tf.GradientTape() as tape:

        tape.watch(sequence_one)
        tape.watch(sequence_two)

        outputs = layer([sequence_one, sequence_two,
                         indicators_one, indicators_two])
        grad = tape.gradient(outputs[0, 0, :], sequence_two)
        print(grad)
