"""Author: Brandon Trabucco, Copyright 2019"""


from best_first.layers.self_attention_block import SelfAttentionBlock
from best_first.layers.dual_attention_block import DualAttentionBlock
import tensorflow as tf


class Transformer(tf.keras.layers.Layer):

    def __init__(
            self,
            num_heads,
            attention_hidden_size,
            dense_hidden_size,
            num_layers,
            hidden_size,
            output_size,
            **kwargs
    ):
        super(Transformer, self).__init__(**kwargs)
        self.hidden_size = hidden_size

        self.input_one_layer = tf.keras.layers.Dense(hidden_size, use_bias=False)
        self.input_two_layer = tf.keras.layers.Dense(hidden_size, use_bias=False)

        self.self_attention_block = SelfAttentionBlock(
            num_heads,
            attention_hidden_size,
            dense_hidden_size,
            num_layers,
            hidden_size,
            **kwargs)

        self.dual_attention_block = DualAttentionBlock(
            num_heads,
            attention_hidden_size,
            dense_hidden_size,
            num_layers,
            hidden_size,
            **kwargs)

        self.output_layer = tf.keras.layers.Dense(output_size, use_bias=False)

    def call(
            self,
            inputs,
            training=True,
            **kwargs
    ):
        sequence_one, sequence_two, *rest = inputs

        if len(rest) > 0:
            indicators_one, *rest = rest
        else:
            indicators_one = tf.ones(tf.shape(sequence_one)[:2])
        if len(rest) > 0:
            indicators_two, *rest = rest
        else:
            indicators_two = tf.ones(tf.shape(sequence_two)[:2])

        depth_ids = tf.cast(tf.range(self.hidden_size, delta=2), tf.float32)
        sequence_one_ids = tf.cast(tf.range(tf.shape(sequence_one)[1]), tf.float32)
        sequence_two_ids = tf.cast(tf.range(tf.shape(sequence_two)[1]), tf.float32)

        depth_one_ids, sequence_one_ids = tf.meshgrid(depth_ids, sequence_one_ids)
        depth_two_ids, sequence_two_ids = tf.meshgrid(depth_ids, sequence_two_ids)

        positional_embeddings_one = tf.concat([
            tf.sin(sequence_one_ids / tf.pow(
                10000.0, 2.0 * depth_one_ids / self.hidden_size)),
            tf.cos(sequence_one_ids / tf.pow(
                10000.0, 2.0 * (depth_one_ids + 1) / self.hidden_size))], 1)

        positional_embeddings_two = tf.concat([
            tf.sin(sequence_two_ids / tf.pow(
                10000.0, 2.0 * depth_two_ids / self.hidden_size)),
            tf.cos(sequence_two_ids / tf.pow(
                10000.0, 2.0 * (depth_two_ids + 1) / self.hidden_size))], 1)

        sequence_one = self.input_one_layer(sequence_one) + positional_embeddings_one
        sequence_two = self.input_two_layer(sequence_two) + positional_embeddings_two

        sequence_one = self.self_attention_block([sequence_one, indicators_one], training=training)
        sequence_two = self.dual_attention_block([
            sequence_two, sequence_one, indicators_two, indicators_one], training=training)

        return self.output_layer(sequence_two)
