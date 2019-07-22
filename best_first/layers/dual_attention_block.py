"""Author: Brandon Trabucco, Copyright 2019"""


from best_first.layers.scaled_dot_product_attention import ScaledDotProductAttention
from best_first.layers.self_attention_block import SelfAttentionBlock
import tensorflow as tf


class DualAttentionBlock(tf.keras.layers.Layer):

    def __init__(
            self,
            num_heads,
            attention_hidden_size,
            dense_hidden_size,
            num_layers,
            hidden_size,
            **kwargs
    ):
        super(DualAttentionBlock, self).__init__(**kwargs)
        self.num_layers = num_layers

        self.attention_layers = [
            ScaledDotProductAttention(
                num_heads,
                attention_hidden_size,
                hidden_size) for i in range(num_layers * 2)]
        self.attention_dropouts = [
            tf.keras.layers.Dropout(0.1) for i in range(num_layers * 2)]
        self.attention_norms = [
            tf.keras.layers.BatchNormalization() for i in range(num_layers * 2)]

        self.dense_hidden_layers = [
            tf.keras.layers.Dense(dense_hidden_size) for i in range(num_layers * 2)]
        self.dense_output_layers = [
            tf.keras.layers.Dense(hidden_size) for i in range(num_layers * 2)]
        self.dense_norms = [
            tf.keras.layers.BatchNormalization() for i in range(num_layers * 2)]

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

        for i in range(self.num_layers):

            x = i * 2
            sequence_one = self.attention_norms[x](
                sequence_one + self.attention_dropouts[x](self.attention_layers[x]([
                    sequence_one,
                    sequence_one,
                    sequence_one,
                    indicators_one,
                    indicators_one,
                    indicators_one])), training=training)
            sequence_one = self.dense_norms[x](
                sequence_one + self.dense_output_layers[x](
                    tf.nn.relu(self.dense_hidden_layers[x](sequence_one))), training=training)

            y = i * 2 + 1
            sequence_one = self.attention_norms[y](
                sequence_one + self.attention_dropouts[y](self.attention_layers[y]([
                    sequence_one,
                    sequence_two,
                    sequence_two,
                    indicators_one,
                    indicators_two,
                    indicators_two])), training=training)
            sequence_one = self.dense_norms[y](
                sequence_one + self.dense_output_layers[y](
                    tf.nn.relu(self.dense_hidden_layers[y](sequence_one))), training=training)

        return sequence_one
