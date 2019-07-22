"""Author: Brandon Trabucco, Copyright 2019"""


from best_first.layers.scaled_dot_product_attention import ScaledDotProductAttention
import tensorflow as tf


class SelfAttentionBlock(tf.keras.layers.Layer):

    def __init__(
            self,
            num_heads,
            attention_hidden_size,
            dense_hidden_size,
            num_layers,
            hidden_size,
            **kwargs
    ):
        super(SelfAttentionBlock, self).__init__(**kwargs)
        self.num_layers = num_layers

        self.attention_layers = [
            ScaledDotProductAttention(
                num_heads,
                attention_hidden_size,
                hidden_size) for i in range(num_layers)]
        self.attention_dropouts = [
            tf.keras.layers.Dropout(0.1) for i in range(num_layers)]
        self.attention_norms = [
            tf.keras.layers.BatchNormalization() for i in range(num_layers)]

        self.dense_hidden_layers = [
            tf.keras.layers.Dense(dense_hidden_size) for i in range(num_layers)]
        self.dense_output_layers = [
            tf.keras.layers.Dense(hidden_size) for i in range(num_layers)]
        self.dense_norms = [
            tf.keras.layers.BatchNormalization() for i in range(num_layers)]

    def call(
            self,
            inputs,
            **kwargs
    ):
        sequence, *rest = inputs

        if len(rest) > 0:
            indicators, *rest = rest
        else:
            indicators = tf.ones(tf.shape(sequence)[:2])

        for x in range(self.num_layers):

            sequence = self.attention_norms[x](
                sequence + self.attention_dropouts[x](self.attention_layers[x]([
                    sequence,
                    sequence,
                    sequence,
                    indicators])))
            sequence = self.dense_norms[x](
                sequence + self.dense_output_layers[x](
                    tf.nn.relu(self.dense_hidden_layers[x](sequence))))

        return sequence
