"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf


class ScaledDotProductAttention(tf.keras.layers.Layer):

    def __init__(
            self,
            num_heads,
            hidden_size,
            output_size,
            **kwargs
    ):
        super(ScaledDotProductAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.query_map = tf.keras.layers.Dense(
            hidden_size * num_heads,
            use_bias=False)
        self.key_map = tf.keras.layers.Dense(
            hidden_size * num_heads,
            use_bias=False)
        self.value_map = tf.keras.layers.Dense(
            hidden_size * num_heads,
            use_bias=False)
        self.output_map = tf.keras.layers.Dense(
            output_size,
            use_bias=False)

    def call(
            self,
            inputs,
            **kwargs
    ):
        queries, keys, values, *rest = inputs
        if len(rest) > 0:
            indicators = rest[0]
        else:
            indicators = tf.ones(tf.shape(values)[:2])
        batch_size = tf.shape(queries)[0]
        num_queries = tf.shape(queries)[1]
        sequence_length = tf.shape(values)[1]
        Q = self.query_map(queries)
        K = self.key_map(keys)
        V = self.value_map(values)
        Q = tf.transpose(
            tf.reshape(Q, [
                batch_size,
                num_queries,
                self.num_heads,
                self.hidden_size]), [0, 2, 1, 3])
        K = tf.transpose(
            tf.reshape(K, [
                batch_size,
                sequence_length,
                self.num_heads,
                self.hidden_size]), [0, 2, 1, 3])
        V = tf.transpose(
            tf.reshape(V, [
                batch_size,
                sequence_length,
                self.num_heads,
                self.hidden_size]), [0, 2, 1, 3])
        S = tf.matmul(tf.expand_dims(tf.expand_dims(indicators, 1), 2) * tf.nn.softmax(
            tf.matmul(Q, tf.transpose(
                K, [0, 1, 3, 2])) / tf.sqrt(float(self.hidden_size))), V)
        return self.output_map(
            tf.reshape(tf.transpose(
                S, [0, 2, 1, 3]), [
                batch_size,
                num_queries,
                self.num_heads * self.hidden_size]))
