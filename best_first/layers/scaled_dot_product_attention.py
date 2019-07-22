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
        self.output_size = output_size
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
        batch_size = tf.shape(queries)[0]
        num_queries = tf.shape(queries)[1]
        sequence_length = tf.shape(values)[1]

        if len(rest) > 0:
            indicators_queries, *rest = rest
        else:
            indicators_queries = tf.ones(tf.shape(queries)[:2])

        if len(rest) > 0:
            indicators_keys, *rest = rest
        else:
            indicators_keys = tf.ones(tf.shape(keys)[:2])

        if len(rest) > 0:
            indicators_values, *rest = rest
        else:
            indicators_values = tf.ones(tf.shape(values)[:2])

        Q = self.query_map(queries)
        K = self.key_map(keys)
        V = self.value_map(values)

        Q = tf.expand_dims(tf.expand_dims(indicators_queries, 1), 3) * tf.transpose(
            tf.reshape(Q, [
                batch_size,
                num_queries,
                self.num_heads,
                self.hidden_size]), [0, 2, 1, 3])
        K = tf.expand_dims(tf.expand_dims(indicators_keys, 1), 3) * tf.transpose(
            tf.reshape(K, [
                batch_size,
                sequence_length,
                self.num_heads,
                self.hidden_size]), [0, 2, 1, 3])
        V = tf.expand_dims(tf.expand_dims(indicators_values, 1), 3) * tf.transpose(
            tf.reshape(V, [
                batch_size,
                sequence_length,
                self.num_heads,
                self.hidden_size]), [0, 2, 1, 3])

        attention_mask = tf.expand_dims(tf.expand_dims(indicators_keys, 1), 2) * tf.nn.softmax(
            tf.matmul(Q, tf.transpose(
                K, [0, 1, 3, 2])) / tf.math.sqrt(float(self.hidden_size)))
        denominator = tf.reduce_sum(attention_mask, axis=(-1), keepdims=True)
        attention_mask = attention_mask / tf.where(
            tf.math.equal(denominator, 0.0), 1.0, denominator)

        return self.output_map(
            tf.reshape(tf.transpose(
                tf.matmul(attention_mask, V), [0, 2, 1, 3]), [
                batch_size,
                num_queries,
                self.num_heads * self.hidden_size]))
