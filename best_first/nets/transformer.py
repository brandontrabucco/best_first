"""Author: Brandon Trabucco, Kavi Gupta, Copyright 2019"""

from best_first.nets.scaled_dot_product_attention import ScaledDotProductAttention
import tensorflow as tf


class Transformer(tf.keras.Model):

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

        self.attention_one_layers = [
            ScaledDotProductAttention(
                num_heads,
                attention_hidden_size,
                hidden_size) for i in range(num_layers)]
        self.attention_one_dropouts = [
            tf.keras.layers.Dropout(0.1) for i in range(num_layers)]
        self.attention_one_norms = [
            tf.keras.layers.BatchNormalization() for i in range(num_layers)]

        self.dense_one_hidden_layers = [
            tf.keras.layers.Dense(dense_hidden_size) for i in range(num_layers)]
        self.dense_one_output_layers = [
            tf.keras.layers.Dense(hidden_size) for i in range(num_layers)]
        self.dense_one_norms = [
            tf.keras.layers.BatchNormalization() for i in range(num_layers)]

        self.attention_two_layers = [
            ScaledDotProductAttention(
                num_heads,
                attention_hidden_size,
                hidden_size) for i in range(num_layers)]
        self.attention_two_dropouts = [
            tf.keras.layers.Dropout(0.1) for i in range(num_layers)]
        self.attention_two_norms = [
            tf.keras.layers.BatchNormalization() for i in range(num_layers)]

        self.dense_two_hidden_layers = [
            tf.keras.layers.Dense(dense_hidden_size) for i in range(num_layers)]
        self.dense_two_output_layers = [
            tf.keras.layers.Dense(hidden_size) for i in range(num_layers)]
        self.dense_two_norms = [
            tf.keras.layers.BatchNormalization() for i in range(num_layers)]

        self.attention_three_layers = [
            ScaledDotProductAttention(
                num_heads,
                attention_hidden_size,
                hidden_size) for i in range(num_layers)]
        self.attention_three_dropouts = [
            tf.keras.layers.Dropout(0.1) for i in range(num_layers)]
        self.attention_three_norms = [
            tf.keras.layers.BatchNormalization() for i in range(num_layers)]

        self.dense_three_hidden_layers = [
            tf.keras.layers.Dense(dense_hidden_size) for i in range(num_layers)]
        self.dense_three_output_layers = [
            tf.keras.layers.Dense(hidden_size) for i in range(num_layers)]
        self.dense_three_norms = [
            tf.keras.layers.BatchNormalization() for i in range(num_layers)]

        self.output_layer = tf.keras.layers.Dense(output_size, use_bias=False)

    def call(
            self,
            inputs,
            **kwargs
    ):
        sequence_one, sequence_two, indicators = inputs

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

        for (attend_one_layer,
             attend_dropout_one_layer,
             attend_norm_one_layer,
             hidden_layer,
             output_layer,
             dense_norm_layer) in zip(
                self.attention_one_layers,
                self.attention_one_dropouts,
                self.attention_one_norms,
                self.dense_one_hidden_layers,
                self.dense_one_output_layers,
                self.dense_one_norms):
            sequence_one = attend_norm_one_layer(
                sequence_one + attend_dropout_one_layer(attend_one_layer([
                    sequence_one,
                    sequence_one,
                    sequence_one])))
            sequence_one = dense_norm_layer(
                sequence_one + output_layer(
                    tf.nn.relu(hidden_layer(sequence_one))))

        for (attend_two_layer,
             attend_dropout_two_layer,
             attend_norm_two_layer,
             attend_three_layer,
             attend_dropout_three_layer,
             attend_norm_three_layer,
             hidden_layer,
             output_layer,
             dense_norm_layer) in zip(
                self.attention_two_layers,
                self.attention_two_dropouts,
                self.attention_two_norms,
                self.attention_three_layers,
                self.attention_three_dropouts,
                self.attention_three_norms,
                self.dense_one_hidden_layers,
                self.dense_one_output_layers,
                self.dense_one_norms):
            sequence_two = attend_norm_two_layer(
                sequence_two + attend_dropout_two_layer(attend_two_layer([
                    sequence_two,
                    sequence_two,
                    sequence_two,
                    indicators])))
            sequence_two = attend_norm_three_layer(
                sequence_two + attend_dropout_three_layer(attend_three_layer([
                    sequence_two,
                    sequence_one,
                    sequence_one])))
            sequence_two = dense_norm_layer(
                sequence_two + output_layer(
                    tf.nn.relu(hidden_layer(sequence_two))))

        return self.output_layer(sequence_two)
