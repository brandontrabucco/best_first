"""Author: Brandon Trabucco, Copyright 2019"""


from best_first.layers.transformer import Transformer
import tensorflow as tf


class BestFirstDecoder(tf.keras.Model):

    def __init__(
            self,
            vocab_size,
            word_embedding_size,
            parts_of_speech_size,
            tag_embedding_size,
            num_heads,
            attention_hidden_size,
            dense_hidden_size,
            num_layers,
            hidden_size,
            output_size,
            **kwargs
    ):
        super(BestFirstDecoder, self).__init__(**kwargs)

        self.word_embedding_layer = tf.keras.layers.Embedding(
            vocab_size,
            word_embedding_size)
        self.tag_embedding_layer = tf.keras.layers.Embedding(
            parts_of_speech_size,
            tag_embedding_size)

        self.stem = Transformer(
            num_heads,
            attention_hidden_size,
            dense_hidden_size,
            num_layers,
            hidden_size,
            output_size)

        self.pointer_logits_layer = tf.keras.layers.Dense(
            1, activation=(lambda x: tf.squeeze(x, -1)))
        self.tag_logits_layer = tf.keras.layers.Dense(
            parts_of_speech_size)
        self.word_logits_layer = tf.keras.layers.Dense(
            vocab_size)

    def call(
            self,
            inputs,
            **kwargs
    ):
        image, words, *rest = inputs

        if len(rest) > 0:
            indicators_image, *rest = rest
        else:
            indicators_image = tf.ones(tf.shape(image)[:2])
        if len(rest) > 0:
            indicators_words, *rest = rest
        else:
            indicators_words = tf.ones(tf.shape(words))

        word_embeddings = self.word_embedding_layer(words)

        hidden_activations = self.stem([
            image, word_embeddings, indicators_image, indicators_words])
        pointer_logits = self.pointer_logits_layer(hidden_activations)

        if len(rest) > 0:
            slot, *rest = rest
        else:
            slot = tf.argmax(pointer_logits, axis=(-1))

        selected_hidden_activations = tf.squeeze(tf.gather(
            hidden_activations, tf.expand_dims(slot, 1), batch_dims=1), 1)
        tag_logits = self.tag_logits_layer(selected_hidden_activations)

        if len(rest) > 0:
            new_tag, *rest = rest
        else:
            new_tag = tf.argmax(tag_logits, axis=(-1))

        tag_embeddings = self.tag_embedding_layer(new_tag)
        word_logits = self.word_logits_layer(tf.concat([
            selected_hidden_activations, tag_embeddings], 1))

        return [pointer_logits, tag_logits, word_logits]
