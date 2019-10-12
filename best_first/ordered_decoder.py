"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from official.transformer.model import model_utils
from official.transformer.v2 import embedding_layer
from official.transformer.v2 import transformer


class OrderedDecoder(tf.keras.Model):

    def __init__(
            self,
            params,
            name=None
    ):
        super(OrderedDecoder, self).__init__(name=name)
        self.params = params

        self.word_embeddings = embedding_layer.EmbeddingSharedWeights(
            tf.cast(params["vocab_size"], tf.int32),
            params["hidden_size"])
        self.tag_embeddings = embedding_layer.EmbeddingSharedWeights(
            tf.cast(params["parts_of_speech_size"], tf.int32),
            params["hidden_size"])

        self.merge_embeddings = tf.keras.layers.Dense(
            params["hidden_size"], use_bias=False)
        self.un_merge_embeddings = tf.keras.layers.Dense(
            params["hidden_size"], use_bias=False)
        self.image_layer = tf.keras.layers.Dense(
            params["hidden_size"], use_bias=False)

        self.encoder = transformer.EncoderStack(params)
        self.decoder = transformer.DecoderStack(params)
        self.pointer_layer = tf.keras.layers.Dense(
            1, activation=lambda x: tf.squeeze(x, -1))

    def get_config(
            self
    ):
        return {"params": self.params}

    def call(
            self,
            images,
            words,
            tags,
            word_indicators=None,
            ground_truth_tag=None,
            ground_truth_slot=None,
            training=False
    ):
        batch_size, image_locations, length = (
            tf.shape(images)[0], tf.shape(images)[1], tf.shape(words)[1])

        # Pass the image features [BATCH, 64, 2048] into an encoder
        images = self.image_layer(images)
        image_attention_bias = tf.zeros([batch_size, 1, 1, image_locations])
        image_attention_bias = tf.cast(image_attention_bias, self.params["dtype"])
        image_padding = tf.zeros_like(images)
        encoder_outputs = self.encoder(
            images, image_attention_bias, image_padding, training=training)

        # Add a positional encoding to the word embeddings
        embedded_inputs = self.merge_embeddings(tf.concat([
            self.word_embeddings(words, mode="embedding"),
            self.tag_embeddings(tags, mode="embedding")], -1))
        pos_encoding = model_utils.get_position_encoding(
            length, self.params["hidden_size"])
        pos_encoding = tf.cast(pos_encoding, self.params["dtype"])
        decoder_inputs = embedded_inputs + pos_encoding

        # Use the decode to merge image and word features
        if word_indicators is None:
            word_indicators = tf.cast(tf.ones_like(words), self.params["dtype"])
        word_attention_bias = -1e9 * (
            1.0 - word_indicators[:, tf.newaxis, tf.newaxis, :])
        decoder_outputs = self.decoder(
            decoder_inputs, encoder_outputs,
            word_attention_bias, image_attention_bias, training=training)

        # Compute the pointer network over slots to insert the next word
        pointer_logits = self.pointer_layer(decoder_outputs)
        if ground_truth_slot is None:
            slot = tf.argmax(pointer_logits, axis=(-1), output_type=tf.int32)
        else:
            slot = ground_truth_slot

        # Use the slot to choose which feature to use for decoding
        slotted_activations = tf.squeeze(
            tf.gather(decoder_outputs, tf.expand_dims(slot, 1), batch_dims=1), 1)

        # Determine a tag to decode next at this slot
        tag_logits = self.tag_embeddings(
            slotted_activations[:, tf.newaxis, :], mode="linear")[:, 0, :]
        if ground_truth_tag is None:
            next_tag = tf.argmax(tag_logits, axis=(-1))
        else:
            next_tag = ground_truth_tag

        # Determine a word to decode next at this slot
        tag_embeddings = self.tag_embeddings(next_tag, mode="embedding")
        word_inputs = self.un_merge_embeddings(tf.concat([
            slotted_activations, tag_embeddings], -1))
        word_logits = self.word_embeddings(
            word_inputs[:, tf.newaxis, :], mode="linear")[:, 0, :]

        return pointer_logits, tag_logits, word_logits
