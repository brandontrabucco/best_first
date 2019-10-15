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

        self.image_layer = tf.keras.layers.Dense(
            params["hidden_size"], activation="relu")
        self.merge_layer_one = tf.keras.layers.Dense(
            params["hidden_size"], activation="relu")

        self.encoder = transformer.EncoderStack(params)
        self.decoder = transformer.DecoderStack(params)

        self.pointer_layer = tf.keras.layers.Dense(
            1, activation=lambda x: tf.squeeze(x, -1))
        self.merge_layer_two = tf.keras.layers.Dense(
            params["hidden_size"], activation="relu")

    def get_config(
            self
    ):
        return {"params": self.params}

    def get_pointer_encodings(
            self,
            images,
            words,
            tags,
            word_paddings=None,
            training=False
    ):
        
        batch_size, image_locations, length = (
            tf.shape(images)[0], tf.shape(images)[1], tf.shape(words)[1])
        if word_paddings is None:
            word_paddings = tf.cast(tf.ones_like(words), self.params["dtype"])

        # Pass the image features [BATCH, 64, 2048] into an encoder
        images = self.image_layer(images)
        image_attention_bias = tf.zeros([batch_size, 1, 1, image_locations])
        image_attention_bias = tf.cast(image_attention_bias, self.params["dtype"])
        image_padding = tf.zeros_like(images)
        encoder_outputs = self.encoder(
            images, 
            image_attention_bias, 
            image_padding, 
            training=training)

        # Add a positional encoding to the word embeddings
        pos_encoding = tf.cast(
            model_utils.get_position_encoding(
                length, self.params["hidden_size"]), self.params["dtype"])
        decoder_inputs = pos_encoding + self.merge_layer_one(tf.concat([
            self.word_embeddings(
                words, mode="embedding", training=training),
            self.tag_embeddings(
                tags, mode="embedding", training=training)], -1), training=training)

        # Use the decoder to merge image and word features
        word_attention_bias = -1e9 * (
            1.0 - word_paddings[:, tf.newaxis, tf.newaxis, :])
        return self.decoder(
            decoder_inputs, 
            encoder_outputs,
            word_attention_bias, 
            image_attention_bias, 
            training=training)

    def get_pointer_logits(
            self,
            pointer_encodings,
            training=False
    ):
        # Compute the pointer network over slots to insert the next word
        return self.pointer_layer(pointer_encodings, training=training)

    def get_tag_logits(
            self,
            slot_encoding,
            training=False
    ):
        # Determine a tag to decode next at this slot
        return self.tag_embeddings(
            slot_encoding[:, tf.newaxis, :], mode="linear", training=training)[:, 0, :]

    def get_word_logits(
            self,
            slot_encoding,
            next_tag,
            training=False
    ):
        # Determine a word to decode next at this slot
        tag_embeddings = self.tag_embeddings(
            next_tag, mode="embedding", training=training)
        word_inputs = self.merge_layer_two(
            tf.concat([slot_encoding, tag_embeddings], -1), training=training)
        return self.word_embeddings(
            word_inputs[:, tf.newaxis, :], mode="linear", training=training)[:, 0, :]

    def call(
            self,
            images,
            words,
            tags,
            word_paddings=None,
            next_tag=None,
            slot=None,
            training=False
    ):
        pointer_encodings = self.get_pointer_encodings(
            images, words, tags, word_paddings=word_paddings, training=training)

        # Compute the pointer network over slots to insert the next word
        pointer_logits = self.get_pointer_logits(
            pointer_encodings, training=training)

        # Use the slot to choose which feature to use for decoding
        if slot is None:
            slot = tf.argmax(pointer_logits, axis=(-1), output_type=tf.int32)
        slot_encodings = tf.squeeze(
            tf.gather(pointer_encodings, tf.expand_dims(slot, 1), batch_dims=1), 1)

        # Determine a tag to decode next at this slot
        tag_logits = self.get_tag_logits(slot_encodings, training=training)

        # Determine a word to decode next at this slot
        if next_tag is None:
            next_tag = tf.argmax(tag_logits, axis=(-1))
        word_logits = self.get_word_logits(
            slot_encodings, next_tag, training=training)

        return pointer_logits, tag_logits, word_logits
