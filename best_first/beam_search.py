"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
import numpy as np


def expand_beam_dim(
        x,
        current_beam_size,
        max_beam_size
):
    batch_size = tf.shape(x)[0]
    return tf.reshape(
        tf.tile(
            x[:, :, tf.newaxis, ...], 
            [1, 1, max_beam_size, *[1 for _ in x.shape[2:]]]), 
        [batch_size, current_beam_size * max_beam_size, *x.shape[2:]])


def insert_into_slot(
        old_data,
        item,
        slot
):
    batch_size, beam_size, length = (
        old_data.shape[0], old_data.shape[1], old_data.shape[2])
    result = np.zeros([
        batch_size, beam_size, length + 1], dtype=np.int32)
    for batch in range(batch_size):
        for beam in range(beam_size):
            for time_step in range(length + 1):
                if tf.equal(slot[batch, beam] + 1, time_step):
                    result[batch, beam, time_step] = item[batch, beam]
                elif tf.greater(slot[batch, beam] + 1, time_step):
                    result[batch, beam, time_step] = old_data[batch, beam, time_step]
                elif tf.less(slot[batch, beam] + 1, time_step):
                    result[batch, beam, time_step] = old_data[batch, beam, time_step - 1]
    return tf.constant(result)


def beam_search(
        images,
        decoder,
        beam_size=7,
        training=False
):
    # pad the images to be the right shape for beam search
    batch_size = tf.shape(images)[0]

    # initialize starting values for every beam
    log_probs = tf.zeros([batch_size, 1], dtype=tf.float32)
    words = tf.tile(tf.constant([[[2, 3]]]), [batch_size, 1, 1])
    tags = tf.tile(tf.constant([[[1, 1]]]), [batch_size, 1, 1])
    slots = tf.zeros([batch_size, 1, 0], dtype=tf.int32)
    closed = tf.fill([batch_size, 1], False)

    # perform a beam search until all beams are closed
    while not tf.reduce_all(closed):
        current_beam_size, length = tf.shape(log_probs)[1], tf.shape(words)[2]

        """STAGE ONE: slot decoding"""

        # Compute the encodings for each slot using the transformer model
        flat_images = tf.reshape(
            tf.tile(images[:, tf.newaxis, :, :], [1, current_beam_size, 1, 1]), 
            [batch_size * current_beam_size, tf.shape(images)[1], tf.shape(images)[2]])
        flat_words = tf.reshape(words, [batch_size * current_beam_size, length])
        flat_tags = tf.reshape(tags, [batch_size * current_beam_size, length])
        flat_pointer_encodings = decoder.get_pointer_encodings(
            flat_images, flat_words, flat_tags, training=training)

        # note the pointer_encodings need to not be flat after this point
        encoding_size = tf.shape(flat_pointer_encodings)[2]
        pointer_encodings = tf.reshape(
            flat_pointer_encodings, [batch_size, current_beam_size, length, encoding_size])

        # compute the pointer logits over slots to insert words next
        flat_pointer_logits = decoder.get_pointer_logits(
            flat_pointer_encodings, training=training)

        # the first iterations might have less than beam_size items available
        max_beam_size = tf.minimum(length, beam_size)

        # mask out elements of beams that are closed that are repeats
        open_beam_mask = 1.0 - tf.cast(
            tf.logical_and( closed[:, :, tf.newaxis],  tf.concat([
                tf.fill([batch_size, current_beam_size, 1], False),
                tf.fill([batch_size, current_beam_size, max_beam_size - 1], True)], 2)), tf.float32)
        closed_beam_bias = -1e9 * (1.0 - open_beam_mask)

        # compute the log probabilities of the top k possible pointers
        pointer_log_probs = tf.reshape(
            tf.nn.log_softmax(flat_pointer_logits), [batch_size, current_beam_size, length])
        pointer_log_probs, slot = tf.math.top_k(pointer_log_probs, k=max_beam_size)
        pointer_log_probs = (open_beam_mask * pointer_log_probs) + closed_beam_bias
        slot_encoding = tf.gather(pointer_encodings, slot, batch_dims=2)

        # the first iterations might have less than beam_size items available
        next_max_beam_size = tf.minimum(current_beam_size * max_beam_size, beam_size)

        log_probs = tf.reshape(
            log_probs[:, :, tf.newaxis] + pointer_log_probs, [batch_size, current_beam_size * max_beam_size])
        log_probs, beam_indices = tf.math.top_k(log_probs, k=next_max_beam_size)

        words = tf.gather(
            expand_beam_dim(words, current_beam_size, max_beam_size), beam_indices, batch_dims=1)
        tags = tf.gather(
            expand_beam_dim(tags, current_beam_size, max_beam_size), beam_indices, batch_dims=1)
        slots = tf.gather(
            expand_beam_dim(slots, current_beam_size, max_beam_size), beam_indices, batch_dims=1)
        closed = tf.gather(
            expand_beam_dim(closed, current_beam_size, max_beam_size), beam_indices, batch_dims=1)

        slot = tf.gather(
            tf.reshape(slot, [batch_size, current_beam_size * max_beam_size]), beam_indices, batch_dims=1)
        slot_encoding = tf.gather(
            tf.reshape(slot_encoding, [
                batch_size, current_beam_size * max_beam_size, encoding_size]), beam_indices, batch_dims=1)

        # if the model points to the final slot, the model is finished and remains finished
        closed = tf.logical_or(closed, tf.equal(slot, length - 1))
        slot = tf.where(closed, tf.fill(tf.shape(slot), length - 1), slot)
        slots = tf.concat([slots, slot[:, :, tf.newaxis]], 2)

        # the first iterations might have less than beam_size items available
        current_beam_size = tf.shape(log_probs)[1]

        """STAGE TWO: tag decoding"""

        # Determine a next tag to decode next at this slot
        flat_slot_encoding = tf.reshape(
            slot_encoding, [batch_size * current_beam_size, encoding_size])
        flat_tag_logits = decoder.get_tag_logits(flat_slot_encoding, training=training)

        # the first iterations might have less than beam_size items available
        max_beam_size = tf.minimum(length, beam_size)

        # mask out elements of beams that are closed that are repeats
        open_beam_mask = 1.0 - tf.cast(
            tf.logical_and( closed[:, :, tf.newaxis],  tf.concat([
                tf.fill([batch_size, current_beam_size, 1], False),
                tf.fill([batch_size, current_beam_size, max_beam_size - 1], True)], 2)), tf.float32)
        closed_beam_bias = -1e9 * (1.0 - open_beam_mask)

        # compute the log probabilities of the top k possible tags
        tag_log_probs = tf.reshape(
            tf.nn.log_softmax(flat_tag_logits), [
                batch_size, current_beam_size, tf.shape(flat_tag_logits)[1]])
        tag_log_probs, next_tag = tf.math.top_k(tag_log_probs, k=max_beam_size)
        tag_log_probs = (open_beam_mask * tag_log_probs) + closed_beam_bias

        # the first iterations might have less than beam_size items available
        next_max_beam_size = tf.minimum(current_beam_size * max_beam_size, beam_size)

        # compute an outer product of every tag with every beam (returns k^2 options)
        log_probs = tf.reshape(
            log_probs[:, :, tf.newaxis] + tag_log_probs, [batch_size, current_beam_size * max_beam_size])
        log_probs, beam_indices = tf.math.top_k(log_probs, k=next_max_beam_size)

        words = tf.gather(
            expand_beam_dim(words, current_beam_size, max_beam_size), beam_indices, batch_dims=1)
        tags = tf.gather(
            expand_beam_dim(tags, current_beam_size, max_beam_size), beam_indices, batch_dims=1)
        slots = tf.gather(
            expand_beam_dim(slots, current_beam_size, max_beam_size), beam_indices, batch_dims=1)
        closed = tf.gather(
            expand_beam_dim(closed, current_beam_size, max_beam_size), beam_indices, batch_dims=1)

        slot = tf.gather(
            expand_beam_dim(slot, current_beam_size, max_beam_size), beam_indices, batch_dims=1)
        slot_encoding = tf.gather(
            expand_beam_dim(slot_encoding, current_beam_size, max_beam_size), beam_indices, batch_dims=1)
        next_tag = tf.gather(
            tf.reshape(next_tag, [batch_size, current_beam_size * max_beam_size]), beam_indices, batch_dims=1)

        # if the model if finished, the predicted tag should be padding (index 0)
        next_tag = tf.where(closed, tf.zeros_like(next_tag), next_tag)
        tags = insert_into_slot(tags, next_tag, slot)
        
        # the first iterations might have less than beam_size items available
        current_beam_size = tf.shape(log_probs)[1]

        """STAGE THREE: word decoding"""

        # Determine a next word to decode next at this slot
        flat_slot_encoding = tf.reshape(
            slot_encoding, [batch_size * beam_size, encoding_size])
        flat_next_tag = tf.reshape(next_tag, [batch_size * beam_size])
        flat_word_logits = decoder.get_word_logits(
            flat_slot_encoding, flat_next_tag, training=training)

        # the first iterations might have less than beam_size items available
        max_beam_size = tf.minimum(length, beam_size)

        # mask out elements of beams that are closed that are repeats
        open_beam_mask = 1.0 - tf.cast(
            tf.logical_and( closed[:, :, tf.newaxis],  tf.concat([
                tf.fill([batch_size, current_beam_size, 1], False),
                tf.fill([batch_size, current_beam_size, max_beam_size - 1], True)], 2)), tf.float32)
        closed_beam_bias = -1e9 * (1.0 - open_beam_mask)

        # compute the log probabilities of the top k possible tags
        word_log_probs = tf.reshape(
            tf.nn.log_softmax(flat_word_logits), [
                batch_size, current_beam_size, tf.shape(flat_word_logits)[1]])
        word_log_probs, next_word = tf.math.top_k(word_log_probs, k=max_beam_size)
        word_log_probs = (word_log_probs * pointer_log_probs) + closed_beam_bias

        # the first iterations might have less than beam_size items available
        next_max_beam_size = tf.minimum(current_beam_size * max_beam_size, beam_size)

        # compute an outer product of every tag with every beam (returns k^2 options)
        log_probs = tf.reshape(
            log_probs[:, :, tf.newaxis] + word_log_probs, [batch_size, current_beam_size * max_beam_size])
        log_probs, beam_indices = tf.math.top_k(log_probs, k=next_max_beam_size)

        words = tf.gather(
            expand_beam_dim(words, current_beam_size, max_beam_size), beam_indices, batch_dims=1)
        tags = tf.gather(
            expand_beam_dim(tags, current_beam_size, max_beam_size), beam_indices, batch_dims=1)
        slots = tf.gather(
            expand_beam_dim(slots, current_beam_size, max_beam_size), beam_indices, batch_dims=1)
        closed = tf.gather(
            expand_beam_dim(closed, current_beam_size, max_beam_size), beam_indices, batch_dims=1)

        slot = tf.gather(
            expand_beam_dim(slot, current_beam_size, max_beam_size), beam_indices, batch_dims=1)
        slot_encoding = tf.gather(
            expand_beam_dim(slot_encoding, current_beam_size, max_beam_size), beam_indices, batch_dims=1)
        next_tag = tf.gather(
            expand_beam_dim(next_tag, current_beam_size, max_beam_size), beam_indices, batch_dims=1)

        # select the top k of the options
        next_word = tf.gather(
            tf.reshape(next_word, [batch_size, current_beam_size * max_beam_size]), beam_indices, batch_dims=1)

        # if the model if finished, the predicted word should be padding (index 0)
        next_word = tf.where(closed, tf.zeros_like(next_word), next_word)
        words = insert_into_slot(words, next_word, slot)

    return words, tags, slots, log_probs
