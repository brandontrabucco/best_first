"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf


def expand_beam_dim(x):
    return tf.reshape(
        tf.tile(
            x[:, :, tf.newaxis, ...], 
            [1, 1, beam_size, *[1 for _ in tf.shape(x)[3:]]]), 
        [batch_size, beam_size ** 2, *tf.shape(x)[3:]])


def beam_search(
        images,
        decoder,
        beam_size=7,
        training=False
):
    # pad the images to be the right shape for beam search
    batch_size = tf.shape(images)[0]
    images = tf.tile(images[:, tf.newaxis, :, :], [1, beam_size, 1, 1])
    flat_images = tf.reshape(
        images, [batch_size * beam_size, tf.shape(images)[2], tf.shape(images)[3]])

    # initialize starting values for every beam
    log_probs = tf.zeros([batch_size, beam_size], dtype=tf.float32)
    beam_words = tf.zeros([batch_size, beam_size, 0], dtype=tf.int32)
    beam_tags = tf.zeros([batch_size, beam_size, 0], dtype=tf.int32)
    beam_slots = tf.zeros([batch_size, beam_size, 0], dtype=tf.int32)
    closed = tf.fill([batch_size, beam_size], False)

    # mask out elements of beams that are closed that are repeats
    open_beam_mask = 1.0 - tf.cast(
        tf.logical_and( closed[:, :, tf.newaxis],  tf.concat([
            tf.fill([batch_size, beam_size, 1], False),
            tf.fill([batch_size, beam_size, beam_size - 1], True)], 2)), tf.float32)
    closed_beam_bias = -1e9 * (1.0 - open_beam_mask)

    # perform a beam search until all beams are closed
    while not tf.reduce_all(closed):
        length = tf.shape(words)[2]
        is_first_iteration = (length.numpy() == 2)

        """STAGE ONE: slot decoding"""

        # Compute the encodings for each slot using the transformer model
        flat_words = tf.reshape(words, [batch_size * beam_size, length])
        flat_tags = tf.reshape(tags, [batch_size * beam_size, length])
        flat_pointer_encodings = self.get_pointer_encodings(
            flat_images, flat_words, flat_tags, training=training)

        # note the pointer_encodings need to not be flat after this point
        encoding_size = tf.shape(flat_pointer_encodings)[2]
        pointer_encodings = tf.reshape(
            flat_pointer_encodings, [batch_size, beam_size, length, encoding_size])

        # compute the pointer logits over slots to insert words next
        flat_pointer_logits = self.get_pointer_logits(
            flat_pointer_encodings, training=training)

        # compute the log probabilities of the top k possible pointers
        pointer_log_probs = tf.reshape(
            tf.log_softmax(flat_pointer_logits), [batch_size, beam_size, length])
        pointer_log_probs, slot = tf.math.top_k(pointer_log_probs, k=beam_size)
        pointer_log_probs = (open_beam_mask * pointer_log_probs) + closed_beam_bias
        slot_encoding = tf.gather(pointer_encodings, slot, batch_dims=2)

        # if this is the first iteration then accept all beams
        if is_first_iteration:
            log_probs = log_probs + pointer_log_probs[:, 0, ...]
            slot = slot[:, 0, ...]
            slot_encoding = slot_encoding[:, 0, ...]

        # compute every combination of pointer and beam (returns k^2 options)
        else:
            log_probs = tf.reshape(
                log_probs[:, :, tf.newaxis] + pointer_log_probs, [batch_size, beam_size ** 2])
            log_probs, beam_indices = tf.math.top_k(log_probs, k=beam_size)

            words = tf.gather(
                expand_beam_dim(words), beam_indices, batch_dims=1)
            tags = tf.gather(
                expand_beam_dim(tags), beam_indices, batch_dims=1)
            slots = tf.gather(
                expand_beam_dim(slots), beam_indices, batch_dims=1)
            closed = tf.gather(
                expand_beam_dim(closed), beam_indices, batch_dims=1)

            slot = tf.gather(
                tf.reshape(slot, [batch_size, beam_size ** 2]), beam_indices, batch_dims=1)
            slot_encoding = tf.gather(
                tf.reshape(slot_encoding, [
                    batch_size, beam_size ** 2, encoding_size]), beam_indices, batch_dims=1)

        # if the model points to the final slot, the model is finished and remains finished
        closed = tf.logical_or(closed, tf.equal(slot, length - 1))
        slot = tf.where(closed, tf.fill(tf.shape(slot), length - 1), slot)
        slots = tf.concat([slots, slot[:, :, tf.newaxis]], 2)

        # mask out elements of beams that are closed that are repeats
        open_beam_mask = 1.0 - tf.cast(
            tf.logical_and( closed[:, :, tf.newaxis],  tf.concat([
                tf.fill([batch_size, beam_size, 1], False),
                tf.fill([batch_size, beam_size, beam_size - 1], True)], 2)), tf.float32)
        closed_beam_bias = -1e9 * (1.0 - open_beam_mask)

        """STAGE TWO: tag decoding"""

        # Determine a next tag to decode next at this slot
        flat_slot_encoding = tf.reshape(
            slot_encoding, [batch_size * beam_size, encoding_size])
        flat_tag_logits = self.get_tag_logits(flat_slot_encoding, training=training)

        # compute the log probabilities of the top k possible tags
        tag_log_probs = tf.reshape(
            tf.log_softmax(flat_tag_logits), [
                batch_size, beam_size, tf.shape(flat_tag_logits)[1]])
        tag_log_probs, next_tag = tf.math.top_k(tag_log_probs, k=beam_size)
        tag_log_probs = (tag_log_probs * pointer_log_probs) + closed_beam_bias

        # compute an outer product of every tag with every beam (returns k^2 options)
        log_probs = tf.reshape(
            log_probs[:, :, tf.newaxis] + tag_log_probs, [batch_size, beam_size ** 2])
        log_probs, beam_indices = tf.math.top_k(log_probs, k=beam_size)

        words = tf.gather(
            expand_beam_dim(words), beam_indices, batch_dims=1)
        tags = tf.gather(
            expand_beam_dim(tags), beam_indices, batch_dims=1)
        slots = tf.gather(
            expand_beam_dim(slots), beam_indices, batch_dims=1)
        closed = tf.gather(
            expand_beam_dim(closed), beam_indices, batch_dims=1)

        slot = tf.gather(
            expand_beam_dim(slot), beam_indices, batch_dims=1)
        slot_encoding = tf.gather(
            expand_beam_dim(slot_encoding), beam_indices, batch_dims=1)
        next_tag = tf.gather(
            tf.reshape(next_tag, [batch_size, beam_size ** 2]), beam_indices, batch_dims=1)

        # if the model if finished, the predicted tag should be padding (index 0)
        next_tag = tf.where(closed, tf.zeros_like(next_tag), next_tag)
        tags = tf.concat([tags, next_tag[:, :, tf.newaxis]], 2)

        """STAGE THREE: word decoding"""

        # Determine a next word to decode next at this slot
        flat_slot_encoding = tf.reshape(
            slot_encoding, [batch_size * beam_size, encoding_size])
        flat_next_tag = tf.reshape(next_tag, [batch_size * beam_size])
        flat_word_logits = self.get_word_logits(
            flat_slot_encoding, flat_next_tag, training=training)

        # compute the log probabilities of the top k possible tags
        word_log_probs = tf.reshape(
            tf.log_softmax(flat_word_logits), [
                batch_size, beam_size, tf.shape(flat_word_logits)[1]])
        word_log_probs, next_word = tf.math.top_k(word_log_probs, k=beam_size)
        word_log_probs = (word_log_probs * pointer_log_probs) + closed_beam_bias

        # compute an outer product of every tag with every beam (returns k^2 options)
        log_probs = tf.reshape(
            log_probs[:, :, tf.newaxis] + word_log_probs, [batch_size, beam_size ** 2])
        log_probs, beam_indices = tf.math.top_k(log_probs, k=beam_size)

        words = tf.gather(
            expand_beam_dim(words), beam_indices, batch_dims=1)
        tags = tf.gather(
            expand_beam_dim(tags), beam_indices, batch_dims=1)
        slots = tf.gather(
            expand_beam_dim(slots), beam_indices, batch_dims=1)
        closed = tf.gather(
            expand_beam_dim(closed), beam_indices, batch_dims=1)

        slot = tf.gather(
            expand_beam_dim(slot), beam_indices, batch_dims=1)
        slot_encoding = tf.gather(
            expand_beam_dim(slot_encoding), beam_indices, batch_dims=1)
        next_tag = tf.gather(
            expand_beam_dim(next_tag), beam_indices, batch_dims=1)

        # select the top k of the options
        next_word = tf.gather(
            tf.reshape(next_word, [batch_size, beam_size ** 2]), beam_indices, batch_dims=1)

        # if the model if finished, the predicted word should be padding (index 0)
        next_word = tf.where(closed, tf.zeros_like(next_word), next_word)
        words = tf.concat([words, next_word[:, :, tf.newaxis]], 2)

    return words, tags, slots, log_probs
