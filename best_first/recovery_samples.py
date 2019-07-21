"""Author: Brandon Trabucco, Copyright 2019"""


import numpy as np
from collections import namedtuple


BestFirstSample = namedtuple("BestFirstSample", [
    "words", "tags", "new_word", "new_tag", "slot"])


def recovery_samples_recursive(
        existing_word_ids,
        existing_tag_ids,
        candidate_word_ids,
        candidate_tag_ids,
        candidate_slots,
        word_ids_weight,
        tag_ids_weight,
        closed_set
):
    samples = []
    if len(existing_word_ids) == 0:
        return samples
    for i, word in enumerate(existing_word_ids):
        new_existing_word_ids = np.append(
            existing_word_ids[:i], existing_word_ids[(i + 1):])
        new_existing_tag_ids = np.append(
            existing_tag_ids[:i], existing_tag_ids[(i + 1):])
        new_candidate_word_ids = np.append(
            candidate_word_ids, existing_word_ids[i:(i + 1)])
        new_candidate_tag_ids = np.append(
            candidate_tag_ids, existing_tag_ids[i:(i + 1)])
        new_candidate_slots = np.append(
            np.where(i < candidate_slots, candidate_slots - 1, candidate_slots), [i])
        key = tuple(new_existing_word_ids)
        if key not in closed_set:
            closed_set.add(key)
            best_index = np.argmax(np.log(
                new_candidate_word_ids * word_ids_weight +
                np.exp(new_candidate_tag_ids * tag_ids_weight)))
            best_word_id = new_candidate_word_ids[best_index]
            best_tag_id = new_candidate_tag_ids[best_index]
            best_slot = new_candidate_slots[best_index]
            samples.append(
                BestFirstSample(
                    words=([2] + new_existing_word_ids.tolist() + [3]),
                    tags=([1] + new_existing_tag_ids.tolist() + [1]),
                    new_word=best_word_id,
                    new_tag=best_tag_id,
                    slot=(1 + best_slot)))
            samples.extend(
                recovery_samples_recursive(
                    new_existing_word_ids,
                    new_existing_tag_ids,
                    new_candidate_word_ids,
                    new_candidate_tag_ids,
                    new_candidate_slots,
                    word_ids_weight,
                    tag_ids_weight,
                    closed_set))
    return samples


def recovery_samples(
        existing_word_ids,
        existing_tag_ids,
        word_ids_weight,
        tag_ids_weight
):
    samples = recovery_samples_recursive(
        existing_word_ids,
        existing_tag_ids,
        np.zeros([], dtype=np.int32),
        np.zeros([], dtype=np.int32),
        np.zeros([], dtype=np.int32),
        word_ids_weight,
        tag_ids_weight,
        set())
    samples.append(
        BestFirstSample(
            words=([2] + existing_word_ids.tolist() + [3]),
            tags=([1] + existing_tag_ids.tolist() + [1]),
            new_word=0,
            new_tag=0,
            slot=(1 + len(existing_word_ids))))
    return samples
