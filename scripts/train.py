"""Author: Brandon Trabucco, Copyright 2019"""


from best_first.data_loader import data_loader
from best_first import load_vocabulary
from best_first import load_parts_of_speech


if __name__ == "__main__":

    vocab = load_vocabulary()
    parts_of_speech = load_parts_of_speech()

    for batch in data_loader():

        image_path = batch["image_path"]
        image = batch["image"]
        new_word = batch["new_word"]
        new_tag = batch["new_tag"]
        slot = batch["slot"]
        words = batch["words"]
        tags = batch["tags"]
        indicators = batch["indicators"]

        print("words: ", vocab.ids_to_words(words))
        print("new_word: ", vocab.ids_to_words(new_word))
        print("tags: ", parts_of_speech.ids_to_words(tags))
        print("new_tag: ", parts_of_speech.ids_to_words(new_tag))
        exit()

