"""Author: Brandon Trabucco, Copyright 2019"""


image_folder = "images/"
image_feature_folder = "image_features/"

caption_folder = "captions/"
caption_feature_folder = "caption_features/"

vocab_file = "vocab.txt"
tagger_file = "tagger.pkl"

tfrecord_folder = "tfrecords/"
num_samples_per_shard = 5096

min_word_frequency = 1
max_caption_length = 13

image_height = 299
image_width = 299

batch_size = 1024
num_epochs = 10
queue_size = 5096
logging_delay = 10

logging_dir = "./"

parts_of_speech = [
    "<pad>",
    "<unk>",
    ".",
    "CONJ",
    "DET",
    "ADP",
    "PRT",
    "PRON",
    "ADV",
    "NUM",
    "ADJ",
    "VERB",
    "NOUN"
]
