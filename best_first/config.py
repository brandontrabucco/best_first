"""Author: Brandon Trabucco, Copyright 2019"""


logging_dir = "./"


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


image_height = 299
image_width = 299


batch_size = 256
num_epochs = 1000
queue_size = 5096


learning_rate = 0.0001
pointer_loss_weight = 1.0
tag_loss_weight = 0.1
word_loss_weight = 0.01


word_embedding_size = 1024
tag_embedding_size = 1024


num_heads = 8
attention_hidden_size = 32
dense_hidden_size = 1024


num_layers = 10
hidden_size = 256
output_size = 1024
