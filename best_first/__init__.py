"""Author: Brandon Trabucco, Copyright 2019"""


from best_first.caption_utils import load_vocabulary
from best_first.caption_utils import load_parts_of_speech
from best_first.ordered_decoder import OrderedDecoder
from best_first.dataset_utils import create_dataset


decoder_params = dict(
    hidden_size=1024,
    dtype="float32",
    num_hidden_layers=6,
    num_heads=16,
    attention_dropout=0.3,
    filter_size=4096,
    relu_dropout=0.3,
    layer_postprocess_dropout=0.3)
