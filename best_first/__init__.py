"""Author: Brandon Trabucco, Copyright 2019"""


from best_first.caption_utils import load_vocabulary
from best_first.caption_utils import load_parts_of_speech
from best_first.ordered_decoder import OrderedDecoder
from best_first.dataset_utils import create_dataset


decoder_params = dict(
    hidden_size=512,
    dtype="float32",
    num_hidden_layers=2,
    num_heads=8,
    attention_dropout=0.1,
    filter_size=512,
    relu_dropout=0.1,
    layer_postprocess_dropout=0.1,
)
