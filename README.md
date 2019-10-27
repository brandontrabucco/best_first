# Non Sequential Decoding Strategies

This repository implements a non sequential auto regressive model that generates captions. The model has the ability to insert new words into already generated sentences.

# Installation

To install our implementation, run the following:

```
git clone https://github.com/brandontrabucco/best_first
cd best_first
git submodule init
git submodule update
pip install -e .
export PYTHONPATH="$PYTHONPATH:path/to/tensorflow/models/"
```

# Building The Datasets


To build the datasets in our experiments, place images and captions into the data folder into the corresponding `images/` and `captions/` folders. Then, run the following build scripts:

```
python data/process_images.py --image_folder data/images/ --image_feature_folder data/image_features/ --batch_size=64  --image_height 299 --image_width 299

python data/process_captions.py --caption_folder data/captions/ --caption_feature_folder data/caption_features_best_first_0_violations/ --ordering "best_first" --max_violations 0 --min_word_frequency 5 --vocab_file data/vocab.txt --max_caption_length 20
python data/create_tfrecords.py --image_feature_folder data/image_features/ --caption_feature_folder data/caption_features_best_first_0_violations/ --tfrecord_folder data/tfrecords_best_first_0_violations/ --num_samples_per_shard 4096

python data/process_captions.py --caption_folder data/captions/ --caption_feature_folder data/caption_features_best_first_0_violations/ --ordering "best_first" --max_violations 1 --min_word_frequency 5 --vocab_file data/vocab.txt --max_caption_length 20
data/python create_tfrecords.py --image_feature_folder data/image_features/ --caption_feature_folder data/caption_features_best_first_1_violations/ --tfrecord_folder data/tfrecords_best_first_0_violations/ --num_samples_per_shard 4096

python data/process_captions.py --caption_folder data/captions/ --caption_feature_folder data/caption_features_best_first_2_violations/ --ordering "best_first" --max_violations 2 --min_word_frequency 5 --vocab_file data/vocab.txt --max_caption_length 20
data/python create_tfrecords.py --image_feature_folder data/image_features/ --caption_feature_folder data/caption_features_best_first_2_violations/ --tfrecord_folder data/tfrecords_best_first_2_violations/ --num_samples_per_shard 4096

python data/process_captions.py --caption_folder data/captions/ --caption_feature_folder data/caption_features_forward_sequential_0_violations/ --ordering "forward_sequential" --max_violations 0 --min_word_frequency 5 --vocab_file data/vocab.txt --max_caption_length 20
data/python create_tfrecords.py --image_feature_folder data/image_features/ --caption_feature_folder data/caption_features_forward_sequential_0_violations/ --tfrecord_folder data/tfrecords_forward_sequential_0_violations/ --num_samples_per_shard 4096

python data/process_captions.py --caption_folder data/captions/ --caption_feature_folder data/caption_features_forward_sequential_0_violations/ --ordering "forward_sequential" --max_violations 1 --min_word_frequency 5 --vocab_file data/vocab.txt --max_caption_length 20
python data/create_tfrecords.py --image_feature_folder data/image_features/ --caption_feature_folder data/caption_features_forward_sequential_1_violations/ --tfrecord_folder data/tfrecords_forward_sequential_0_violations/ --num_samples_per_shard 4096

python data/process_captions.py --caption_folder data/captions/ --caption_feature_folder data/caption_features_forward_sequential_2_violations/ --ordering "forward_sequential" --max_violations 2 --min_word_frequency 5 --vocab_file data/vocab.txt --max_caption_length 20
python data/create_tfrecords.py --image_feature_folder data/image_features/ --caption_feature_folder data/caption_features_forward_sequential_2_violations/ --tfrecord_folder data/tfrecords_forward_sequential_2_violations/ --num_samples_per_shard 4096

python data/process_captions.py --caption_folder data/captions/ --caption_feature_folder data/caption_features_backward_sequential_0_violations/ --ordering "backward_sequential" --max_violations 0 --min_word_frequency 5 --vocab_file data/vocab.txt --max_caption_length 20
python data/create_tfrecords.py --image_feature_folder data/image_features/ --caption_feature_folder data/caption_features_backward_sequential_0_violations/ --tfrecord_folder data/tfrecords_backward_sequential_0_violations/ --num_samples_per_shard 4096

python data/process_captions.py --caption_folder data/captions/ --caption_feature_folder data/caption_features_backward_sequential_0_violations/ --ordering "backward_sequential" --max_violations 1 --min_word_frequency 5 --vocab_file data/vocab.txt --max_caption_length 20
python data/create_tfrecords.py --image_feature_folder data/image_features/ --caption_feature_folder data/caption_features_backward_sequential_1_violations/ --tfrecord_folder data/tfrecords_backward_sequential_0_violations/ --num_samples_per_shard 4096

python data/process_captions.py --caption_folder data/captions/ --caption_feature_folder data/caption_features_backward_sequential_2_violations/ --ordering "backward_sequential" --max_violations 2 --min_word_frequency 5 --vocab_file data/vocab.txt --max_caption_length 20
python data/create_tfrecords.py --image_feature_folder data/image_features/ --caption_feature_folder data/caption_features_backward_sequential_2_violations/ --tfrecord_folder data/tfrecords_backward_sequential_2_violations/ --num_samples_per_shard 4096

python data/process_captions.py --caption_folder data/captions/ --caption_feature_folder data/caption_features_random_0_violations/ --ordering "random_word" --max_violations 0 --min_word_frequency 5 --vocab_file data/vocab.txt --max_caption_length 20
python data/create_tfrecords.py --image_feature_folder data/image_features/ --caption_feature_folder data/caption_features_random_0_violations/ --tfrecord_folder data/tfrecords_backward_sequential_0_violations/ --num_samples_per_shard 4096

python data/process_captions.py --caption_folder data/captions/ --caption_feature_folder data/caption_features_random_0_violations/ --ordering "random_word" --max_violations 1 --min_word_frequency 5 --vocab_file data/vocab.txt --max_caption_length 20
python data/create_tfrecords.py --image_feature_folder data/image_features/ --caption_feature_folder data/caption_features_random_1_violations/ --tfrecord_folder data/tfrecords_random_0_violations/ --num_samples_per_shard 4096

python data/process_captions.py --caption_folder data/captions/ --caption_feature_folder data/caption_features_random_2_violations/ --ordering "random_word" --max_violations 2 --min_word_frequency 5 --vocab_file data/vocab.txt --max_caption_length 20
python data/create_tfrecords.py --image_feature_folder data/image_features/ --caption_feature_folder data/caption_features_random_2_violations/ --tfrecord_folder data/tfrecords_random_2_violations/ --num_samples_per_shard 4096
 
python scripts/train.py --tfrecord_folder data/tfrecords_best_first_0_violations/ --vocab_file data/vocab.txt --logging_dir best_first_0_violations --model best_first_0_violations/model.ckpt
python scripts/train.py --tfrecord_folder data/tfrecords_best_first_1_violations/ --vocab_file data/vocab.txt --logging_dir best_first_1_violations --model best_first_1_violations/model.ckpt
python scripts/train.py --tfrecord_folder data/tfrecords_best_first_2_violations/ --vocab_file data/vocab.txt --logging_dir best_first_2_violations --model best_first_2_violations/model.ckptckpt

python scripts/train.py --tfrecord_folder data/tfrecords_forward_sequential_0_violations/ --vocab_file data/vocab.txt --logging_dir forward_sequential_0_violations --model forward_sequential_0_violations/model.ckpt
python scripts/train.py --tfrecord_folder data/tfrecords_forward_sequential_1_violations/ --vocab_file data/vocab.txt --logging_dir forward_sequential_1_violations --model forward_sequential_1_violations/model.ckpt
python scripts/train.py --tfrecord_folder data/tfrecords_forward_sequential_2_violations/ --vocab_file data/vocab.txt --logging_dir forward_sequential_2_violations --model forward_sequential_2_violations/model.ckpt

python scripts/train.py --tfrecord_folder data/tfrecords_backward_sequential_0_violations/ --vocab_file data/vocab.txt --logging_dir backward_sequential_0_violations --model backward_sequential_0_violations/model.ckpt
python scripts/train.py --tfrecord_folder data/tfrecords_backward_sequential_1_violations/ --vocab_file data/vocab.txt --logging_dir backward_sequential_1_violations --model backward_sequential_1_violations/model.ckpt
python scripts/train.py --tfrecord_folder data/tfrecords_backward_sequential_2_violations/ --vocab_file data/vocab.txt --logging_dir backward_sequential_2_violations --model backward_sequential_2_violations/model.ckpt

python scripts/train.py --tfrecord_folder data/tfrecords_random_0_violations/ --vocab_file data/vocab.txt --logging_dir random_0_violations --model random_0_violations/model.ckpt
python scripts/train.py --tfrecord_folder data/tfrecords_random_1_violations/ --vocab_file data/vocab.txt --logging_dir random_1_violations --model random_1_violations/model.ckpt
python scripts/train.py --tfrecord_folder data/tfrecords_random_2_violations/ --vocab_file data/vocab.txt --logging_dir random_2_violations --model random_2_violations/model.ckpt
```
