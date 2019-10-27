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

python data/process_captions.py --caption_folder data/captions/ --caption_feature_folder data/caption_features_best_first_2_violations/ --ordering "best_first" --max_violations 2 --min_word_frequency 5 --vocab_file data/vocab.txt --max_caption_length 20

python data/create_tfrecords.py --image_feature_folder data/image_features/ --caption_feature_folder data/caption_features_best_first_2_violations/ --tfrecord_folder data/tfrecords_best_first_2_violations/ --num_samples_per_shard 4096

python scripts/train.py --tfrecord_folder data/tfrecords_best_first_2_violations/ --vocab_file data/vocab.txt --logging_dir best_first_2_violations --model best_first_2_violations/model.ckptckpt
```
