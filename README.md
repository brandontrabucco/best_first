# Best First

Non Sequential Decoding Strategies. To replicate our results, run the following commands.

```
git clone https://github.com/brandontrabucco/best_first

cd best_first

conda create -n best_first python=3.6

conda activate best_first

pip install -e .

export PYTHONPATH="$PYTHONPATH:path/to/tensorflow/models/"

cd data

python process_images.py --image_folder images/ --image_feature_folder image_features/ --batch_size=64  --image_height 299 --image_width 299



python process_captions.py --caption_folder captions/ --caption_feature_folder caption_features_best_first_0_violations/ --ordering "best_first" --max_violations 0 --min_word_frequency 5 --vocab_file vocab.txt --max_caption_length 20

python create_tfrecords.py --image_feature_folder image_features/ --caption_feature_folder caption_features_best_first_0_violations/ --tfrecord_folder tfrecords_best_first_0_violations/ --num_samples_per_shard 4096

python process_captions.py --caption_folder captions/ --caption_feature_folder caption_features_best_first_0_violations/ --ordering "best_first" --max_violations 1 --min_word_frequency 5 --vocab_file vocab.txt --max_caption_length 20

python create_tfrecords.py --image_feature_folder image_features/ --caption_feature_folder caption_features_best_first_1_violations/ --tfrecord_folder tfrecords_best_first_0_violations/ --num_samples_per_shard 4096

python process_captions.py --caption_folder captions/ --caption_feature_folder caption_features_best_first_2_violations/ --ordering "best_first" --max_violations 2 --min_word_frequency 5 --vocab_file vocab.txt --max_caption_length 20

python create_tfrecords.py --image_feature_folder image_features/ --caption_feature_folder caption_features_best_first_2_violations/ --tfrecord_folder tfrecords_best_first_2_violations/ --num_samples_per_shard 4096



python process_captions.py --caption_folder captions/ --caption_feature_folder caption_features_forward_sequential_0_violations/ --ordering "forward_sequential" --max_violations 0 --min_word_frequency 5 --vocab_file vocab.txt --max_caption_length 20

python create_tfrecords.py --image_feature_folder image_features/ --caption_feature_folder caption_features_forward_sequential_0_violations/ --tfrecord_folder tfrecords_forward_sequential_0_violations/ --num_samples_per_shard 4096

python process_captions.py --caption_folder captions/ --caption_feature_folder caption_features_forward_sequential_0_violations/ --ordering "forward_sequential" --max_violations 1 --min_word_frequency 5 --vocab_file vocab.txt --max_caption_length 20

python create_tfrecords.py --image_feature_folder image_features/ --caption_feature_folder caption_features_forward_sequential_1_violations/ --tfrecord_folder tfrecords_forward_sequential_0_violations/ --num_samples_per_shard 4096

python process_captions.py --caption_folder captions/ --caption_feature_folder caption_features_forward_sequential_2_violations/ --ordering "forward_sequential" --max_violations 2 --min_word_frequency 5 --vocab_file vocab.txt --max_caption_length 20

python create_tfrecords.py --image_feature_folder image_features/ --caption_feature_folder caption_features_forward_sequential_2_violations/ --tfrecord_folder tfrecords_forward_sequential_2_violations/ --num_samples_per_shard 4096



python process_captions.py --caption_folder captions/ --caption_feature_folder caption_features_backward_sequential_0_violations/ --ordering "backward_sequential" --max_violations 0 --min_word_frequency 5 --vocab_file vocab.txt --max_caption_length 20

python create_tfrecords.py --image_feature_folder image_features/ --caption_feature_folder caption_features_backward_sequential_0_violations/ --tfrecord_folder tfrecords_backward_sequential_0_violations/ --num_samples_per_shard 4096

python process_captions.py --caption_folder captions/ --caption_feature_folder caption_features_backward_sequential_0_violations/ --ordering "backward_sequential" --max_violations 1 --min_word_frequency 5 --vocab_file vocab.txt --max_caption_length 20

python create_tfrecords.py --image_feature_folder image_features/ --caption_feature_folder caption_features_backward_sequential_1_violations/ --tfrecord_folder tfrecords_backward_sequential_0_violations/ --num_samples_per_shard 4096

python process_captions.py --caption_folder captions/ --caption_feature_folder caption_features_backward_sequential_2_violations/ --ordering "backward_sequential" --max_violations 2 --min_word_frequency 5 --vocab_file vocab.txt --max_caption_length 20

python create_tfrecords.py --image_feature_folder image_features/ --caption_feature_folder caption_features_backward_sequential_2_violations/ --tfrecord_folder tfrecords_backward_sequential_2_violations/ --num_samples_per_shard 4096



python process_captions.py --caption_folder captions/ --caption_feature_folder caption_features_random_0_violations/ --ordering "random_word" --max_violations 0 --min_word_frequency 5 --vocab_file vocab.txt --max_caption_length 20

python create_tfrecords.py --image_feature_folder image_features/ --caption_feature_folder caption_features_random_0_violations/ --tfrecord_folder tfrecords_backward_sequential_0_violations/ --num_samples_per_shard 4096

python process_captions.py --caption_folder captions/ --caption_feature_folder caption_features_random_0_violations/ --ordering "random_word" --max_violations 1 --min_word_frequency 5 --vocab_file vocab.txt --max_caption_length 20

python create_tfrecords.py --image_feature_folder image_features/ --caption_feature_folder caption_features_random_1_violations/ --tfrecord_folder tfrecords_random_0_violations/ --num_samples_per_shard 4096

python process_captions.py --caption_folder captions/ --caption_feature_folder caption_features_random_2_violations/ --ordering "random_word" --max_violations 2 --min_word_frequency 5 --vocab_file vocab.txt --max_caption_length 20

python create_tfrecords.py --image_feature_folder image_features/ --caption_feature_folder caption_features_random_2_violations/ --tfrecord_folder tfrecords_random_2_violations/ --num_samples_per_shard 4096



cd ../scripts
 
python train.py --tfrecord_folder tfrecords_best_first_0_violations/ --vocab_file ../data/vocab.txt --logging_dir ../best_first_0_violations --model ../best_first_0_violations/model.ckpt
 
python train.py --tfrecord_folder tfrecords_best_first_1_violations/ --vocab_file ../data/vocab.txt --logging_dir ../best_first_1_violations --model ../best_first_1_violations/model.ckpt
 
python train.py --tfrecord_folder tfrecords_best_first_2_violations/ --vocab_file ../data/vocab.txt --logging_dir ../best_first_2_violations --model ../best_first_2_violations/model.ckptckpt


 
python train.py --tfrecord_folder tfrecords_forward_sequential_0_violations/ --vocab_file ../data/vocab.txt --logging_dir ../forward_sequential_0_violations --model ../forward_sequential_0_violations/model.ckpt
 
python train.py --tfrecord_folder tfrecords_forward_sequential_1_violations/ --vocab_file ../data/vocab.txt --logging_dir ../forward_sequential_1_violations --model ../forward_sequential_1_violations/model.ckpt
 
python train.py --tfrecord_folder tfrecords_forward_sequential_2_violations/ --vocab_file ../data/vocab.txt --logging_dir ../forward_sequential_2_violations --model ../forward_sequential_2_violations/model.ckpt


 
python train.py --tfrecord_folder tfrecords_backward_sequential_0_violations/ --vocab_file ../data/vocab.txt --logging_dir ../backward_sequential_0_violations --model ../backward_sequential_0_violations/model.ckpt
 
python train.py --tfrecord_folder tfrecords_backward_sequential_1_violations/ --vocab_file ../data/vocab.txt --logging_dir ../backward_sequential_1_violations --model ../backward_sequential_1_violations/model.ckpt
 
python train.py --tfrecord_folder tfrecords_backward_sequential_2_violations/ --vocab_file ../data/vocab.txt --logging_dir ../backward_sequential_2_violations --model ../backward_sequential_2_violations/model.ckpt


 
python train.py --tfrecord_folder tfrecords_random_0_violations/ --vocab_file ../data/vocab.txt --logging_dir ../random_0_violations --model ../random_0_violations/model.ckpt
 
python train.py --tfrecord_folder tfrecords_random_1_violations/ --vocab_file ../data/vocab.txt --logging_dir ../random_1_violations --model ../random_1_violations/model.ckpt
 
python train.py --tfrecord_folder tfrecords_random_2_violations/ --vocab_file ../data/vocab.txt --logging_dir ../random_2_violations --model ../random_2_violations/model.ckpt

```
