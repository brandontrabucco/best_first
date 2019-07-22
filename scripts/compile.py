"""Author: Brandon Trabucco, Copyright 2019"""


from best_first.backend.process_captions import process_captions
from best_first.backend.process_images import process_images
from best_first.backend.create_tfrecords import create_tfrecords


if __name__ == "__main__":

    process_captions()
    process_images()
    create_tfrecords()
