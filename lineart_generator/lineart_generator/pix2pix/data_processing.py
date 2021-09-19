# Code samples from https://www.tensorflow.org/tutorials/generative/pix2pix,
# Accessed June 19, 2021.
#
# Content is licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Data processing functions.

Code samples modified slightly to break into separate modules.

"""

import json
from math import ceil
import os

import tensorflow as tf
from tensorflow.math import ceil, divide, minimum, multiply, not_equal


IMG_WIDTH = 256
IMG_HEIGHT = 256


def make_target_path(image_path):

    head, im_filename = os.path.split(image_path)

    data_dir, im_dir = os.path.split(head)
    target_dir = os.path.join(data_dir, 'lineart_' + im_dir.split('_')[1])

    return os.path.join(target_dir, im_filename)


def read_decode_image(image_path):
    # Read and decode an image file to a uint8 tensor
    image = tf.io.read_file(image_path)
    im_tensor = tf.image.decode_jpeg(image)
    return im_tensor


def load(image_path, target_path):
    
    input_image = read_decode_image(image_path)
    target_image = read_decode_image(target_path)

    h, w = (tf.shape(input_image)[0], 
            tf.shape(input_image)[1]) 
    h, w = (tf.cast(h, tf.float32), 
            tf.cast(w, tf.float32))

    h_t, w_t = (tf.shape(target_image)[0], 
                tf.shape(target_image)[1])
    h_t, w_t = (tf.cast(h_t, tf.float32), 
                tf.cast(w_t, tf.float32))

    if not_equal(h_t, h) or not_equal(w_t, w):
        print(h_t, h)
        print('resize target')
        target_image = resize_single(target_image, h, w)

    if h < IMG_HEIGHT or w < IMG_WIDTH:
        # print('make larger')
        min_dim = minimum(h, w)
        factor = tf.cast(ceil(divide(IMG_HEIGHT, min_dim)), tf.float32)
        input_image, target_image = resize(input_image, target_image, multiply(factor, h), multiply(factor, w))

    if h > IMG_HEIGHT or w > IMG_WIDTH:
        # print('random crop')
        input_image, target_image = random_crop(input_image, target_image)
    
    # Convert both images to float32 tensors
    input_image = tf.cast(input_image, tf.float32)
    target_image = tf.cast(target_image, tf.float32)

    return input_image, target_image


def resize(input_image, target_image, height, width):
    input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    target_image = tf.image.resize(target_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return input_image, target_image


def resize_single(image, height, width):
    print('resize target')
    image = tf.image.resize(image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return image


def random_crop(input_image, target_image):
    stacked_image = tf.stack([input_image, target_image], axis=0)
    cropped_image = tf.image.random_crop(
        stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])

    return cropped_image[0], cropped_image[1]


# Normalizing the images to [-1, 1]
def normalize(input_image, target_image):
    input_image = (input_image / 127.5) - 1
    target_image = (target_image / 127.5) - 1

    return input_image, target_image


@tf.function()
def random_jitter(input_image, target_image):
    # Resizing to 286x286
    input_image, target_image = resize(input_image, target_image, 286, 286)

    # Random cropping back to 256x256
    input_image, target_image = random_crop(input_image, target_image)

    if tf.random.uniform(()) > 0.5:
        # Random mirroring
        input_image = tf.image.flip_left_right(input_image)
        target_image = tf.image.flip_left_right(target_image)

    return input_image, target_image


def load_image_train(path_tuple):

    image_file, target_file = path_tuple[0], path_tuple[1]

    input_image, target_image = load(image_file, target_file)
    input_image, target_image = random_jitter(input_image, target_image)
    input_image, target_image = normalize(input_image, target_image)

    return input_image, target_image


def load_image_test(path_tuple):

    image_file, target_file = path_tuple[0], path_tuple[1]

    input_image, target_image = load(image_file, target_file)
    input_image, target_image = resize(input_image, target_image,
                                    IMG_HEIGHT, IMG_WIDTH)
    input_image, target_image = normalize(input_image, target_image)

    return input_image, target_image


def get_image_paths(parent_dir):

    image_paths = []
    for input_dir in ['photo_obv', 'photo_rev']:
        root = os.path.join(parent_dir, input_dir)
        image_paths.extend([os.path.join(root, file) for file in os.listdir(root)])
    
    paired_paths = []
    for image_path in image_paths:
        paired_paths.append((image_path, make_target_path(image_path)))

    return paired_paths


def prepare_datasets(data_config):
    parent_dir, buffer_size, batch_size = (data_config['parent_directory'], 
                                           data_config['buffer_size'],  
                                           data_config['batch_size'])

    train_paths = get_image_paths(os.path.join(parent_dir, 'train'))
    train_dataset = tf.data.Dataset.from_tensor_slices(train_paths)
    train_dataset = train_dataset.map(load_image_train,
                                      num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.shuffle(buffer_size)
    train_dataset = train_dataset.batch(batch_size)

    test_paths = get_image_paths(os.path.join(parent_dir, 'test'))
    test_dataset = tf.data.Dataset.from_tensor_slices(test_paths)
    test_dataset = test_dataset.map(load_image_test)
    test_dataset = test_dataset.batch(batch_size)

    return train_dataset, test_dataset


if __name__ == '__main__':
    config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'training_data.json')
    train_dataset, test_dataset = prepare_datasets(config_path)