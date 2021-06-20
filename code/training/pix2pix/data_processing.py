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
import os

import tensorflow as tf


IMG_WIDTH = 256
IMG_HEIGHT = 256


def load(image_file):
    
    # Read and decode an image file to a uint8 tensor
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)

    # Split each image tensor into two tensors:
    # - one with a real building facade image
    # - one with an architecture label image
    w = tf.shape(image)[1]
    w = w // 2
    input_image = image[:, w:, :]
    target_image = image[:, :w, :]

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


def load_image_train(image_file):
    input_image, target_image = load(image_file)
    input_image, target_image = random_jitter(input_image, target_image)
    input_image, target_image = normalize(input_image, target_image)

    return input_image, target_image


def load_image_test(image_file):
    input_image, target_image = load(image_file)
    input_image, target_image = resize(input_image, target_image,
                                    IMG_HEIGHT, IMG_WIDTH)
    input_image, target_image = normalize(input_image, target_image)

    return input_image, target_image


def read_dataset_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)


def prepare_datasets(config_path):
    data_config = read_dataset_config(config_path)
    parent_dir, buffer_size, batch_size = (data_config['parent_directory'], 
                                           data_config['buffer_size'],  
                                           data_config['batch_size'])

    train_dataset = tf.data.Dataset.list_files([os.path.join(parent_dir, 'train/photo_rev/*.jpg'),
                                                os.path.join(parent_dir + 'train/photo_obv/*.jpg')])
    train_dataset = train_dataset.map(load_image_train,
                                      num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.shuffle(buffer_size)
    train_dataset = train_dataset.batch(batch_size)

    test_dataset = tf.data.Dataset.list_files([os.path.join(parent_dir, 'test/photo_rev/*.jpg'),
                                               os.path.join(parent_dir, 'test/photo_obv/*.jpg')])
    test_dataset = test_dataset.map(load_image_test)
    test_dataset = test_dataset.batch(batch_size)

    return train_dataset, test_dataset