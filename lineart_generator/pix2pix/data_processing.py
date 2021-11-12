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

"""Data loading and preprocessing to create model training/testing data sets.

Code samples modified slightly to break into separate modules.

"""

import math
import os

import numpy as np
import tensorflow as tf
from tensorflow.image import ResizeMethod
from tensorflow.math import ceil, divide, minimum, multiply, not_equal


IMG_WIDTH = 256
IMG_HEIGHT = 256


def make_target_path(image_path):
    """Construct line art path given photo path."""
    head, im_filename = os.path.split(image_path)

    data_dir, im_dir = os.path.split(head)
    target_dir = os.path.join(data_dir, 'lineart_' + im_dir.split('_')[1])

    return os.path.join(target_dir, im_filename)


def read_decode_image(image_path):
    """Read image file and convert to a uint8 tensor."""
    image = tf.io.read_file(image_path)
    im_tensor = tf.image.decode_jpeg(image)
    return im_tensor


def load(image_path, target_path):
    """Load image files and create correctly sized/typed tensors for data set.

    Args:
        image_path (str): Path to photo of tablet face.
        target_path (str): Path to line art of tablet face.

    Returns:
        input_image (tf.Tensor, dtype 'float32', dimensions IMG_HEIGHT x IMG_WIDTH x 3):
            Tensor (photo) ready for further data augmentation or direct model
            input.
        target_image (tf.Tensor, dtype 'float32', dimensions IMG_HEIGHT x IMG_WIDTH x 3):
            Tensor (line art) ready for further data augmentation or direct
            model input.

    """
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
        min_dim = minimum(h, w)
        factor = tf.cast(ceil(divide(IMG_HEIGHT, min_dim)), tf.float32)
        input_image, target_image = resize(input_image, target_image,
                                           multiply(factor, h),
                                           multiply(factor, w))

    if h > IMG_HEIGHT or w > IMG_WIDTH:
        input_image, target_image = random_crop(input_image, target_image)

    input_image = tf.cast(input_image, tf.float32)
    target_image = tf.cast(target_image, tf.float32)

    return input_image, target_image


def resize(input_image, target_image, height, width):
    """Resize input (photo) and target (line art).

    Args:
        input_image (tf.Tensor, dtype 'uint8', dimensions m x n x 3): Tensor
            (photo) to be resized.
        target_image (tf.Tensor, dtype 'uint8', dimensions m x n x 3): Tensor
            (lineart) to be resized.
        height (tf.Tensor, dtype 'float32', dimensions 0): Output height.
        width (tf.Tensor, dtype 'float32', dimensions 0): Output width.

    Returns:
        input_image (tf.Tensor, dtype 'uint8', dimensions height x width x 3):
            Resized image tensor (photo).
        target_image (tf.Tensor, dtype 'uint8', dimensions height x width x 3):
            Resized image tensor (line art).

    """
    input_image = tf.image.resize(input_image, [height, width],
                                  method=ResizeMethod.NEAREST_NEIGHBOR)
    target_image = tf.image.resize(target_image, [height, width],
                                   method=ResizeMethod.NEAREST_NEIGHBOR)

    return input_image, target_image


def resize_single(image, height, width):
    """No longer used because resizing accomplished in data preprocessing."""
    image = tf.image.resize(image, [height, width],
                            method=ResizeMethod.NEAREST_NEIGHBOR)
    return image


def random_crop(input_image, target_image):
    """Crop random matching portion from image and target.

    Args:
        input_image (tf.Tensor, dtype 'uint8', dimensions m x n x 3): Tensor
            (photo) to be cropped.
        target_image (tf.Tensor, dtype 'uint8', dimensions m x n x 3): Tensor
            (lineart) to be cropped.

    Returns:
        cropped_image[0] (tf.Tensor, dtype 'uint8', dimensions IMG_HEIGHT x IMG_WIDTH x 3):
            Cropped image tensor (photo).
        cropped_image[1] (tf.Tensor, dtype 'uint8', dimensions IMG_HEIGHT x IMG_WIDTH x 3):
            Cropped image tensor (line art).

    """
    stacked_image = tf.stack([input_image, target_image], axis=0)
    cropped_image = tf.image.random_crop(
        stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])

    return cropped_image[0], cropped_image[1]


def normalize(input_image, target_image):
    """Normalize pixel values to [-1, 1]."""
    input_image = (input_image / 127.5) - 1
    target_image = (target_image / 127.5) - 1

    return input_image, target_image


def random_jitter(input_image, target_image):
    """Random jitter and horizontal flipping of input images.

    Args:
        input_image (tf.Tensor, dtype 'uint8', dimensions m x n x 3): Tensor
            (photo) for data augmentation.
        target_image (tf.Tensor, dtype 'uint8', dimensions m x n x 3): Tensor
            (lineart) to be data augmentation.

    Returns:
        input_image (tf.Tensor, dtype 'uint8', dimensions IMG_HEIGHT x IMG_WIDTH x 3):
            Augmented image tensor (photo).
        target_image (tf.Tensor, dtype 'uint8', dimensions IMG_HEIGHT x IMG_WIDTH x 3):
            Augmented image tensor (line art).

    """

    # Jitter by resizing then random crop.
    input_image, target_image = resize(input_image, target_image, 286, 286)
    input_image, target_image = random_crop(input_image, target_image)

    # Random horizontal flipping.
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        target_image = tf.image.flip_left_right(target_image)

    return input_image, target_image


def load_image_train(path_tuple):
    """Load and preprocess training data set example."""
    image_file, target_file = path_tuple[0], path_tuple[1]

    input_image, target_image = load(image_file, target_file)
    input_image, target_image = random_jitter(input_image, target_image)
    input_image, target_image = normalize(input_image, target_image)

    return input_image, target_image


def load_image_test(path_tuple):
    """Load and preprocess test data set example."""
    image_file, target_file = path_tuple[0], path_tuple[1]

    input_image, target_image = load(image_file, target_file)
    input_image, target_image = resize(input_image, target_image,
                                       IMG_HEIGHT, IMG_WIDTH)
    input_image, target_image = normalize(input_image, target_image)

    return input_image, target_image


def get_image_paths(parent_dir):
    """ List all photo + line art paired paths in directory.

    Args:
        parent_dir (str): Directory containing matched photos and line art for
            individual tablet faces.

    Returns:
        paired_paths (list of tuples (str, str)): Tuples comprise paired paths
            (photo path, line art path) for a tablet face.

    """
    image_paths = []
    for input_dir in ['photo_obv', 'photo_rev']:
        root = os.path.join(parent_dir, input_dir)
        image_paths.extend([os.path.join(root, file) for file
                            in os.listdir(root)])

    paired_paths = []
    for image_path in image_paths:
        paired_paths.append((image_path, make_target_path(image_path)))

    return paired_paths


def split_datasets(all_paths, train_proportion, rand_generator):
    """Randomly assigns tablet faces to training and test sets.

    Args:
        all_paths (list of tuples (str, str)): Tuples comprise paired paths
            (photo path, line art path) for a tablet face.
        train_proportion (float): Proportion (between 0 and 1) of image
            examples to use for model training.  Other images will be used for
            the test set.

    Returns:
        training_pairs (list of tuples (str, str)): Paired paths
            (photo path, line art path) for tablet faces used to train model.
        test_pairs (list of tuples (str, str)): Paired paths
            (photo path, line art path) for tablet faces used to test model.

    """
    training_pairs, test_pairs = [], []

    num_training_examples = min(
        round(train_proportion * len(all_paths)),
        len(all_paths))

    num_training_examples = max(1, num_training_examples)

    rand_generator.shuffle(all_paths)
    
    training_pairs = all_paths[:num_training_examples]
    test_pairs = all_paths[num_training_examples:]

    if train_proportion < 1 and len(test_pairs) == 0:
        test_pairs.append(training_pairs.pop())
    elif train_proportion == 1:
        test_pairs = training_pairs

    return training_pairs, test_pairs


def prepare_datasets(data_dir, train_proportion, rand_generator):
    """Create train and test datasets.

    Args:
        data_dir (str): Directory containing matched photographs and line art
            for individual tablet faces.
        train_proportion (float): Proportion (between 0 and 1) of image
            examples to use for model training.  Other images will comprise the
            test dataset.
    Returns:
        train_dataset (tf.data.Dataset): Preprocessed, shuffled, and batched
            images used for model training.
        test_dataset (tf.data.Dataset): Preprocessed and batched
            images used for model testing.
        rand_generator (np.random.default_rng): Random Generator to drive data
            set splitting.

    """
    paired_paths = get_image_paths(data_dir)
    train_paths, test_paths = split_datasets(paired_paths, train_proportion,
                                             rand_generator)

    batch_size = 1   # Suggested by pix2pix authors
    buffer_size = min(100, len(train_paths))

    train_dataset = tf.data.Dataset.from_tensor_slices(train_paths)
    train_dataset = train_dataset.map(load_image_train,
                                      num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.shuffle(buffer_size)
    train_dataset = train_dataset.batch(batch_size)

    test_dataset = tf.data.Dataset.from_tensor_slices(test_paths)
    test_dataset = test_dataset.map(load_image_test)

    # Ensure test dataset has at least 50 images for tf.summary image vis.
    if len(test_paths) < 50:
        repeat = math.ceil(50/len(test_paths))
    test_dataset = test_dataset.repeat(repeat).batch(batch_size)

    return train_dataset, test_dataset
