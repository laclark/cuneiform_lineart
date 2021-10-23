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

"""Main training script for pix2pix.

Code samples modified slightly to break into separate modules.

Note: Objects referred to as tf.Tensors may be either
'EagerTensors' (via eager execution) or standard tensorflow 'Tensors'
(handles to nodes in computational graph).

"""
import argparse
import io
import os
import time
from datetime import datetime

from matplotlib import figure
from matplotlib import pyplot as plt
import tensorflow as tf

from lineart_generator.data_munging.cdli_data_preparation import PROCESSED_DATA_DIR
import lineart_generator.pix2pix.data_processing as dt
import lineart_generator.pix2pix.discriminator as ds
import lineart_generator.pix2pix.generator as gn


curr_path = os.path.abspath(__file__)
path_sections = curr_path.split(os.path.sep)
TRAINING_DIR = os.path.join(os.path.sep.join(path_sections[:-4]), 'models')

GENERATOR_OPTIMIZER = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
DISCRIMINATOR_OPTIMIZER = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

DISCRIMINATOR = ds.Discriminator()
GENERATOR = gn.Generator()


def name_model():
    """Create default name for model training directory."""
    now = datetime.now()
    date_time = now.strftime("%d-%m-%Y__%H_%M_%S")
    return f'{date_time}__model'


def create_training_dir(training_dir, model_name):
    """Paths for storing training checkpoints and summary data."""
    checkpoint_dir = os.path.join(training_dir,
                                  model_name,
                                  'ckpts')
    log_dir = os.path.join(training_dir,
                           model_name,
                           'tf_event_logs')
    return log_dir, checkpoint_dir


def generate_image(model, input, target):
    """Generate figure showing data set photo patch, ground truth line art and
    model-generated line art.

    Args:
        model (tf.keras.Model): Generator used for prediction.
        input (tf.Tensor, dtype 'float32', dimensions batch_size x 256 x 256 x 3):
            Generator photo input.
        target (tf.Tensor, dtype 'float32', dimensions batch_size x 256 x 256 x 3):
            Generator target (line art).

    Returns:
        fig (matplotlib.figure.Figure): Comparison figure.

    """
    prediction = model(input, training=True)

    display_list = [input[0], target[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    fig = figure.Figure()

    for i in range(3):
        axis = fig.add_subplot(1, 3, i + 1)
        axis.set_title(title[i])
        # Getting the pixel values in the [0, 1] range to plot.
        axis.imshow(display_list[i] * 0.5 + 0.5)
        axis.set_axis_off()

    return fig


def fig_to_tf_summary(figs):
    """Convert matplotlib figures to tensor for tf.summary.

    Code adapted from https://www.tensorflow.org/tensorboard/image_summaries.

    """
    im_tensors = []

    for fig in figs:
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        im_tensor = tf.image.decode_png(buf.getvalue(), channels=4)
        im_tensors.append(tf.expand_dims(im_tensor, axis=0))

    summary_ims = tf.concat(im_tensors, axis=0)

    return summary_ims


def create_summary_figs(summary_writer, test_ds, num_summary_ims, epoch,
                        global_step, frequency=None):
    """Generate predicted images and save to tf.summary for monitoring.

    Images are created and saved every epoch or at the given frequency (per
    number of epochs).

    """
    if frequency is None or (epoch + 1) % frequency == 0:
        figs = []
        for example_input, example_target in test_ds.take(num_summary_ims):
            figs.append(generate_image(GENERATOR, example_input,
                        example_target))

        with summary_writer.as_default():
            tf.summary.image(f"Test data, Epoch {epoch + 1}",
                             fig_to_tf_summary(figs), step=global_step)


def add_losses_to_summary(summary_writer, losses, step):
    with summary_writer.as_default():
        for name, loss in losses.items():
            tf.summary.scalar(name, loss, step=step)


def save_checkpoint(checkpoint, epoch, frequency):
    """Save checkpoint at given frequency."""
    def name_checkpoint():
        return os.path.join(checkpoint.dir, f"epoch_{epoch + 1:05d}")
    if (epoch + 1) % frequency == 0:
        checkpoint.save(file_prefix=name_checkpoint())


def train_step(input_image, target):
    """Run a single training step and return losses."""
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = GENERATOR(input_image, training=True)

        disc_real_output = DISCRIMINATOR([input_image, target], training=True)
        disc_generated_output = DISCRIMINATOR([input_image, gen_output],
                                              training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = gn.generator_loss(
                                                        disc_generated_output,
                                                        gen_output,
                                                        target)
        disc_loss = ds.discriminator_loss(disc_real_output,
                                          disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_total_loss,
                                            GENERATOR.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss,
                                                 DISCRIMINATOR.trainable_variables)

    GENERATOR_OPTIMIZER.apply_gradients(zip(generator_gradients,
                                            GENERATOR.trainable_variables))
    DISCRIMINATOR_OPTIMIZER.apply_gradients(zip(discriminator_gradients,
                                            DISCRIMINATOR.trainable_variables))

    return {'gen_total_loss': gen_total_loss,
            'gen_gan_loss': gen_gan_loss,
            'gen_l1_loss': gen_l1_loss,
            'disc_loss': disc_loss}


def fit(checkpoint, summary_writer, epochs, train_ds, test_ds, save_frequency):
    """Control model training, checkpoint saving, and tf.summary generation."""
    global_step = 0

    for epoch in range(epochs):
        start = time.time()

        num_summary_ims = 2
        create_summary_figs(summary_writer, test_ds, num_summary_ims,
                            epoch, global_step, frequency=save_frequency)

        print("Epoch: ", epoch + 1)

        # Training step
        for n, (input_image, target) in train_ds.enumerate():
            print('.', end='')
            if (n+1) % 100 == 0:
                print()
            losses = train_step(input_image, target)
            add_losses_to_summary(summary_writer, losses,
                                  global_step + n.numpy() + 1)

        save_checkpoint(checkpoint, epoch, save_frequency)

        print()

        global_step += n.numpy()

        print('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                           time.time()-start))


def train_lineart_generator(training_dir, model_name, data_dir,
                            train_proportion, epochs, save_frequency):
    """
    Args:
        training_dir (str): Parent directory containing model training
            subdirectories.
        model_name (str): Name of subdirectory holding training artifacts.
        data_dir (str): Directory containing matched photographs and line art
            for individual tablet faces.
        train_proportion (float): Proportion (between 0 and 1) of image
            examples to use for model training.  Other images will comprise the
            test dataset.
        epochs (int): Number of training epochs.
        save_frequency (int): Save model checkpoint every N epochs.

    Returns:
        None
    """
    log_dir, checkpoint_dir = create_training_dir(training_dir, model_name)

    train_dataset, test_dataset = dt.prepare_datasets(data_dir,
                                                      train_proportion)

    checkpoint = tf.train.Checkpoint(generator_optimizer=GENERATOR_OPTIMIZER,
                                     discriminator_optimizer=DISCRIMINATOR_OPTIMIZER,
                                     generator=GENERATOR,
                                     discriminator=DISCRIMINATOR)
    checkpoint.dir = checkpoint_dir

    summary_writer = tf.summary.create_file_writer(log_dir)

    fit(checkpoint, summary_writer, epochs, train_dataset, test_dataset,
        save_frequency)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--training_dir',
        help='Parent directory containing model training artifacts.',
        default=TRAINING_DIR,
        )

    parser.add_argument(
        '--model_name',
        help='Name of subdirectory holding training artifacts.',
        default=name_model(),
        )

    parser.add_argument(
        '--data_dir',
        help=('Directory containing matched photographs and line art for'
              ' individual tablet faces.'),
        default=PROCESSED_DATA_DIR,
        )

    parser.add_argument(
        '--train_proportion',
        help=('Proportion (between 0 and 1) of image examples to use for'
              ' model training.  All other images will be used for the test'
              ' set.'),
        default=0.7,
        )

    parser.add_argument(
        '--epochs',
        help='Number of training epochs.',
        default=25,
        )

    parser.add_argument(
        '--save_every_n_epochs',
        help='Save checkpoint every N epochs.',
        default=1,
        )

    args = parser.parse_args()

    training_dir = args.training_dir
    model_name = args.model_name
    data_dir = args.data_dir
    train_proportion = float(args.train_proportion)
    epochs = int(args.epochs)
    save_frequency = int(args.save_every_n_epochs)

    train_lineart_generator(
        training_dir,
        model_name,
        data_dir,
        train_proportion,
        epochs,
        save_frequency)
