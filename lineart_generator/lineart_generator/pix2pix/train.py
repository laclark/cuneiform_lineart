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

"""
import argparse
import io
import tensorflow as tf
import json
import os
import time
import datetime

from matplotlib import pyplot as plt

import lineart_generator.pix2pix.data_processing as dt
import lineart_generator.pix2pix.discriminator as ds
import lineart_generator.pix2pix.generator as gn
from lineart_generator.data_munging.cdli_data_preparation import PROCESSED_DATA_DIR


curr_path = os.path.abspath(__file__)
path_sections = curr_path.split(os.path.sep)
TRAINING_DIR = os.path.join(os.path.sep.join(path_sections[:-4]), 'models')

GENERATOR_OPTIMIZER = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
DISCRIMINATOR_OPTIMIZER = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

DISCRIMINATOR = ds.Discriminator()
GENERATOR = gn.Generator()


def read_configuration(config_path):
    with open(config_path, 'r') as f:
        configs = json.load(f) 
        return configs['training'], configs['dataset']


def create_training_dir(training_dir, model_name):

    checkpoint_dir = os.path.join(training_dir, 
                                  model_name,
                                  'ckpts')
    log_dir = os.path.join(training_dir, 
                           model_name,
                           'tf_event_logs')
    return log_dir, checkpoint_dir


def generate_image(model, test_input, tar):
    prediction = model(test_input, training=True)

    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    fig, axes = plt.subplots(1, 3)

    for i in range(3):
        axes[i].set_title(title[i])
        # Getting the pixel values in the [0, 1] range to plot.
        axes[i].imshow(display_list[i] * 0.5 + 0.5)
        axes[i].set_axis_off()
    
    return fig


def fig_to_tf_summary(figs):

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


@tf.function
def train_step(input_image, target, epoch):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = GENERATOR(input_image, training=True)

        disc_real_output = DISCRIMINATOR([input_image, target], training=True)
        disc_generated_output = DISCRIMINATOR([input_image, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = gn.generator_loss(disc_generated_output, gen_output, target)
        disc_loss = ds.discriminator_loss(disc_real_output, disc_generated_output)

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


def add_losses_to_summary(summary_writer, losses, step):
    with summary_writer.as_default():
        for name, loss in losses.items():
            tf.summary.scalar(name, loss, step=step)


def save_checkpoint(checkpoint, epoch, frequency):
    def name_checkpoint():
        return os.path.join(checkpoint.dir, f"epoch_{epoch + 1:05d}")
    if (epoch + 1) % frequency == 0:
        checkpoint.save(file_prefix=name_checkpoint())


def fit(checkpoint, summary_writer, epochs, train_ds, test_ds):
    global_step = 0

    for epoch in range(epochs):
        start = time.time()

        num_summary_ims = 5
        figs = []
        for example_input, example_target in test_ds.take(num_summary_ims):
            figs.append(generate_image(GENERATOR, example_input, example_target))

        with summary_writer.as_default():
            tf.summary.image(f"Test data, Epoch {epoch}", fig_to_tf_summary(figs), step=global_step)

        print("Epoch: ", epoch)

        # Training step
        for n, (input_image, target) in train_ds.enumerate():
            print('.', end='')
            if (n+1) % 100 == 0:
                print()
            losses = train_step(input_image, target, epoch)
            add_losses_to_summary(summary_writer, losses, global_step + n.numpy() + 1)
            
        frequency = 5
        save_checkpoint(checkpoint, epoch, frequency)

        print()
        
        global_step += n.numpy()

        print('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                           time.time()-start))


def train_lineart_generator(training_dir, model_name, data_dir, train_proportion, epochs):
    log_dir, checkpoint_dir = create_training_dir(training_dir, model_name)

    train_dataset, test_dataset = dt.prepare_datasets(data_dir, train_proportion)

    checkpoint = tf.train.Checkpoint(generator_optimizer=GENERATOR_OPTIMIZER,
                                     discriminator_optimizer=DISCRIMINATOR_OPTIMIZER,
                                     generator=GENERATOR,
                                     discriminator=DISCRIMINATOR)
    checkpoint.dir = checkpoint_dir

    summary_writer = tf.summary.create_file_writer(log_dir)

    fit(checkpoint, summary_writer, epochs, train_dataset, test_dataset)


if __name__ == '__main__':
    epochs = 10
    model_name = 'toy_1'
    train_proportion = 0.5

    train_lineart_generator(
        TRAINING_DIR, 
        model_name, 
        PROCESSED_DATA_DIR, 
        train_proportion, 
        epochs)