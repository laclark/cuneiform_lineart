import json
import os
import shutil
import tempfile
import unittest

from lineart_generator.pix2pix import train


class TestTrainLineartGeneratorSystem(unittest.TestCase):
    def setUp(self):
        """set up model training dir and write config file
        """
        tmp_dir = tempfile.TemporaryDirectory()

        self.temp_dir = tmp_dir
        self.dir_path = tmp_dir.name

        data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'testing_data', 'input_images')

        training_config = {
            "training": {
                "training_dir": self.dir_path,
                "model_name": "test"
                },
            "dataset": {
                "parent_directory": data_dir,
                "buffer_size": 2,
                "batch_size": 1
                }
            }

        self.config_path = os.path.join(self.dir_path, 'config.json')

        with open(self.config_path, 'w') as f:
            json.dump(training_config, f)

    def tearDown(self):
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir.name):
            shutil.rmtree(self.temp_dir.name)

    def test_model_trains_without_exception(self):
        """ Given a training configuration,
            When a model is trained,
            It should complete without errors.
        """
        epochs = 1
        train.train_lineart_generator(epochs, self.config_path)


if __name__ == "__main__":
    unittest.main()