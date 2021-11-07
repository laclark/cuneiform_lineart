import os
import tempfile
import unittest
from unittest.mock import patch

import lineart_generator
import lineart_generator.data_munging.cdli_data_download as dd
import lineart_generator.data_munging.cdli_data_preparation as dp
from lineart_generator.pix2pix.train import train_lineart_generator


class TestCuneiformQuickStart(unittest.TestCase):
    def setUp(self):
        """Set up data + training dirs, define quickstart funcs + args."""
        curr_dir = os.path.dirname(__file__)
        self.quickstart_tablet_path = os.path.join(
            curr_dir, 
            '..', 
            '..',
            'quickstart_tablet_ids.txt')

        self.temp_dir = tempfile.TemporaryDirectory()
        self.model_dir = os.path.join(self.temp_dir.name, 'models')
        self.raw_data_path = os.path.join(self.temp_dir.name, 'data', 'raw_data')
        self.processed_data_path = os.path.join(self.temp_dir.name, 'data', 'processed_data')

        os.makedirs(self.model_dir)
        os.makedirs(self.raw_data_path)

        # Creating patch objects here rather than using decorators allows us to
        # use setUp, tearDown
        self.mock_raw_data_dir = patch.object(
            lineart_generator.data_munging.cdli_data_download,
            'RAW_DATA_DIR',
            self.raw_data_path
            )

        self.prep_mock_raw_data_dir = patch.object(
            lineart_generator.data_munging.cdli_data_preparation,
            'RAW_DATA_DIR',
            self.raw_data_path
            )

        self.mock_processed_data_dir = patch.object(
            lineart_generator.data_munging.cdli_data_preparation,
            'PROCESSED_DATA_DIR',
            self.processed_data_path
            )

        self.train_mock_processed_data_dir = patch.object(
            lineart_generator.pix2pix.train,
            'PROCESSED_DATA_DIR',
            self.processed_data_path
            )
        
        self.quick_start_inputs = (
            (dd.download_selected_tablet_images, {
                'cdli_dir': None,
                'cdli_list_path': self.quickstart_tablet_path,
                'filters': {},
                'max_ims': None,
                },
            ),
            (dp.process_tablets, { 
                'cdli_ids': dd.read_tablet_text(self.quickstart_tablet_path),
                'data_set_name': 'quickstart_data',
                },
            ),
            (train_lineart_generator, {        
                'training_dir': self.model_dir,
                'model_name': 'quickstart_model',
                'data_dir': os.path.join(self.processed_data_path, 'quickstart_data'),
                'train_proportion': 0.8,
                'epochs': 1,
                'save_frequency': 1,
                },
            ),
        )

    def test_quick_start(self):
        with (self.mock_raw_data_dir, self.prep_mock_raw_data_dir,
              self.mock_processed_data_dir, self.train_mock_processed_data_dir):
            for func, kwargs in self.quick_start_inputs:
                func(**kwargs)

    def tearDown(self):
        self.temp_dir.cleanup()


if __name__ == "__main__":
    unittest.main()