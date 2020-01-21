import unittest
from pathlib import Path
from data import SpatialAudioDatasetWaveform

class TestDataProcessing(unittest.TestCase):
    def test_SpatialAudioDatasetWaveform_length(self):
        dataset = SpatialAudioDatasetWaveform('./sample_data')
        n_data = len(list(filter(lambda x: x.is_dir(), Path('./sample_data').glob('*/'))))
        assert len(dataset) == n_data, "expect len(dataset) to be {}".format(n_data)

    def test_SpatialAudioDatasetWaveform_format(self):
        dataset = SpatialAudioDatasetWaveform('./sample_data')
        x = dataset[0]
        assert len(x) == 3, "expect data to have mixed_data, gt_data, and locs"
        assert len(x[2]) == 3, "expect locs to have 3 dimensions"
        assert x[0].shape[0] == x[1].shape[1] and x[0].shape[1] == x[1].shape[2], \
            "expect input and output to have the same shape"

if __name__ == '__main__':
    unittest.main()
