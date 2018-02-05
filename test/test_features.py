from abnet3.features import FeaturesGenerator
import numpy as np
import h5features
import tempfile
from pathlib import Path
import h5py
import pytest
import shutil

class TestFeatures:


    def test_stack(self):
        features_generator = FeaturesGenerator()
        features = np.random.rand(100, 40)
        result_features = features_generator.stack_fbanks(features, nframes=7)
        x, y = result_features.shape

        assert x == 100
        assert y == 280


    def test_normalization(self):

        tempdir = Path(tempfile.mkdtemp())
        h5f = tempdir / 'h5.features'

        features = [np.random.rand(100, 40)]
        items = ['file1']
        times = [np.arange(features[0].shape[0], dtype=float) * 0.01 + 0.0025]

        h5features.write(h5f, '/features/',
                         items, times,
                         features)

        features_generator = FeaturesGenerator()
        h5f_mean_var = tempdir / 'h5-normalized.features'
        mean, variance = features_generator.mean_variance_normalisation(h5f, h5f_mean_var)

        assert mean == pytest.approx(np.mean(features[0]))
        assert variance == pytest.approx(np.std(features[0]))

        # check that the new file has 0 mean and 1 variance
        dset = list(h5py.File(h5f_mean_var).keys())[0]
        data = h5py.File(h5f_mean_var)[dset]['features'][:]
        assert np.mean(data) == pytest.approx(0)
        assert np.std(data) == pytest.approx(1)

        shutil.rmtree(tempdir)

