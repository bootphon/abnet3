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

        features = [np.full((100, 40), 1.0), np.full((150, 40), 2.0)]
        items = ['file1', 'file2']
        times = [np.arange(features[0].shape[0], dtype=float) * 0.01 + 0.0025]
        times.append(np.arange(features[1].shape[0], dtype=float) * 0.01 + 0.0025)

        h5features.write(h5f, '/features/',
                         items, times,
                         features)

        features_generator = FeaturesGenerator()
        h5f_mean_var = tempdir / 'h5-normalized.features'
        mean, variance = features_generator.mean_variance_normalisation(h5f, h5f_mean_var)

        assert mean == pytest.approx((1.0 * 1 + 2.0 * 1.5) / 2.5)
        assert variance == pytest.approx(np.std(np.vstack(features)))

        # check that the new file has 0 mean and 1 variance
        dset = list(h5py.File(h5f_mean_var).keys())[0]
        data = h5py.File(h5f_mean_var)[dset]['features'][:]
        assert np.mean(data) == pytest.approx(0)
        assert np.std(data) == pytest.approx(1)
        shutil.rmtree(tempdir)

    def test_normalization_per_file(self):

        tempdir = Path(tempfile.mkdtemp())
        h5f = tempdir / 'h5.features'

        feature1 = np.vstack([np.full((100, 40), 1.0), np.full((100, 40), -1.0)])
        feature2 = np.vstack([np.full((100, 40), 1.0), np.full((100, 40), 2.0)])
        features = [feature1, feature2]
        items = ['file1', 'file2']
        times = [np.arange(feature1.shape[0], dtype=float) * 0.01 + 0.0025]
        times.append(np.arange(feature2.shape[0], dtype=float) * 0.01 + 0.0025)

        h5features.write(h5f, '/features/',
                         items, times,
                         features)

        h5f_mean_var = tempdir / 'h5-normalized.features'
        features_generator = FeaturesGenerator(normalization=True, norm_per_file=True)
        meansvars = features_generator.mean_var_norm_per_file(h5f, h5f_mean_var)

        assert meansvars == [
            ('file1', 0, np.std(feature1)),
            ('file2', 1.5, np.std(feature2)),
        ]

        reader = h5features.Reader(h5f_mean_var)
        data = reader.read()
        for file in data.items():
            assert np.mean(data.dict_features()[file]) == pytest.approx(0)
            assert np.std(data.dict_features()[file]) == pytest.approx(1)

        shutil.rmtree(tempdir)
