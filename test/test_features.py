from abnet3.features import FeaturesGenerator
import numpy as np
import h5features
import tempfile
from pathlib import Path
import h5py
import pytest
import shutil


class TestFeatures:

    def test_save_load_mean_variance(self):

        mean, variance = np.random.randn(10), np.random.randn(10)

        features_generator = FeaturesGenerator()
        tmp = tempfile.NamedTemporaryFile()
        print(tmp.name)
        features_generator.save_mean_variance(mean, variance, tmp.name)

        saved = features_generator.load_mean_variance(tmp.name)

        assert (saved['mean'] == mean).all()
        assert (saved['variance'] == variance).all()

    def test_stack(self):
        features_generator = FeaturesGenerator()
        features = np.random.rand(100, 40)
        result_features = features_generator.stack_fbanks(features, nframes=7)
        x, y = result_features.shape

        assert x == 100
        assert y == 280


    def test_normalization(self):

        tempdir = Path(tempfile.mkdtemp())
        h5f = str(tempdir / 'h5.features')

        features = [np.full((100, 40), 1.0), np.full((150, 40), 2.0)]
        items = ['file1', 'file2']
        times = [np.arange(features[0].shape[0], dtype=np.float32) * 0.01 + 0.0025]
        times.append(
            np.arange(features[1].shape[0], dtype=np.float32) * 0.01 + 0.0025)

        h5features.write(h5f, '/features/',
                         items, times,
                         features)

        features_generator = FeaturesGenerator()
        h5f_mean_var = str(tempdir / 'h5-normalized.features')
        mean, variance = features_generator.mean_variance_normalisation(
            h5f, h5f_mean_var)

        stacked_features = np.vstack(features)
        assert mean == pytest.approx(np.mean(stacked_features, axis=0))
        assert variance == pytest.approx(np.std(stacked_features, axis=0))

        # check that the new file has 0 mean and 1 variance
        dset = list(h5py.File(h5f_mean_var).keys())[0]
        data = h5py.File(h5f_mean_var)[dset]['features'][:]
        means = np.mean(data, axis=0)
        assert all(means == pytest.approx(0.0, abs=1e-6))
        assert all(np.std(data, axis=0) == pytest.approx(1.0, abs=1e-6))
        shutil.rmtree(str(tempdir))

    def test_normalization_per_file(self):

        tempdir = Path(tempfile.mkdtemp())
        h5f = str(tempdir / 'h5.features')

        feature1 = np.vstack([np.full((100, 40), 1.), np.full((100, 40), -1.)])
        feature2 = np.vstack([np.full((100, 40), 1.), np.full((100, 40), 2.)])
        features = [feature1, feature2]
        items = ['file1', 'file2']
        times = [np.arange(feature1.shape[0], dtype=float) * 0.01 + 0.0025]
        times.append(np.arange(feature2.shape[0], dtype=float) * 0.01 + 0.0025)

        h5features.write(h5f, '/features/',
                         items, times,
                         features)

        h5f_mean_var = str(tempdir / 'h5-normalized.features')
        features_generator = FeaturesGenerator(normalization=True,
                                               norm_per_file=True)
        meansvars = features_generator.mean_var_norm_per_file(h5f,
                                                              h5f_mean_var)

        assert meansvars[0][0] == 'file1'
        assert all(meansvars[0][1] == np.mean(feature1, axis=0))
        assert all(meansvars[0][2] == np.std(feature1, axis=0))

        assert meansvars[1][0] == 'file2'
        assert all(meansvars[1][1] == np.mean(feature2, axis=0))
        assert all(meansvars[1][2] == np.std(feature2, axis=0))


        reader = h5features.Reader(h5f_mean_var)
        data = reader.read()
        for file in data.items():
            assert np.mean(data.dict_features()[file]) == pytest.approx(0)
            assert np.std(data.dict_features()[file]) == pytest.approx(1)


        shutil.rmtree(str(tempdir))

    def test_normalization_with_VAD(self):
        # paths
        tempdir = Path(tempfile.mkdtemp())
        h5f = str(tempdir / 'h5.features')
        vad_file = str(tempdir / 'vad')

        # write VAD data for file 1
        with open(vad_file, 'w') as vad1:
            vad1.write("file start stop\n"
                       "file1 0.0025 0.5000\n"
                       "file1 0.7525 1.000\n")

        items = ['file1', 'file2']

        # generate data
        feature1 = np.vstack([np.full((50, 40), 1.0), np.full((50, 40), -1.0)])
        feature2 = np.vstack([np.full((50, 40), 1.0), np.full((50, 40), -1.0)])
        features = [feature1, feature2]
        times = [np.arange(feature1.shape[0], dtype=float) * 0.01 + 0.0025]
        times.append(np.arange(feature2.shape[0], dtype=float) * 0.01 + 0.0025)
        h5features.write(h5f, '/features/',
                         items, times,
                         features)

        h5f_mean_var = str(tempdir / 'h5-normalized.features')
        features_generator = FeaturesGenerator(normalization=True,
                                               norm_per_file=True)
        mean, var = features_generator.mean_variance_normalisation(
            h5f, h5f_mean_var, vad_file=vad_file)

        print(mean)
        print(np.mean(np.vstack([feature1[:75], feature2]), axis=0))
        assert mean == pytest.approx(np.mean(np.vstack([feature1[:75], feature2]), axis=0))
        assert var == pytest.approx(np.std(np.vstack([feature1[:75], feature2]), axis=0))

        reader = h5features.Reader(h5f_mean_var)
        data = reader.read()
        assert data.dict_features()['file1'] == pytest.approx(
            (feature1 - mean) / var)

        assert data.dict_features()['file2'] == pytest.approx(
            (feature2 - mean) / var)

        shutil.rmtree(str(tempdir))

    def test_norm_per_file_with_VAD(self):

        # paths
        tempdir = Path(tempfile.mkdtemp())
        h5f = str(tempdir / 'h5.features')
        vad_path = str(tempdir / 'vad')

        # write VAD data for file 1
        with open(str(vad_path), 'w') as vad1:
            vad1.write("file start stop\n"
                       "file1 0.0025 0.5000\n"
                       "file1 0.7525 1.000\n")

        items = ['file1', 'file2']

        # generate data
        feature1 = np.vstack([np.full((50, 40), 1.0), np.full((50, 40), -1.0)])
        feature2 = np.vstack([np.full((50, 40), 1.0), np.full((50, 40), -1.0)])
        features = [feature1, feature2]
        times = [np.arange(feature1.shape[0], dtype=float) * 0.01 + 0.0025]
        times.append(np.arange(feature2.shape[0], dtype=float) * 0.01 + 0.0025)
        h5features.write(h5f, '/features/',
                         items, times,
                         features)

        h5f_mean_var = str(tempdir / 'h5-normalized.features')
        features_generator = FeaturesGenerator(normalization=True,
                                               norm_per_file=True)
        meansvars = features_generator.mean_var_norm_per_file(
            h5f, h5f_mean_var, vad_file=str(vad_path))

        assert meansvars[0][0] == 'file1'
        print(meansvars)

        assert all(meansvars[0][1] == np.mean(feature1[:75], axis=0))
        assert all(meansvars[0][2] == np.std(feature1[:75], axis=0))

        assert meansvars[1][0] == 'file2'
        assert all(meansvars[1][1] == np.mean(feature2, axis=0))
        assert all(meansvars[1][2] == np.std(feature2, axis=0))

        reader = h5features.Reader(h5f_mean_var)
        data = reader.read()
        assert data.dict_features()['file1'] == pytest.approx(
            (feature1 - np.mean(feature1[:75])) / np.std(feature1[:75]))

        assert np.mean(data.dict_features()['file2']) == pytest.approx(0)
        assert np.std(data.dict_features()['file2']) == pytest.approx(1)

        shutil.rmtree(str(tempdir))
