from abnet3.features import FeaturesGenerator
import numpy as np

class TestFeatures:


    def test_stack_features(self):
        features_generator = FeaturesGenerator()

        features = np.random.rand(100, 40)
        result_features = features_generator.stack_fbanks(features, nframes=7)
        x, y = result_features.shape

        assert x == 100
        assert y == 280
