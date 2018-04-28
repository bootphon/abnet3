from abnet3.dataloader import PairsDataLoader
import numpy as np


class MockFeaturesAccessor:

    def get(self, f, e, s):
        return np.ones((10, 3))  # 10 frames of size 3

def test_pair_loader_loading():
    sampler_pairs = PairsDataLoader(
        pairs_path='data/pairs_knn.txt',
        features_path=None,
        id_to_file=None, ratio_split_train_test=0.7,
        train_iterations=2, test_iterations=2,
        proportion_positive_pairs=0.5
    )

    # test loading data
    sampler_pairs.load_pairs()

    total_pairs = 15
    expected_train_pairs = int(0.7*total_pairs)
    expected_test_pairs = total_pairs - expected_train_pairs
    assert len(sampler_pairs.pairs['train']) == expected_train_pairs
    assert len(sampler_pairs.pairs['test']) == expected_test_pairs

    assert all([len(x) == 6 for x in sampler_pairs.pairs['train']])
    assert all([len(x) == 6 for x in sampler_pairs.pairs['test']])


def test_pair_loader_iterator():
    sampler_pairs = PairsDataLoader(
        pairs_path='data/pairs_knn.txt',
        features_path=None,
        id_to_file=None, ratio_split_train_test=0.7,
        train_iterations=2, test_iterations=3,
        proportion_positive_pairs=0.5
    )

    sampler_pairs.features = MockFeaturesAccessor()

    iterator = sampler_pairs.batch_iterator(train_mode=True)
    n = 0
    for _ in iterator:
        n += 1
    assert n == 2

    iterator = sampler_pairs.batch_iterator(train_mode=False)
    n = 0
    for _ in iterator:
        n += 1
    assert n == 3
