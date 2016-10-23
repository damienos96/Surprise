"""
Module for testing the Dataset class
"""

import os
import pytest

from pyrec import Dataset
from pyrec import Reader


reader = Reader(line_format='user item rating', sep=' ', skip_lines=3,
                rating_scale=(1, 5))

def test_split():
    """Test the split method."""

    custom_dataset_path = (os.path.dirname(os.path.realpath(__file__)) +
                           '/custom_dataset')
    data = Dataset.load_from_file(file_path=custom_dataset_path, reader=reader)

    # Test n_folds parameter
    data.split(5)
    assert len(list(data.folds)) == 5

    with pytest.raises(ValueError):
        data.split(10)
        for fold in data.folds:
            pass

    with pytest.raises(ValueError):
        data.split(1)
        for fold in data.folds:
            pass

    # Test the shuffle parameter
    data.split(n_folds=3, shuffle=False)
    testsets_a = [testset for (_, testset) in data.folds]
    data.split(n_folds=3, shuffle=False)
    testsets_b = [testset for (_, testset) in data.folds]
    assert testsets_a == testsets_b

    data.split(n_folds=3, shuffle=True)
    testsets_b = [testset for (_, testset) in data.folds]
    assert testsets_a != testsets_b
    # Note : there's a chance that the above test fails, just by lack of luck.
    # This is probably not a good thing.

    # Ensure that folds are the same if split is not called again
    testsets_a = [testset for (_, testset) in data.folds]
    testsets_b = [testset for (_, testset) in data.folds]
    assert testsets_a == testsets_b

def test_trainset_testset():
    """Test the construct_trainset and construct_testset methods."""

    current_dir = os.path.dirname(os.path.realpath(__file__))
    folds_files = [(current_dir + '/custom_train',
                    current_dir + '/custom_test')]

    data = Dataset.load_from_folds(folds_files=folds_files, reader=reader)

    for trainset, testset in data.folds:
        pass # just need trainset and testset to be set

    # test rm:
    rm = trainset.rm
    assert rm[0, 0] == 4
    assert rm[1, 0] == 4
    assert rm[3, 1] == 5
    assert rm[40, 20000] == 0 # not in the trainset

    # test ur
    ur = trainset.ur
    assert ur[0] == [(0, 4)]
    assert ur[1] == [(0, 4), (1, 2)]
    assert ur[40] == [] # not in the trainset

    # test ir
    ir = trainset.ir
    assert ir[0] == [(0, 4), (1, 4), (2, 1)]
    assert ir[1] == [(1, 2), (2, 1), (3, 5)]
    assert ir[20000] == [] # not in the trainset

    # test n_users, n_items, r_min, r_max
    assert trainset.n_users == 4
    assert trainset.n_items == 2
    assert trainset.r_min == 1
    assert trainset.r_max == 5

    # test raw2inner: ensure inner ids are given in proper order
    raw2inner_id_users = trainset.raw2inner_id_users
    for i in range(4):
        assert raw2inner_id_users['user' + str(i)] == i

    raw2inner_id_items = trainset.raw2inner_id_items
    for i in range(2):
        assert raw2inner_id_items['item' + str(i)] == i

    # test testset:
    assert testset[0] == (3, 0, 5)  # user3 item0 5
    assert testset[1] == (0, 1, 1)  # user0 item1 1
    assert testset[2][0].startswith('unknown')
    assert testset[2][1].startswith('unknown')