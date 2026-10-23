import numpy as np

from scripts.prepare_db15k_clean_split import group_key, make_holdout, split_stats


def test_holdout_groups_reversals_and_preserves_coverage():
    # Dense graph permits a holdout without losing entity or relation coverage.
    train = np.array(
        [(h, r, t) for r in (0, 1) for h in range(6) for t in range(6) if h != t],
        dtype=np.uint64,
    )
    test = np.array([(0, 0, 1)], dtype=np.uint64)
    kept, valid, excluded = make_holdout(train, test, 12, 2022)
    stats = split_stats({"train": kept, "valid": valid, "test": test})
    assert len(valid) == 12
    assert set(map(tuple, excluded)) == {(0, 0, 1), (1, 0, 0)}
    assert not any(stats["same_relation_undirected_group_overlaps"].values())
    assert set(kept[:, 0]) | set(kept[:, 2]) == set(range(6))
    assert set(kept[:, 1]) == {0, 1}
    assert len(kept) + len(valid) + len(excluded) == len(train)
    for actual, repeated in zip(
        (kept, valid, excluded), make_holdout(train, test, 12, 2022)
    ):
        np.testing.assert_array_equal(actual, repeated)


def test_holdout_retains_rare_entity_and_relation():
    train = np.array(
        [(h, 0, t) for h in range(5) for t in range(5) if h != t] + [(5, 1, 0)],
        dtype=np.uint64,
    )
    test = np.array([(6, 0, 1)], dtype=np.uint64)
    kept, valid, _ = make_holdout(train, test, 4, 7)
    assert (5, 1, 0) in set(map(tuple, kept))
    assert group_key((5, 1, 0)) not in {group_key(row) for row in valid}
    assert len(valid) == 4
