import pytest

from embeddingvariable import Embedding


def test_exact():

    variable = Embedding("foo", corpus=["the same string", "another string"])

    assert variable.comparator("the same string", "the same string") == pytest.approx(
        1.0, 0.001
    )


def test_different():

    variable = Embedding("foo", corpus=["the same string", "another string"])

    assert variable.comparator("the same string", "another string") == pytest.approx(
        0.7577, 0.001
    )
