import pytest

from embeddingvariable import Embedding


def test_shape():

    variable = Embedding("foo", corpus=["the same string", "another string"])

    variable.embeddings.shape[0] == 2


def test_exact():

    variable = Embedding("foo", corpus=["the same string", "another string"])

    assert variable.comparator("the same string", "the same string") == pytest.approx(
        1.0, 0.001
    )


def test_different():

    variable = Embedding("foo", corpus=["the same string", "another string"])

    assert variable.comparator("the same string", "another string") < 1.0
