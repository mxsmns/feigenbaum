import pytest
from feigenbaum import generation


def test_iterate():
    assert list(generation.iterate(0.5, 0.65, 5)) == [
        0.1625,
        0.08846093750000002,
        0.05241314002380372,
        0.03228290180482176,
        0.0203064654363233,
    ]


@pytest.mark.parametrize(
    "x,alpha,precision,expected",
    [(0.5, 2.3, 4, 0.5652), (0.65, 0.5, 4, 0)],
)
def test_find_stable_value(x: float, alpha: float, precision: int, expected: float):
    stable_value = generation.find_stable_value(x, alpha)
    assert round(stable_value.value, precision) == expected


def test_find_stable_value_exceeds_max_iterations():
    with pytest.raises(IndexError):
        generation.find_stable_value(0.5, 2.3, max_iterations=4)
