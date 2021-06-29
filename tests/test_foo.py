from feigenbaum import generation


def test_iterate():
    assert list(generation.iterate(0.5, 0.65, 5)) == [
        0.1625,
        0.08846093750000002,
        0.05241314002380372,
        0.03228290180482176,
        0.0203064654363233,
    ]
