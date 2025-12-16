import pytest

from gazetteer import Gazetteer


def test_gazetteer_multi_process(sample_coords):
    gz = Gazetteer(mode=2)
    for data in gz.search(list(sample_coords.keys())):
        assert sample_coords[(data.lon, data.lat)] == data.result


def test_gazetteer_single_process(sample_coords):
    gz = Gazetteer(mode=1)
    for data in gz.search(list(sample_coords.keys())):
        assert sample_coords[(data.lon, data.lat)] == data.result


def test_invalid_inputs():
    gz = Gazetteer()
    invalid_coordinates = (
        12,
        72,
    )  # valid coordinates should be a list of tuple [(12,72)]
    with pytest.raises(TypeError):
        gz.search([])
        gz.search(invalid_coordinates)
        gz.search(list(invalid_coordinates))
