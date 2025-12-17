def test_import():
    from gazetteer import Gazetteer
    from gazetteer.reverse_geocode import GeocoderResultBaseModel

    gz = Gazetteer()
    coordinates = [(-74.0060, 40.7128)]
    for data in gz.search(coordinates):
        assert isinstance(data, GeocoderResultBaseModel)
