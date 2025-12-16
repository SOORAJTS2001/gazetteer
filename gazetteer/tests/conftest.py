import pytest

from gazetteer.reverse_geocode import LocationBaseModel

test_location = {
    (77.5946, 12.9716): LocationBaseModel(
        lat=13.03574629491685,
        lon=77.524003926496,
        name="Bengaluru North",
        admin1="India",
        admin2="Karnataka",
    ),
    (72.8777, 19.076): LocationBaseModel(
        lat=19.142645665794415,
        lon=72.8793817783488,
        name="Mumbai Suburban",
        admin1="India",
        admin2="Maharashtra",
    ),
    (88.3639, 22.5726): LocationBaseModel(
        lat=22.527204124804495,
        lon=88.3584794638589,
        name="Kolkata",
        admin1="India",
        admin2="West Bengal",
    ),
    (80.2707, 13.0827): LocationBaseModel(
        lat=13.063089231492407,
        lon=80.24385249184856,
        name="Chennai",
        admin1="India",
        admin2="Tamil Nadu",
    ),
    (77.1025, 28.7041): LocationBaseModel(
        lat=28.712129903710974,
        lon=77.06717119560813,
        name="Rohini",
        admin1="India",
        admin2="Delhi",
    ),
    (-74.006, 40.7128): LocationBaseModel(
        lat=40.77488420403987,
        lon=-73.97077035729478,
        name="New York",
        admin1="United States",
        admin2="New York",
    ),
    (-118.2437, 34.0522): LocationBaseModel(
        lat=34.32028359063148,
        lon=-118.2249356314733,
        name="Los Angeles",
        admin1="United States",
        admin2="California",
    ),
    (-87.6298, 41.8781): LocationBaseModel(
        lat=41.839994668505874,
        lon=-87.81668339710608,
        name="Cook",
        admin1="United States",
        admin2="Illinois",
    ),
    (-122.4194, 37.7749): LocationBaseModel(
        lat=37.75572557178133,
        lon=-122.43966236546086,
        name="San Francisco",
        admin1="United States",
        admin2="California",
    ),
    (-95.3698, 29.7604): LocationBaseModel(
        lat=29.857393896937875,
        lon=-95.39322578298103,
        name="Harris",
        admin1="United States",
        admin2="Texas",
    ),
    (139.6917, 35.6895): LocationBaseModel(
        lat=35.70058005169901,
        lon=139.70874071879186,
        name="Shinjuku",
        admin1="Japan",
        admin2="Tokyo",
    ),
    (116.4074, 39.9042): LocationBaseModel(
        lat=39.91083978282277,
        lon=116.41029261357018,
        name="Dongcheng District",
        admin1="China",
        admin2="Beijing",
    ),
    (121.4737, 31.2304): LocationBaseModel(
        lat=31.219070996621216,
        lon=121.47952712135906,
        name="Huangpu District",
        admin1="China",
        admin2="Shanghai",
    ),
    (126.978, 37.5665): LocationBaseModel(
        lat=37.565238636343864,
        lon=126.97456448030223,
        name="Sogong-dong",
        admin1="South Korea",
        admin2="Seoul",
    ),
    (103.8198, 1.3521): LocationBaseModel(
        lat=1.3775050831036277,
        lon=103.80140368512508,
        name="CENTRAL WATER CATCHMENT",
        admin1="Singapore",
        admin2="South West",
    ),
    (2.3522, 48.8566): LocationBaseModel(
        lat=48.85656038755335,
        lon=2.342193457446841,
        name="Paris",
        admin1="France",
        admin2="Île-de-France",
    ),
    (-0.1276, 51.5072): LocationBaseModel(
        lat=51.512940751481,
        lon=-0.15993885252936887,
        name="Westminster",
        admin1="United Kingdom",
        admin2="England",
    ),
    (13.405, 52.52): LocationBaseModel(
        lat=52.501913055352006,
        lon=13.40138613283014,
        name="Berlin",
        admin1="Germany",
        admin2="Berlin",
    ),
    (151.2093, -33.8688): LocationBaseModel(
        lat=-33.889908034431414,
        lon=151.20315680592586,
        name="Sydney",
        admin1="Australia",
        admin2="New South Wales",
    ),
    (-58.3816, -34.6037): LocationBaseModel(
        lat=-34.60764266508075,
        lon=-58.37193772327205,
        name="Comuna 1",
        admin1="Argentina",
        admin2="Buenos Aires",
    ),
}


@pytest.fixture
def sample_coords(scope="session"):
    return test_location
