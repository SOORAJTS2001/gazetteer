# Gazetteer
[![CI](https://github.com/SOORAJTS2001/gazetteer/actions/workflows/ci.yml/badge.svg)](
https://github.com/SOORAJTS2001/gazetteer/actions/workflows/ci.yml
)
![Python](https://img.shields.io/badge/python-3.12%20|%203.13%20|%203.14-blue)
![License](https://img.shields.io/github/license/SOORAJTS2001/gazetteer)
[![Documentation Status](https://app.readthedocs.org/projects/gazetteer/badge/?version=latest)](https://gazetteer.readthedocs.io/en/latest/)
[![PyPI version](https://badge.fury.io/py/python-gazetteer.svg)](https://badge.fury.io/py/python-gazetteer)

A simple, fully offline, accurate boundary based reverse geocoding library in python

Docs - https://gazetteer.readthedocs.io/en/latest/
### Reverse Geocoding
It is a process of finding valid address out of coordinates (lat, long), that includes county,state,country etc..

### Advantage
- This is a boundary based geocoding library which provides significantly more accurate result than the point based computing counterparts
- Provide data upto 3 administrative divisions
  - ADM0 - Country
  - ADM1 - State
  - ADM2/ADM3 - Towns, Cities, Regions (boundaries which does not have ADM3, ADM2 (districts) will be provided)
- Boundary validation is computed on the fly, which makes it extremely fast and efficient
- No max usage, rate limits since everything runs on yours device fully offline
- No boundary ambiguity or invalid address near border regions, since it is validated by boundary itself
- Covers over 145,000+ boundaries of 210+ countries

### How to use gazetteer
#### Installation
```bash
pip install python-gazetteer
```
#### Usage
```python
from gazetteer import Gazetteer #import the library
gz = Gazetteer() #gazetteer instance
coordinates = [(-74.0060,40.7128),(76.2673,9.9312)] # list of tuples, in order longitude,latitude
for data in gz.search(coordinates): # gz.search() returns a generator
    print(data) # GeocoderResultBaseModel
```
#### Result
```bash
lat=40.7128 lon=-74.006 result=LocationBaseModel(lat=40.77488420403987, lon=-73.97077035729478, name='New York', admin1='United States', admin2='New York')
lat=9.9312 lon=76.2673 result=LocationBaseModel(lat=9.96268442863286, lon=76.32995967426373, name='Kanayannur', admin1='India', admin2='Kerala')
```
#### Return Base Model
```python
from pydantic import BaseModel, Field
class LocationBaseModel(BaseModel):
    lat: float = Field(..., description="Centroid latitude of the nearest neighbor")
    lon: float = Field(..., description="Centroid longitude of the nearest neighbor")
    name: str = Field(..., description="Name of the nearest ADM3(eg towns,cities)/ADM2(eg districts) neighbour")
    admin1: str = Field(..., description="Name of the primary administrative division ADM0 (country)")
    admin2: str = Field(
        ...,
        description="Name of the secondary administrative division ADM1 (eg: state)",
    )


class GeocoderResultBaseModel(BaseModel):
    lat: float = Field(..., description="Given latitude")
    lon: float = Field(..., description="Given longitude")
    result: LocationBaseModel
```
#### For large dataset (100000+) coordinate pairs use
```python
from gazetteer import Gazetteer
gz = Gazetteer(mode=2)
```
This switches to multiprocessing mode, utilizing all cores in your system

If the given coordinates are not found or found inside a water boundary like sea/ocean, it should return `None` instead of `LocationBaseModel`
### How it works
#### Data Management
- Boundary Data is sourced from [Geoboundaries](https://www.geoboundaries.org/)
- Data is stored in two places, ``sqlite`` db and a ``csv`` file
- A simplified boundary is stored inside ``sqlite`` db in the format of ``WKB``(Well Known Library)
- The metadata for locations is stored as ``csv``
#### Computation
  - Uses ``KDTree`` nearest neighbour algorithm from ``scipy`` to find the closest boundary point
  - Validates the nearest neighbour using the boundary provided in the ``sqlite``

***Note: Regions that doesn't have an available ADM3 boundary will return ADM2 instead as the nearest neighbour***

### Acknowledgements

- [Geoboundaries](https://www.geoboundaries.org/) - A big shout out for making their amazing dataset free and open-source
- [Ajay Thampi](https://github.com/thampiman/reverse-geocoder) - A  part of this implementation was inspired by his
  [reverse_geocoder](https://github.com/thampiman/reverse-geocoder)

## License
This project is licensed under the GNU Lesser General Public License (LGPL v2.1).
See the LICENSE file for details.
