# RetroGeo - An Offline Reverse Geocoding Library

A boundary based reverse geocoding library, which converts given lat and long into address

```python
from RetroGeo import GeoLocator
rev = GeoLocator()
locations = [(-73.2404,43.2342)] #lon,lat
result = rev.query(locations)
print(result)
```

```bash
{(-73.2404, 43.2342): LocationBaseModel(lat=-73.09269723242618, lon=43.03582962989797, name='Bennington', admin1='United States', admin2='Vermont')}
```
