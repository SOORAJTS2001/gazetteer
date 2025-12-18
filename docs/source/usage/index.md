# Usage
## How to use it
There are mainly two modes of using it, they could be chosen according to the use cases

`mode=1` for smaller dataset <br>
`mode=2` for larger dataset above 100000+ pairs

They are essentially spawning multiple process to enable parallel computation for larger dataset,
**default value for `mode` is 1**

```python
from gazetteer import Gazetteer #import the library
gz = Gazetteer() #gazetteer instance
coordinates = [(-74.0060,40.7128),(76.2673,9.9312)] # list of tuples, in order longitude,latitude
for data in gz.search(coordinates): # gz.search() returns a generator
    print(data) # GeocoderResultBaseModel
```

```bash
lat=40.7128 lon=-74.006 result=LocationBaseModel(lat=40.77488420403987, lon=-73.97077035729478, name='New York', admin1='United States', admin2='New York')
lat=9.9312 lon=76.2673 result=LocationBaseModel(lat=9.96268442863286, lon=76.32995967426373, name='Kanayannur', admin1='India', admin2='Kerala')
```

for larger dataset
```python
from gazetteer import Gazetteer
gz = Gazetteer(mode=2)
```

Make sure that you are providing  `coordinates` as list of tuples, in order of `(longitude,latitude)`, other wise
it would return error

Here `gz.search(coordinates)` returns a generator instead of a plain list, this helps to do computation on the fly,
hence the code below doesn't have to wait for completing the computation of the entire dataset

However, if having a one time list is required, one could do
```python
from gazetteer import Gazetteer
gz = Gazetteer()
coordinates = [(-74.0060,40.7128),(76.2673,9.9312)]
data = list(gz.search(coordinates))
print(data)
```
```bash
[GeocoderResultBaseModel(lat=40.7128, lon=-74.006, result=LocationBaseModel(lat=40.77488420403987, lon=-73.97077035729478, name='New York', admin1='United States', admin2='New York')), GeocoderResultBaseModel(lat=9.9312, lon=76.2673, result=LocationBaseModel(lat=9.96268442863286, lon=76.32995967426373, name='Kanayannur', admin1='India', admin2='Kerala'))]
```
