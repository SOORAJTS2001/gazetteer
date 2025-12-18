# Implementation
## Why it was made ?

Latitude and longitude are precise but not meaningful on their own. Most systems and people reason in terms of places - cities, regions, and addresses, not raw coordinates. Converting lat - long into an address adds semantic context, making location data understandable, searchable, and actionable in real-world applications.

This method of converting coordinates into sensible locations is known as reverse geocoding
They are heavily used in almost every customer facing applications like delivery, travel booking etc

Conventional way of doing was using commercial API's like Google Maps, Mapbox
They are very much accurate, but costly in nature, also they come with various rate limits.
For large data intensive applications this may incur as a limitation

Hence, there are Offline Reverse Geocoding libraries, they would be completely offline, since it runs on your machine
there would be no rate limiting
But they come at a cost of accuracy, their normal working procedure is to use a large point based dataset
(most likely cities) coordinates and finding the nearest neighbour from the given coordinates using algorithms like
`KDTree` from scipy

## Problem with the conventional approach
Suppose we have a situation like this
```{image} ../_static/img.png
:alt: Reverse geocoding flow
:width: 600px
:align: center
```
Here, we are trying to find address of this given location (blue point), but the closes point
to it is Lisbon, If you are naively using the nearest neighbour algorithm it is going to return
the address of Lisbon, but that is wrong because the point is visibly inside the boundary of Vermount

## Our approach
Instead of focusing entirely on points, we would be considering boundaries also.
- Step 1
    - Find the nearest neighbouring boundaries from the given coordinate
    - This could be done by computing the centroid of polygons and use `KDTree`
      algorithm to get the nearest **boundaries**
    - By default, the nearest 3 neighbours are considered

- Step 2
   - Check whether which boundary encloses the given point

This approach gives a validation that the given point is actually inside that boundary

## Challenges
**Storage**: This is one of the biggest challenges faced, because boundary data is huge, but
Geoboundaries provide their simplified geometries free and open source, even though the total size
was around 1.5 GB, so it was converted to WKB and stored inside sqlite for querying and filtering, hence the overall
size reduced to around 90 MB

**Speed**: It was impractical to check boundary enclosure for every boundary there is, hence computed
their centroid instead and use them for primary layer of filtering, and enclosure was validated
on the fly, saving time and space
