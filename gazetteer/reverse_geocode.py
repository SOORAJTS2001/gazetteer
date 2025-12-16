""" A Fast, Offline Reverse Geocoder in Python

A Python library for offline reverse geocoding. It improves on an existing library
called reverse_geocode developed by Richard Penman and Ajay Thampi.
"""
import csv
import sqlite3
import sys
from importlib.resources import files

import numpy as np
from pydantic import BaseModel, Field
from shapely import wkb
from shapely.geometry import Point

if sys.platform == 'win32':
    # Windows C long is 32 bits, and the Python int is too large to fit inside.
    # Use the limit appropriate for a 32-bit integer as the max file size
    csv.field_size_limit(2 ** 31 - 1)
else:
    csv.field_size_limit(sys.maxsize)
from scipy.spatial import KDTree
from . import KD_Tree

# Schema of the cities file created by this library
RG_COLUMNS = ['name', 'shape_id', 'lon', 'lat', 'admin1', 'admin2']

DB_PATH = files("gazetteer.data") / "data.db"
FILENAME = files("gazetteer.data") / "geo-boundaries.csv"

DEFAULT_K = 3


class LocationBaseModel(BaseModel):
    lat: float = Field(..., description="Centroid latitude of the nearest neighbor")
    lon: float = Field(..., description="Centroid longitude of the nearest neighbor")
    name: str = Field(..., description="Name of the nearest neighbour(")
    admin1: str = Field(..., description="Name of the primary administrative division (e.g., country)")
    admin2: str = Field(..., description="Name of the secondary administrative division (e.g., state or province)")


class GeocoderResultBaseModel(BaseModel):
    lat: float = Field(..., description="Given latitude")
    lon: float = Field(..., description="Given longitude")
    result: LocationBaseModel


def singleton(cls):
    """
    Function to get single instance of the RGeocoder class
    """
    instances = {}

    def getinstance(**kwargs):
        """
        Creates a new RGeocoder instance if not created already
        """
        if cls not in instances:
            instances[cls] = cls(**kwargs)
        return instances[cls]

    return getinstance


@singleton
class Gazetteer(object):
    """
    The main reverse geocoder class
    """

    def __init__(self, mode: int = 1):
        """ Class Instantiation
        Args:
        mode (int): Library supports the following two modes:
                    - 1 = Single-threaded K-D Tree
                    - 2 = Multi-threaded K-D Tree (Default)
        stream (io.StringIO): An in-memory stream of a custom data source
        """
        self.mode = mode
        coordinates, self.locations = self._load()
        self.conn = sqlite3.connect(DB_PATH)
        self.curr = self.conn.cursor()
        if self.mode == 1:  # Single-process
            self.tree = KDTree(coordinates)
        else:  # Multi-process
            self.tree = KD_Tree.cKDTree_MP(coordinates)

    def _load(self):
        """
        Function that loads a custom data source
        Args:
        stream (io.StringIO): An in-memory stream of a custom data source.
                              The format of the stream must be a comma-separated file
                              with header containing the columns defined in RG_COLUMNS.
        """
        with open(FILENAME, mode='r', newline='') as file:
            stream_reader = csv.DictReader(file)
            header = stream_reader.fieldnames
            if header != RG_COLUMNS:
                raise csv.Error(f'Inputs should contain the columns defined in {RG_COLUMNS}')

            # Load all the coordinates and locations
            geo_coords, locations = [], []
            for row in stream_reader:
                geo_coords.append((row['lon'], row['lat']))
                locations.append(row)
            return geo_coords, locations

    def safe_load(self, blob):
        geom = wkb.loads(blob)
        return geom[0] if isinstance(geom, np.ndarray) else geom

    def query_shape(self, filters: list[str]) -> list:

        placeholders = ",".join(["(?)"] * len(filters))

        query = f"""
            SELECT name, shape_id, coordinates
            FROM location_data
            WHERE shape_id IN ({placeholders});
        """

        self.curr.execute(query, filters)
        rows = self.curr.fetchall()

        lookup = {
            shape_id: self.safe_load(blob)
            for name, shape_id, blob in rows
        }

        return [lookup.get(shape_id) for shape_id in filters]

    def geo_contains(self, search_location: [float, float], indexes: list[int]):
        search_location = Point(*search_location)
        filters = [self.locations[index].get("shape_id") for index in indexes]
        for index, geometry in zip(indexes, self.query_shape(filters)):
            if geometry.contains(search_location):
                return GeocoderResultBaseModel(lat=search_location.y, lon=search_location.x,
                                               result=LocationBaseModel(**self.locations[index]))
        return GeocoderResultBaseModel(lat=search_location.y, lon=search_location.x, result=None)

    def query(self, coordinates):
        """
        Function to query the K-D tree to find the nearest city
        Args:
        coordinates (list): List of tuple coordinates, i.e. [(latitude, longitude)]
        """
        if self.mode == 1:
            _, indices = self.tree.query(coordinates, k=DEFAULT_K)
        else:
            _, indices = self.tree.pquery(coordinates, k=DEFAULT_K)

        def _iter():
            for position, indexes_ in enumerate(indices):
                yield self.geo_contains(coordinates[position], indexes_)

        return _iter()

    def search(self, geo_coords):
        """
        Function to query for a list of coordinates
        """
        if not isinstance(geo_coords, tuple) and not isinstance(geo_coords, list):
            raise TypeError('Expecting a tuple or a tuple/list of tuples')
        elif not isinstance(geo_coords[0], tuple):
            geo_coords = [geo_coords]
        return self.query(geo_coords)
