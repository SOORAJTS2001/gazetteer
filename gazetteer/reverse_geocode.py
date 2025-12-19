"""A Fast, Offline Reverse Geocoder in Python

A Python library for offline reverse geocoding. It improves on an existing library
called reverse_geocode developed by Ajay Thampi.
"""

import csv
import sqlite3
import sys
from collections.abc import Iterable
from importlib.resources import files

import numpy as np
from pydantic import BaseModel, Field
from shapely import wkb
from shapely.geometry import Point

if sys.platform == "win32":
    # Windows C long is 32 bits, and the Python int is too large to fit inside.
    # Use the limit appropriate for a 32-bit integer as the max file size
    csv.field_size_limit(2**31 - 1)
else:
    csv.field_size_limit(sys.maxsize)
from scipy.spatial import KDTree

from . import KD_Tree

# Schema of the geo_boundaries file created by this library
RG_COLUMNS: list = ["name", "shape_id", "lon", "lat", "admin1", "admin2"]

DB_PATH: str = str(files("gazetteer.data") / "data.db")
FILENAME: str = str(files("gazetteer.data") / "geo-boundaries.csv")

DEFAULT_K: int = 3


class LocationBaseModel(BaseModel):
    lat: float = Field(..., description="Centroid latitude of the nearest neighbor")
    lon: float = Field(..., description="Centroid longitude of the nearest neighbor")
    name: str = Field(..., description="Name of the nearest neighbour(")
    admin1: str = Field(..., description="Name of the primary administrative division (e.g., country)")
    admin2: str = Field(
        ...,
        description="Name of the secondary administrative division (e.g., state or province)",
    )


class GeocoderResultBaseModel(BaseModel):
    lat: float = Field(..., description="Given latitude")
    lon: float = Field(..., description="Given longitude")
    result: LocationBaseModel | None


def singleton(cls):
    """
    Function to get single instance of the Gazetteer class
    """
    instances = {}

    def getinstance(**kwargs):
        """
        Creates a new Gazetteer instance if not created already
        """
        if cls not in instances:
            instances[cls] = cls(**kwargs)
        return instances[cls]

    return getinstance


@singleton
class Gazetteer:
    """
    The main reverse geocoder class
    """

    def __init__(self, mode: int = 1):
        """Class Instantiation
        params:
        mode (int): Library supports the following two modes:
                    - 1 = Single-process K-D Tree (Default)
                    - 2 = Multi-process K-D Tree for large dataset
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
        To extract coordinates and it's csv data from the given file
        params:
        returns:
        geo_coords: list of tuples (lon, lat)
        locations: list of rows of csv including its index
        """
        with open(FILENAME, newline="") as file:
            stream_reader: csv.DictReader = csv.DictReader(file)
            header = stream_reader.fieldnames
            if header != RG_COLUMNS:
                raise csv.Error(f"Inputs should contain the columns defined in {RG_COLUMNS}")

            # Load all the coordinates and locations
            geo_coords, locations = [], []
            for row in stream_reader:
                geo_coords.append((row["lon"], row["lat"]))
                locations.append(row)
            return geo_coords, locations

    def _safe_load(self, blob):
        geom = wkb.loads(blob)
        return geom[0] if isinstance(geom, np.ndarray) else geom

    def _query_shape(self, filters: list[str]) -> list:
        """
        To query the shape from SQLite file
        params:
        filters (list[str]): List of filters to apply to the query
        returns:
        Geometry of the given shape_id
        """

        placeholders = ",".join(["(?)"] * len(filters))

        query = f"""
            SELECT name, shape_id, coordinates
            FROM location_data
            WHERE shape_id IN ({placeholders});
        """

        self.curr.execute(query, filters)
        rows = self.curr.fetchall()

        lookup = {shape_id: self._safe_load(blob) for name, shape_id, blob in rows}

        return [lookup.get(shape_id) for shape_id in filters]

    def geo_contains(self, search_location: [float, float], indexes: list[int]) -> GeocoderResultBaseModel:
        """
        Verifies whether the location taken from indexes contains the given search_location
        params:
        search_location: [float, float] of format [lon,lat]
        returns:
        The most matching GeocoderResultBaseModel, if there is no match return result inside
        GeocoderResultBaseModel as None
        """
        search_location = Point(*search_location)
        filters = [self.locations[index].get("shape_id") for index in indexes]
        for index, geometry in zip(indexes, self._query_shape(filters), strict=True):
            if geometry.contains(search_location):
                return GeocoderResultBaseModel(
                    lat=search_location.y,
                    lon=search_location.x,
                    result=LocationBaseModel(**self.locations[index]),
                )
        return GeocoderResultBaseModel(lat=search_location.y, lon=search_location.x, result=None)

    def query(self, coordinates: list[tuple[float, float]]) -> Iterable[GeocoderResultBaseModel]:
        """
        Function to query the K-D tree to find the location, according to the given mode, switches between
        single process and multiprocess
        params:
        coordinates (list): List of tuple coordinates, i.e. [(longitude, latitude)]
        returns:
        Gazetteer Iterator of reverse geocoded locations of type GeocoderResultBaseModel,
        """
        if self.mode == 1:
            _, indices = self.tree.query(coordinates, k=DEFAULT_K)
        else:
            _, indices = self.tree.pquery(coordinates, k=DEFAULT_K)

        def _iter():
            for position, indexes_ in enumerate(indices):
                yield self.geo_contains(coordinates[position], indexes_)

        return _iter()

    def search(self, geo_coords) -> Iterable[GeocoderResultBaseModel]:
        """
        params:
        geo_coords (list): List of tuple coordinates, i.e. [(longitude, latitude)]
        returns:
        Gazetteer Iterator of reverse geocoded locations of type GeocoderResultBaseModel,
        """
        if not geo_coords:
            raise TypeError("Coordinates cannot be empty")
        if not isinstance(geo_coords, list) and not isinstance(geo_coords[0], tuple):
            raise TypeError(f"Coordinates must be a list of tuples {type(geo_coords)}: {geo_coords}")
        return self.query(geo_coords)
