import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any, Union, Set

import numpy
from PIL.Image import open as pil_open, Image as pil_Image
import rasterio

import ee
from enum import Enum

import requests
from ee import ImageCollection, Geometry, Image
from ee.batch import Export
from numpy import ndarray, vectorize
from rasterio.io import DatasetWriter

# ee.Initialize()
from utils import download_file_from_url, unzip_file

tmp_dir: Path = Path("tmp")
tmp_dir.mkdir(parents=True, exist_ok=True)


class DataSets(Enum):
    modis_reflectance_500m = "MODIS/006/MCD43A4"
    modis_terra_land_water = "MODIS/006/MOD44W"


class ModisTerraLandWater(Enum):
    water_mask = "water_mask"
    water_mask_QA = "water_mask_QA"


def fetch_data():
    start_date: datetime = datetime(2015, 1, 1)
    end_date: datetime = datetime(2015, 2, 1)
    # africa_geometry: Geometry = ee.Geometry.Rectangle([-22.96874821, 33.29640894, 55.07812679, -39.20562293])
    chesapeake_bay_geometry: Geometry = ee.Geometry.Rectangle([-78, 40, -75, 36])

    modis: ImageCollection = ee.ImageCollection(DataSets.modis_terra_land_water.value) \
                               .filter(ee.Filter.date(start_date, end_date)) \
                               .filterBounds(chesapeake_bay_geometry) \
                               .select([ModisTerraLandWater.water_mask.value, ModisTerraLandWater.water_mask_QA.value]) \
                               .limit(1)

    modis_image: Image = modis.first().select([ModisTerraLandWater.water_mask.value])
    modis_image_reduced = modis_image.reduceResolution(
        reducer=ee.Reducer.mean(),
        bestEffort=True
    )
    # video_args: Dict = {
    #     "dimensions": 768,
    #     "region": chesapeake_bay_geometry,
    #     "framesPerSecond": 7,
    #     "crs": "EPSG:3857",
    #     "min": 10,
    #     "max": 500,
    #     "palette": ['blue', 'purple', 'cyan', 'green', 'yellow', 'red']
    # }
    #
    # print(modis.getVideoThumbURL(video_args))

    image_args: Dict = {
        "name": "landmask_image",
        "dimensions": 768,
        "region": chesapeake_bay_geometry,
        "crs": "EPSG:3857",
        "bands": [ModisTerraLandWater.water_mask.value],
        "min": 0,
        "max": 1,
        "palette": ['blue', 'purple', 'cyan', 'green', 'yellow', 'red']
    }

    url: str = modis_image_reduced.getDownloadURL(image_args)

    print(url)
    download_file_from_url(url, tmp_dir / Path("landmask_image.zip"))
    unzip_file(tmp_dir / Path("landmask_image.zip"))

    print("finished")


def land_mask(source: float, mask: float):
    return numpy.nan if mask < 0.5 else source


def manipulate_image():
    dataset: rasterio.DatasetReader
    land_mask_data: ndarray
    with rasterio.open("landmask_image.water_mask.tif", mode="r") as dataset:
        land_mask_data = dataset.read(1)

    source_data: ndarray
    with rasterio.open("processed.tif", mode="r") as dataset:
        source_data = dataset.read(1)

    land_mask_vectorized = numpy.vectorize(land_mask)

    source_masked: ndarray = land_mask_vectorized(source_data, land_mask_data)
    # pixel_values: Set = set()
    # for pixel in band1.flat:
    #     pixel_values.add(pixel)
    # print(pixel_values)

    output: DatasetWriter
    with rasterio.open("processed_masked.tif",
                       mode="w",
                       **dataset.meta) as output:
        output.write(source_masked, 1)


if __name__ == '__main__':
    # fetch_data()
    manipulate_image()
