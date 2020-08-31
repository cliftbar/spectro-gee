import zipfile
from datetime import datetime
from pathlib import Path
from time import sleep, time
from typing import Dict, Optional, Any, Union

import numpy
from PIL.Image import open as pil_open, Image as pil_Image
import rasterio

import ee
from enum import Enum

import requests
from ee import ImageCollection, Geometry, Image
from ee.batch import Export, Task
from numpy import ndarray, vectorize
from rasterio.io import DatasetWriter

from utils import DataSets, download_file_from_url

ee.Initialize()


tmp_dir: Path = Path("tmp")
tmp_dir.mkdir(parents=True, exist_ok=True)


class Landsat8Bands(Enum):
    band4_645nm = "B3"
    band5_859nm = "B4"


def masker(image: Image):
    cloudShadowBitMask = 1 << 3
    cloudsBitMask = 1 << 5
    qa: Image = image.select('pixel_qa')

    mask = qa.bitwiseAnd(cloudShadowBitMask).eq(0).And(qa.bitwiseAnd(cloudsBitMask).eq(0))
    return image.updateMask(mask)


def fetch_data():
    start_date: datetime = datetime(2019, 1, 1)
    end_date: datetime = datetime(2019, 6, 1)
    # africa_geometry: Geometry = ee.Geometry.Rectangle([-22.96874821, 33.29640894, 55.07812679, -39.20562293])
    chesapeake_bay_geometry: Geometry = ee.Geometry.Rectangle([-78, 40, -75, 36])

    landsat8: ImageCollection = ee.ImageCollection(DataSets.landsat8_reflectance_30m.value) \
                                  .filter(ee.Filter.date(start_date, end_date)) \
                                  .filterBounds(chesapeake_bay_geometry)
                                  # .select([Landsat8Bands.band4_645nm.value])
                                  # .limit(24)

    landsat8_image: Image = landsat8.map(masker).median().select([Landsat8Bands.band4_645nm.value, Landsat8Bands.band5_859nm.value])
    scale_m: int = 30

    url: str = landsat8_image.getDownloadURL({
        "name": f"high_resolution_landsat8_{scale_m}m_{datetime.now().isoformat()}",
        "region": chesapeake_bay_geometry,
        "crs": "EPSG:3857",
        "scale": scale_m,
        "bands": [Landsat8Bands.band4_645nm.value, Landsat8Bands.band5_859nm.value],
        "maxPixels": 210313503
    })
    print(url)
    download_file_from_url(url, tmp_dir / Path("landsat8_image.zip"))

    # chesapeake_bay_geometry_high: Geometry = ee.Geometry.Rectangle([-77.08, 37.33, -76.9, 37.2])

    # export_task: Task = Export.image.toDrive(
    #     image=landsat8_image,
    #     folder="aquaculture-exports",
    #     description=f"high_resolution_landsat8_{scale_m}m_{datetime.now().isoformat()}",
    #     region=chesapeake_bay_geometry,
    #     crs="EPSG:3857",
    #     scale=scale_m,
    #     maxPixels=210313503
    # )
    # start_time: int = int(time())
    # export_task.start()
    # while export_task.active():
    #     print(export_task.status())
    #     sleep(1)
    # end_time: int = int(time())
    # completion_status: Dict = export_task.status()
    # print(f"Batch Time: {end_time - start_time}")
    # if completion_status["state"] == "FAILED":
    #     print(completion_status)
    # else:
    #     print(completion_status["destination_uris"])
    #
    #     i: int = 0
    #     for uri in completion_status["destination_uris"]:
    #         download_file_from_url(uri, tmp_dir / Path(f"high_resolution_landsat8_{scale_m}m_{i}.tif"))

    print("finished")


def clipper_645(x):
    return float(x) if 1 <= x <= 500 else numpy.nan


def clipper_859(x):
    return float(x) if 1 <= x <= 800 else numpy.nan


def reflectance_to_turbidity_645nm(reflectance: float) -> float:
    A_T: float = 228.1
    C_T: float = 0.1641

    turbidity: float
    if .0001 <= reflectance <= .05:
        turbidity = (A_T * reflectance) / (1 - (reflectance / C_T))
    else:
        turbidity = numpy.nan
    return turbidity


def reflectance_to_turbidity_859nm(reflectance: float) -> float:
    A_T: float = 3078.9
    C_T: float = 0.2112

    turbidity: float
    if 0.0001 <= reflectance <= .08:
        turbidity: float = (A_T * reflectance) / (1 - (reflectance / C_T))
    else:
        turbidity = numpy.nan
    return turbidity


def blending_scaler(reflectance: float) -> float:
    return (reflectance - 0.05) / (0.07 - 0.05)


def turbidity_blender(reflectance_645: float, reflectance_859: float) -> float:
    scaled_reflectance_645 = reflectance_645 * 0.0001
    scaled_reflectance_859 = reflectance_859 * 0.0001

    # return reflectance_to_turbidity_645nm(scaled_reflectance_645)
    # return reflectance_to_turbidity_859nm(scaled_reflectance_859)
    if scaled_reflectance_645 < 0.05:
        return reflectance_to_turbidity_645nm(scaled_reflectance_645)
    elif 0.07 < scaled_reflectance_645:
        return reflectance_to_turbidity_859nm(scaled_reflectance_859)
    else:
        weight: float = blending_scaler(scaled_reflectance_645)
        turbidity_645: float = reflectance_to_turbidity_645nm(scaled_reflectance_645)
        turbidity_859: float = reflectance_to_turbidity_859nm(scaled_reflectance_859)
        # return numpy.nan
        if numpy.isnan(turbidity_645) or numpy.isnan(turbidity_859):
            return numpy.nan
        else:
            return ((1 - weight) * turbidity_645) \
                   + (weight * turbidity_859)


def manipulate_image():
    dataset: rasterio.DatasetReader
    with rasterio.open("modis_image.Nadir_Reflectance_Band1.tif", mode="r") as dataset1:
        band1: ndarray = dataset1.read(1)

    with rasterio.open("modis_image.Nadir_Reflectance_Band2.tif", mode="r") as dataset2:
        band2: ndarray = dataset2.read(1)

    assert len(band1) == len(band2)

    blender_vectorized = vectorize(turbidity_blender)
    blended_turbidity: ndarray = blender_vectorized(band1, band2)

    output: DatasetWriter
    with rasterio.open("processed.tif",
                       mode="w",
                       **dataset1.meta) as output:

        output.write(blended_turbidity, 1)


if __name__ == '__main__':
    fetch_data()
    # manipulate_image()
