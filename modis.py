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

ee.Initialize()


tmp_dir: Path = Path("tmp")
tmp_dir.mkdir(parents=True, exist_ok=True)


def download_url(url, save_path: Path, chunk_size=128):
    r = requests.get(url, stream=True)
    with save_path.open('wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)

    with zipfile.ZipFile(save_path, "r") as zip_ref:
        zip_ref.extractall()


class DataSets(Enum):
    modis_reflectance_500m = "MODIS/006/MCD43A4"


class ModisBands(Enum):
    band1_645nm = "Nadir_Reflectance_Band1"
    band2_859nm = "Nadir_Reflectance_Band2"


def fetch_data():
    start_date: datetime = datetime(2020, 1, 1)
    end_date: datetime = datetime(2020, 2, 1)
    # africa_geometry: Geometry = ee.Geometry.Rectangle([-22.96874821, 33.29640894, 55.07812679, -39.20562293])
    chesapeake_bay_geometry: Geometry = ee.Geometry.Rectangle([-78, 40, -75, 36])

    modis: ImageCollection = ee.ImageCollection(DataSets.modis_reflectance_500m.value) \
                               .filter(ee.Filter.date(start_date, end_date)) \
                               .filterBounds(chesapeake_bay_geometry) \
                               .select([ModisBands.band1_645nm.value, ModisBands.band2_859nm.value]) \
                               .limit(24)

    modis_image: Image = modis.first().select([ModisBands.band1_645nm.value, ModisBands.band2_859nm.value])
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
        "name": "modis_image",
        "dimensions": 768,
        "region": chesapeake_bay_geometry,
        "crs": "EPSG:3857",
        "bands": [ModisBands.band1_645nm.value, ModisBands.band2_859nm.value],
        "palette": ['blue', 'purple', 'cyan', 'green', 'yellow', 'red'],
        "scale": 30
    }

    # url: str = modis_image_reduced.getDownloadURL(image_args)
    # print(url)
    # download_url(url, tmp_dir / Path("modis_image.zip"))

    chesapeake_bay_geometry_high: Geometry = ee.Geometry.Rectangle([-77.08, 37.33, -76.9, 37.2])
    scale_m: int = 30
    export_task: Task = Export.image.toDrive(
        image=modis_image,
        folder="aquaculture/modis",
        description=f"high_resolution_modis_{scale_m}m_{datetime.now().isoformat()}",
        region=chesapeake_bay_geometry,
        crs="EPSG:3857",
        scale=scale_m,
        maxPixels=210313503
    )
    start_time: int = int(time())
    export_task.start()
    while export_task.active():
        print(export_task.status())
        sleep(1)
    end_time: int = int(time())
    completion_status: Dict = export_task.status()
    print(f"Batch Time: {end_time - start_time}")
    if completion_status["state"] == "FAILED":
        print(completion_status)
    else:
        print(completion_status["destination_uris"])

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
