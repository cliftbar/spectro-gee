import zipfile
from enum import Enum
from pathlib import Path

import requests


def download_file_from_url(url: str, save_path: Path, chunk_size=128) -> None:
    r = requests.get(url, stream=True)
    with save_path.open('wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)


def unzip_file(archive_path: Path, save_path: Path = None, delete_archive: bool = False) -> None:
    if save_path is None:
        save_path = archive_path
    with zipfile.ZipFile(archive_path, "r") as zip_ref:
        zip_ref.extractall(save_path)

    if delete_archive:
        archive_path.unlink()


class DataSets(Enum):
    modis_reflectance_500m = "MODIS/006/MCD43A4"
    landsat8_reflectance_30m = "LANDSAT/LC08/C01/T1_SR"
